# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Flow matching noise scheduler: noising, denoising steps, and training targets/weights.

In LingBot-VA, video and action each own a **decoupled** FlowMatchScheduler instance
(video snr_shift=5.0, action snr_shift=1.0) that never interfere. Core math:

- Noising (forward process): ``x_t = (1-sigma)*x0 + sigma*eps``, with sigma in [0,1]
  linearly mapped from the timestep;
- Denoising target (velocity): ``v = eps - x0``; the model predicts v and one update is
  ``x_{t-1} = x_t + v*(sigma_{t-1} - sigma_t)``;
- SNR shift: ``sigma' = shift*sigma / (1 + (shift-1)*sigma)``; shift > 1 pushes the sigma
  distribution toward the high-noise region (video has rich detail and needs more
  sampling steps at high noise), shift = 1 means no shift (actions);
- Training mode: additionally provides bell-shaped timestep weights (bsmntw_weighing,
  Gaussian weighting) that peak at intermediate noise levels and vanish at both ends
  (near-clean / near-pure-noise).

Used by both wan_va_server.py (inference denoising loops) and train.py
(_add_noise / compute_loss).
"""
import math
import torch

class FlowMatchScheduler():
    """Flow matching scheduler: manages the discrete sigma/timestep sequence and provides noising and denoising steps.

    Attributes:
        sigmas: discrete noise-level sequence (descending, sigma_max -> sigma_min),
            already SNR-shifted.
        timesteps: ``sigmas * num_train_timesteps``, matching external timestep values.
        linear_timesteps_weights: bell-shaped timestep weights for training mode.
        training: whether in training mode (set by set_timesteps(training=True)).
    """

    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        """Initialize the scheduler and immediately build the default timestep sequence.

        Args:
            num_inference_steps: default number of inference steps (can be overridden
                later via set_timesteps).
            num_train_timesteps: total number of training timesteps;
                timestep = sigma * this value.
            shift: SNR shift coefficient (sigma' = shift*sigma/(1+(shift-1)*sigma));
                video uses 5.0, action uses 1.0.
            sigma_max / sigma_min: upper/lower bounds of the noise level.
            inverse_timesteps: whether to flip the sigma sequence (ascending).
            extra_one_step: when True, generate one extra step and drop the last entry so
                the final sigma is exactly sigma_min (enabled for all inference in this
                repo; with sigma_min=0 the last step has sigma=0, i.e. fully clean).
            reverse_sigmas: whether to use 1-sigma (reversed noise semantics).
            exponential_shift / exponential_shift_mu: use the exponential form
                sigma' = e^mu/(e^mu + 1/sigma - 1) instead of the linear shift
                (SD3-style dynamic shift).
            shift_terminal: terminal shift correction that rescales the end of the sigma
                sequence onto this value.
        """
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self,
                      num_inference_steps=100,
                      denoising_strength=1.0,
                      training=False,
                      shift=None,
                      dynamic_shift_len=None):
        """(Re)build the discrete sigma/timestep sequences; with training=True also compute bell-shaped weights.

        Args:
            num_inference_steps: number of denoising steps at inference (in training mode
                it also serves as the scale parameter of the weighting Gaussian).
            denoising_strength: starting noise-strength ratio (1.0 = start from pure
                noise; < 1 for img2img scenarios).
            training: when True, compute and store bsmntw_weighing (bell-shaped Gaussian
                timestep weights peaking at intermediate noise levels) used to weight
                the training loss.
            shift: override the instance's SNR shift coefficient.
            dynamic_shift_len: sequence length used by exponential_shift to compute mu
                dynamically.
        """
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max -
                                        self.sigma_min) * denoising_strength
        if self.extra_one_step:
            # Take one extra step then drop the last entry: guarantees the final sigma
            # is exactly sigma_min instead of overshooting past it
            self.sigmas = torch.linspace(sigma_start, self.sigma_min,
                                         num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min,
                                         num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            # Exponential (SD3-style) dynamic shift: mu is linearly interpolated from
            # the sequence length
            mu = self.calculate_shift(
                dynamic_shift_len
            ) if dynamic_shift_len is not None else self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            # SNR shift: sigma' = shift*sigma / (1 + (shift-1)*sigma).
            # shift > 1 pushes sampling points toward the high-noise region (video uses
            # 5.0); shift = 1 is the identity (actions use 1.0)
            self.sigmas = self.shift * self.sigmas / (
                1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            # Terminal correction: scale (1-sigma) globally so the sequence's final
            # sigma lands exactly on shift_terminal
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        # Timesteps map linearly to sigma: t = sigma * num_train_timesteps
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            # Bell-shaped (bsmntw) training weights: a Gaussian centered at
            # num_inference_steps/2, shifted to be non-negative and normalized so the
            # weights sum to num_inference_steps. Intermediate noise levels get the
            # largest weight -- that region has a moderate SNR and the strongest
            # learning signal, while both ends (near-clean / near-pure-noise) contribute
            # little
            x = self.timesteps
            y = torch.exp(
                -2 * ((x - num_inference_steps / 2) / num_inference_steps)**2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps /
                                           y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        """One flow-matching denoising step (Euler step).

        From x_t = (1-sigma)*x0 + sigma*eps and velocity v = eps - x0 we get
        dx/dsigma = v, hence ``x_{t-1} = x_t + v*(sigma_{t-1} - sigma_t)``
        (sigma decreases, so sigma_{t-1} - sigma_t < 0).

        Args:
            model_output: model-predicted velocity, same shape as sample.
            timestep: current timestep value (scalar or 0-d tensor); sigma is looked up
                by nearest neighbor.
            sample: current noisy sample x_t.
            to_final: when True, jump straight to the endpoint (sigma_=0, fully clean).

        Returns:
            The sample after one denoising step, x_{t-1}, same shape as sample.
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        # Nearest-neighbor lookup of the current timestep's position in the discrete sequence
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            # Already at the last step: target sigma_ is 0 (or 1 in reversed modes),
            # i.e. fully denoised
            sigma_ = 1 if (self.inverse_timesteps
                           or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stablized):
        """Back out the effective velocity at a given timestep from a "stabilized" sample (used to pull a sample back to that noise level).

        Args:
            timestep: target timestep value.
            sample: current sample (treated as the noised x_t).
            sample_stablized: the corresponding clean sample (treated as x0).

        Returns:
            The effective model_output = (x_t - x0)/sigma, ready to feed into ``step``.
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def add_noise(self, original_samples, noise, timestep, t_dim=2):
        """Forward noising: x_t = (1-sigma)*x0 + sigma*eps, supporting per-frame independent timesteps (diffusion forcing).

        Args:
            original_samples: clean samples x0, e.g. video latents [B,C,F,H,W].
            noise: Gaussian noise eps, same shape as original_samples.
            timestep: timestep tensor [T]; when T > 1 each element is an independent
                noise level for one frame.
            t_dim: the dimension along which timesteps broadcast (default 2, i.e. the
                frame dim F of [B,C,F,H,W]).

        Returns:
            The noised x_t, same shape as original_samples.
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep = timestep[None]
        # Nearest-neighbor sigma lookup per timestep, then reshape so only t_dim is
        # non-1 for broadcasting
        timestep_id = torch.argmin((self.timesteps[:, None] - timestep).abs(),
                                   dim=0)
        shape = [1] * noise.ndim
        shape[t_dim] = timestep_id.shape[0]
        sigma = self.sigmas[timestep_id].to(original_samples).view(shape)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        """Training regression target: the flow-matching velocity eps - x0.

        Args:
            sample: clean sample x0.
            noise: Gaussian noise eps.
            timestep: unused (kept for interface consistency).

        Returns:
            target = noise - sample, same shape as the inputs.
        """
        target = noise - sample
        return target

    def training_weight(self, timestep):
        """Look up the bell-shaped training weight for given timesteps (see set_timesteps(training=True)).

        Args:
            timestep: timestep tensor [T].

        Returns:
            Weight tensor [T], used to weight the per-timestep loss.
        """
        timestep_id = torch.argmin(
            (self.timesteps[:, None].to(timestep.device) -
             timestep[None]).abs(),
            dim=0)
        weights = self.linear_timesteps_weights.to(
            timestep.device)[timestep_id].to(timestep.device)
        return weights

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        """Linearly interpolate mu for the dynamic shift by sequence length (used in exponential_shift mode).

        Args:
            image_seq_len: current token sequence length.
            base_seq_len / max_seq_len: sequence lengths at the interpolation endpoints.
            base_shift / max_shift: shift values at the interpolation endpoints.

        Returns:
            float, mu = m*image_seq_len + b (longer sequences get a larger shift).
        """
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
