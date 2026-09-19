# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Model loaders and VAE helper utilities.

Responsibilities of this file:
1. Four ``load_*`` factory functions that load the sub-models required by
   LingBot-VA training/inference (Wan2.2 VAE, UMT5 text encoder, T5 tokenizer,
   and the core ``WanTransformer3DModel``) from pretrained weight directories,
   moving them to the requested dtype/device.
2. ``patchify``: folds spatial dimensions of a video tensor into the channel
   dimension by patch_size; the generic pixel-packing front end of the Wan2.2
   VAE.
3. ``WanVAEStreamingWrapper``: a streaming (chunk-wise incremental) wrapper
   around the Wan2.2 VAE encoder. It exploits the feat_cache mechanism of the
   causal 3D convolutions to encode video chunks as they arrive, so the
   inference server can encode real observation frames online without
   re-encoding the whole history.

Position in the architecture: used by ``wan_va_server.py`` (inference service)
and the training scripts to assemble models; a support module for ``model.py``.
"""
import torch
from diffusers import AutoencoderKLWan
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    """Load the Wan2.2 video VAE (diffusers ``AutoencoderKLWan``).

    Used to encode pixel videos into latents (offline preprocessing of
    training data, encoding real observation frames at inference time) and to
    decode denoised latents back into videos.

    Args:
        vae_path: Directory of the pretrained VAE weights.
        torch_dtype: Parameter precision after loading (e.g. torch.bfloat16).
        torch_device: Target device (e.g. 'cuda').

    Returns:
        AutoencoderKLWan: The VAE moved to the requested dtype/device.
    """
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    """Load the UMT5 text encoder that turns task prompts into text embeddings.

    The resulting embeddings serve as the KV of the Transformer
    cross-attention (usually precomputed offline as text_emb during training,
    encoded online at inference).

    Args:
        text_encoder_path: Directory of the pretrained UMT5 weights.
        torch_dtype: Parameter precision.
        torch_device: Target device.

    Returns:
        UMT5EncoderModel: The text encoder moved to the requested dtype/device.
    """
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, ):
    """Load the T5-family tokenizer, used together with ``load_text_encoder``.

    Args:
        tokenizer_path: Directory of the tokenizer weights/vocabulary.

    Returns:
        T5TokenizerFast: The tokenizer instance.
    """
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, )
    return tokenizer


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    **kwargs
):
    """Load the core autoregressive diffusion Transformer (``WanTransformer3DModel``).

    Args:
        transformer_path: Directory of the pretrained transformer weights.
        torch_dtype: Parameter precision (usually bfloat16 for train/inference).
        torch_device: Target device.
        **kwargs: Extra config forwarded to ``from_pretrained`` (e.g. attn_mode).

    Returns:
        WanTransformer3DModel: The model moved to the requested dtype/device.
    """
    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
        **kwargs
    )
    return model.to(torch_device)


def patchify(x, patch_size):
    """Fold the spatial dims of a video tensor into channels by patch_size (pixel packing).

    This is the front-end preprocessing of the Wan2.2 VAE: pixels in a
    patch_size x patch_size neighborhood are concatenated into the channel
    dim, lowering the spatial resolution seen by the subsequent 3D
    convolutions and improving throughput.

    Args:
        x: Input video tensor, shape [B, C, F, H, W].
        patch_size: Spatial patch side length; returned unchanged when None or 1.

    Returns:
        torch.Tensor: Patchified tensor, shape
        [B, C*patch_size^2, F, H//patch_size, W//patch_size].
    """
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size,
               width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames,
               height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:
    """Streaming (chunk-wise incremental) wrapper around the Wan2.2 VAE encoder.

    The temporal convolutions of the Wan2.2 VAE are causal 3D convolutions
    (WanCausalConv3d) that keep the previous chunk's temporal boundary
    features in an internal ``feat_cache``. This wrapper exposes that
    mechanism: feeding only the newly arrived video chunk yields the
    corresponding latents without re-encoding the whole history — the
    inference server relies on this when encoding real observation frames
    online.

    Attributes:
        vae: The wrapped AutoencoderKLWan instance.
        encoder: The VAE's encoder submodule.
        quant_conv: Quantization conv after the encoder (produces the latent
            distribution parameters).
        enc_conv_num: Number of causal 3D conv layers in the encoder, i.e.
            the number of feat_cache slots.
        feat_cache: List of temporal boundary feature caches, one per causal
            conv layer.
    """

    def __init__(self, vae_model):
        """Initialize the wrapper, count causal conv layers, and clear feat_cache.

        Args:
            vae_model: An AutoencoderKLWan instance (its encoder must use
                causal 3D convolutions).
        """
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        # Prefer the conv-layer count cached internally by diffusers; otherwise
        # walk the encoder and count WanCausalConv3d layers manually (each
        # causal conv owns one feat_cache slot)
        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        """Reset the temporal boundary caches of all causal convs (call before encoding a new video)."""
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        """Stream-encode one video chunk and return the corresponding latents.

        Internally writes this chunk's boundary features into ``feat_cache``
        for the next chunk's causal convolutions to read, so calls must be
        made in temporal order; call ``clear_cache`` before encoding a new
        video.

        Args:
            x_chunk: Pixel video chunk, shape [B, C, F_chunk, H, W] (if the
                VAE config has a patch_size, patchify is applied first).

        Returns:
            torch.Tensor: Encoder output after quant_conv (latent distribution
            parameters).
        """
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk,
                           feat_cache=self.feat_cache,
                           feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc
