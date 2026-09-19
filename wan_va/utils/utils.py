# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""General utilities: RoPE grid-id generation, sequence<->patch restoration, async saving, and training helpers.

The most critical piece is ``get_mesh_id`` -- the source of RoPE position encodings for
the interleaved video/action sequence: action tokens use fractional time positions
(f + k/17) with spatial coordinates set to -1, so the 16 action sub-steps within a frame
sit precisely between the two surrounding video frames on the time axis (the literal
implementation of the paper's "interleaved sequence"). ``data_seq_to_patch`` is the
inverse transform that restores the transformer's patch-sequence output back to the
[B,C,F,H,W] latent layout used by the inference denoising loops.
"""
import concurrent.futures

import numpy as np
import torch

# Single-threaded background executor: async saving never blocks the inference main loop
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

__all__ = ['get_mesh_id', 'save_async', 'data_seq_to_patch']


def data_seq_to_patch(
    patch_size,
    data_seq,
    latent_num_frames,
    latent_height,
    latent_width,
    batch_size=1,
):
    """Restore the transformer's flattened patch sequence back into a latent tensor.

    Internally the model patchifies by (p_t, p_h, p_w) and flattens into a token
    sequence; this function is the exact inverse, matching the patch_embedding layout
    in model.py.

    Args:
        patch_size: (p_t, p_h, p_w), e.g. (1, 2, 2).
        data_seq: patch sequence [B, N_token, C_patch], where
            N_token = (F/p_t)*(H/p_h)*(W/p_w) and C_patch = C*p_t*p_h*p_w.
        latent_num_frames / latent_height / latent_width: target latent F/H/W.
        batch_size: batch size (2 when CFG is enabled).

    Returns:
        torch.Tensor of shape [B, C, F, H, W].
    """
    p_t, p_h, p_w = patch_size
    post_patch_num_frames = latent_num_frames // p_t
    post_patch_height = latent_height // p_h
    post_patch_width = latent_width // p_w

    # First reshape into a (frames, height, width, p_t, p_h, p_w, channels) grid, then
    # permute/flatten to merge the intra-patch dims back into the spatial and channel
    # dims, recovering the [B,C,F,H,W] layout
    data_patch = data_seq.reshape(batch_size, post_patch_num_frames,
                                  post_patch_height, post_patch_width, p_t,
                                  p_h, p_w, -1)
    data_patch = data_patch.permute(0, 7, 1, 4, 2, 5, 3, 6)
    data_patch = data_patch.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return data_patch


def get_mesh_id(f, h, w, t, f_w=1, f_shift=0, action=False):
    """Generate the RoPE grid ids (four rows: time / height / width / modality).

    Video tokens: integer time positions frame_st_id..frame_st_id+f-1 and spatial
    positions (hh, ww). Action tokens (action=True): **fractional** time positions
    f_shift + k/(h+1) (i.e. f + 1/17, 2/17, ..., 16/17), placing the 16 action sub-steps
    of a frame precisely between the two surrounding video frames; spatial coordinates
    are set to -1 (actions have no spatial position and take no part in spatial RoPE).

    Args:
        f / h / w: temporal/height/width sizes of the token grid (post-patch sizes for
            video; for actions h = action_per_frame).
        t: modality tag (video=0, action=1), concatenated as the 4th row of grid_id.
        f_w: temporal step coefficient.
        f_shift: temporal start offset (i.e. the server's frame_st_id).
        action: whether these are action tokens (enables fractional time positions and
            -1 spatial coordinates).

    Returns:
        torch.Tensor of shape [4, f*h*w], rows being (ff, hh, ww, t).
    """
    f_idx = torch.arange(f_shift, f + f_shift) * f_w
    h_idx = torch.arange(h)
    w_idx = torch.arange(w)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing='ij')
    if action:
        # Fractional time offsets k/(h+1), k=1..h: action sub-steps land between
        # adjacent video frames
        ff_offset = (torch.ones([h]).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + ff_offset
        # Spatial coordinate -1: marks "no spatial position", distinguishing action
        # tokens from video tokens
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1

    grid_id = torch.cat(
        [
            ff.unsqueeze(0),
            hh.unsqueeze(0),
            ww.unsqueeze(0),
        ],
        dim=0,
    ).flatten(1)
    # Append the modality tag t as the 4th row so the model can tell the video/action
    # streams apart
    grid_id = torch.cat([grid_id, torch.full_like(grid_id[:1], t)], dim=0)
    return grid_id


def save_async(obj, file_path):
    """
    todo

    Asynchronously save an object (torch.save / np.save) on a background single thread,
    without blocking the inference main loop. CUDA tensors are first copied back to CPU
    (dicts are copied value-by-value) before being submitted to the executor, avoiding
    race conditions.

    Args:
        obj: object to save (tensor / ndarray / dict of tensors / any picklable object).
        file_path: destination file path.
    """
    if torch.is_tensor(obj) or (isinstance(obj, dict) and any(
            torch.is_tensor(v) for v in obj.values())):
        if torch.is_tensor(obj):
            if obj.is_cuda:
                obj = obj.cpu()
        elif isinstance(obj, dict):
            obj = {
                k: v.cpu() if torch.is_tensor(v) else v
                for k, v in obj.items()
            }
        executor.submit(torch.save, obj, file_path)
    elif isinstance(obj, np.ndarray):
        obj_copy = obj.copy()
        executor.submit(np.save, file_path, obj_copy)
    else:
        executor.submit(torch.save, obj, file_path)

def sample_timestep_id(
    batch_size: int = 1,
    min_timestep_bd: float = 0.0,
    max_timestep_bd: float = 1.0,
    num_train_timesteps: int = 1000,
):
    """Uniformly sample timestep ids in [min_timestep_bd, max_timestep_bd).

    During training this is called with batch_size=F (the frame count), i.e. **each
    latent frame independently samples its own timestep** -- the key implementation of
    diffusion forcing (see train.py:_add_noise).

    Args:
        batch_size: number of samples.
        min_timestep_bd / max_timestep_bd: sampling interval bounds (normalized to [0,1]).
        num_train_timesteps: total number of timesteps (determines the id upper bound).

    Returns:
        torch.LongTensor [batch_size] with values in [0, num_train_timesteps-1].
    """
    u = torch.rand(size=[batch_size])
    u = u * (max_timestep_bd - min_timestep_bd) + min_timestep_bd
    timestep_id = (u * num_train_timesteps).clamp(min=0, max=num_train_timesteps - 1).to(torch.int64)
    return timestep_id


def warmup_constant_lambda(current_step, warmup_steps=1000):
    """Warmup-then-constant lambda function for lr_scheduler.LambdaLR.

    Args:
        current_step: current training step.
        warmup_steps: number of linear warmup steps.

    Returns:
        float: current_step/warmup_steps during warmup (ramping linearly to 1),
        constant 1.0 afterwards.
    """
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0