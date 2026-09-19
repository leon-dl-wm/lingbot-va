# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""LingBot-VA core model: an autoregressive diffusion Transformer built on the Wan2.2 video DiT.

This file is the algorithmic core of the whole repository, implementing the
unified video-action interleaved-sequence model for "world modeling
(predicting future video) + robot action generation". Along the time axis the
sequence reads v0 -> a0 -> v1 -> a1 ...:
- video latent [B,48,F,H,W] -> patchify(1,2,2) + Linear(48*4 -> 3072) = video tokens;
- action [B,30,F,16,1] -> Linear(30 -> 3072) = action tokens
  (30 = unified action-space dim, 16 = action_per_frame, i.e. 16 control
  sub-steps per latent frame).

Key classes:
- ``FlexAttnFunc``: training-time FlexAttention wrapper. ``init_mask`` assigns
  three ids (frame/noise/seq) to every token of the concatenated sequence
  [noisy_video | clean_video | noisy_action | clean_action];
  ``_get_mask_mod`` combines them into the structured mask implementing all
  attention constraints of "block-causal AR + Diffusion Forcing + sliding
  window".
- ``WanTimeTextImageEmbedding``: sinusoidal timestep encoding + MLP producing
  per-token scale-shift modulation vectors (video/action each own an
  independent instance = the MoT dual-stream modulation).
- ``WanRotaryPosEmbed``: 3D (f/h/w) RoPE. Action tokens use fractional time
  positions f + k/17 (k=1..16) with hh=ww=-1, placing them exactly between
  the two surrounding video frames.
- ``WanAttention``: self-/cross-attention plus the three-state KV-cache pool
  state machine (update_cache=0 write-then-rollback / =1 commit imagined
  frames / =2 commit real observations). Inference uses no mask; causality is
  fully guaranteed by the cache contents.
- ``WanTransformerBlock``: MoT dual-stream Transformer block; video/action
  share attention weights but get independent timestep modulation.
- ``WanTransformer3DModel``: top-level model. ``forward_train`` is the
  training path (one big sequence + FlexAttention mask); ``forward`` is the
  inference path (KV cache + separate video/action modes);
  ``create_empty_cache`` & friends manage the cache-pool lifecycle.
"""
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from typing import Callable, ClassVar
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
    and_masks,
    or_masks
)
from functools import partial

try:
    from flash_attn_interface import flash_attn_func
except Exception:
    try:
        from flash_attn import flash_attn_func
    except Exception:
        # flash-attn is only needed for attn_mode='flashattn'. Training uses
        # 'flex' and inference can use 'torch', so keep the import optional
        # (e.g. on aarch64/Blackwell where prebuilt flash-attn is unavailable).
        flash_attn_func = None

__all__ = ['WanTransformer3DModel']


def custom_sdpa(q, k, v):
    """Mask-free SDPA attention (used when attn_mode='torch', inference path).

    Inputs/output all use the flash-attn style [B, S, N, D] layout; internally
    transposed to the [B, N, S, D] layout required by PyTorch SDPA and back,
    keeping the same interface as flash_attn_func.

    Args:
        q: query, shape [B, S, N, D].
        k: key, shape [B, S_kv, N, D].
        v: value, shape [B, S_kv, N, D].

    Returns:
        torch.Tensor: attention output, shape [B, S, N, D].
    """
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2))
    return out.transpose(1, 2)

class FlexAttnFunc(nn.Module):
    """Training-time FlexAttention wrapper (attn_mode='flex').

    The compiled ``flex_attention`` / ``create_block_mask`` callables and the
    structured BlockMasks are cached as **class variables**: the sequence
    layout is fixed for the whole training run, so the mask is built once in
    ``init_mask`` and shared by all layers and all steps, avoiding repeated
    compilation.

    ClassVar:
        flex_attn: torch.compile'd flex_attention kernel.
        compiled_create_block_mask: torch.compile'd BlockMask builder.
        attention_mask: self-attention BlockMask (written by init_mask).
        cross_attention_mask: text cross-attention BlockMask (written by init_mask).
    """
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, dynamic=True, 
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    attention_mask: ClassVar[BlockMask] = None
    cross_attention_mask: ClassVar[BlockMask] = None

    def __init__(
        self, 
        is_cross=False,
    ) -> None:
        """Initialize.

        Args:
            is_cross: True for text cross-attention (uses
                cross_attention_mask), False for self-attention (uses
                attention_mask).
        """
        super().__init__()
        self.is_cross = is_cross
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        """Run FlexAttention with the cached BlockMask.

        Args:
            query: shape [B, S, N, D] (during training B=1, the whole batch
                is flattened into the sequence dim S).
            key: shape [B, S_kv, N, D].
            value: shape [B, S_kv, N, D].
            dtype: target half-precision type (fp16/bf16), required by the
                flex_attention kernel.

        Returns:
            torch.Tensor: attention output, shape [B, S, N, D].
        """
        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        # the flex_attention kernel only supports half precision; cast all
        # three operands to the same dtype
        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)
        
        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        # pick the class-level BlockMask cache for self- vs cross-attention
        block_mask = FlexAttnFunc.cross_attention_mask if self.is_cross else FlexAttnFunc.attention_mask

        # kernel_options manually pin the block sizes, tuned for this model's
        # sequence length / head dim
        x_out = FlexAttnFunc.flex_attn(q_varlen, k_varlen, v_varlen, block_mask=block_mask, kernel_options = {
                                                    "BLOCK_M": 64,
                                                    "BLOCK_N": 64,
                                                    "BLOCK_M1": 32,
                                                    "BLOCK_N1": 64,
                                                    "BLOCK_M2": 64,
                                                    "BLOCK_N2": 32,
                                                })

        x_out = rearrange(x_out, "b n s d -> b s n d")
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_mask(
        latent_shape, 
        action_shape, 
        padded_length, 
        chunk_size,
        window_size,
        patch_size,
        device,
    ):
        """Build the FlexAttention BlockMasks (self + cross) for the training sequence.

        The training sequence layout is [noisy_video | clean_video |
        noisy_action | clean_action] (batch flattened into the sequence, tail
        padded to a multiple of 128). This function assigns three ids per
        token, which ``_get_mask_mod`` then combines into the structured mask:
        - seq_ids: sample index, isolating different batch samples flattened
          into the same sequence (padding = -1);
        - frame_ids: causal time-axis index. Video chunk k -> 2k, action chunk
          k -> 2k+1, enforcing the causal order v_k before a_k before
          v_{k+1} (actions can see the future frames of their own chunk; the
          next video chunk can see past actions);
        - noise_ids: noisy segment = 0 / clean segment = 1, distinguishing
          "tokens being denoised" from "clean history used as condition".

        Results are stored in the class variables ``attention_mask`` /
        ``cross_attention_mask`` and shared globally.

        Args:
            latent_shape: noisy video latent shape [B, C, L_F, L_H, L_W].
            action_shape: noisy action shape [B, C, A_F, A_H, A_W].
            padded_length: number of padding tokens appended (128 alignment).
            chunk_size: AR chunk size (latent frames per chunk, randomly 1..4
                during training).
            window_size: sliding attention window |Δframe| ≤ window_size
                (randomly sampled during training).
            patch_size: video patchify sizes (p1, p2, p3).
            device: device of the mask tensors.
        """
        torch._inductor.config.realize_opcount_threshold = 100
        B, _, L_F, L_H, L_W = latent_shape
        _, _, A_F, A_H, A_W = action_shape

        # --- seq_ids: which batch sample each token belongs to ---
        # video tokens are enumerated over the patchified (F, H/p2, W/p3)
        # grid; noisy and clean segments share the same layout, hence the
        # latent part is repeated twice (same for actions)
        latent_seq_id = torch.arange(B)[:, None, None, None].\
            expand(-1, L_F // patch_size[0], L_H // patch_size[1], L_W // patch_size[2]).flatten()
        action_seq_id = torch.arange(B)[:, None, None, None].expand(-1, A_F, A_H, A_W).flatten()
        seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)

        # --- frame_ids: causal time-axis indices ---
        # video frame f belongs to chunk f//chunk_size and gets index 2k;
        # action frames get 2k+1, so within one chunk the action sits exactly
        # after the video (the key to the interleaved causal order)
        latent_frame_id = torch.arange(L_F)[None, :, None, None].expand(B, -1, L_H // patch_size[1], L_W // patch_size[2])[None].flatten()
        action_frame_id = torch.arange(A_F)[None, :, None, None].expand(B, -1, A_H, A_W)[None].flatten()
        frame_ids = torch.cat([latent_frame_id // chunk_size * 2] * 2 + [action_frame_id // chunk_size * 2 + 1] * 2)

        # --- noise_ids: mark noisy(0)/clean(1) following the 4-segment layout ---
        noise_ids = torch.cat(
            [
                torch.zeros_like(latent_frame_id),
                torch.ones_like(latent_frame_id),
                torch.zeros_like(action_frame_id),
                torch.ones_like(action_frame_id),
            ]
        )

        # padding tokens get -1 in all three ids and are excluded by seq_mask
        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)

        # combine the mask rules and compile into a BlockMask (B=1, H=1:
        # shared across all batch samples and heads)
        mask_mod = FlexAttnFunc._get_mask_mod(seq_ids.long().to(device), frame_ids.long().to(device), noise_ids.long().to(device), window_size)
        block_mask = FlexAttnFunc.compiled_create_block_mask(
                mask_mod, 1, 1, len(seq_ids), len(seq_ids), device=device, _compile=True
            )
        FlexAttnFunc.attention_mask = block_mask

        # --- cross-attention mask: text KV is fixed at 512 tokens, so only
        # seq_id alignment is needed — video/action tokens of sample b attend
        # only to text prompt b ---
        text_seq_ids = torch.arange(B)[:, None].expand(-1, 512).flatten()
        mask_mod_cross = FlexAttnFunc._get_cross_mask_mod(seq_ids.long().to(device), text_seq_ids.long().to(device))
        block_mask_cross = FlexAttnFunc.compiled_create_block_mask(
                mask_mod_cross, 1, 1, len(seq_ids), len(text_seq_ids), device=device, _compile=True
            )
        FlexAttnFunc.cross_attention_mask = block_mask_cross
    
    @staticmethod
    @torch.no_grad()
    def _get_cross_mask_mod(seq_ids, text_seq_ids):
        """Build the text cross-attention mask_mod: alignment by sample identity only.

        Args:
            seq_ids: sample index of every main-sequence token, padding = -1,
                shape [S].
            text_seq_ids: sample index of every text KV token, shape [B*512].

        Returns:
            Callable: FlexAttention mask_mod(b, h, q_idx, kv_idx) -> bool;
            the rule is that query and key belong to the same sample and
            neither side is padding.
        """
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_ids[q_idx] == text_seq_ids[kv_idx]) & (seq_ids[q_idx] >=0 ) & (text_seq_ids[kv_idx] >= 0)
        return seq_mask
    
    @staticmethod
    @torch.no_grad()
    def _get_mask_mod(seq_ids, frame_ids, noise_ids, window_size):
        """Build the structured self-attention mask_mod (the core of this file: causal AR + Diffusion Forcing).

        The mask is the OR of three "noise-segment combination" rules, then
        AND-ed with sample isolation and the sliding window:

        | query -> key    | rule                    | semantics |
        |-----------------|-------------------------|-----------|
        | clean -> clean  | frame_kv <= frame_q     | block-causal among clean history |
        | noisy -> clean  | frame_kv <  frame_q     | strictly causal, **excludes the clean segment of the current chunk**, otherwise denoising would directly see the answer (leakage) |
        | noisy -> noisy  | frame_kv == frame_q     | bidirectional only within the same chunk (joint denoising) |
        | all             | AND \\|Δframe\\| <= window AND same seq | sliding window + batch-sample isolation |

        Note that clean -> noisy appears in no rule, i.e. the clean segment
        never sees the noisy segment — clean-history representations do not
        depend on the current noise, which strictly matches the inference
        semantics where the KV cache only stores clean frames.

        Args:
            seq_ids: sample index per token (padding = -1), shape [S].
            frame_ids: causal frame index per token (video chunk k -> 2k,
                action chunk k -> 2k+1), shape [S].
            noise_ids: noisy = 0 / clean = 1 (padding = -1), shape [S].
            window_size: sliding-window radius (in frame_id distance).

        Returns:
            Callable: the combined FlexAttention mask_mod.
        """
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # sample isolation: different batch samples are flattened into the
            # same sequence and must not see each other; padding tokens
            # (id = -1) are excluded by this rule as well
            return (seq_ids[q_idx] == seq_ids[kv_idx]) & (seq_ids[q_idx] >=0 ) & (seq_ids[kv_idx] >= 0)
        
        def block_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # block-causal: may attend to the same chunk and earlier ones (<=)
            return (frame_ids[kv_idx] <= frame_ids[q_idx])
        
        def block_causal_mask_exclude_self(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # strictly block-causal: only earlier chunks (<), self chunk excluded
            return (frame_ids[kv_idx] < frame_ids[q_idx])
        
        def block_self_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # same chunk only: tokens within one chunk see each other bidirectionally
            return (frame_ids[kv_idx] == frame_ids[q_idx])
        
        def clean2clean_mask(
                b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # both query and key are in the clean segment (noise_id = 1)
            return (noise_ids[q_idx] == 1) & (noise_ids[kv_idx] == 1)
        
        def noise2clean_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # query in the noisy segment (denoising target), key in the clean
            # segment (history condition)
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 1)
        def noise2noise_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            # both query and key in the noisy segment (joint denoising of one chunk)
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 0)
        
        def block_window_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor, window_size: int
        ):
            # sliding window: bound the attention span |Δframe| <= window,
            # mirroring the finite capacity of the KV-cache pool at inference
            # (hardware-level sliding window)
            return ((frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size)

        # OR of the three combination rules: a noisy token may only see
        # "earlier clean + same-chunk noisy"
        mask_list = []
        mask_list.append(and_masks(clean2clean_mask, block_causal_mask))
        mask_list.append(and_masks(noise2clean_mask, block_causal_mask_exclude_self))
        mask_list.append(and_masks(noise2noise_mask, block_self_mask))
        mask = or_masks(*mask_list)
        # AND with the two global constraints: sample isolation + sliding window
        mask = and_masks(mask, seq_mask)
        mask = and_masks(mask, partial(block_window_mask, window_size=window_size))
        return mask
       
class WanTimeTextImageEmbedding(nn.Module):
    """Timestep / text conditioning embedding module.

    Timestep path: scalar t -> sinusoidal frequency encoding (Timesteps) ->
    MLP (TimestepEmbedding) -> temb, then SiLU + Linear projects it into 6
    groups of scale-shift-gate modulation parameters (time_proj_dim = 6*dim).
    Because Diffusion Forcing gives **every token its own timestep**
    (per-frame independent noise), the input is per-token [B, L] instead of
    per-sample [B].

    Text path: text_embedder projects the UMT5 output (4096-dim) to the model
    dim, serving as the KV of cross-attention.

    MoT dual stream: the model holds two independent instances of this class
    (condition_embedder for video, condition_embedder_action for action), so
    video/action each learn their own noise-level modulation while attention
    weights remain shared.
    """

    def __init__(
        self,
        dim,
        time_freq_dim,
        time_proj_dim,
        text_embed_dim,
        pos_embed_seq_len,
    ):
        """Initialize the embedding sub-layers.

        Args:
            dim: model hidden dim (inner_dim = 3072).
            time_freq_dim: frequency dim of the sinusoidal timestep encoding.
            time_proj_dim: projection dim of the modulation parameters (= 6*dim).
            text_embed_dim: output dim of the text encoder (4096 for UMT5).
            pos_embed_seq_len: reserved positional-embedding sequence length
                (currently unused).
        """
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim,
                                        flip_sin_to_cos=True,
                                        downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim,
                                               time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim,
                                                       dim,
                                                       act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        dtype=None,
    ):
        """Encode per-token timesteps into modulation vectors.

        Args:
            timestep: per-token diffusion timesteps, shape [B, L].
            dtype: target precision of temb (usually matches hidden_states,
                e.g. bf16).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - temb: [B, L, dim], used by the output layer's scale-shift;
            - timestep_proj: [B, L, 6*dim], source of the 6 modulation groups
              inside each block.
        """
        B, L = timestep.shape
        timestep = timestep.reshape(-1)
        timestep = self.timesteps_proj(timestep)
        # time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(B, L, -1), timestep_proj.reshape(B, L, -1)


class WanRotaryPosEmbed(nn.Module):
    """3D rotary position embedding (RoPE) producing complex frequencies from (frame, height, width) grid ids.

    head_dim is split into three sections encoding the f/h/w coordinate axes
    respectively (f_dim + h_dim + w_dim = head_dim). Each token's position is
    given by the externally provided grid_ids [3, L] (rows are the f/h/w
    coordinates):
    - video tokens: integer frame index + spatial coordinates on the patch grid;
    - action tokens: **fractional time positions** f + k/17 (k=1..16, produced
      by utils.get_mesh_id) with hh=ww=-1 (no spatial position). The
      fractional positions place the 16 action sub-steps of one frame exactly
      between the two surrounding video frames — the literal implementation,
      at the positional-encoding level, of the interleaved sequence
      v0 -> a0 -> v1.

    Since RoPE depends only on coordinate differences, cached history tokens
    need no position recomputation at inference time.
    """
    def __init__(
        self,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        """Initialize and precompute the frequency bases of the f/h/w sections.

        Args:
            attention_head_dim: dim per attention head (split into f/h/w sections).
            patch_size: video patchify sizes (kept for reference).
            max_seq_len: maximum supported sequence length (kept for reference).
            theta: RoPE frequency base.
        """
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        # Precompute and register buffers
        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()
        self.f_freqs_base = f_freqs_base
        self.h_freqs_base = h_freqs_base
        self.w_freqs_base = w_freqs_base

    def _precompute_freqs_base(self):
        """Precompute the RoPE frequency bases 1/(theta^(2k/dim)) for the f/h/w sections.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: frequency bases of the three
            sections (double precision, dims f_dim//2, h_dim//2, w_dim//2).
        """
        # freqs_base = 1.0 / (theta ** (2k / dim))
        f_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.f_dim, 2)[:(self.f_dim // 2)].double() / self.f_dim))
        h_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.h_dim, 2)[:(self.h_dim // 2)].double() / self.h_dim))
        w_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.w_dim, 2)[:(self.w_dim // 2)].double() / self.w_dim))
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids):
        """Generate complex-valued rotation frequencies from grid coordinates.

        Args:
            grid_ids: (f, h, w) coordinates of the tokens, shape [3, L]. For
                action tokens the f row may be fractional (e.g. f + k/17) and
                the h/w rows are -1.

        Returns:
            torch.Tensor: complex frequencies freqs_cis, shape [L, head_dim//2],
            multiplied with paired dims of q/k to apply the rotation.
        """
        with torch.no_grad():
            # coordinate x frequency base -> phase angle; computed in double
            # precision so phases stay accurate for long sequences and
            # fractional coordinates
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(grid_ids.device)
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(grid_ids.device)
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(grid_ids.device)
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            # complex form of e^{i*theta}; the rotation is applied later via
            # complex multiplication
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis


class WanAttention(torch.nn.Module):
    """Attention module (self / text-cross) plus the inference-time KV-cache pool.

    Self-attention (cross_attention_dim_head=None) additionally maintains
    ``attn_caches``: a fixed-capacity, token-level KV pool which, together
    with three auxiliary arrays, forms a three-state state machine — inference
    uses no mask at all; causality is fully guaranteed by *what is in the
    pool*:

    - ``update_cache=0``: write -> attention -> rollback (restore_cache). Used
      for intermediate denoising steps: the temporary KV takes part in this
      attention call but does not pollute the long-term history;
    - ``update_cache=1``: commit with is_pred=True. After a chunk finishes
      denoising, the "imagined future frames" are written so the subsequent
      action denoising can read them (imagine first, then act);
    - ``update_cache=2`` (any value other than 0/1): commit with
      is_pred=False, writing real observation frames (the server's
      _compute_kv_cache path).

    Pool structure (k/v are [B, total_tolen, N, D]):
    - mask[total_tolen]: whether a slot is occupied;
    - id[total_tolen]: write generation (monotonically increasing); when the
      pool is full the oldest entries by id are evicted — a hardware-level
      sliding window;
    - is_pred[total_tolen]: whether the slot holds an imagined frame; when a
      real observation arrives, clear_pred_cache drops all imagined frames at
      once (closed-loop error correction).

    For cross-attention (cross_attention_dim_head not None) attn_caches is
    always None: text KV is recomputed every call and never cached.
    """

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        eps=1e-5,
        dropout=0.0,
        cross_attention_dim_head=None,
        attn_mode='torch',
    ):
        """Initialize QKV projections, QK-RMSNorm, and the attention operator.

        Args:
            dim: input/output hidden dim.
            heads: number of attention heads.
            dim_head: dim per head (inner_dim = dim_head * heads).
            eps: RMSNorm numerical-stability term.
            dropout: output dropout probability.
            cross_attention_dim_head: when not None this is cross-attention
                (text KV) and no KV cache is created.
            attn_mode: 'torch' (SDPA, inference) / 'flashattn' (inference) /
                'flex' (FlexAttention + structured mask, training).
        """
        super().__init__()
        if attn_mode == 'torch':
            self.attn_op = custom_sdpa
        elif attn_mode == 'flashattn':
            if flash_attn_func is None:
                raise ImportError(
                    "attn_mode='flashattn' requires flash-attn, which is not installed. "
                    "Install flash-attn or set attn_mode to 'torch' (inference) / 'flex' (training)."
                )
            self.attn_op = flash_attn_func
        elif attn_mode == 'flex':
            self.attn_op = FlexAttnFunc(cross_attention_dim_head is not None)
        else:
            raise ValueError(
                f"Unsupported attention mode: {attn_mode}, only support torch and flashattn"
            )

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList([
            torch.nn.Linear(self.inner_dim, dim, bias=True),
            torch.nn.Dropout(dropout),
        ])
        self.norm_q = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        # only self-attention keeps a KV-cache pool; cross-attention (text) is always None
        self.attn_caches = {} if cross_attention_dim_head is None else None

    def clear_pred_cache(self, cache_name):
        """Drop all "imagined frames" (slots with is_pred=True) from the cache pool.

        Called by the server when a real observation arrives: imagined future
        frames only serve as conditioning for action denoising and never enter
        the long-term history — after this one-shot clear they are replaced by
        real frames, providing closed-loop error correction that prevents
        error accumulation over long autoregressive rollouts.

        Args:
            cache_name: name of the cache pool (e.g. 'pos').
        """
        if self.attn_caches is None:
            return
        cache = self.attn_caches[cache_name]
        is_pred = cache['is_pred']
        cache['mask'][is_pred] = False

    def clear_cache(self, cache_name):
        """Clear the given cache pool entirely (called on session reset).

        Args:
            cache_name: name of the cache pool.
        """
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = None

    def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim,
                      device, dtype, batch_size):
        """Allocate the fixed-capacity KV-cache pool and its three auxiliary arrays.

        The pool capacity total_tolen is computed by ``create_empty_cache``
        from the sliding-window budget:
        (attn_window//2) * (video_chunk_tokens + action_chunk_tokens).

        Args:
            cache_name: name of the cache pool.
            total_tolen: total number of token slots in the pool.
            num_head: number of attention heads.
            head_dim: dim per head.
            device: tensor device.
            dtype: tensor precision (matches inference precision, e.g. bf16).
            batch_size: batch size (includes both conditional/unconditional
                copies under CFG).
        """
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = {
            'k':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'v':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            # id: write generation, initially -1; eviction picks the oldest ids
            'id':
            torch.full((total_tolen, ), -1, device=device),
            # mask: slot-occupancy flags
            "mask":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
            # is_pred: whether the slot holds an "imagined future frame"
            "is_pred":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
        }

    def allocate_slots(self, cache_name, key_size):
        """Allocate key_size free slots in the pool, evicting the oldest entries by id when full.

        Evicting the oldest = a hardware-level sliding window: history frames
        outside the attention window are naturally pushed out, matching the
        training-time mask semantics |Δframe| <= attn_window.

        Args:
            cache_name: name of the cache pool.
            key_size: number of tokens to write this time.

        Returns:
            torch.Tensor: indices of the allocated slots, shape [key_size].
        """
        cache = self.attn_caches[cache_name]
        mask = cache["mask"]
        ids = cache["id"]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            # not enough free slots: sort occupied slots by write generation
            # (id, ascending) and evict the oldest `need` of them (they are
            # necessarily outside the attention window anyway)
            used = mask.nonzero(as_tuple=False).squeeze(-1)

            used_ids = ids[used]
            order = torch.argsort(used_ids)
            need = key_size - free.numel()
            to_free = used[order[:need]]

            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        assert free.numel() >= key_size
        return free[:key_size]

    def _next_cache_id(self, cache_name):
        """Generate the write-generation id for this write (max id + 1; 0 for an empty pool).

        Args:
            cache_name: name of the cache pool.

        Returns:
            torch.Tensor: scalar generation id.
        """
        ids = self.attn_caches[cache_name]['id']
        mask = self.attn_caches[cache_name]['mask']

        if mask.any():
            return ids[mask].max() + 1
        else:
            return torch.tensor(0, device=ids.device, dtype=ids.dtype)

    def update_cache(self, cache_name, key, value, is_pred):
        """Write a batch of K/V into the cache pool (commit or temporary write, per caller semantics).

        Args:
            cache_name: name of the cache pool.
            key: K to write, shape [B, S, N, D] (already rotated by RoPE).
            value: V to write, shape [B, S, N, D].
            is_pred: True for imagined frames (update_cache=1), False for real
                observations (update_cache=2) or temporary writes of
                intermediate denoising steps (update_cache=0, rolled back
                right after via restore_cache).

        Returns:
            torch.Tensor: indices of the written slots, for restore_cache rollback.
        """
        cache = self.attn_caches[cache_name]

        key_size = key.shape[1]
        slots = self.allocate_slots(cache_name, key_size)

        new_id = self._next_cache_id(cache_name)

        cache['k'][:, slots] = key
        cache['v'][:, slots] = value
        cache['mask'][slots] = True
        cache['id'][slots] = new_id
        cache['is_pred'][slots] = is_pred
        return slots

    def restore_cache(self, cache_name, slots):
        """Rollback: release the given slots (called after attention for update_cache=0 intermediate steps).

        Only the mask flags are cleared, not the data, so slots can be reused
        immediately — this guarantees the intermediate steps of the denoising
        loop never leave noisy KV in the long-term history.

        Args:
            cache_name: name of the cache pool.
            slots: slot indices returned by update_cache.
        """
        self.attn_caches[cache_name]['mask'][slots] = False

    def forward(
        self,
        q,
        k,
        v,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
    ):
        """Attention forward: QKV projection -> QK-norm -> RoPE -> (KV-cache read/write) -> attention.

        Args:
            q: query input, shape [B, S, C].
            k: key input, shape [B, S_kv, C] (text embeddings for cross-attention).
            v: value input, shape [B, S_kv, C].
            rotary_emb: complex RoPE frequencies [S, D//2, 1]; None for cross-attention.
            update_cache: three-state KV-cache semantics — 0: write ->
                attention -> rollback (intermediate denoising step); 1: commit
                with is_pred=True (imagined frame); other (2): commit with
                is_pred=False (real observation).
            cache_name: name of the cache pool.

        Returns:
            torch.Tensor: attention output, shape [B, S, C].
        """
        kv_cache = self.attn_caches[
            cache_name] if (self.attn_caches is not None) and (cache_name in self.attn_caches) else None

        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query)
        query = query.unflatten(2, (self.heads, -1))
        key = self.norm_k(key)
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:

            def apply_rotary_emb(x, freqs):
                # rotation via complex multiplication: view each pair of
                # adjacent dims as one complex number and multiply by e^{i*theta}.
                # float64 intermediate precision avoids phase-error
                # accumulation over long sequences
                x_out = torch.view_as_complex(
                    x.to(torch.float64).reshape(x.shape[0], x.shape[1],
                                                x.shape[2], -1, 2))
                x_out = torch.view_as_real(x_out * freqs).flatten(3)
                return x_out.to(x.dtype)
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        slots = None
        if kv_cache is not None and kv_cache['k'] is not None:
            # inference path: first write the current K/V into the pool (the
            # absolute RoPE phases are already baked into `key`, so history
            # tokens need no recomputation), then attend over *all valid
            # slots in the pool* — no mask is used; causality is guaranteed
            # by the pool contents themselves
            slots = self.update_cache(cache_name,
                                      key,
                                      value,
                                      is_pred=(update_cache == 1))
            key_pool = self.attn_caches[cache_name]['k']
            value_pool = self.attn_caches[cache_name]['v']
            mask = self.attn_caches[cache_name]['mask']
            valid = mask.nonzero(as_tuple=False).squeeze(-1)
            key = key_pool[:, valid]
            value = value_pool[:, valid]

        hidden_states = self.attn_op(query, key, value)

        if update_cache == 0:
            # intermediate denoising step: roll back this write so the
            # long-term history is not polluted
            if kv_cache is not None and kv_cache['k'] is not None:
                self.restore_cache(cache_name, slots)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class WanTransformerBlock(nn.Module):
    """MoT (Mixture-of-Transformers) dual-stream Transformer block.

    Structure: self-attention (attn1, optionally with KV cache) -> text
    cross-attention (attn2) -> FFN, each with a residual connection. Video
    and action tokens **share all attention/FFN weights**, but their timestep
    modulation parameters come per-token from their respective
    condition_embedder (condition_embedder for video,
    condition_embedder_action for action), so both token types in the same
    interleaved sequence receive independent scale-shift-gate modulation —
    this is the MoT dual stream: modality-split modulation parameters +
    shared sequence mixing.

    There are 6 modulation groups in total (shift_msa/scale_msa/gate_msa for
    self-attention, c_shift_msa/c_scale_msa/c_gate_msa for the FFN), obtained
    by adding scale_shift_table (a learnable base table) to temb (the
    per-token timestep embedding).
    """

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode: str = "flashattn",
    ):
        """Initialize the block's sub-layers.

        Args:
            dim: hidden dim.
            ffn_dim: FFN inner dim.
            num_heads: number of attention heads.
            cross_attn_norm: whether to add an affine LayerNorm (norm2) before
                cross-attention.
            eps: numerical-stability term for LayerNorm/RMSNorm.
            attn_mode: attention implementation ('torch'/'flashattn'/'flex').
        """
        super().__init__()
        self.attn_mode = attn_mode

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode=attn_mode,
        )
        self.norm2 = FP32LayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim,
                               inner_dim=ffn_dim,
                               activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # learnable base table of the 6 modulation groups; added to the
        # per-token temb, then chunked into shift/scale/gate (self-attention)
        # and c_shift/c_scale/c_gate (FFN)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
    ) -> torch.Tensor:
        """Block forward: modulated self-attention -> text cross-attention -> modulated FFN.

        Args:
            hidden_states: video/action token sequence, shape [B, L, C]
                (during training B=1 with the whole batch flattened into L).
            encoder_hidden_states: text embeddings (KV of cross-attention),
                shape [B, L_text, C].
            temb: per-token timestep modulation parameters, shape [B, L, 6*C]
                (video/action tokens come from their independent
                condition_embedders).
            rotary_emb: complex RoPE frequencies; not used by cross-attention.
            update_cache: three-state KV-cache semantics (0/1/2) forwarded to
                self-attention.
            cache_name: name of the KV-cache pool.

        Returns:
            torch.Tensor: updated hidden_states, shape [B, L, C].
        """
        # produce the 6 per-token modulation groups: base table + temb, then
        # split as [B, 6, L, C]
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
            rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(6, dim=1)
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)
        # 1. Self-attention
        # adaLN modulation: affine the normalized tokens by per-token
        # scale/shift; the output is gated and added residually
        norm_hidden_states = (self.norm1(hidden_states.float()) *
                              (1. + scale_msa) +
                              shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states,
                                 norm_hidden_states,
                                 norm_hidden_states,
                                 rotary_emb,
                                 update_cache=update_cache,
                                 cache_name=cache_name)
        hidden_states = (hidden_states.float() +
                         attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        # text conditioning injection: no KV cache (update_cache=0), text
        # embeddings are recomputed every call
        norm_hidden_states = self.norm2(
            hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states,
                                 encoder_hidden_states,
                                 encoder_hidden_states,
                                 None,
                                 update_cache=0,
                                 cache_name=cache_name)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        # the FFN is likewise modulated by per-token c_scale/c_shift/c_gate
        norm_hidden_states = (self.norm3(hidden_states.float()) *
                              (1. + c_scale_msa) +
                              c_shift_msa).type_as(hidden_states)

        ff_output = self.ffn(norm_hidden_states)

        hidden_states = (hidden_states.float() +
                         ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states


class WanTransformer3DModel(ModelMixin, ConfigMixin):
    r"""
    TODO

    LingBot-VA top-level model: an autoregressive diffusion Transformer built
    on the Wan2.2 video DiT that jointly denoises video latents and actions
    over a single interleaved sequence.

    Two forward paths:
    - ``forward_train`` (training): concatenates [noisy_video | clean_video |
      noisy_action | clean_action] into one big sequence (batch flattened,
      tail padded to a multiple of 128) and uses the FlexAttention structured
      mask to simulate every causal/window/chunk combination in one pass;
    - ``forward`` (inference): processes only the current chunk's noisy
      tokens; the historical context lives in each layer's attn1 KV-cache
      pool, and causality is guaranteed by the cache contents (not a mask).
      ``action_mode`` selects whether video or action is being denoised.

    Input/output embeddings:
    - video: latent [B,48,F,H,W] -> patchify(1,2,2) -> patch_embedding_mlp
      Linear(48*4 -> inner_dim); the output goes through proj_out back to
      latent space (unpatchified by the caller);
    - action: [B,30,F,16,1] -> action_embedder Linear(30 -> inner_dim); the
      output goes through action_proj_out back to the 30-dim unified action
      space;
    - text: UMT5 embedding -> condition_embedder.text_embedder.

    MoT dual stream: condition_embedder (video) and condition_embedder_action
    (action) are two independent sets of timestep-modulation parameters
    (initialized via deepcopy); attention/FFN weights are shared.
    """
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
                                        # "patch_embedding", 
                                        "patch_embedding_mlp",
                                        "condition_embedder", 
                                        'condition_embedder_action',
                                        "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", 
                             "scale_shift_table", 
                             "scale_shift_table_action",
                             "norm1", 
                             'action_norm1',
                             'text_norm1',
                             "norm2", 
                             'action_norm2',
                             'text_norm2',
                             "norm3",
                             'action_norm3',
                             'text_norm3'
                             ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(self,
                 patch_size=[1, 2, 2],
                 num_attention_heads=24,
                 attention_head_dim=128,
                 in_channels=48,
                 out_channels=48,
                 action_dim=30,
                 text_dim=4096,
                 freq_dim=256,
                 ffn_dim=14336,
                 num_layers=30,
                 cross_attn_norm=True,
                 eps=1e-06,
                 rope_max_seq_len=1024,
                 pos_embed_seq_len=None,
                 attn_mode="torch"):
        r"""
        TODO

        Initialize the model structure.

        Args:
            patch_size: patchify sizes (p1,p2,p3)=(1,2,2) for the video latent.
            num_attention_heads: number of attention heads.
            attention_head_dim: dim per head (inner_dim = heads * head_dim = 3072).
            in_channels: video latent channels (48 for the Wan2.2 VAE).
            out_channels: output latent channels (48).
            action_dim: unified action-space dim (30).
            text_dim: text encoder output dim (4096 for UMT5).
            freq_dim: frequency dim of the sinusoidal timestep encoding.
            ffn_dim: FFN inner dim.
            num_layers: number of Transformer blocks.
            cross_attn_norm: whether to add a LayerNorm before cross-attention.
            eps: numerical-stability term of the norm layers.
            rope_max_seq_len: maximum sequence length supported by RoPE.
            pos_embed_seq_len: reserved positional-embedding length (unused).
            attn_mode: attention implementation ('torch'/'flashattn' for
                inference, 'flex' for training).
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size,
                                      rope_max_seq_len)
        # video token embedding: 48*1*2*2=192 dims after patchify -> inner_dim
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2],
            inner_dim)
        # action token embedding: 30-dim unified action -> inner_dim
        self.action_embedder = nn.Linear(action_dim, inner_dim)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        # MoT dual stream: actions use independent timestep-modulation
        # parameters (decoupled from video)
        self.condition_embedder_action = deepcopy(self.condition_embedder)

        self.blocks = nn.ModuleList([
            WanTransformerBlock(inner_dim,
                                ffn_dim,
                                num_attention_heads,
                                cross_attn_norm,
                                eps,
                                attn_mode=attn_mode) for _ in range(num_layers)
        ])

        # output heads: after norm_out + per-token scale-shift, video goes
        # through proj_out (back to 48*4 dims, unpatchified by the caller)
        # and action through action_proj_out (back to 30 dims)
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim,
                                  out_channels * math.prod(patch_size))
        self.action_proj_out = nn.Linear(inner_dim, action_dim)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)

    def clear_cache(self, cache_name):
        """Clear the self-attention KV-cache pools of all layers (called on session reset).

        Args:
            cache_name: name of the cache pool.
        """
        for block in self.blocks:
            block.attn1.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name):
        """Drop the imagined frames (slots with is_pred=True) from all layers' KV caches.

        Called by the server when a real observation arrives, providing
        closed-loop error correction: imagined future frames only serve to
        condition action generation and are always replaced by real
        observations, never entering the long-term history.

        Args:
            cache_name: name of the cache pool.
        """
        for block in self.blocks:
            block.attn1.clear_pred_cache(cache_name)

    def create_empty_cache(self, cache_name, attn_window,
                           latent_token_per_chunk, action_token_per_chunk,
                           device, dtype, batch_size):
        """Allocate fixed-capacity KV-cache pools for all layers' self-attention.

        Capacity follows the sliding-window budget: attn_window is counted in
        frame_ids, and video chunks occupy even frames while action chunks
        occupy odd frames, so each modality gets attn_window//2 chunks' worth
        of tokens:
        total_tolen = (attn_window//2) * (video_chunk_tokens + action_chunk_tokens).

        Args:
            cache_name: name of the cache pool (e.g. 'pos').
            attn_window: attention window (72 in the deployment config).
            latent_token_per_chunk: tokens per video chunk.
            action_token_per_chunk: tokens per action chunk.
            device: tensor device.
            dtype: tensor precision (e.g. bf16).
            batch_size: batch size (doubled under CFG).
        """
        total_tolen = (attn_window // 2) * latent_token_per_chunk + (
            attn_window // 2) * action_token_per_chunk
        for block in self.blocks:
            block.attn1.init_kv_cache(cache_name, total_tolen,
                                      self.num_attention_heads,
                                      self.attention_head_dim, device, dtype, batch_size)
    
    def _input_embed(self, latents, input_type='latent'):
        """Embed raw tensors into token sequences according to the input type.

        Args:
            latents: input tensor. For 'latent': video latent [B,48,F,H,W];
                for 'action': actions [B,30,F,16,1]; for 'text': UMT5
                embeddings [B,L_text,4096].
            input_type: 'latent' / 'action' / 'text'.

        Returns:
            torch.Tensor: token sequence. video: [B, F*(H/2)*(W/2), 3072];
            action: [B, F*16*1, 3072]; text: [B, L_text, 3072].
        """
        if input_type == 'latent':
            # patchify(1,2,2): fold 2x2 spatial neighborhoods into channels
            # -> 192 dims per token
            hidden_states = rearrange(
                latents,
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            hidden_states = self.patch_embedding_mlp(hidden_states)
        elif input_type == 'action':
            # the action (h,w)=(16,1) dims are the 16 control sub-steps per
            # frame; flatten them directly into the sequence
            hidden_states = rearrange(latents, 'b c f h w -> b (f h w) c')
            hidden_states = self.action_embedder(hidden_states)
        elif input_type == 'text':
            hidden_states = self.condition_embedder.text_embedder(latents)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        return hidden_states

    def _time_embed(self, timesteps, H, W, dtype, action_mode=False):
        """Expand per-frame timesteps into per-token modulation vectors (training path).

        Under Diffusion Forcing each frame has its own timestep: a video frame
        covers (H/p2)*(W/p3) tokens and an action frame covers H*W (=16)
        tokens, so repeat_interleave broadcasts frame-level t to token level.

        Args:
            timesteps: per-frame timesteps, shape [B, F] (batch already flattened).
            H: latent height (L_H for video, 16 for action).
            W: latent width (L_W for video, 1 for action).
            dtype: output precision.
            action_mode: when True use condition_embedder_action and skip the
                spatial patch scaling (actions have no spatial patchify).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - temb: [1, B*F*tokens_per_frame, dim];
            - timestep_proj: [1, L, 6, dim], the 6 modulation groups per block.
        """
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            timesteps,
            (H // pach_scale_h) *
            (W // pach_scale_w), dim=1)  # L
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C
        return temb, timestep_proj

    def forward_train(self, input_dict):
        """Training forward: one big sequence + the FlexAttention structured mask.

        Sequence layout [noisy_video | clean_video | noisy_action |
        clean_action] (batch flattened into the sequence dim, tail padded to a
        multiple of 128). The noisy segments are the denoising targets, the
        clean segments are the history condition; mask rules are documented in
        ``FlexAttnFunc._get_mask_mod``. chunk_size/window_size are randomly
        resampled every training step, so a single training run supports any
        chunk/window configuration at inference time.

        Args:
            input_dict: training input dict containing:
                - latent_dict: noisy_latents [B,48,F,H,W], latent (clean
                  condition, same shape), text_emb [B,512,4096],
                  grid_id [B,3,L], timesteps/cond_timesteps [B,F];
                - action_dict: noisy_latents/latent [B,30,F,16,1], grid_id,
                  timesteps/cond_timesteps;
                - chunk_size / window_size: the AR configuration randomly
                  sampled for this step.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - latent_hidden_states: predicted video velocity field
              [B, F*H*W, 48*4] (patch-expanded layout, unpatchified by the caller);
            - action_hidden_states: predicted action velocity field [B, F*16, 30].
        """
        # cast everything to bf16 (mixed-precision training)
        input_dict['latent_dict']['noisy_latents'] = input_dict['latent_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['latent_dict']['latent'] = input_dict['latent_dict']['latent'].to(torch.bfloat16)
        input_dict['action_dict']['noisy_latents'] = input_dict['action_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['action_dict']['latent'] = input_dict['action_dict']['latent'].to(torch.bfloat16)

        latent_dict = input_dict['latent_dict']
        action_dict = input_dict['action_dict']
        batch_size = latent_dict['noisy_latents'].shape[0]

        # embed the four segments then flatten(0,1): the batch dim is folded
        # into the sequence dim (one big sequence with B=1); sample isolation
        # is handled by the mask's seq_ids
        latent_hidden_states = self._input_embed(latent_dict['noisy_latents'], input_type='latent').flatten(0, 1)[None]
        action_hidden_states = self._input_embed(action_dict['noisy_latents'], input_type='action').flatten(0, 1)[None]
        text_hidden_states = self._input_embed(latent_dict["text_emb"], input_type='text')

        text_hidden_states = text_hidden_states.flatten(0, 1)[None]

        condition_latent_hidden_states = self._input_embed(latent_dict['latent'], input_type='latent').flatten(0, 1)[None]
        condition_action_hidden_states = self._input_embed(action_dict['latent'], input_type='action').flatten(0, 1)[None]

        # concatenate into the training layout: [noisy_v | clean_v | noisy_a | clean_a]
        hidden_states = torch.cat([latent_hidden_states, 
                                   condition_latent_hidden_states,
                                   action_hidden_states, 
                                   condition_action_hidden_states], dim=1)


        # RoPE grid ids are concatenated in the same layout: noisy/clean
        # segments share positions (two noise views of the same tokens), so
        # each grid_id is repeated twice
        latent_grid_id = latent_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        action_grid_id = action_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        full_grid_id = torch.cat([latent_grid_id] * 2 + [action_grid_id] * 2, dim=2)

        rotary_emb = self.rope(full_grid_id)[:, :, None] 

        # per-token timesteps: `timesteps` for the noisy segments,
        # `cond_timesteps` for the clean segments (the clean condition
        # segments may also be noised, see noisy_cond_prob in train.py)
        latent_time_steps = torch.cat(
            [latent_dict['timesteps'].flatten(0, 1), latent_dict['cond_timesteps'].flatten(0, 1)]
        )[None]
        action_time_steps = torch.cat(
            [action_dict['timesteps'].flatten(0, 1), action_dict['cond_timesteps'].flatten(0, 1)]
        )[None]
        # video/action go through their own condition_embedder (MoT dual-stream modulation)
        latent_temb, latent_timestep_proj =self._time_embed(latent_time_steps, 
                        latent_dict['noisy_latents'].shape[-2], 
                        latent_dict['noisy_latents'].shape[-1], 
                        dtype=hidden_states.dtype, 
                        action_mode=False)
        action_temb, action_timestep_proj = self._time_embed(action_time_steps,
                        action_dict['noisy_latents'].shape[-2], 
                        action_dict['noisy_latents'].shape[-1], 
                        dtype=hidden_states.dtype, 
                        action_mode=True)
        temb = torch.cat([latent_temb, action_temb], dim=1)
        timestep_proj = torch.cat([latent_timestep_proj, action_timestep_proj], dim=1)

        # pad to a multiple of 128: block-alignment requirement of the
        # FlexAttention kernel
        total_length = hidden_states.shape[1]
        padded_length = (128 - total_length % 128) % 128
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

        # record the five segment lengths; only noisy_video / noisy_action are
        # kept for the loss at the output
        split_list = [latent_hidden_states.shape[1], 
                      condition_latent_hidden_states.shape[1], 
                      action_hidden_states.shape[1], 
                      condition_action_hidden_states.shape[1],
                      padded_length]

        # build the structured mask for this step's randomly sampled
        # chunk_size/window_size (cached at class level)
        FlexAttnFunc.init_mask(latent_dict['noisy_latents'].shape, 
                               action_dict['noisy_latents'].shape, 
                               padded_length, 
                               input_dict["chunk_size"],
                               window_size=input_dict['window_size'],
                               patch_size=self.patch_size,
                               device=hidden_states.device
                               )

        # training uses no KV cache (update_cache=False); causality is fully
        # expressed by the mask
        for block in self.blocks:
            hidden_states = block(hidden_states,
                                         text_hidden_states,
                                         timestep_proj,
                                         rotary_emb,
                                         update_cache=False)
        # output adaLN: per-token scale/shift modulation, then project back to
        # each modality's space
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table,
                                 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
        hidden_states = (self.norm_out(hidden_states.float()) *
                                (1. + scale) +
                                shift).type_as(hidden_states)
        # keep only the noisy segments' predictions (clean segments and
        # padding are discarded) and restore the batch dim
        latent_hidden_states, _, action_hidden_states, _, _ = torch.split(hidden_states, split_list, dim=1)
        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(latent_hidden_states,
                                             '1 (b l) (n c) -> b (l n) c',
                                             n=math.prod(self.patch_size), b=batch_size)  #
        action_hidden_states = self.action_proj_out(action_hidden_states)
        action_hidden_states = rearrange(action_hidden_states,
                                             '1 (b l) c -> b l c',
                                             b=batch_size)  #

        return latent_hidden_states, action_hidden_states

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]

        Note (the actual interface is the code below; the English docstring above
        is the original Wan signature): inference forward. Each call processes
        only the **current chunk**'s noisy tokens; the historical context is
        provided by each layer's attn1 KV-cache pool, with no mask involved.
        The server's AR main loop calls this function twice per chunk:
        1) action_mode=False to denoise video (intermediate steps use
           update_cache=0, write-then-rollback; the final t=0 step uses
           update_cache=1 to commit the clean imagined frames into the cache);
        2) action_mode=True to denoise action (update_cache=0, reading the
           just-committed imagined video KV — "imagine the future first, then
           act").
        Real observation frames are committed with update_cache=2 via the
        server's _compute_kv_cache.

        Args (actual parameters):
            input_dict: contains noisy_latents (video [B,48,F,H,W] / action
                [B,30,F,16,1]), text_emb [B,512,4096], grid_id [3,L],
                timesteps [B,F].
            update_cache: three-state KV-cache semantics (0 rollback /
                1 commit imagined frames / 2 commit real observations).
            cache_name: name of the cache pool.
            action_mode: True when denoising action tokens (uses the action
                embedder/modulation/output head), False for video.
            train_mode: when True, dispatch to forward_train.

        Returns:
            torch.Tensor: video mode returns the predicted velocity field
            [B, F*H*W, 48*4] (patch-expanded); action mode returns
            [B, F*16, 30].
        """
        if train_mode:
            return self.forward_train(input_dict)
        if action_mode:  # action input emb
            # action tokens: the (h,w)=(16,1) control sub-steps are flattened
            # into the sequence, then Linear(30 -> C)
            latent_hidden_states = rearrange(input_dict['noisy_latents'],
                                             'b c f h w -> b (f h w) c')
            latent_hidden_states = self.action_embedder(
                latent_hidden_states)  # B L1 C
        else:  # latent input emb
            # video tokens: patchify(1,2,2) then Linear(48*4 -> C)
            latent_hidden_states = rearrange(
                input_dict['noisy_latents'],
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            latent_hidden_states = self.patch_embedding_mlp(
                latent_hidden_states)
        text_hidden_states = self.condition_embedder.text_embedder(
            input_dict["text_emb"])  # B L2 C

        # grid_id is produced by the server: action tokens carry fractional f
        # coordinates f + k/17, placed exactly between the two surrounding
        # video frames (h/w = -1, no spatial position)
        latent_grid_id = input_dict['grid_id']
        rotary_emb = self.rope(latent_grid_id)[:, :, None]  # 1 L 1 C
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])

        # broadcast per-frame timesteps to per-token (diffusion forcing:
        # independent noise level per frame)
        latent_time_steps = torch.repeat_interleave(
            input_dict['timesteps'],
            (input_dict['noisy_latents'].shape[-2] // pach_scale_h) *
            (input_dict['noisy_latents'].shape[-1] // pach_scale_w), dim=1)  # L
        # MoT dual stream: video/action use their own condition_embedder for modulation
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=latent_hidden_states.dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C

        # self-attention with the KV cache: update_cache semantics are
        # forwarded to every layer's attn1; the attention KV = all valid slots
        # in the pool (clean history frames + this write)
        for block in self.blocks:
            latent_hidden_states = block(latent_hidden_states,
                                         text_hidden_states,
                                         timestep_proj,
                                         rotary_emb,
                                         update_cache=update_cache,
                                         cache_name=cache_name)
        # output adaLN modulation, then each modality uses its own output head
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table,
                                 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(latent_hidden_states.device).squeeze(1)
        scale = scale.to(latent_hidden_states.device).squeeze(1)
        latent_hidden_states = (self.norm_out(latent_hidden_states.float()) *
                                (1. + scale) +
                                shift).type_as(latent_hidden_states)

        if action_mode:
            # action head: project straight back to the 30-dim unified action space
            latent_hidden_states = self.action_proj_out(latent_hidden_states)
        else:
            # video head: project back to 48*4 dims and expand into the patch
            # layout (the caller unpatchifies)
            latent_hidden_states = self.proj_out(latent_hidden_states)
            latent_hidden_states = rearrange(latent_hidden_states,
                                             'b l (n c) -> b (l n) c',
                                             n=math.prod(self.patch_size))  #

        return latent_hidden_states


if __name__ == '__main__':
    # smoke test: instantiate the model with the deployment config and print its structure
    model = WanTransformer3DModel(patch_size=[1, 2, 2],
                                  num_attention_heads=24,
                                  attention_head_dim=128,
                                  in_channels=48,
                                  out_channels=48,
                                  action_dim=30,
                                  text_dim=4096,
                                  freq_dim=256,
                                  ffn_dim=14336,
                                  num_layers=30,
                                  cross_attn_norm=True,
                                  eps=1e-6,
                                  rope_max_seq_len=1024,
                                  pos_embed_seq_len=None,
                                  attn_mode="torch")
    print(model)
