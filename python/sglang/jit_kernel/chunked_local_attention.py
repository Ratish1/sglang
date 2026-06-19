# SPDX-License-Identifier: Apache-2.0
"""Workspace-backed chunked local-causal attention.

This helper adapts small fixed chunks with bounded local K/V cache into SGLang's
varlen FlashAttention wrapper. Callers own model-specific projection and RoPE;
this module owns static packing, varlen metadata, and in-place cache commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


@dataclass
class LocalCausalVarlenWorkspace:
    q_pack: torch.Tensor
    k_pack: torch.Tensor
    v_pack: torch.Tensor
    next_cache_k: torch.Tensor
    next_cache_v: torch.Tensor
    next_cache_pos: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    k_lens: torch.Tensor
    batch_arange: torch.Tensor
    max_batch_size: int
    max_chunk_len: int
    context: int
    num_heads: int
    head_dim: int

    @classmethod
    def create(
        cls,
        *,
        max_batch_size: int,
        max_chunk_len: int,
        context: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> LocalCausalVarlenWorkspace:
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        if max_chunk_len < 1:
            raise ValueError(f"max_chunk_len must be >= 1, got {max_chunk_len}")
        if context < 1:
            raise ValueError(f"context must be >= 1, got {context}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {head_dim}")

        return cls(
            q_pack=torch.empty(
                max_batch_size * max_chunk_len,
                num_heads,
                head_dim,
                device=device,
                dtype=dtype,
            ),
            k_pack=torch.empty(
                max_batch_size * (context + max_chunk_len),
                num_heads,
                head_dim,
                device=device,
                dtype=dtype,
            ),
            v_pack=torch.empty(
                max_batch_size * (context + max_chunk_len),
                num_heads,
                head_dim,
                device=device,
                dtype=dtype,
            ),
            next_cache_k=torch.empty(
                max_batch_size,
                num_heads,
                context,
                head_dim,
                device=device,
                dtype=dtype,
            ),
            next_cache_v=torch.empty(
                max_batch_size,
                num_heads,
                context,
                head_dim,
                device=device,
                dtype=dtype,
            ),
            next_cache_pos=torch.empty(
                max_batch_size,
                context,
                device=device,
                dtype=torch.long,
            ),
            cu_q=torch.empty(max_batch_size + 1, device=device, dtype=torch.int32),
            cu_k=torch.empty(max_batch_size + 1, device=device, dtype=torch.int32),
            k_lens=torch.empty(max_batch_size, device=device, dtype=torch.int32),
            batch_arange=torch.arange(
                max_batch_size + 1, device=device, dtype=torch.int32
            ),
            max_batch_size=max_batch_size,
            max_chunk_len=max_chunk_len,
            context=context,
            num_heads=num_heads,
            head_dim=head_dim,
        )


def _validate_bhtd(name: str, tensor: torch.Tensor) -> tuple[int, int, int, int]:
    if tensor.dim() != 4:
        raise ValueError(f"{name} must be 4D [B, H, T, D], got {tuple(tensor.shape)}")
    return tuple(int(dim) for dim in tensor.shape)  # type: ignore[return-value]


if triton is not None:

    @triton.jit
    def _pack_q_kernel(
        q,
        q_pack,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_t: tl.constexpr,
        q_stride_d: tl.constexpr,
        total: tl.constexpr,
        num_heads: tl.constexpr,
        chunk_len: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offs = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offs < total
        d = offs % head_dim
        h = (offs // head_dim) % num_heads
        t = (offs // (head_dim * num_heads)) % chunk_len
        b = offs // (head_dim * num_heads * chunk_len)
        src = b * q_stride_b + h * q_stride_h + t * q_stride_t + d * q_stride_d
        values = tl.load(q + src, mask=mask)
        tl.store(q_pack + offs, values, mask=mask)

    @triton.jit
    def _pack_kv_kernel(
        k,
        v,
        cache_k,
        cache_v,
        k_pack,
        v_pack,
        cu_k,
        k_lens,
        k_stride_b: tl.constexpr,
        k_stride_h: tl.constexpr,
        k_stride_t: tl.constexpr,
        k_stride_d: tl.constexpr,
        cache_stride_b: tl.constexpr,
        cache_stride_h: tl.constexpr,
        cache_stride_t: tl.constexpr,
        cache_stride_d: tl.constexpr,
        total_slots: tl.constexpr,
        num_heads: tl.constexpr,
        chunk_len: tl.constexpr,
        context: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offs = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offs < total_slots
        d = offs % head_dim
        h = (offs // head_dim) % num_heads
        local_t = (offs // (head_dim * num_heads)) % (context + chunk_len)
        b = offs // (head_dim * num_heads * (context + chunk_len))
        cached_len = tl.load(k_lens + b, mask=mask, other=0) - chunk_len
        seq_len = cached_len + chunk_len
        valid = mask & (local_t < seq_len)
        dst_t = tl.load(cu_k + b, mask=mask, other=0) + local_t
        dst = (dst_t * num_heads + h) * head_dim + d
        from_cache = local_t < cached_len
        cache_t = context - cached_len + local_t
        cur_t = local_t - cached_len
        cache_src = (
            b * cache_stride_b
            + h * cache_stride_h
            + cache_t * cache_stride_t
            + d * cache_stride_d
        )
        cur_src = b * k_stride_b + h * k_stride_h + cur_t * k_stride_t + d * k_stride_d
        cache_mask = valid & from_cache
        cur_mask = valid & (local_t >= cached_len)
        k_value = tl.where(
            from_cache,
            tl.load(cache_k + cache_src, mask=cache_mask, other=0.0),
            tl.load(k + cur_src, mask=cur_mask, other=0.0),
        )
        v_value = tl.where(
            from_cache,
            tl.load(cache_v + cache_src, mask=cache_mask, other=0.0),
            tl.load(v + cur_src, mask=cur_mask, other=0.0),
        )
        tl.store(k_pack + dst, k_value, mask=valid)
        tl.store(v_pack + dst, v_value, mask=valid)

    @triton.jit
    def _build_next_cache_kernel(
        k,
        v,
        cache_k,
        cache_v,
        cache_pos,
        offset,
        next_cache_k,
        next_cache_v,
        next_cache_pos,
        k_stride_b: tl.constexpr,
        k_stride_h: tl.constexpr,
        k_stride_t: tl.constexpr,
        k_stride_d: tl.constexpr,
        cache_stride_b: tl.constexpr,
        cache_stride_h: tl.constexpr,
        cache_stride_t: tl.constexpr,
        cache_stride_d: tl.constexpr,
        total_kv: tl.constexpr,
        total_pos: tl.constexpr,
        num_heads: tl.constexpr,
        chunk_len: tl.constexpr,
        context: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offs = tl.program_id(0) * block_size + tl.arange(0, block_size)
        kv_mask = offs < total_kv
        d = offs % head_dim
        h = (offs // head_dim) % num_heads
        c = (offs // (head_dim * num_heads)) % context
        b = offs // (head_dim * num_heads * context)
        current_only = chunk_len >= context
        from_cache = (not current_only) & (c < (context - chunk_len))
        cache_t = c + chunk_len
        cur_t = tl.where(
            current_only,
            chunk_len - context + c,
            c - (context - chunk_len),
        )
        cache_src = (
            b * cache_stride_b
            + h * cache_stride_h
            + cache_t * cache_stride_t
            + d * cache_stride_d
        )
        cur_src = b * k_stride_b + h * k_stride_h + cur_t * k_stride_t + d * k_stride_d
        dst = ((b * num_heads + h) * context + c) * head_dim + d
        cache_mask = kv_mask & from_cache
        cur_mask = kv_mask & (~from_cache)
        k_value = tl.where(
            from_cache,
            tl.load(cache_k + cache_src, mask=cache_mask, other=0.0),
            tl.load(k + cur_src, mask=cur_mask, other=0.0),
        )
        v_value = tl.where(
            from_cache,
            tl.load(cache_v + cache_src, mask=cache_mask, other=0.0),
            tl.load(v + cur_src, mask=cur_mask, other=0.0),
        )
        tl.store(next_cache_k + dst, k_value, mask=kv_mask)
        tl.store(next_cache_v + dst, v_value, mask=kv_mask)

        pos_mask = offs < total_pos
        c_pos = offs % context
        b_pos = offs // context
        pos_from_cache = (chunk_len < context) & (c_pos < (context - chunk_len))
        cache_pos_src = b_pos * context + c_pos + chunk_len
        cur_pos = tl.load(offset + b_pos, mask=pos_mask, other=0) + tl.where(
            chunk_len >= context,
            chunk_len - context + c_pos,
            c_pos - (context - chunk_len),
        )
        pos_value = tl.where(
            pos_from_cache,
            tl.load(cache_pos + cache_pos_src, mask=pos_mask & pos_from_cache),
            cur_pos,
        )
        tl.store(next_cache_pos + offs, pos_value, mask=pos_mask)

    @triton.jit
    def _commit_cache_kernel(
        cache_k,
        cache_v,
        cache_pos,
        offset,
        next_cache_k,
        next_cache_v,
        next_cache_pos,
        exec_mask,
        total_kv: tl.constexpr,
        total_pos: tl.constexpr,
        batch_size: tl.constexpr,
        context: tl.constexpr,
        chunk_len: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offs = tl.program_id(0) * block_size + tl.arange(0, block_size)
        kv_mask = offs < total_kv
        elems_per_batch = total_kv // batch_size
        b = offs // elems_per_batch
        active = tl.load(exec_mask + b, mask=kv_mask, other=0).to(tl.int1)
        tl.store(
            cache_k + offs,
            tl.load(next_cache_k + offs, mask=kv_mask),
            mask=kv_mask & active,
        )
        tl.store(
            cache_v + offs,
            tl.load(next_cache_v + offs, mask=kv_mask),
            mask=kv_mask & active,
        )

        pos_mask = offs < total_pos
        b_pos = offs // context
        active_pos = tl.load(exec_mask + b_pos, mask=pos_mask, other=0).to(tl.int1)
        tl.store(
            cache_pos + offs,
            tl.load(next_cache_pos + offs, mask=pos_mask),
            mask=pos_mask & active_pos,
        )
        batch_mask = offs < batch_size
        active_batch = tl.load(exec_mask + offs, mask=batch_mask, other=0).to(tl.int1)
        old_offset = tl.load(offset + offs, mask=batch_mask, other=0)
        tl.store(offset + offs, old_offset + chunk_len, mask=batch_mask & active_batch)


def _can_use_triton(q: torch.Tensor) -> bool:
    return triton is not None and q.is_cuda


def _fill_metadata(
    offset: torch.Tensor,
    workspace: LocalCausalVarlenWorkspace,
    *,
    batch_size: int,
    chunk_len: int,
    context: int,
) -> None:
    workspace.cu_q[: batch_size + 1].copy_(
        workspace.batch_arange[: batch_size + 1] * chunk_len
    )
    cached_lens = torch.clamp(offset[:batch_size], min=0, max=context).to(torch.int32)
    workspace.k_lens[:batch_size].copy_(cached_lens + chunk_len)
    workspace.cu_k[0].zero_()
    workspace.cu_k[1 : batch_size + 1].copy_(
        torch.cumsum(workspace.k_lens[:batch_size], dim=0)
    )


def _pack_python(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    cache_pos: torch.Tensor,
    workspace: LocalCausalVarlenWorkspace,
    *,
    batch_size: int,
    chunk_len: int,
) -> tuple[int, int]:
    """CPU/test fallback for the CUDA Triton pack path."""
    workspace.cu_q[0].zero_()
    workspace.cu_k[0].zero_()
    total_k = 0
    max_k = 0
    for batch_idx in range(batch_size):
        q_start = batch_idx * chunk_len
        q_next = q_start + chunk_len
        workspace.q_pack[q_start:q_next].copy_(
            q[batch_idx].transpose(0, 1).contiguous()
        )
        valid_k = cache_pos[batch_idx] >= 0
        cached_len = int(valid_k.sum().item())
        k_next = total_k + cached_len + chunk_len
        if cached_len:
            cache_end = total_k + cached_len
            workspace.k_pack[total_k:cache_end].copy_(
                cache_k[batch_idx, :, valid_k, :].transpose(0, 1)
            )
            workspace.v_pack[total_k:cache_end].copy_(
                cache_v[batch_idx, :, valid_k, :].transpose(0, 1)
            )
            current_start = cache_end
        else:
            current_start = total_k
        workspace.k_pack[current_start:k_next].copy_(k[batch_idx].transpose(0, 1))
        workspace.v_pack[current_start:k_next].copy_(v[batch_idx].transpose(0, 1))
        workspace.cu_q[batch_idx + 1] = q_next
        workspace.cu_k[batch_idx + 1] = k_next
        workspace.k_lens[batch_idx] = cached_len + chunk_len
        total_k = k_next
        max_k = max(max_k, cached_len + chunk_len)
    return total_k, max_k


def _update_cache_python(
    k: torch.Tensor,
    v: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    cache_pos: torch.Tensor,
    offset: torch.Tensor,
    exec_mask: torch.Tensor,
    workspace: LocalCausalVarlenWorkspace,
    *,
    batch_size: int,
    chunk_len: int,
    context: int,
) -> None:
    """CPU/test fallback for the CUDA Triton cache-update path."""
    pos_q = offset.view(-1, 1) + torch.arange(
        chunk_len, device=offset.device, dtype=offset.dtype
    ).view(1, -1)
    if chunk_len >= context:
        workspace.next_cache_k[:batch_size].copy_(k[:, :, -context:, :])
        workspace.next_cache_v[:batch_size].copy_(v[:, :, -context:, :])
        workspace.next_cache_pos[:batch_size].copy_(pos_q[:, -context:])
    else:
        keep = context - chunk_len
        workspace.next_cache_k[:batch_size, :, :keep, :].copy_(cache_k[:, :, -keep:, :])
        workspace.next_cache_k[:batch_size, :, keep:, :].copy_(k)
        workspace.next_cache_v[:batch_size, :, :keep, :].copy_(cache_v[:, :, -keep:, :])
        workspace.next_cache_v[:batch_size, :, keep:, :].copy_(v)
        workspace.next_cache_pos[:batch_size, :keep].copy_(cache_pos[:, -keep:])
        workspace.next_cache_pos[:batch_size, keep:].copy_(pos_q)

    exec_mask_kv = exec_mask.to(device=cache_k.device, dtype=torch.bool).view(
        -1, 1, 1, 1
    )
    exec_mask_pos = exec_mask.to(device=cache_pos.device, dtype=torch.bool).view(-1, 1)
    cache_k.copy_(
        torch.where(exec_mask_kv, workspace.next_cache_k[:batch_size], cache_k)
    )
    cache_v.copy_(
        torch.where(exec_mask_kv, workspace.next_cache_v[:batch_size], cache_v)
    )
    cache_pos.copy_(
        torch.where(exec_mask_pos, workspace.next_cache_pos[:batch_size], cache_pos)
    )
    offset.copy_(
        torch.where(
            exec_mask.to(device=offset.device, dtype=torch.bool),
            offset + chunk_len,
            offset,
        )
    )


def local_causal_varlen_attention_with_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    cache_pos: torch.Tensor,
    offset: torch.Tensor,
    exec_mask: torch.Tensor,
    workspace: LocalCausalVarlenWorkspace,
    *,
    context: int,
    flash_attn_varlen_func: Callable[..., torch.Tensor] | None = None,
    window_size: tuple[int, int] | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    batch_size, num_heads, chunk_len, head_dim = _validate_bhtd("q", q)
    if _validate_bhtd("k", k) != (batch_size, num_heads, chunk_len, head_dim):
        raise ValueError("k must have the same [B, H, T, D] shape as q")
    if _validate_bhtd("v", v) != (batch_size, num_heads, chunk_len, head_dim):
        raise ValueError("v must have the same [B, H, T, D] shape as q")
    if _validate_bhtd("cache_k", cache_k) != (batch_size, num_heads, context, head_dim):
        raise ValueError("cache_k must have shape [B, H, context, D]")
    if _validate_bhtd("cache_v", cache_v) != (batch_size, num_heads, context, head_dim):
        raise ValueError("cache_v must have shape [B, H, context, D]")
    if cache_pos.shape != (batch_size, context):
        raise ValueError(
            f"cache_pos must have shape {(batch_size, context)}, got {tuple(cache_pos.shape)}"
        )
    if offset.shape != (batch_size,):
        raise ValueError(
            f"offset must have shape {(batch_size,)}, got {tuple(offset.shape)}"
        )
    if exec_mask.shape != (batch_size,):
        raise ValueError(
            f"exec_mask must have shape {(batch_size,)}, got {tuple(exec_mask.shape)}"
        )
    if batch_size > workspace.max_batch_size:
        raise ValueError(
            f"batch size {batch_size} exceeds workspace max_batch_size "
            f"{workspace.max_batch_size}"
        )
    if chunk_len > workspace.max_chunk_len:
        raise ValueError(
            f"chunk length {chunk_len} exceeds workspace max_chunk_len "
            f"{workspace.max_chunk_len}"
        )
    if context != workspace.context:
        raise ValueError(f"context mismatch: {context} != {workspace.context}")
    if num_heads != workspace.num_heads or head_dim != workspace.head_dim:
        raise ValueError(
            "attention head shape does not match workspace: "
            f"got H={num_heads}, D={head_dim}; "
            f"workspace H={workspace.num_heads}, D={workspace.head_dim}"
        )

    if flash_attn_varlen_func is None:
        from sglang.jit_kernel.flash_attention import flash_attn_varlen_func
    if window_size is None:
        window_size = (context, 0)

    total_q = batch_size * chunk_len
    total_k_capacity = batch_size * (context + chunk_len)
    use_triton = _can_use_triton(q)

    if use_triton:
        _fill_metadata(
            offset,
            workspace,
            batch_size=batch_size,
            chunk_len=chunk_len,
            context=context,
        )
        max_k = context + chunk_len
        block_size = 256
        _pack_q_kernel[(triton.cdiv(total_q * num_heads * head_dim, block_size),)](
            q,
            workspace.q_pack,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            total_q * num_heads * head_dim,
            num_heads,
            chunk_len,
            head_dim,
            block_size,
        )
        _pack_kv_kernel[
            (triton.cdiv(total_k_capacity * num_heads * head_dim, block_size),)
        ](
            k,
            v,
            cache_k,
            cache_v,
            workspace.k_pack,
            workspace.v_pack,
            workspace.cu_k,
            workspace.k_lens,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            cache_k.stride(0),
            cache_k.stride(1),
            cache_k.stride(2),
            cache_k.stride(3),
            total_k_capacity * num_heads * head_dim,
            num_heads,
            chunk_len,
            context,
            head_dim,
            block_size,
        )
        k_pack = workspace.k_pack[:total_k_capacity]
        v_pack = workspace.v_pack[:total_k_capacity]
    else:
        total_k, max_k = _pack_python(
            q,
            k,
            v,
            cache_k,
            cache_v,
            cache_pos,
            workspace,
            batch_size=batch_size,
            chunk_len=chunk_len,
        )
        k_pack = workspace.k_pack[:total_k]
        v_pack = workspace.v_pack[:total_k]

    out_pack = flash_attn_varlen_func(
        workspace.q_pack[:total_q],
        k_pack,
        v_pack,
        workspace.cu_q[: batch_size + 1],
        workspace.cu_k[: batch_size + 1],
        chunk_len,
        max_k,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=window_size,
    )
    out = out_pack.view(batch_size, chunk_len, num_heads, head_dim).permute(0, 2, 1, 3)

    if use_triton:
        block_size = 256
        total_cache_kv = batch_size * num_heads * context * head_dim
        total_cache_pos = batch_size * context
        grid = (triton.cdiv(max(total_cache_kv, total_cache_pos), block_size),)
        _build_next_cache_kernel[grid](
            k,
            v,
            cache_k,
            cache_v,
            cache_pos,
            offset,
            workspace.next_cache_k,
            workspace.next_cache_v,
            workspace.next_cache_pos,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            cache_k.stride(0),
            cache_k.stride(1),
            cache_k.stride(2),
            cache_k.stride(3),
            total_cache_kv,
            total_cache_pos,
            num_heads,
            chunk_len,
            context,
            head_dim,
            block_size,
        )
        _commit_cache_kernel[grid](
            cache_k,
            cache_v,
            cache_pos,
            offset,
            workspace.next_cache_k,
            workspace.next_cache_v,
            workspace.next_cache_pos,
            exec_mask,
            total_cache_kv,
            total_cache_pos,
            batch_size,
            context,
            chunk_len,
            block_size,
        )
    else:
        _update_cache_python(
            k,
            v,
            cache_k,
            cache_v,
            cache_pos,
            offset,
            exec_mask,
            workspace,
            batch_size=batch_size,
            chunk_len=chunk_len,
            context=context,
        )
    return out.contiguous()
