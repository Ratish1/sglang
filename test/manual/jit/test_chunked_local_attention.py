import pytest
import torch

from sglang.jit_kernel.chunked_local_attention import (
    LocalCausalVarlenWorkspace,
    local_causal_varlen_attention_with_cache,
)


def _expected_pack(q, k, v, cache_k, cache_v, cache_pos):
    q_chunks = []
    k_chunks = []
    v_chunks = []
    cu_q = [0]
    cu_k = [0]
    max_k = 0
    for batch_idx in range(q.shape[0]):
        valid_k = cache_pos[batch_idx] >= 0
        q_i = q[batch_idx].transpose(0, 1).contiguous()
        k_i = torch.cat(
            [cache_k[batch_idx, :, valid_k, :], k[batch_idx]], dim=1
        ).transpose(0, 1)
        v_i = torch.cat(
            [cache_v[batch_idx, :, valid_k, :], v[batch_idx]], dim=1
        ).transpose(0, 1)
        q_chunks.append(q_i)
        k_chunks.append(k_i)
        v_chunks.append(v_i)
        cu_q.append(cu_q[-1] + q_i.shape[0])
        cu_k.append(cu_k[-1] + k_i.shape[0])
        max_k = max(max_k, int(k_i.shape[0]))
    return (
        torch.cat(q_chunks, dim=0),
        torch.cat(k_chunks, dim=0),
        torch.cat(v_chunks, dim=0),
        torch.tensor(cu_q, dtype=torch.int32),
        torch.tensor(cu_k, dtype=torch.int32),
        max_k,
    )


def test_local_causal_workspace_matches_reference_pack_and_updates_cache():
    bsz, heads, chunk_len, context, head_dim = 2, 3, 4, 5, 2
    q = torch.randn(bsz, heads, chunk_len, head_dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cache_k = torch.randn(bsz, heads, context, head_dim)
    cache_v = torch.randn_like(cache_k)
    cache_pos = torch.tensor([[-1, -1, 7, 8, 9], [-1, 3, 4, 5, 6]])
    offset = torch.tensor([10, 7], dtype=torch.long)
    exec_mask = torch.tensor([True, True])

    expected_q, expected_k, expected_v, expected_cu_q, expected_cu_k, expected_max_k = (
        _expected_pack(q, k, v, cache_k, cache_v, cache_pos)
    )
    cache_k_ptr = cache_k.data_ptr()
    cache_v_ptr = cache_v.data_ptr()
    cache_pos_ptr = cache_pos.data_ptr()
    seen = {}

    def fake_flash(q_pack, k_pack, v_pack, cu_q, cu_k, max_q, max_k, **kwargs):
        seen["q"] = q_pack.clone()
        seen["k"] = k_pack.clone()
        seen["v"] = v_pack.clone()
        seen["cu_q"] = cu_q.cpu().clone()
        seen["cu_k"] = cu_k.cpu().clone()
        seen["max_q"] = max_q
        seen["max_k"] = max_k
        seen["kwargs"] = kwargs
        return q_pack + 1

    workspace = LocalCausalVarlenWorkspace.create(
        max_batch_size=bsz,
        max_chunk_len=chunk_len,
        context=context,
        num_heads=heads,
        head_dim=head_dim,
        device=q.device,
        dtype=q.dtype,
    )

    out = local_causal_varlen_attention_with_cache(
        q,
        k,
        v,
        cache_k,
        cache_v,
        cache_pos,
        offset,
        exec_mask,
        workspace,
        context=context,
        flash_attn_varlen_func=fake_flash,
        window_size=(context - 1, 0),
    )

    assert torch.equal(seen["q"], expected_q)
    assert torch.equal(seen["k"], expected_k)
    assert torch.equal(seen["v"], expected_v)
    assert torch.equal(seen["cu_q"], expected_cu_q)
    assert torch.equal(seen["cu_k"], expected_cu_k)
    assert seen["max_q"] == chunk_len
    assert seen["max_k"] == expected_max_k
    assert seen["kwargs"]["causal"] is True
    assert seen["kwargs"]["window_size"] == (context - 1, 0)
    assert out.shape == q.shape
    assert torch.equal(out, q + 1)
    assert cache_k.data_ptr() == cache_k_ptr
    assert cache_v.data_ptr() == cache_v_ptr
    assert cache_pos.data_ptr() == cache_pos_ptr
    assert torch.equal(
        cache_pos,
        torch.tensor([[9, 10, 11, 12, 13], [6, 7, 8, 9, 10]]),
    )
    assert torch.equal(offset, torch.tensor([14, 11]))


def test_local_causal_workspace_preserves_inactive_cache_rows():
    bsz, heads, chunk_len, context, head_dim = 2, 2, 3, 4, 2
    q = torch.randn(bsz, heads, chunk_len, head_dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cache_k = torch.randn(bsz, heads, context, head_dim)
    cache_v = torch.randn_like(cache_k)
    cache_pos = torch.tensor([[0, 1, 2, 3], [20, 21, 22, 23]])
    original_k_row_1 = cache_k[1].clone()
    original_v_row_1 = cache_v[1].clone()
    offset = torch.tensor([4, 24], dtype=torch.long)
    exec_mask = torch.tensor([True, False])

    def fake_flash(q_pack, *args, **kwargs):
        return q_pack

    workspace = LocalCausalVarlenWorkspace.create(
        max_batch_size=bsz,
        max_chunk_len=chunk_len,
        context=context,
        num_heads=heads,
        head_dim=head_dim,
        device=q.device,
        dtype=q.dtype,
    )

    _ = local_causal_varlen_attention_with_cache(
        q,
        k,
        v,
        cache_k,
        cache_v,
        cache_pos,
        offset,
        exec_mask,
        workspace,
        context=context,
        flash_attn_varlen_func=fake_flash,
    )

    assert torch.equal(cache_pos[0], torch.tensor([3, 4, 5, 6]))
    assert torch.equal(cache_pos[1], torch.tensor([20, 21, 22, 23]))
    assert torch.equal(cache_k[1], original_k_row_1)
    assert torch.equal(cache_v[1], original_v_row_1)
    assert torch.equal(offset, torch.tensor([7, 24]))


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("context", [5, 7])
@pytest.mark.parametrize("chunk_len", [1, 3])
def test_local_causal_workspace_reference_path_uses_cache_positions(
    batch_size, context, chunk_len
):
    heads, head_dim = 2, 2
    q = torch.randn(batch_size, heads, chunk_len, head_dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cache_k = torch.randn(batch_size, heads, context, head_dim)
    cache_v = torch.randn_like(cache_k)
    cache_pos = torch.full((batch_size, context), -1, dtype=torch.long)
    for batch_idx in range(batch_size):
        valid = min(context, batch_idx + 1)
        cache_pos[batch_idx, -valid:] = torch.arange(valid)
    offset = torch.full((batch_size,), context * 2, dtype=torch.long)
    exec_mask = torch.ones(batch_size, dtype=torch.bool)
    _, _, _, expected_cu_q, expected_cu_k, expected_max_k = _expected_pack(
        q, k, v, cache_k, cache_v, cache_pos
    )
    seen = {}

    def fake_flash(q_pack, k_pack, v_pack, cu_q, cu_k, max_q, max_k, **kwargs):
        seen["cu_q"] = cu_q.cpu().clone()
        seen["cu_k"] = cu_k.cpu().clone()
        seen["max_q"] = max_q
        seen["max_k"] = max_k
        return q_pack

    workspace = LocalCausalVarlenWorkspace.create(
        max_batch_size=batch_size,
        max_chunk_len=chunk_len,
        context=context,
        num_heads=heads,
        head_dim=head_dim,
        device=q.device,
        dtype=q.dtype,
    )

    _ = local_causal_varlen_attention_with_cache(
        q,
        k,
        v,
        cache_k,
        cache_v,
        cache_pos,
        offset,
        exec_mask,
        workspace,
        context=context,
        flash_attn_varlen_func=fake_flash,
    )

    assert torch.equal(seen["cu_q"], expected_cu_q)
    assert torch.equal(seen["cu_k"], expected_cu_k)
    assert seen["max_q"] == chunk_len
    assert seen["max_k"] == expected_max_k
