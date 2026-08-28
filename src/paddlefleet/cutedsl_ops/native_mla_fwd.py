# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Forward-only native MLA attention using the SM90 HMMA CuTe kernel.

The producer warpgroup consumes the token-level index tensor directly and
fills a double-buffered shared K/V tile.  The public tensors remain native:

* Q is ``[B, S, 16, 192]``;
* K is ``[B, Skv, 192]``;
* V is ``[B, Skv, 128]``;
* indices are ``[B, S, topk]``.

No gathered K/V workspace or separate gather launch is used by the primary
entry point.
"""

import time
from typing import Final

import paddle
from cutlass import BFloat16, Float16, Float32, cute

from .dlpack import paddle_to_cute_tensor


def _load_hmma_kernel():
    """Build the PaddleFleet-owned HMMA copy without importing cudnn."""
    from .native_mla_hmma import HopperSelectAttentionFwd

    return HopperSelectAttentionFwd


HopperSelectAttentionFwd = _load_hmma_kernel()


_BLOCK_SIZE: Final = 16
_COMPILED: dict[tuple[object, ...], object] = {}


def _trace_stage(stage: str) -> None:
    print(
        f"[native_mla_cutedsl] {stage} t={time.perf_counter():.3f}",
        flush=True,
    )


def _cutlass_dtype(dtype):
    dtype_name = str(dtype).split(".")[-1]
    if dtype in ("bfloat16", paddle.bfloat16) or dtype_name == "bfloat16":
        return BFloat16
    if dtype in ("float16", paddle.float16) or dtype_name == "float16":
        return Float16
    raise TypeError(
        f"native MLA CuTe forward expects float16/bfloat16, got {dtype}"
    )


def _check_inputs(q, k, v, indices):
    if HopperSelectAttentionFwd is None:
        raise ImportError("CuTe native MLA local HMMA kernel is unavailable")
    if q.ndim != 4 or k.ndim != 3 or v.ndim != 3 or indices.ndim != 3:
        raise ValueError(
            "expected Q[B,S,H,D], K[B,Sk,D], V[B,Sk,Dv], indices[B,S,K]"
        )
    b, s, h, dqk = q.shape
    if h != 16 or dqk != 192 or k.shape[0] != b or v.shape[0] != b:
        raise ValueError(
            "native CuTe MLA currently supports Q[B,S,16,192], "
            "K[B,Sk,192], and V[B,Sk,128]"
        )
    if k.shape[2] != 192 or v.shape[2] != 128:
        raise ValueError("native CuTe MLA requires K D=192 and V D=128")
    if tuple(indices.shape[:2]) != (b, s):
        raise ValueError("indices must have shape [B,S,topk]")
    if indices.dtype != paddle.int32:
        raise TypeError("indices must be int32")
    valid = (indices >= 0) & (indices < k.shape[1])
    if not bool(valid.all()):
        raise ValueError(
            "native CuTe MLA direct-indexed forward currently requires "
            "fully populated token rows; invalid indices need a masked "
            "producer path"
        )
    if q.dtype not in (paddle.float16, paddle.bfloat16):
        raise TypeError("Q/K/V must use float16 or bfloat16")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Q, K, and V must have the same dtype")
    if indices.shape[2] <= 0 or indices.shape[2] % _BLOCK_SIZE:
        raise ValueError("topk must be a positive multiple of 16")


def _make_gathered_kv(k, v, indices):
    """Gather token rows and return a 16-row-tiled KV workspace."""
    b, s, topk = indices.shape
    sk = k.shape[1]
    flat_k = k.reshape([b * sk, 192])
    flat_v = v.reshape([b * sk, 128])
    valid = (indices >= 0) & (indices < sk)
    if not bool(valid.all()):
        raise ValueError(
            "native CuTe MLA currently requires fully populated token top-k "
            "rows; pass a row mask/fused gather for causal prefix rows"
        )
    flat_indices = indices.reshape([-1]).cast("int64")
    gathered_k = paddle.gather(flat_k, flat_indices, axis=0).reshape(
        [b, s * topk, 192]
    )
    gathered_v = paddle.gather(flat_v, flat_indices, axis=0).reshape(
        [b, s * topk, 128]
    )
    return gathered_k.contiguous(), gathered_v.contiguous()


def _make_block_metadata(batch_size, seq_len, topk):
    blocks_per_row = topk // _BLOCK_SIZE
    row_bases = (
        paddle.arange(seq_len, dtype="int32").reshape([seq_len, 1])
        * blocks_per_row
    )
    block_ids = row_bases + paddle.arange(
        blocks_per_row, dtype="int32"
    ).reshape([1, blocks_per_row])
    block_ids = block_ids.tile([batch_size, 1, 1]).reshape(
        [batch_size * seq_len, 1, blocks_per_row]
    )
    block_counts = paddle.full(
        [batch_size * seq_len, 1], blocks_per_row, dtype="int32"
    )
    topk_lengths = paddle.full([batch_size * seq_len, 1], topk, dtype="int32")
    q_offsets = paddle.arange(batch_size + 1, dtype="int32") * seq_len
    kv_offsets = paddle.arange(batch_size + 1, dtype="int32") * (seq_len * topk)
    return (
        block_ids.contiguous(),
        block_counts.contiguous(),
        topk_lengths.contiguous(),
        q_offsets.contiguous(),
        kv_offsets.contiguous(),
    )


def _make_index_metadata(indices, kv_len):
    """Flatten per-batch token indices without materializing K/V."""
    b, s, topk = indices.shape
    index_rows = indices.reshape([b * s, 1, topk]).contiguous()
    blocks_per_row = topk // _BLOCK_SIZE
    block_counts = paddle.full([b * s, 1], blocks_per_row, dtype="int32")
    topk_lengths = (
        (indices >= 0).astype("int32").sum(axis=-1).reshape([b * s, 1])
    )
    block_counts = paddle.ceil(topk_lengths.cast("float32") / _BLOCK_SIZE).cast(
        "int32"
    )
    q_offsets = paddle.arange(b + 1, dtype="int32") * s
    kv_offsets = paddle.arange(b + 1, dtype="int32") * kv_len
    return (
        index_rows,
        block_counts.contiguous(),
        topk_lengths.contiguous(),
        q_offsets.contiguous(),
        kv_offsets.contiguous(),
    )


def _compile_native_mla(
    q,
    k,
    v,
    o,
    lse,
    row_max,
    block_ids,
    block_counts,
    topk_lengths,
    attn_sink,
    q_offsets,
    kv_offsets,
    seq_len,
    sm_scale,
):
    key = (
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        tuple(block_ids.shape),
        q.dtype,
    )
    compiled = _COMPILED.get(key)
    if compiled is not None:
        return compiled

    kernel = HopperSelectAttentionFwd(
        head_dim=192,
        value_dim=128,
        GQA_group_size=16,
        block_size=_BLOCK_SIZE,
        index_tiles=block_ids.shape[2],
        dtype=_cutlass_dtype(q.dtype),
        acc_dtype=Float32,
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    _trace_stage("before_compile")
    compiled = cute.compile(
        kernel,
        Q=paddle_to_cute_tensor(q, assumed_align=16, leading_dim=3),
        K=paddle_to_cute_tensor(k, assumed_align=16, leading_dim=3),
        V=paddle_to_cute_tensor(v, assumed_align=16, leading_dim=3),
        O=paddle_to_cute_tensor(o, assumed_align=16, leading_dim=3),
        L=paddle_to_cute_tensor(lse, assumed_align=4, leading_dim=2),
        M=paddle_to_cute_tensor(row_max, assumed_align=4, leading_dim=2),
        block_indices=paddle_to_cute_tensor(
            block_ids, assumed_align=4, leading_dim=2
        ),
        block_counts=paddle_to_cute_tensor(
            block_counts, assumed_align=4, leading_dim=1
        ),
        topk_lengths=paddle_to_cute_tensor(
            topk_lengths, assumed_align=4, leading_dim=1
        ),
        attn_sink=paddle_to_cute_tensor(
            attn_sink, assumed_align=4, leading_dim=0
        ),
        max_length=seq_len,
        seq_offsets_q=paddle_to_cute_tensor(
            q_offsets, assumed_align=4, leading_dim=0
        ),
        seq_offsets_k=paddle_to_cute_tensor(
            kv_offsets, assumed_align=4, leading_dim=0
        ),
        softmax_scale=Float32(sm_scale),
        stream=fake_stream,
        options="--enable-tvm-ffi",
    )
    _trace_stage("after_compile")
    _COMPILED[key] = compiled
    return compiled


def native_mla_cutedsl_fwd_prepared(
    q,
    gathered_k,
    gathered_v,
    block_ids,
    block_counts,
    q_offsets,
    kv_offsets,
    *,
    sm_scale=None,
):
    """Run the CuTe kernel after token gather has been materialized."""
    b, s, _, _ = q.shape
    topk = gathered_k.shape[1] // s
    sm_scale = 192**-0.5 if sm_scale is None else float(sm_scale)

    q_flat = q.reshape([1, b * s, 16, 192]).contiguous()
    k_flat = gathered_k.reshape([1, b * s * topk, 1, 192]).contiguous()
    v_flat = gathered_v.reshape([1, b * s * topk, 1, 128]).contiguous()
    o_flat = paddle.empty([1, b * s, 16, 128], dtype=q.dtype)
    lse = paddle.empty([1, b * s, 16], dtype="float32")
    row_max = paddle.empty([1, b * s, 16], dtype="float32")

    # Keep this helper for benchmark/regression callers.  It now feeds the
    # same direct-indexed kernel using identity token indices into the
    # already-materialized workspace.
    token_ids = paddle.arange(b * s * topk, dtype="int32").reshape(
        [b * s, 1, topk]
    )
    topk_lengths = block_counts * _BLOCK_SIZE
    attn_sink = paddle.full([16], -1e30, dtype="float32")
    compiled = _compile_native_mla(
        q_flat,
        k_flat,
        v_flat,
        o_flat,
        lse,
        row_max,
        token_ids,
        block_counts,
        topk_lengths,
        attn_sink,
        q_offsets,
        kv_offsets,
        s,
        sm_scale,
    )
    _trace_stage("before_launch")
    compiled(
        q_flat,
        k_flat,
        v_flat,
        o_flat,
        lse,
        row_max,
        token_ids,
        block_counts,
        topk_lengths,
        attn_sink,
        s,
        q_offsets,
        kv_offsets,
        Float32(sm_scale),
    )
    _trace_stage("after_launch")
    return o_flat.reshape([b, s, 16, 128])


def native_mla_cutedsl_fwd(
    q,
    k,
    v,
    indices,
    *,
    sm_scale=None,
):
    """Run native MLA sparse forward with CuTe HMMA on SM90."""
    _check_inputs(q, k, v, indices)
    b, s, _, _ = q.shape
    index_rows, block_counts, topk_lengths, q_offsets, kv_offsets = (
        _make_index_metadata(indices, k.shape[1])
    )
    q_flat = q.reshape([1, b * s, 16, 192]).contiguous()
    k_flat = k.reshape([1, b * k.shape[1], 1, 192]).contiguous()
    v_flat = v.reshape([1, b * v.shape[1], 1, 128]).contiguous()
    o_flat = paddle.empty([1, b * s, 16, 128], dtype=q.dtype)
    lse = paddle.empty([1, b * s, 16], dtype="float32")
    row_max = paddle.empty([1, b * s, 16], dtype="float32")
    scale = 192**-0.5 if sm_scale is None else float(sm_scale)
    attn_sink = paddle.full([16], -1e30, dtype="float32")
    compiled = _compile_native_mla(
        q_flat,
        k_flat,
        v_flat,
        o_flat,
        lse,
        row_max,
        index_rows,
        block_counts,
        topk_lengths,
        attn_sink,
        q_offsets,
        kv_offsets,
        s,
        scale,
    )
    _trace_stage("before_launch")
    compiled(
        q_flat,
        k_flat,
        v_flat,
        o_flat,
        lse,
        row_max,
        index_rows,
        block_counts,
        topk_lengths,
        attn_sink,
        s,
        q_offsets,
        kv_offsets,
        Float32(scale),
    )
    _trace_stage("after_launch")
    return o_flat.reshape([b, s, 16, 128])


__all__ = ["native_mla_cutedsl_fwd", "native_mla_cutedsl_fwd_prepared"]
