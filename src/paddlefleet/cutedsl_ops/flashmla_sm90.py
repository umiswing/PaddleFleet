# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CuTeDSL SM90 FlashMLA-style sparse-prefill forward.

This is the first local migration step for FlashMLA's SM90 sparse prefill
phase1.  The device implementation is deliberately kept separate from the
production DSA path while it is being brought up.

The execution shape mirrors FlashMLA phase1:

* two consumer warp groups perform QK, online softmax and PV;
* one producer warp group directly gathers indexed K/V rows into staged
  shared memory;
* Q is moved with TMA and the indexed K/V path uses ordinary global loads;
* the value output is normalized by the online-softmax accumulator.

The current local HMMA implementation processes four ``GQA=16`` groups for
the shared MQA KV head.  This preserves the absorbed-MQA result for
``H=64`` while leaving the exact FlashMLA ``B_H=64`` WGMMA schedule as the
next optimization step.

Supported first-port layout:

* Q: ``[B, S, 64, 576]``
* K: ``[B, Sk, 576]``
* V: ``[B, Sk, 512]``
* indices: ``[B, S, TopK]`` int32, valid prefix followed by ``-1``
* output: ``[B, S, 64, 512]``

The producer performs safe loads for invalid entries and the consumer masks
their logits.  This module does not modify or import the third-party FlashMLA
implementation.
"""

from __future__ import annotations

from typing import Final

import paddle
from cutlass import BFloat16, Float16, Float32, cute

from .dlpack import paddle_to_cute_tensor
from .native_mla_hmma import HopperSelectAttentionFwd

_QK_DIM: Final = 576
_V_DIM: Final = 512
_Q_HEADS: Final = 64
_KV_HEADS: Final = 1
_GQA_GROUP: Final = _Q_HEADS // _KV_HEADS
_TOPK_BLOCK: Final = 64

_COMPILED: dict[tuple[object, ...], object] = {}


def _cutlass_dtype(dtype):
    if dtype in (paddle.bfloat16, "bfloat16"):
        return BFloat16
    if dtype in (paddle.float16, "float16"):
        return Float16
    raise TypeError(f"FlashMLA CuTeDSL expects float16/bfloat16, got {dtype}")


def _check_inputs(q, kv, indices, topk_lengths, attn_sink):
    if q.ndim != 4 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError(
            "expected q[B,S,H,Dqk], kv[B,Sk,Dqk], indices[B,S,topk]"
        )
    b, s, h, dqk = q.shape
    if h != _Q_HEADS or dqk != _QK_DIM:
        raise ValueError(
            "FlashMLA SM90 first port requires "
            f"q[B,S,{_Q_HEADS},{_QK_DIM}], got {list(q.shape)}"
        )
    if kv.shape[0] != b or kv.shape[2] != _QK_DIM:
        raise ValueError(f"kv must be [B,Sk,{_QK_DIM}], got {list(kv.shape)}")
    if tuple(indices.shape[:2]) != (b, s):
        raise ValueError("indices must have shape [B,S,topk]")
    if indices.dtype != paddle.int32:
        raise TypeError("indices must be int32")
    if indices.shape[2] <= 0 or indices.shape[2] % _TOPK_BLOCK:
        raise ValueError(
            f"topk must be a positive multiple of {_TOPK_BLOCK}; "
            f"got {indices.shape[2]}"
        )
    invalid = (indices < -1) | (indices >= kv.shape[1])
    if bool(invalid.any()):
        raise ValueError("indices must contain valid token ids or -1 padding")
    if topk_lengths is not None:
        if topk_lengths.shape != [b, s]:
            raise ValueError("topk_lengths must have shape [B,S]")
        if topk_lengths.dtype != paddle.int32:
            raise TypeError("topk_lengths must be int32")
        if bool((topk_lengths < 0).any()) or bool(
            (topk_lengths > indices.shape[2]).any()
        ):
            raise ValueError("topk_lengths must lie in [0, topk]")
    if attn_sink is not None:
        if attn_sink.shape != [_Q_HEADS]:
            raise ValueError(f"attn_sink must have shape [{_Q_HEADS}]")
        if attn_sink.dtype != paddle.float32:
            raise TypeError("attn_sink must be float32")
    if q.dtype not in (paddle.float16, paddle.bfloat16):
        raise TypeError("q and kv must use float16 or bfloat16")
    if kv.dtype != q.dtype:
        raise TypeError("q and kv must have the same dtype")


def _make_metadata(indices, topk_lengths, kv_len):
    b, s, topk = indices.shape
    # The local HMMA kernel consumes [flattened query, kv-head, token].  The
    # four KV heads are views of one shared MQA tensor, so this expansion does
    # not duplicate the KV allocation.
    index_rows = indices.reshape([b * s, 1, topk]).expand(
        [b * s, _KV_HEADS, topk]
    )
    index_rows = index_rows.contiguous()
    if topk_lengths is None:
        topk_lengths = (indices >= 0).astype("int32").sum(axis=-1)
    topk_lengths = (
        topk_lengths.reshape([b * s, 1]).expand([b * s, _KV_HEADS]).contiguous()
    )
    block_counts = paddle.ceil(topk_lengths.cast("float32") / _TOPK_BLOCK).cast(
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


def _compile(
    q,
    kv,
    v,
    out,
    lse,
    row_max,
    indices,
    block_counts,
    topk_lengths,
    attn_sink,
    q_offsets,
    kv_offsets,
    seq_len,
    sm_scale,
    reuse_kv,
    kv_stages,
):
    key = (
        tuple(q.shape),
        tuple(kv.shape),
        tuple(out.shape),
        tuple(indices.shape),
        q.dtype,
        bool(reuse_kv),
        int(kv_stages),
    )
    compiled = _COMPILED.get(key)
    if compiled is not None:
        return compiled

    kernel = HopperSelectAttentionFwd(
        head_dim=_QK_DIM,
        value_dim=_V_DIM,
        GQA_group_size=_GQA_GROUP,
        block_size=_TOPK_BLOCK,
        index_tiles=indices.shape[2],
        dtype=_cutlass_dtype(q.dtype),
        acc_dtype=Float32,
        kv_stages=kv_stages,
        reuse_kv=reuse_kv,
        query_tiles=1,
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel,
        Q=paddle_to_cute_tensor(q, assumed_align=16, leading_dim=3),
        K=paddle_to_cute_tensor(kv, assumed_align=16, leading_dim=3),
        V=paddle_to_cute_tensor(v, assumed_align=16, leading_dim=3),
        O=paddle_to_cute_tensor(out, assumed_align=16, leading_dim=3),
        L=paddle_to_cute_tensor(lse, assumed_align=4, leading_dim=2),
        M=paddle_to_cute_tensor(row_max, assumed_align=4, leading_dim=2),
        block_indices=paddle_to_cute_tensor(
            indices, assumed_align=4, leading_dim=2
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
    _COMPILED[key] = compiled
    return compiled


def flashmla_sm90_sparse_prefill(
    q,
    kv,
    indices,
    *,
    sm_scale=None,
    topk_lengths=None,
    attn_sink=None,
    reuse_kv=True,
    kv_stages=1,
):
    """Run the standalone local FlashMLA-style SM90 sparse-prefill forward.

    This entry is intentionally forward-only and is not wired into
    ``slashmla_sparse_attention`` yet.  It returns the latent output and the
    natural-log LSE:

    ``output: [B,S,64,512]``
    ``lse:    [B,S,64]``

    ``reuse_kv`` selects the experimental single-stage shared-memory reuse
    schedule.  It defaults to ``True``; ``False`` is only intended for
    standalone A/B validation against the independent K/V baseline.
    ``kv_stages`` controls the compile-time K/V stage count and is currently
    restricted to one or two stages.
    """
    _check_inputs(q, kv, indices, topk_lengths, attn_sink)
    if not isinstance(reuse_kv, bool):
        raise TypeError("reuse_kv must be a compile-time bool")
    if not isinstance(kv_stages, int) or kv_stages not in (1, 2):
        raise ValueError("kv_stages must be one of {1, 2}")
    b, s, _, _ = q.shape
    scale = _QK_DIM**-0.5 if sm_scale is None else float(sm_scale)
    index_rows, block_counts, topk_lengths, q_offsets, kv_offsets = (
        _make_metadata(indices, topk_lengths, kv.shape[1])
    )

    q_flat = q.reshape([1, b * s, _Q_HEADS, _QK_DIM]).contiguous()
    kv_flat = kv.reshape([1, b * kv.shape[1], 1, _QK_DIM]).expand(
        [1, b * kv.shape[1], _KV_HEADS, _QK_DIM]
    )
    kv_flat = kv_flat.contiguous()
    # The producer reads V as a separate tensor.  Use a zero-copy view with
    # the leading absorbed latent channels and the replicated KV-head stride.
    v_flat = (
        kv[..., :_V_DIM]
        .reshape([1, b * kv.shape[1], 1, _V_DIM])
        .expand([1, b * kv.shape[1], _KV_HEADS, _V_DIM])
    )
    v_flat = v_flat.contiguous()
    if attn_sink is None:
        attn_sink = paddle.full([_Q_HEADS], -1e30, dtype="float32")
    else:
        attn_sink = attn_sink.contiguous()
    out = paddle.empty([1, b * s, _Q_HEADS, _V_DIM], dtype=q.dtype)
    row_sum = paddle.empty([1, b * s, _Q_HEADS], dtype="float32")
    row_max = paddle.empty([1, b * s, _Q_HEADS], dtype="float32")

    compiled = _compile(
        q_flat,
        kv_flat,
        v_flat,
        out,
        row_sum,
        row_max,
        index_rows,
        block_counts,
        topk_lengths,
        attn_sink,
        q_offsets,
        kv_offsets,
        s,
        scale,
        reuse_kv,
        kv_stages,
    )
    compiled(
        q_flat,
        kv_flat,
        v_flat,
        out,
        row_sum,
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
    row_sum = row_sum.reshape([b, s, _Q_HEADS])
    row_max = row_max.reshape([b, s, _Q_HEADS])
    lse = paddle.where(
        row_sum > 0.0,
        paddle.log(row_sum) + row_max,
        paddle.full_like(row_sum, float("-inf")),
    )
    return out.reshape([b, s, _Q_HEADS, _V_DIM]), lse


__all__ = ["flashmla_sm90_sparse_prefill"]
