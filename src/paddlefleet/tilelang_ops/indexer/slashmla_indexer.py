# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Standalone TileLang token indexer for SlashMLA.

This is intentionally independent from the CSA indexer.  It scores the
absorbed MLA query against the shared latent key and returns token-level
indices.  The final top-k merge is performed by Paddle over bounded key
chunks; TileLang still owns the QK score kernel.
"""

import os

import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)


try:
    import tilelang
    from tilelang import language as T

    HAS_TILELANG = True
except ImportError:
    HAS_TILELANG = False


def _tilelang_dtype(x):
    if x.dtype == paddle.bfloat16:
        return "bfloat16"
    if x.dtype == paddle.float16:
        return "float16"
    if x.dtype == paddle.float32:
        return "float"
    raise TypeError(f"SlashMLA TileLang indexer does not support {x.dtype}.")


if HAS_TILELANG:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _slashmla_score_kernel(
        heads,
        dim,
        block_k,
        key_offset,
        dtype="bfloat16",
        threads=128,
    ):
        batch = T.dynamic("batch")
        seq_len = T.dynamic("seq_len")
        key_seq_len = T.dynamic("key_seq_len")
        q_shape = [batch, seq_len, heads, dim]
        k_shape = [batch, key_seq_len, dim]
        range_shape = [batch, seq_len, 2]
        score_shape = [batch, seq_len, block_k]

        @T.prim_func
        def kernel(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(k_shape, dtype),
            ValidRange: T.Tensor(range_shape, "int32"),
            Scores: T.Tensor(score_shape, "float"),
        ):
            with T.Kernel(seq_len, batch, threads=threads) as (bx, by):
                q_shared = T.alloc_shared([heads, dim], dtype)
                k_shared = T.alloc_shared([block_k, dim], dtype)
                logits = T.alloc_fragment([heads, block_k], "float")
                scores = T.alloc_fragment([block_k], "float")

                for h_i, d_i in T.Parallel(heads, dim):
                    q_shared[h_i, d_i] = Q[by, bx, h_i, d_i]

                valid_start = ValidRange[by, bx, 0]
                valid_end = ValidRange[by, bx, 1]
                for k_i, d_i in T.Parallel(block_k, dim):
                    pos = key_offset + k_i
                    k_shared[k_i, d_i] = T.if_then_else(
                        pos < key_seq_len,
                        K[by, pos, d_i],
                        0,
                    )

                T.gemm(
                    q_shared,
                    k_shared,
                    logits,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.reduce_max(logits, scores, dim=0)
                for k_i in T.Parallel(block_k):
                    pos = key_offset + k_i
                    valid = (
                        (pos >= valid_start)
                        & (pos < valid_end)
                        & (pos < key_seq_len)
                    )
                    Scores[by, bx, k_i] = T.if_then_else(
                        valid,
                        scores[k_i],
                        -T.infinity("float"),
                    )

        return kernel


def slashmla_tilelang_topk(query, key, topk, valid_range):
    """Return token-level ``[B, S, K]`` indices using the standalone kernel."""
    if not HAS_TILELANG:
        raise RuntimeError("TileLang is required for SlashMLA indexer.")
    if query.ndim != 4 or key.ndim != 3:
        raise ValueError(
            "SlashMLA TileLang indexer expects query [B,S,H,D] and "
            f"key [B,S,D], got {list(query.shape)} and {list(key.shape)}."
        )
    if valid_range.shape != [query.shape[0], query.shape[1], 2]:
        raise ValueError(
            f"valid_range must be [B,S,2], got {list(valid_range.shape)}."
        )

    b, s, h, d = query.shape
    sk = key.shape[1]
    requested = min(int(topk), sk)
    if requested <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")
    block_k = int(os.environ.get("FLEET_SLASHMLA_TILELANG_BLOCK_K", "256"))
    if block_k <= 0 or block_k & (block_k - 1):
        raise ValueError(
            "FLEET_SLASHMLA_TILELANG_BLOCK_K must be a power of 2."
        )
    threads = int(os.environ.get("FLEET_SLASHMLA_TILELANG_THREADS", "128"))
    dtype = _tilelang_dtype(query)
    query = query.contiguous()
    key = key.contiguous()
    valid_range = valid_range.cast("int32").contiguous()

    best_scores = paddle.full([b, s, requested], -float("inf"), dtype="float32")
    best_indices = paddle.full([b, s, requested], -1, dtype="int64")
    for offset in range(0, sk, block_k):
        kernel = _slashmla_score_kernel(
            heads=h,
            dim=d,
            block_k=block_k,
            key_offset=offset,
            dtype=dtype,
            threads=threads,
        )
        scores = paddle.empty([b, s, block_k], dtype="float32")
        kernel(query, key, valid_range, scores)
        cols = paddle.arange(offset, offset + block_k, dtype="int64").reshape(
            [1, 1, block_k]
        )
        cols = cols.expand([b, s, block_k])
        merged_scores = paddle.concat([best_scores, scores], axis=-1)
        merged_indices = paddle.concat([best_indices, cols], axis=-1)
        best_scores, positions = paddle.topk(
            merged_scores, k=requested, axis=-1
        )
        best_indices = paddle.take_along_axis(
            merged_indices, positions, axis=-1
        )
    best_indices = paddle.where(
        paddle.isfinite(best_scores),
        best_indices,
        paddle.full_like(best_indices, -1),
    )
    return best_indices.cast("int32")
