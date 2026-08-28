# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""TileLang direct candidate expansion and fine-stage score kernel.

Each CTA owns one query row.  It consumes the selected compressed-block ids,
expands each selected block into contiguous token positions, adds the current
partial block, and computes

    score[token] = sum_h dot(query[h], token_key[token])

with FP32 accumulation.  The kernel returns candidate token ids and raw FP32
scores only; it does not perform top-k.  The caller can keep using Paddle's
existing large top-k implementation while this kernel remains focused on
contiguous KV loading and QK score computation.
"""

import paddle

paddle.enable_compat(scope={"tilelang"}, silent=True)

try:
    import tilelang
    from tilelang import language as T

    HAS_TILELANG = True
except ImportError:
    HAS_TILELANG = False


def _tilelang_dtype(tensor):
    if tensor.dtype == paddle.bfloat16:
        return "bfloat16"
    if tensor.dtype == paddle.float16:
        return "float16"
    if tensor.dtype == paddle.float32:
        return "float"
    raise TypeError(
        f"CSA fine score TileLang kernel does not support {tensor.dtype}."
    )


if HAS_TILELANG:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _csa_fine_score_kernel(
        heads,
        dim,
        block_topk,
        ratio,
        candidate_count,
        block_tile=32,
        dtype="bfloat16",
        threads=128,
    ):
        batch = T.dynamic("batch")
        seq_len = T.dynamic("seq_len")
        token_seq_len = T.dynamic("token_seq_len")
        num_blocks = T.dynamic("num_blocks")

        q_shape = [batch, seq_len, heads, dim]
        k_shape = [batch, token_seq_len, dim]
        selected_shape = [batch, seq_len, block_topk]
        starts_shape = [batch, num_blocks]
        range_shape = [batch, seq_len, 2]
        query_offset_shape = [1]
        group_count = block_topk + 1
        num_tiles = ratio // block_tile
        score_shape = [batch, seq_len, group_count, num_tiles, block_tile]
        index_shape = [batch, seq_len, group_count, num_tiles, block_tile]

        @T.prim_func
        def kernel(
            Query: T.Tensor(q_shape, dtype),
            TokenKey: T.Tensor(k_shape, dtype),
            SelectedBlocks: T.Tensor(selected_shape, "int32"),
            BlockStarts: T.Tensor(starts_shape, "int32"),
            ValidRange: T.Tensor(range_shape, "int32"),
            QueryOffset: T.Tensor(query_offset_shape, "int32"),
            Scores: T.Tensor(score_shape, "float"),
            TokenIds: T.Tensor(index_shape, "int32"),
        ):
            with T.Kernel(seq_len, batch, threads=threads) as (bx, by):
                q_shared = T.alloc_shared([heads, dim], dtype)
                q_sum = T.alloc_fragment([dim], "float")
                key_shared = T.alloc_shared([block_tile, dim], dtype)
                products = T.alloc_fragment([block_tile, dim], "float")
                tile_scores = T.alloc_fragment([block_tile], "float")
                tile_ids = T.alloc_fragment([block_tile], "int32")
                duplicate_flags = T.alloc_fragment([block_topk], "float")
                duplicate_sum = T.alloc_fragment([1], "float")
                tile_scores_shared = T.alloc_shared([block_tile], "float")
                tile_ids_shared = T.alloc_shared([block_tile], "int32")

                for h_i, d_i in T.Parallel(heads, dim):
                    q_shared[h_i, d_i] = Query[by, bx, h_i, d_i]
                T.sync_threads()

                T.reduce_sum(q_shared, q_sum, dim=0)

                valid_start = ValidRange[by, bx, 0]
                valid_end = ValidRange[by, bx, 1]
                query_pos = QueryOffset[0] + bx
                current_start = (
                    valid_start + ((query_pos - valid_start) // ratio) * ratio
                )
                for block_i in T.Parallel(block_topk):
                    block_id = SelectedBlocks[by, bx, block_i]
                    safe_block_id = T.max(block_id, 0)
                    selected_start = BlockStarts[by, safe_block_id]
                    duplicate_flags[block_i] = T.if_then_else(
                        (block_id >= 0) & (selected_start == current_start),
                        T.cast(1, "float"),
                        T.cast(0, "float"),
                    )
                T.reduce_sum(
                    duplicate_flags,
                    duplicate_sum,
                    dim=0,
                    clear=True,
                )
                current_duplicate = duplicate_sum[0] > 0

                # Selected complete blocks.  Each iteration reads one
                # contiguous [ratio, dim] token block.
                for block_i in T.serial(block_topk):
                    block_id = SelectedBlocks[by, bx, block_i]
                    safe_block_id = T.max(block_id, 0)
                    block_start = BlockStarts[by, safe_block_id]
                    block_valid = block_id >= 0

                    for tile_i in T.serial(num_tiles):
                        tile_start = block_start + tile_i * block_tile
                        for token_i, d_i in T.Parallel(block_tile, dim):
                            token_pos = tile_start + token_i
                            safe_token_pos = T.if_then_else(
                                (token_pos >= 0) & (token_pos < token_seq_len),
                                token_pos,
                                0,
                            )
                            key_shared[token_i, d_i] = T.if_then_else(
                                block_valid
                                & (token_pos >= valid_start)
                                & (token_pos < valid_end)
                                & (token_pos >= 0)
                                & (token_pos < token_seq_len),
                                TokenKey[by, safe_token_pos, d_i],
                                T.cast(0, dtype),
                            )
                        T.sync_threads()
                        for token_i, d_i in T.Parallel(block_tile, dim):
                            products[token_i, d_i] = (
                                T.cast(key_shared[token_i, d_i], "float")
                                * q_sum[d_i]
                            )
                        T.reduce_sum(products, tile_scores, dim=1)
                        for token_i in T.Parallel(block_tile):
                            token_pos = tile_start + token_i
                            valid = (
                                block_valid
                                & (token_pos >= valid_start)
                                & (token_pos < valid_end)
                                & (token_pos >= 0)
                                & (token_pos < token_seq_len)
                            )
                            tile_scores[token_i] = T.if_then_else(
                                valid,
                                tile_scores[token_i],
                                -T.infinity("float"),
                            )
                            tile_ids[token_i] = T.if_then_else(
                                valid, token_pos, -1
                            )
                        T.copy(tile_scores, tile_scores_shared)
                        T.copy(tile_ids, tile_ids_shared)
                        T.sync_threads()
                        T.copy(
                            tile_scores_shared,
                            Scores[by, bx, block_i, tile_i, :],
                        )
                        T.copy(
                            tile_ids_shared,
                            TokenIds[by, bx, block_i, tile_i, :],
                        )

                # The current block is not guaranteed to exist in the
                # compressed-key buffer, so append it after selected blocks.
                for tile_i in T.serial(num_tiles):
                    tile_start = current_start + tile_i * block_tile
                    for token_i, d_i in T.Parallel(block_tile, dim):
                        token_pos = tile_start + token_i
                        safe_token_pos = T.if_then_else(
                            (token_pos >= 0) & (token_pos < token_seq_len),
                            token_pos,
                            0,
                        )
                        key_shared[token_i, d_i] = T.if_then_else(
                            (not current_duplicate)
                            & (token_pos >= valid_start)
                            & (token_pos < valid_end)
                            & (token_pos >= 0)
                            & (token_pos < token_seq_len),
                            TokenKey[by, safe_token_pos, d_i],
                            T.cast(0, dtype),
                        )
                    T.sync_threads()
                    for token_i, d_i in T.Parallel(block_tile, dim):
                        products[token_i, d_i] = (
                            T.cast(key_shared[token_i, d_i], "float")
                            * q_sum[d_i]
                        )
                    T.reduce_sum(products, tile_scores, dim=1)
                    for token_i in T.Parallel(block_tile):
                        token_pos = tile_start + token_i
                        valid = (
                            (not current_duplicate)
                            & (token_pos >= valid_start)
                            & (token_pos < valid_end)
                            & (token_pos >= 0)
                            & (token_pos < token_seq_len)
                        )
                        tile_scores[token_i] = T.if_then_else(
                            valid,
                            tile_scores[token_i],
                            -T.infinity("float"),
                        )
                        tile_ids[token_i] = T.if_then_else(valid, token_pos, -1)
                    T.copy(tile_scores, tile_scores_shared)
                    T.copy(tile_ids, tile_ids_shared)
                    T.sync_threads()
                    T.copy(
                        tile_scores_shared,
                        Scores[by, bx, block_topk, tile_i, :],
                    )
                    T.copy(
                        tile_ids_shared,
                        TokenIds[by, bx, block_topk, tile_i, :],
                    )

        return kernel


def csa_fine_score_fwd(
    query,
    token_key,
    selected_indices,
    block_starts,
    valid_range,
    ratio,
    query_offset=0,
    block_tile=32,
    threads=128,
):
    """Return raw SUM scores and token ids without performing top-k.

    ``query_offset`` is the global position of the first query in this
    chunk.  It is required when the caller tiles the sequence because the
    current partial block is defined in global token coordinates.
    """
    if not HAS_TILELANG:
        raise RuntimeError("TileLang is required for csa_fine_score_fwd.")
    if query.ndim != 4 or token_key.ndim != 3:
        raise ValueError(
            "query must be [B,Q,H,D] and token_key must be [B,S,D]."
        )
    if selected_indices.ndim != 3:
        raise ValueError("selected_indices must be [B,Q,block_topk].")
    if block_starts.ndim != 2:
        raise ValueError("block_starts must be [B,C].")
    if valid_range.ndim != 3 or valid_range.shape[-1] != 2:
        raise ValueError("valid_range must be [B,Q,2].")
    b, q_len, heads, dim = [int(x) for x in query.shape]
    bk_b, token_len, key_dim = [int(x) for x in token_key.shape]
    si_b, si_q, block_topk = [int(x) for x in selected_indices.shape]
    bs_b, num_blocks = [int(x) for x in block_starts.shape]
    vr_b, vr_q, _ = [int(x) for x in valid_range.shape]
    ratio = int(ratio)
    query_offset = int(query_offset)
    block_tile = int(block_tile)
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}.")
    if block_tile <= 0 or ratio % block_tile != 0:
        raise ValueError(
            f"ratio ({ratio}) must be divisible by block_tile ({block_tile})."
        )
    if (
        b != bk_b
        or b != si_b
        or b != bs_b
        or b != vr_b
        or q_len != si_q
        or q_len != vr_q
        or key_dim != dim
    ):
        raise ValueError("CSA fine score input shapes are incompatible.")
    if block_topk <= 0 or num_blocks <= 0:
        raise ValueError("block_topk and num_blocks must be positive.")

    candidate_count = (block_topk + 1) * ratio
    dtype = _tilelang_dtype(query)
    query = query.contiguous()
    token_key = token_key.contiguous()
    selected_indices = selected_indices.cast("int32").contiguous()
    block_starts = block_starts.cast("int32").contiguous()
    valid_range = valid_range.cast("int32").contiguous()
    query_offset_tensor = paddle.full([1], query_offset, dtype="int32")
    num_tiles = ratio // block_tile
    scores = paddle.empty(
        [b, q_len, block_topk + 1, num_tiles, block_tile],
        dtype="float32",
    )
    token_ids = paddle.empty(
        [b, q_len, block_topk + 1, num_tiles, block_tile],
        dtype="int32",
    )
    kernel = _csa_fine_score_kernel(
        heads=heads,
        dim=dim,
        block_topk=block_topk,
        ratio=ratio,
        candidate_count=candidate_count,
        block_tile=block_tile,
        dtype=dtype,
        threads=int(threads),
    )
    kernel(
        query,
        token_key,
        selected_indices,
        block_starts,
        valid_range,
        query_offset_tensor,
        scores,
        token_ids,
    )
    return (
        scores.reshape([b, q_len, candidate_count]),
        token_ids.reshape([b, q_len, candidate_count]),
    )


__all__ = ["csa_fine_score_fwd", "HAS_TILELANG"]
