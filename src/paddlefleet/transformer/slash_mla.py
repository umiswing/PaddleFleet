# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Token-sparse absorbed MLA and HCA+MLA helpers.

The original SLASH implementation was written for shared-KV VHA.  This module
keeps its token-level routing contract, but scores the absorbed MLA query
against the shared latent key and uses the absorbed-MQA sparse backend for the
actual attention.

The indexer is deliberately kept behind small function boundaries. The legacy
TileLang implementation returns token-level indices directly, while the
HCA+MLA path uses a no-grad Paddle hierarchical indexer: compressed-block
coarse top-k followed by token-level fine top-k inside the selected blocks.
"""

from __future__ import annotations

import os
import warnings

import paddle


def _is_slashmla_flash_mla_available() -> bool:
    """Whether the FlashMLA sparse forward kernel is available.

    SlashMLA uses the local CuTeDSL backward on SM90, so it must not reuse
    ``is_dsa_available``: that helper deliberately requires SM100+ and also
    checks the cuDNN DSA backward dependency. SlashMLA only needs the FlashMLA
    sparse forward operator here.
    """
    try:
        import paddlefleet_ops

        from paddlefleet.cudnn_ops.attn import csa_sparse_attn_fwd_cudnn

        if (
            not paddlefleet_ops.is_flash_mla_available()
            or csa_sparse_attn_fwd_cudnn._flash_mla_sparse_fwd is None
        ):
            return False
    except (ImportError, RuntimeError, AttributeError):
        return False
    return True


class SlashMLAHCACompressor(paddle.nn.Layer):
    """Non-overlapping HCA pooling over the shared absorbed MLA token KV.

    The compressor keeps the token-KV representation unchanged: it learns only
    a per-channel gate (plus an intra-block positional bias) and pools each
    complete ``ratio``-token block into one KV vector. The gate and positional
    bias are zero-initialized, so the initial compressor is an exact block mean.
    """

    def __init__(self, key_dim, ratio=128):
        super().__init__()
        key_dim = int(key_dim)
        ratio = int(ratio)
        if key_dim <= 0:
            raise ValueError(f"HCA key_dim must be positive, got {key_dim}.")
        if ratio <= 0:
            raise ValueError(f"HCA ratio must be positive, got {ratio}.")
        self.key_dim = key_dim
        self.ratio = ratio
        self.gate_proj = paddle.nn.Linear(
            key_dim,
            key_dim,
            bias_attr=False,
            weight_attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.0)
            ),
        )
        self.ape = self.create_parameter(
            shape=[ratio, key_dim],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0.0),
        )

    def forward(self, token_kv, mask=None):
        if token_kv.ndim != 3:
            raise ValueError(
                "SlashMLA HCA compressor expects token KV [B,S,D], got "
                f"{list(token_kv.shape)}."
            )
        if int(token_kv.shape[-1]) != self.key_dim:
            raise ValueError(
                "SlashMLA HCA compressor key width mismatch: expected "
                f"{self.key_dim}, got {token_kv.shape[-1]}."
            )

        b, s, d = [int(x) for x in token_kv.shape]
        block_starts = _hca_block_starts(mask, b, s, self.ratio)
        n_blocks = int(block_starts.shape[1])
        if n_blocks == 0:
            return paddle.zeros([b, 0, d], dtype=token_kv.dtype), block_starts

        offsets = paddle.arange(self.ratio, dtype="int64").reshape(
            [1, 1, self.ratio]
        )
        token_indices = block_starts.unsqueeze(-1) + offsets
        block_valid = block_starts >= 0
        token_indices = paddle.where(
            block_valid.unsqueeze(-1),
            token_indices,
            paddle.full_like(token_indices, -1),
        )
        block_tokens = _batched_gather_tokens(token_kv, token_indices)

        gate_input = block_tokens.cast(self.gate_proj.weight.dtype)
        gate_logits = self.gate_proj(gate_input).cast("float32")
        gate_logits = gate_logits + self.ape.reshape(
            [1, 1, self.ratio, self.key_dim]
        )
        weights = paddle.nn.functional.softmax(gate_logits, axis=2)
        weights = weights * block_valid.reshape([b, n_blocks, 1, 1]).cast(
            weights.dtype
        )
        compressed = (block_tokens.cast(weights.dtype) * weights).sum(axis=2)
        return compressed.cast(token_kv.dtype), block_starts


def _env_enabled(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _document_metadata(mask, batch_size, seq_len):
    """Return broadcast document ends and starts for a flashmask layout."""
    if mask is None:
        return None, paddle.zeros([batch_size, seq_len], dtype="int64")
    if mask.ndim == 4:
        if (
            mask.shape[0] not in (1, batch_size)
            or mask.shape[2] != seq_len
            or mask.shape[3] != 1
        ):
            raise ValueError(
                "SlashMLA expects attn_mask_startend_row_indices with shape "
                f"[1|batch, num_masks, seq, 1], got {list(mask.shape)}."
            )
        doc_ends = mask[:, 0, :, 0]
    elif mask.ndim == 2:
        if mask.shape[0] not in (1, batch_size) or mask.shape[1] != seq_len:
            raise ValueError(
                "SlashMLA expects a 2-D mask with shape [1|batch, seq], "
                f"got {list(mask.shape)}."
            )
        doc_ends = mask
    else:
        raise ValueError(
            "SlashMLA document mask must be 2-D or 4-D, "
            f"got ndim={mask.ndim} shape={list(mask.shape)}."
        )
    doc_ends = doc_ends.cast("int64")
    if doc_ends.shape[0] == 1 and batch_size > 1:
        doc_ends = doc_ends.expand([batch_size, seq_len])

    positions = paddle.arange(seq_len, dtype="int64").unsqueeze(0)
    positions = positions.expand([batch_size, seq_len])
    boundary = paddle.zeros([batch_size, seq_len], dtype="bool")
    boundary[:, 0] = True
    boundary[:, 1:] = (positions[:, 1:] == doc_ends[:, :-1]) & (
        doc_ends[:, 1:] != doc_ends[:, :-1]
    )
    starts = paddle.cummax(
        paddle.where(boundary, positions, paddle.zeros_like(positions)), axis=1
    )[0]
    return doc_ends, starts


def _hca_block_starts(mask, batch_size, seq_len, ratio):
    """Return compact document-local starts for complete HCA blocks."""
    ratio = int(ratio)
    max_blocks = int(seq_len) // ratio
    if max_blocks == 0:
        return paddle.empty([batch_size, 0], dtype="int64")

    doc_ends, doc_starts = _document_metadata(mask, batch_size, seq_len)
    if doc_ends is None:
        doc_ends = paddle.full([batch_size, seq_len], seq_len, dtype="int64")
    positions = paddle.arange(seq_len, dtype="int64").reshape([1, seq_len])
    positions = positions.expand([batch_size, seq_len])
    pos_in_doc = positions - doc_starts
    is_complete_start = (pos_in_doc % ratio == 0) & (
        positions + ratio <= doc_ends
    )
    candidates = paddle.where(
        is_complete_start,
        positions,
        paddle.full_like(positions, seq_len),
    )
    candidates = paddle.sort(candidates, axis=1)[:, :max_blocks]
    return paddle.where(
        candidates < seq_len,
        candidates,
        paddle.full_like(candidates, -1),
    )


def _batched_gather_tokens(token_kv, token_indices):
    """Gather [B,...] token rows without expanding a [B,S,...] source."""
    b, s, d = [int(x) for x in token_kv.shape]
    indices = token_indices.cast("int64")
    valid = indices >= 0
    safe = indices.clip(0, max(s - 1, 0))
    batch_offsets = paddle.arange(b, dtype="int64").reshape(
        [b] + [1] * (indices.ndim - 1)
    )
    flat_indices = safe + batch_offsets * s
    gathered = paddle.gather(
        token_kv.reshape([b * s, d]), flat_indices.flatten(), axis=0
    ).reshape([*list(indices.shape), d])
    return gathered * valid.unsqueeze(-1).cast(gathered.dtype)


def build_slashmla_offsets(mask, batch_size, seq_len):
    """Build the packed-document offsets used by the legacy SLASH contract."""
    if mask is None:
        return paddle.arange(0, batch_size + 1, dtype="int32") * seq_len
    _, starts = _document_metadata(mask, batch_size, seq_len)
    offsets = []
    for batch_idx in range(batch_size):
        boundary = paddle.zeros([seq_len], dtype="bool")
        boundary[0] = True
        boundary[1:] = starts[batch_idx, 1:] != starts[batch_idx, :-1]
        batch_starts = paddle.nonzero(boundary).flatten().cast("int32")
        offsets.append(batch_starts + batch_idx * seq_len)
    offsets.append(paddle.full([1], batch_size * seq_len, dtype="int32"))
    return paddle.concat(offsets)


def _valid_range(mask, batch_size, seq_len):
    """Return document-relative causal [start, end) ranges."""
    _, starts = _document_metadata(mask, batch_size, seq_len)
    positions = paddle.arange(seq_len, dtype="int64").unsqueeze(0)
    ends = positions.expand([batch_size, seq_len]) + 1
    return paddle.stack([starts, ends], axis=-1).cast("int32")


def _validate_valid_range(valid_range, batch_size, seq_len, key_len):
    expected = [batch_size, seq_len, 2]
    if list(valid_range.shape) != expected:
        raise ValueError(
            f"SlashMLA valid_range must have shape {expected}, got "
            f"{list(valid_range.shape)}."
        )
    positions = paddle.arange(seq_len, dtype="int64").reshape([1, seq_len])
    positions = positions.expand([batch_size, seq_len])
    ranges = valid_range.cast("int64")
    invalid = (
        (ranges[:, :, 0] < 0)
        | (ranges[:, :, 0] > positions)
        | (ranges[:, :, 1] <= positions)
        | (ranges[:, :, 1] < ranges[:, :, 0])
        | (ranges[:, :, 1] > key_len)
    )
    if bool(paddle.any(invalid).item()):
        raise ValueError(
            "SlashMLA valid_range must satisfy "
            "0 <= start <= query_position < end <= key_length."
        )


def _paddle_topk(query, key, topk, valid_range, chunk_size=512):
    """Memory-bounded token top-k fallback.

    ``query`` is [B, S, H, D] and ``key`` is [B, S, D].  The head-wise
    selection is reduced with max, matching the shared-token routing semantics
    needed by the downstream SWA layers.
    """
    b, sq, _, dim = query.shape
    sk = key.shape[1]
    k = min(int(topk), sk)
    q = query.transpose([0, 2, 1, 3]).cast("float32")
    best_scores = paddle.full([b, sq, k], -float("inf"), dtype="float32")
    best_indices = paddle.full([b, sq, k], -1, dtype="int32")
    ranges = valid_range.cast("int64")
    for start in range(0, sk, chunk_size):
        end = min(start + chunk_size, sk)
        scores = paddle.matmul(
            q,
            key[:, start:end, :]
            .cast("float32")
            .transpose([0, 2, 1])
            .unsqueeze(1),
        ).max(axis=1)
        columns = paddle.arange(start, end, dtype="int64").reshape([1, 1, -1])
        columns = columns.expand([b, sq, end - start])
        allowed = (columns >= ranges[:, :, 0:1]) & (columns < ranges[:, :, 1:2])
        scores = paddle.where(
            allowed, scores, paddle.full_like(scores, -float("inf"))
        )
        merged_scores = paddle.concat([best_scores, scores], axis=-1)
        merged_indices = paddle.concat(
            [best_indices.cast("int64"), columns], axis=-1
        )
        best_scores, positions = paddle.topk(merged_scores, k=k, axis=-1)
        best_indices = paddle.take_along_axis(
            merged_indices, positions, axis=-1
        ).cast("int32")
    return best_indices


def _next_power_of_two(value):
    result = 1
    while result < value:
        result <<= 1
    return result


def _cutedsl_topk(query, key, topk, valid_range):
    """Adapt the legacy SM90 VHA indexer to shared-key SlashMLA.

    The legacy kernel takes ``[B,S,2,D]`` grouped KV and reduces the query
    heads inside each group.  SlashMLA instead has one shared latent key, so
    the adapter presents two identical KV groups.  The resulting score is
    ``2 * sum_h dot(q_h, key)``: the factor two is positive and therefore does
    not change the ranking, while the sum-over-head behavior is exactly the
    behavior of the legacy SLASH indexer.  A single kernel invocation also
    keeps the candidate set unique and avoids one launch per query head.
    """
    if query.shape[-1] not in (32, 64):
        raise ValueError(
            "CuTeDSL SlashMLA adapter requires slashmla_dim to be 32 or 64 "
            f"for the legacy SM90 kernel, got D={query.shape[-1]}."
        )
    if query.shape[0] != key.shape[0]:
        raise ValueError(
            "CuTeDSL SlashMLA query and key must share the batch dimension, got "
            f"{query.shape[0]} and {key.shape[0]}."
        )
    if query.shape[1] > key.shape[1]:
        raise ValueError(
            "CuTeDSL SlashMLA requires query length <= key length for causal "
            f"routing, got query length {query.shape[1]} and key length "
            f"{key.shape[1]}."
        )
    if query.shape[2] not in (8, 16, 32, 64):
        raise ValueError(
            "CuTeDSL SlashMLA requires query heads in {8, 16, 32, 64} "
            f"for the legacy SM90 kernel, got H={query.shape[2]}."
        )
    if valid_range.shape != [query.shape[0], query.shape[1], 2]:
        raise ValueError(
            "CuTeDSL SlashMLA adapter requires valid_range [B,S,2], got "
            f"{list(valid_range.shape)}."
        )

    from paddlefleet.cutedsl_ops import dsa_sparse_vha_topk_two_stage_cutedsl

    b, s, _, dim = [int(x) for x in query.shape]
    sk = int(key.shape[1])
    requested = min(int(topk), sk)
    if requested <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")
    padded = min(32768, _next_power_of_two(requested))
    if padded < 256:
        padded = 256
    if padded > 32768:
        raise ValueError(
            "CuTeDSL SlashMLA topk must fit the legacy 32K radix bucket, got "
            f"topk={requested}."
        )

    valid_range = valid_range.cast("int32").contiguous()
    doc_start = valid_range[:, :, 0]
    doc_end = valid_range[:, :, 1]
    # The old implementation intentionally has two KV groups.  ``expand`` is
    # followed by contiguous so the two groups have independent valid strides
    # in the DLPack/CuTe layout while holding exactly the same key values.
    grouped_key = key.unsqueeze(2).expand([b, sk, 2, dim]).contiguous()
    local_indices = dsa_sparse_vha_topk_two_stage_cutedsl(
        query,
        grouped_key,
        requested,
        padded,
        doc_start,
        doc_end,
        sm_scale=1.0,
        key_bits=32,
        sort_output=False,
    )[:, :, :requested]
    # The legacy output is document-local. SlashMLA and FlashMLA consume token
    # indices in the batch's packed sequence, so restore the document start
    # before returning.
    return paddle.where(
        local_indices >= 0,
        local_indices + doc_start.unsqueeze(-1),
        paddle.full_like(local_indices, -1),
    ).cast("int32")


def slashmla_topk(query, key, topk, slash_dim, mask=None):
    """Compute token-level [B, S, K] indices for absorbed MLA."""
    if query.ndim != 4 or key.ndim != 3:
        raise ValueError(
            f"SlashMLA indexer expects query [B,S,H,D] and key [B,S,D], "
            f"got {list(query.shape)} and {list(key.shape)}."
        )
    if query.shape[0] != key.shape[0]:
        raise ValueError(
            "SlashMLA query and key must share the batch dimension, got "
            f"{query.shape[0]} and {key.shape[0]}."
        )
    if int(topk) <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")
    requested_dim = int(slash_dim)
    if requested_dim <= 0:
        raise ValueError(f"slashmla_dim must be positive, got {slash_dim}.")
    if requested_dim > query.shape[-1] or requested_dim > key.shape[-1]:
        raise ValueError(
            "slashmla_dim cannot exceed query/key dimensions, got "
            f"slashmla_dim={requested_dim}, query_dim={query.shape[-1]}, "
            f"key_dim={key.shape[-1]}."
        )
    select_dim = requested_dim
    query = query[..., query.shape[-1] - select_dim :].contiguous()
    key = key[..., key.shape[-1] - select_dim :].contiguous()
    valid_range = _valid_range(mask, query.shape[0], query.shape[1])
    _validate_valid_range(
        valid_range,
        int(query.shape[0]),
        int(query.shape[1]),
        int(key.shape[1]),
    )

    backend = os.environ.get(
        "FLEET_SLASHMLA_INDEXER",
        os.environ.get("FLEET_SLASH_INDEXER", "cutedsl"),
    ).lower()
    if backend == "cutedsl":
        try:
            result = _cutedsl_topk(query, key, topk, valid_range)
        except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
            if not _env_enabled("FLEET_SLASHMLA_CUTEDSL_FALLBACK", True):
                raise RuntimeError(
                    "SlashMLA CuTeDSL indexer is unavailable. It requires the "
                    "CuTe DSL/cuda/cutlass runtime on an SM90 device; set "
                    "FLEET_SLASHMLA_CUTEDSL_FALLBACK=1 to use Paddle fallback."
                ) from exc
            warnings.warn(
                "SlashMLA CuTeDSL indexer unavailable; falling back to Paddle: "
                f"{exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            raise RuntimeError("SlashMLA CuTeDSL fallback not implemented")
        return result
    if backend == "tilelang":
        try:
            from paddlefleet.tilelang_ops.indexer.slashmla_indexer import (
                slashmla_tilelang_topk,
            )

            return slashmla_tilelang_topk(
                query,
                key,
                min(int(topk), key.shape[1]),
                valid_range,
            )
        except (
            ImportError,
            ModuleNotFoundError,
            RuntimeError,
            NotImplementedError,
        ):
            # The fallback is useful for correctness tests and for machines
            # where the optional TileLang extension was not built.
            pass
    elif backend != "paddle":
        raise ValueError(
            "FLEET_SLASHMLA_INDEXER must be 'tilelang', 'cutedsl', or "
            f"'paddle', got {backend!r}."
        )
    return _paddle_topk(query, key, topk, valid_range)


def slashmla_hca_topk(
    query,
    token_key,
    compressed_key,
    block_starts,
    topk,
    slash_dim,
    block_topk,
    ratio=128,
    mask=None,
    query_chunk_size=128,
    stage1_backend=None,
    fine_backend=None,
):
    """Run compressed-block top-k followed by token top-k, without autograd."""
    with paddle.no_grad():
        from paddlefleet.cutedsl_ops.csa_two_stage_indexer import (
            csa_two_stage_topk_no_grad,
        )

        valid_range = _valid_range(
            mask,
            int(query.shape[0]),
            int(query.shape[1]),
        )
        result = csa_two_stage_topk_no_grad(
            query,
            token_key,
            compressed_key,
            block_starts,
            topk=topk,
            block_topk=block_topk,
            ratio=ratio,
            valid_range=valid_range,
            slash_dim=slash_dim,
            query_chunk_size=query_chunk_size,
            stage1_backend=(
                os.environ.get("FLEET_SLASHMLA_HCA_STAGE1", "tilelang")
                if stage1_backend is None
                else stage1_backend
            ),
            fine_backend=(
                os.environ.get("FLEET_SLASHMLA_HCA_FINE", "paddle")
                if fine_backend is None
                else fine_backend
            ),
        )
    result.stop_gradient = True
    return result


def _slashmla_hca_topk_impl(
    query,
    token_key,
    compressed_key,
    block_starts,
    topk,
    slash_dim,
    block_topk,
    ratio,
    mask,
    query_chunk_size,
):
    if query.ndim != 4 or token_key.ndim != 3 or compressed_key.ndim != 3:
        raise ValueError(
            "SlashMLA HCA indexer expects query [B,S,H,D], token key [B,S,D], "
            "and compressed key [B,C,D]."
        )
    b, sq, _, query_dim = [int(x) for x in query.shape]
    if int(token_key.shape[0]) != b or int(compressed_key.shape[0]) != b:
        raise ValueError("SlashMLA HCA indexer inputs must share batch size.")
    if int(token_key.shape[1]) != sq:
        raise ValueError(
            "SlashMLA HCA indexer requires query/token-key lengths to match, "
            f"got {sq} and {token_key.shape[1]}."
        )
    select_dim = int(slash_dim)
    if (
        select_dim <= 0
        or select_dim > query_dim
        or select_dim > int(token_key.shape[-1])
        or select_dim > int(compressed_key.shape[-1])
    ):
        raise ValueError(
            "slash_dim must be positive and fit query/token/compressed widths, "
            f"got {select_dim}."
        )
    ratio = int(ratio)
    block_topk = int(block_topk)
    requested_topk = int(topk)
    query_chunk_size = int(query_chunk_size)
    if ratio <= 0 or block_topk <= 0 or requested_topk <= 0:
        raise ValueError("ratio, block_topk, and topk must all be positive.")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive.")

    n_blocks = int(compressed_key.shape[1])
    if list(block_starts.shape) != [b, n_blocks]:
        raise ValueError(
            "SlashMLA HCA block_starts must have shape "
            f"[{b}, {n_blocks}], got {list(block_starts.shape)}."
        )

    q_index = query[..., :select_dim].cast("float32")
    token_index_key = token_key[..., :select_dim].cast("float32")
    compressed_index_key = compressed_key[..., :select_dim].cast("float32")
    valid_range = _valid_range(mask, b, sq).cast("int64")
    block_starts_i64 = block_starts.cast("int64")
    block_ends = block_starts_i64 + ratio
    offsets = paddle.arange(ratio, dtype="int64")
    result_chunks = []

    for q_start in range(0, sq, query_chunk_size):
        q_end = min(q_start + query_chunk_size, sq)
        q_chunk = q_index[:, q_start:q_end]
        chunk_len = q_end - q_start
        ranges = valid_range[:, q_start:q_end]
        query_positions = paddle.arange(q_start, q_end, dtype="int64").reshape(
            [1, chunk_len, 1]
        )

        if n_blocks > 0:
            coarse_scores = paddle.einsum(
                "bqhd,bcd->bqhc", q_chunk, compressed_index_key
            ).sum(axis=2)
            coarse_valid = (
                (
                    block_starts_i64.reshape([b, 1, n_blocks])
                    >= ranges[:, :, 0:1]
                )
                & (block_ends.reshape([b, 1, n_blocks]) <= query_positions + 1)
                & (block_starts_i64.reshape([b, 1, n_blocks]) >= 0)
            )
            coarse_scores = paddle.where(
                coarse_valid,
                coarse_scores,
                paddle.full_like(coarse_scores, -1.0e30),
            )
            coarse_k = min(block_topk, n_blocks)
            coarse_values, coarse_indices = paddle.topk(
                coarse_scores, k=coarse_k, axis=-1
            )
            coarse_indices = paddle.where(
                coarse_values > -1.0e20,
                coarse_indices,
                paddle.full_like(coarse_indices, -1),
            )
            safe_block_indices = coarse_indices.clip(0, max(n_blocks - 1, 0))
            starts_table = block_starts_i64.unsqueeze(1).expand(
                [b, chunk_len, n_blocks]
            )
            selected_starts = paddle.take_along_axis(
                starts_table, safe_block_indices, axis=2
            )
            selected_starts = paddle.where(
                coarse_indices >= 0,
                selected_starts,
                paddle.full_like(selected_starts, -1),
            )
        else:
            selected_starts = paddle.empty([b, chunk_len, 0], dtype="int64")

        # Keep the current partial block as a fine candidate. HCA only has
        # completed blocks, but SparseMLA must still serve early/current tokens.
        doc_start = ranges[:, :, 0]
        query_pos_2d = query_positions.squeeze(-1).expand([b, chunk_len])
        current_start = (
            doc_start + ((query_pos_2d - doc_start) // ratio) * ratio
        )
        current_is_duplicate = (
            (selected_starts == current_start.unsqueeze(-1)).any(axis=-1)
            if int(selected_starts.shape[-1]) > 0
            else paddle.zeros([b, chunk_len], dtype="bool")
        )
        selected_candidates = selected_starts.unsqueeze(-1) + offsets.reshape(
            [1, 1, 1, ratio]
        )
        selected_candidates = paddle.where(
            selected_starts.unsqueeze(-1) >= 0,
            selected_candidates,
            paddle.full_like(selected_candidates, -1),
        ).reshape([b, chunk_len, -1])
        current_candidates = current_start.unsqueeze(-1) + offsets.reshape(
            [1, 1, ratio]
        )
        current_candidates = paddle.where(
            current_is_duplicate.unsqueeze(-1),
            paddle.full_like(current_candidates, -1),
            current_candidates,
        )
        candidates = paddle.concat(
            [selected_candidates, current_candidates], axis=-1
        )
        candidate_valid = (
            (candidates >= ranges[:, :, 0:1])
            & (candidates < ranges[:, :, 1:2])
            & (candidates >= 0)
            & (candidates < int(token_key.shape[1]))
        )
        candidates = paddle.where(
            candidate_valid, candidates, paddle.full_like(candidates, -1)
        )

        gathered_key = _batched_gather_tokens(token_index_key, candidates)
        fine_scores = paddle.einsum(
            "bqhd,bqnd->bqhn", q_chunk, gathered_key
        ).sum(axis=2)
        fine_scores = paddle.where(
            candidate_valid,
            fine_scores,
            paddle.full_like(fine_scores, -1.0e30),
        )
        fine_k = min(requested_topk, int(candidates.shape[-1]))
        fine_values, fine_positions = paddle.topk(
            fine_scores, k=fine_k, axis=-1
        )
        fine_indices = paddle.take_along_axis(
            candidates, fine_positions, axis=-1
        )
        fine_indices = paddle.where(
            fine_values > -1.0e20,
            fine_indices,
            paddle.full_like(fine_indices, -1),
        )
        result_chunks.append(fine_indices.cast("int32"))

    return paddle.concat(result_chunks, axis=1)


def _paddle_hca_attention(
    query,
    compressed_key,
    block_starts,
    ratio,
    mask,
    sm_scale,
    kv_lora_rank,
    query_chunk_size=128,
):
    """Causal attention over completed HCA blocks using Paddle operators."""
    b, sq, h, _ = [int(x) for x in query.shape]
    n_blocks = int(compressed_key.shape[1])
    if n_blocks == 0:
        return paddle.zeros([b, sq, h, int(kv_lora_rank)], dtype=query.dtype)

    valid_range = _valid_range(mask, b, sq).cast("int64")
    starts = block_starts.cast("int64")
    ends = starts + int(ratio)
    compressed_f32 = compressed_key.cast("float32")
    compressed_value = compressed_f32[..., : int(kv_lora_rank)]
    outputs = []

    for q_start in range(0, sq, int(query_chunk_size)):
        q_end = min(q_start + int(query_chunk_size), sq)
        chunk_len = q_end - q_start
        q_chunk = query[:, q_start:q_end].cast("float32")
        scores = paddle.einsum(
            "bqhd,bcd->bqhc", q_chunk, compressed_f32
        ) * float(sm_scale)
        ranges = valid_range[:, q_start:q_end]
        query_positions = paddle.arange(q_start, q_end, dtype="int64").reshape(
            [1, chunk_len, 1]
        )
        valid_blocks = (
            (starts.reshape([b, 1, n_blocks]) >= ranges[:, :, 0:1])
            & (ends.reshape([b, 1, n_blocks]) <= query_positions + 1)
            & (starts.reshape([b, 1, n_blocks]) >= 0)
        )
        scores = paddle.where(
            valid_blocks.unsqueeze(2),
            scores,
            paddle.full_like(scores, -1.0e30),
        )
        weights = paddle.nn.functional.softmax(scores, axis=-1)
        weights = weights * valid_blocks.unsqueeze(2).cast(weights.dtype)
        latent_output = paddle.einsum(
            "bqhc,bcr->bqhr", weights, compressed_value
        )
        outputs.append(latent_output.cast(query.dtype))
    return paddle.concat(outputs, axis=1)


def _hca_compressed_indices(block_starts, ratio, mask, batch_size, seq_len):
    """Build per-query local indices for completed HCA blocks.

    The CSA/FlashMLA sparse kernels consume local columns in ``kv``.  HCA's
    ``block_starts`` are token positions, so this helper deliberately maps the
    compact compressed-block column ``c`` to local index ``c`` rather than
    passing the original token start.  ``-1`` entries retain the packed-document
    and causal holes required by the sparse backend.
    """
    n_blocks = int(block_starts.shape[1])
    if n_blocks == 0:
        return paddle.empty(
            [batch_size, seq_len, 0],
            dtype="int32",
        )

    valid_range = _valid_range(mask, batch_size, seq_len).cast("int64")
    starts = block_starts.cast("int64")
    ends = starts + int(ratio)
    query_positions = paddle.arange(seq_len, dtype="int64").reshape(
        [1, seq_len, 1]
    )
    valid = (
        (starts.reshape([batch_size, 1, n_blocks]) >= valid_range[:, :, 0:1])
        & (ends.reshape([batch_size, 1, n_blocks]) <= query_positions + 1)
        & (starts.reshape([batch_size, 1, n_blocks]) >= 0)
    )
    compressed_indices = paddle.arange(n_blocks, dtype="int32").reshape(
        [1, 1, n_blocks]
    )
    compressed_indices = compressed_indices.expand(
        [batch_size, seq_len, n_blocks]
    )
    return paddle.where(
        valid,
        compressed_indices,
        paddle.full_like(compressed_indices, -1),
    )


def _csa_hca_attention(
    query,
    compressed_key,
    block_starts,
    ratio,
    mask,
    sm_scale,
    kv_lora_rank,
    query_chunk_size=128,
):
    """Run HCA attention through the CSA FlashMLA/DSA attention pair.

    This path intentionally does not call ``CompressedSparseAttention.forward``:
    that higher-level layer would run a second hidden-state compressor.  Instead,
    it feeds the already-compressed MLA token KV to the same sparse attention
    kernel pair used by CSA.  The absorbed-MQA wrapper is used because the MLA
    layout is asymmetric (query/key width 576, value width 512).
    """
    b, sq, h, _ = [int(x) for x in query.shape]
    n_blocks = int(compressed_key.shape[1])
    if n_blocks == 0:
        return paddle.zeros(
            [b, sq, h, int(kv_lora_rank)],
            dtype=query.dtype,
        )

    compressed_indices = _hca_compressed_indices(
        block_starts,
        ratio,
        mask,
        b,
        sq,
    )
    requested_backend = os.environ.get("FLEET_SLASHMLA_BACKEND", "auto").lower()
    if requested_backend not in {"auto", "cudnn", "paddle"}:
        raise ValueError(
            "FLEET_SLASHMLA_BACKEND must be 'auto', 'cudnn', or 'paddle', "
            f"got {requested_backend!r}."
        )

    use_fallback = requested_backend == "paddle"
    flash_mla_available = None
    if not use_fallback:
        try:
            flash_mla_available = _is_slashmla_flash_mla_available()
            if requested_backend == "cudnn" and not flash_mla_available:
                raise RuntimeError(
                    "FLEET_SLASHMLA_BACKEND=cudnn requires the FlashMLA "
                    "sparse forward kernel, but it is unavailable on this "
                    "device."
                )
            use_fallback = not flash_mla_available
        except (ImportError, RuntimeError):
            if requested_backend == "cudnn":
                raise
            use_fallback = True

    if use_fallback:
        return _paddle_hca_attention(
            query,
            compressed_key,
            block_starts,
            ratio,
            mask,
            sm_scale,
            kv_lora_rank,
            query_chunk_size=query_chunk_size,
        )

    from paddlefleet.fusions.mqa_sparse_attn import mqa_sparse_attn

    latent_output = mqa_sparse_attn(
        query,
        compressed_key,
        compressed_indices,
        float(sm_scale),
        int(kv_lora_rank),
        attn_sink=None,
        use_slashmla=True,
    )
    return latent_output.reshape([b, sq, h, int(kv_lora_rank)])


def _deabsorb_value(latent_output, value_weight, value_dim):
    """Convert [B,S,H,R] latent output to [B,S,H*Dv]."""
    b, s, h, _ = latent_output.shape
    out = paddle.einsum("bshr,hrv->bshv", latent_output, value_weight)
    return out.reshape([b, s, h * value_dim])


def _paddle_sparse_attention(
    query, latent_key, token_indices, sm_scale, value_dim
):
    """Autograd-safe token gather fallback for non-SM100 machines."""
    b, s, h, _ = query.shape
    k = token_indices.shape[-1]
    safe_indices = token_indices.cast("int64").clip(0, latent_key.shape[1] - 1)
    key_for_gather = latent_key.unsqueeze(1).expand(
        [b, s, latent_key.shape[1], latent_key.shape[2]]
    )
    gathered = paddle.take_along_axis(
        key_for_gather,
        safe_indices.unsqueeze(-1).expand([b, s, k, latent_key.shape[2]]),
        axis=2,
    )
    valid = token_indices >= 0
    logits = paddle.einsum("bshd,bskd->bhsk", query, gathered)
    logits = logits * float(sm_scale)
    logits = paddle.where(
        valid.unsqueeze(1), logits, paddle.full_like(logits, -float("inf"))
    )
    weights = paddle.nn.functional.softmax(logits, axis=-1)
    weights = paddle.where(
        valid.unsqueeze(1), weights, paddle.zeros_like(weights)
    )
    latent_output = paddle.einsum(
        "bhsk,bskd->bshd", weights, gathered[..., :value_dim]
    )
    return latent_output


def slashmla_sparse_attention(
    query,
    shared_key,
    value_weight,
    topk_indices,
    sm_scale,
    kv_lora_rank,
    value_dim,
    attn_sink=None,
):
    """Run FlashMLA/cuDNN sparse MQA and de-absorb its latent output."""
    latent_key = shared_key.squeeze(2) if shared_key.ndim == 4 else shared_key
    token_indices = topk_indices.clone()
    token_indices.stop_gradient = True
    requested_backend = os.environ.get(
        "FLEET_SLASHMLA_BACKEND", "cudnn"
    ).lower()
    if requested_backend not in {"auto", "cudnn", "paddle"}:
        raise ValueError(
            "FLEET_SLASHMLA_BACKEND must be 'auto', 'cudnn', or 'paddle', "
            f"got {requested_backend!r}."
        )
    use_fallback = requested_backend == "paddle"
    flash_mla_available = None
    try:
        flash_mla_available = _is_slashmla_flash_mla_available()
        if requested_backend == "cudnn" and not flash_mla_available:
            raise RuntimeError(
                "FLEET_SLASHMLA_BACKEND=cudnn requires the FlashMLA sparse "
                "forward kernel, but it is unavailable on this device."
            )
        use_fallback = use_fallback or (
            requested_backend == "auto" and not flash_mla_available
        )
    except (ImportError, RuntimeError):
        if requested_backend == "cudnn":
            raise
        if requested_backend == "auto":
            raise ImportError(
                "Unable to import cudnn_ops or function is not available."
            )
    if use_fallback:
        raise ImportError("paddle backend is not available.")
        latent_output = _paddle_sparse_attention(
            query,
            latent_key,
            token_indices,
            sm_scale,
            int(kv_lora_rank),
        )
    else:
        from paddlefleet.fusions.mqa_sparse_attn import mqa_sparse_attn

        latent_output = mqa_sparse_attn(
            query,
            latent_key,
            token_indices,
            float(sm_scale),
            int(kv_lora_rank),
            attn_sink=attn_sink,
            use_slashmla=True,
        )
    latent_output = latent_output.reshape(
        [query.shape[0], query.shape[1], query.shape[2], kv_lora_rank]
    )
    output = _deabsorb_value(latent_output, value_weight, value_dim)
    return output


def slashmla_hca_sparse_attention(
    query,
    shared_key,
    value_weight,
    hca_compressor,
    topk,
    slash_dim,
    block_topk,
    sm_scale,
    kv_lora_rank,
    value_dim,
    mask=None,
    alpha=1.0,
    query_chunk_size=128,
    attn_sink=None,
    stage1_backend=None,
    fine_backend=None,
):
    """Return ``O_hca + alpha * O_sparse`` over one shared MLA token KV."""
    latent_key = shared_key.squeeze(2) if shared_key.ndim == 4 else shared_key
    if latent_key.ndim != 3:
        raise ValueError(
            "SlashMLA HCA+SparseMLA expects shared key [B,S,D] or [B,S,1,D], "
            f"got {list(shared_key.shape)}."
        )
    if not isinstance(hca_compressor, paddle.nn.Layer):
        raise TypeError("hca_compressor must be a Paddle Layer.")

    compressed_key, block_starts = hca_compressor(latent_key, mask=mask)
    token_indices = slashmla_hca_topk(
        query,
        latent_key,
        compressed_key,
        block_starts,
        topk,
        slash_dim,
        block_topk,
        ratio=hca_compressor.ratio,
        mask=mask,
        query_chunk_size=query_chunk_size,
        stage1_backend=stage1_backend,
        fine_backend=fine_backend,
    )
    hca_latent = _csa_hca_attention(
        query,
        compressed_key,
        block_starts,
        hca_compressor.ratio,
        mask,
        sm_scale,
        kv_lora_rank,
        query_chunk_size=query_chunk_size,
    )
    hca_output = _deabsorb_value(hca_latent, value_weight, value_dim)
    sparse_output = slashmla_sparse_attention(
        query,
        latent_key,
        value_weight,
        token_indices,
        sm_scale,
        kv_lora_rank,
        value_dim,
        attn_sink=attn_sink,
    )
    output = hca_output + float(alpha) * sparse_output
    return output, token_indices
