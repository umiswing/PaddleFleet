# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Two-stage CSA token indexer.

The CSA routing problem has two different key resolutions:

1. select compressed blocks using query-vs-compressed-K scores;
2. select token positions only inside those blocks.

This implementation keeps both stages query-tiled and candidate-tiled.  It
never constructs a full ``[B, S, S, D]`` or ``[B, S, S, H]`` score tensor.
The interface intentionally mirrors the HCA/CSA call site rather than the
legacy VHA two-stage kernel, whose grouped KV contract is not applicable to
CSA.
"""

from __future__ import annotations

import os

import paddle


def _profile_enabled():
    return os.environ.get(
        "FLEET_SLASHMLA_INDEXER_PROFILE", ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def _profile_start(enabled):
    if not enabled or not paddle.is_compiled_with_cuda():
        return None
    event = paddle.device.cuda.Event(enable_timing=True)
    event.record()
    return event


def _profile_end(start_event, timings, label):
    if start_event is None:
        return
    end_event = paddle.device.cuda.Event(enable_timing=True)
    end_event.record()
    end_event.synchronize()
    timings[label] = timings.get(label, 0.0) + float(
        start_event.elapsed_time(end_event)
    )


def _profile_report(timings, backend, shape):
    if not timings:
        return
    total = sum(timings.values())
    details = " ".join(
        f"{name}={value:.3f}ms" for name, value in timings.items()
    )
    print(
        "[CSA indexer profile] "
        f"backend={backend} shape={shape} total={total:.3f}ms {details}",
        flush=True,
    )


def _compressed_valid_range(block_starts, valid_range, ratio, seq_len):
    """Map token-valid ranges to contiguous compressed-block ranges."""
    _, sq = [int(x) for x in valid_range.shape[:2]]
    starts = block_starts.cast("int64")
    q_pos = paddle.arange(seq_len, dtype="int64").reshape([1, sq, 1])
    token_start = valid_range[:, :, 0].cast("int64").unsqueeze(-1)
    token_end = valid_range[:, :, 1].cast("int64").unsqueeze(-1)
    before_doc = (starts[:, None, :] >= 0) & (starts[:, None, :] < token_start)
    compressed_start = before_doc.cast("int32").sum(axis=-1)
    completed = (
        (starts[:, None, :] >= token_start)
        & (starts[:, None, :] + int(ratio) <= q_pos + 1)
        & (starts[:, None, :] + int(ratio) <= token_end)
        & (starts[:, None, :] >= 0)
    )
    compressed_count = completed.cast("int32").sum(axis=-1)
    return paddle.stack(
        [compressed_start, compressed_start + compressed_count], axis=-1
    ).contiguous()


def _coarse_topk_paddle(
    q_chunk,
    compressed_index_key,
    starts,
    ranges,
    positions,
    block_topk,
    ratio,
):
    """Reference coarse stage with one small block top-k per query."""
    n_blocks = int(compressed_index_key.shape[1])
    if n_blocks == 0:
        return paddle.empty(
            [q_chunk.shape[0], q_chunk.shape[1], 0], dtype="int64"
        )
    block_valid = (
        (starts[:, None, :] >= ranges[:, :, None, 0])
        & ((starts + int(ratio))[:, None, :] <= positions + 1)
        & (starts[:, None, :] >= 0)
    )
    scores = paddle.einsum("bqhd,bcd->bqhc", q_chunk, compressed_index_key).sum(
        axis=2
    )
    scores = paddle.where(
        block_valid, scores, paddle.full_like(scores, -1.0e30)
    )
    coarse_k = min(int(block_topk), n_blocks)
    values, indices = paddle.topk(scores, coarse_k, axis=-1)
    return paddle.where(
        values > -1.0e20, indices, paddle.full_like(indices, -1)
    ).cast("int64")


def _coarse_topk_tilelang(
    query,
    compressed_key,
    block_starts,
    valid_range,
    block_topk,
    ratio,
):
    """Use the CSA streaming bitonic top-k kernel for the coarse stage."""
    from paddlefleet.tilelang_ops.indexer.csa_indexer import (
        csa_indexer_topk_fwd,
    )

    b, sq, heads, dim = [int(x) for x in query.shape]
    n_blocks = int(compressed_key.shape[1])
    if n_blocks == 0:
        return paddle.empty([b, sq, 0], dtype="int64")
    compressed_range = _compressed_valid_range(
        block_starts, valid_range, ratio, sq
    )
    weights = paddle.ones([b, sq, heads], dtype="float32")
    indices, _ = csa_indexer_topk_fwd(
        query[..., :dim].contiguous(),
        compressed_key.contiguous(),
        weights,
        ratio=int(ratio),
        topk_effective=min(int(block_topk), n_blocks),
        valid_range=compressed_range,
        block_K=32,
        relu_scores=False,
    )
    return indices.cast("int64")


def _coarse_topk_cutedsl_paddle(
    query,
    compressed_key,
    starts,
    ranges,
    block_topk,
    ratio,
):
    """CuTeDSL QK score stage followed by the unfused Paddle top-k helper."""
    from paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn import (
        cudnn_indexer_topk,
    )
    from paddlefleet.cutedsl_ops.two_stage_indexer import (
        dsa_sparse_vha_score_cutedsl,
    )

    b, sq, heads, dim = [int(x) for x in query.shape]
    n_blocks = int(compressed_key.shape[1])
    if n_blocks == 0:
        return paddle.empty([b, sq, 0], dtype="int64")
    if heads not in (8, 16, 32, 64) or dim not in (32, 64):
        raise ValueError(
            "CuTeDSL CSA stage-1 requires query shape [B,S,H,D] with "
            f"H in {{8, 16, 32, 64}} and D in {{32, 64}}, got "
            f"H={heads}, D={dim}."
        )

    compressed_range = _compressed_valid_range(starts, ranges, ratio, sq).cast(
        "int32"
    )
    row_start = compressed_range[..., 0]
    row_count = (compressed_range[..., 1] - compressed_range[..., 0]).clip(
        min=0
    )
    grouped_key = (
        compressed_key.unsqueeze(2).expand([b, n_blocks, 2, dim]).contiguous()
    )
    scores = dsa_sparse_vha_score_cutedsl(
        query.contiguous(),
        grouped_key,
        row_count,
        row_start=row_start,
        sm_scale=1.0,
        score_width=n_blocks,
    )
    # The CuTeDSL score buffer is row-local: score[:, :, j] corresponds to
    # compressed column row_start + j.  A zero-based range makes the existing
    # cuDNN wrapper's document remap a no-op, after which we restore the
    # global compressed-column offset.
    local_range = paddle.stack(
        [
            paddle.zeros_like(row_start),
            row_count,
        ],
        axis=-1,
    )
    local_indices, _ = cudnn_indexer_topk(
        scores,
        sq=sq,
        ratio=ratio,
        topk=block_topk,
        valid_range=local_range,
    )
    valid = local_indices >= 0
    global_indices = local_indices + row_start.unsqueeze(-1)
    return paddle.where(
        valid,
        global_indices,
        paddle.full_like(global_indices, -1),
    ).cast("int64")


def _batched_gather_tokens(token_key, indices):
    b, s, d = [int(x) for x in token_key.shape]
    safe = indices.cast("int64").clip(0, max(s - 1, 0))
    valid = indices >= 0
    offsets = paddle.arange(b, dtype="int64").reshape(
        [b] + [1] * (indices.ndim - 1)
    )
    flat = safe + offsets * s
    gathered = paddle.gather(
        token_key.reshape([b * s, d]), flat.flatten(), axis=0
    ).reshape([*list(indices.shape), d])
    return gathered * valid.unsqueeze(-1).cast(gathered.dtype)


def _pad_topk(indices, scores, topk):
    width = int(indices.shape[-1])
    if width >= topk:
        return indices[..., :topk], scores[..., :topk]
    pad = topk - width
    indices = paddle.nn.functional.pad(indices, (0, pad), value=-1)
    scores = paddle.nn.functional.pad(scores, (0, pad), value=-1.0e30)
    return indices, scores


def _cutedsl_topk_token_ids(scores, token_ids, topk):
    """Run CuTeDSL _TwoStageTopK over CSA score-only outputs."""


def csa_two_stage_topk(
    query,
    token_key,
    compressed_key,
    block_starts,
    topk,
    block_topk,
    ratio,
    valid_range,
    slash_dim=None,
    query_chunk_size=128,
    candidate_chunk_size=512,
    stage1_backend="tilelang",
    fine_backend="paddle",
):
    """Return token-level top-k indices from CSA's coarse and fine stages.

    Args:
        query: ``[B, S, H, D]`` query heads.
        token_key: ``[B, S, D]`` token-level key used by fine selection.
        compressed_key: ``[B, C, D]`` one key per completed block.
        block_starts: ``[B, C]`` token start of each compressed block.
        valid_range: ``[B, S, 2]`` token-level document/causal range.

    Returns:
        ``int32 [B, S, topk]`` token indices; invalid entries are ``-1``.
    """
    if query.ndim != 4 or token_key.ndim != 3 or compressed_key.ndim != 3:
        raise ValueError(
            "CSA two-stage indexer expects query [B,S,H,D], token key [B,S,D], "
            "and compressed key [B,C,D]."
        )
    b, sq, heads, qdim = [int(x) for x in query.shape]
    if list(token_key.shape[:2]) != [b, sq]:
        raise ValueError("query and token_key must have matching [B,S].")
    if int(block_starts.shape[0]) != b or int(block_starts.shape[1]) != int(
        compressed_key.shape[1]
    ):
        raise ValueError("block_starts must match compressed_key [B,C].")
    if list(valid_range.shape) != [b, sq, 2]:
        raise ValueError("valid_range must have shape [B,S,2].")

    topk = int(topk)
    block_topk = int(block_topk)
    ratio = int(ratio)
    query_chunk_size = int(query_chunk_size)
    profile = _profile_enabled()
    profile_timings = {}
    if query_chunk_size <= 0 or int(candidate_chunk_size) <= 0:
        raise ValueError("chunk sizes must be positive.")
    del candidate_chunk_size
    stage1_backend = str(stage1_backend).lower()
    fine_backend = str(fine_backend).lower()
    select_dim = qdim if slash_dim is None else int(slash_dim)
    if topk <= 0 or block_topk <= 0 or ratio <= 0:
        raise ValueError("topk, block_topk, and ratio must be positive.")
    if stage1_backend not in (
        "tilelang",
        "auto",
        "paddle",
        "cutedsl_paddle_topk",
    ):
        raise ValueError(
            "stage1_backend must be 'tilelang', 'auto', 'paddle', or "
            "'cutedsl_paddle_topk', "
            f"got {stage1_backend!r}"
        )
    if fine_backend not in ("paddle", "tilelang", "auto"):
        raise ValueError(
            "fine_backend must be 'paddle', 'tilelang', or 'auto', "
            f"got {fine_backend!r}"
        )
    if select_dim > qdim or select_dim > int(token_key.shape[-1]):
        raise ValueError("slash_dim must fit query and token-key widths.")
    if select_dim > int(compressed_key.shape[-1]):
        raise ValueError("slash_dim must fit compressed-key width.")

    q_index = query[..., :select_dim].cast("float32")
    q_index_kernel = query[..., :select_dim].contiguous()
    token_index_key = token_key[..., :select_dim].cast("float32")
    compressed_index_key = compressed_key[..., :select_dim].cast("float32")
    starts = block_starts.cast("int64")
    valid_range = valid_range.cast("int64")
    n_blocks = int(compressed_key.shape[1])
    offsets = paddle.arange(ratio, dtype="int64")
    outputs = []
    coarse_indices_all = None
    if n_blocks > 0 and stage1_backend in ("tilelang", "auto"):
        profile_start = _profile_start(profile)
        try:
            coarse_indices_all = _coarse_topk_tilelang(
                q_index_kernel,
                compressed_key[..., :select_dim],
                starts,
                valid_range,
                block_topk,
                ratio,
            )
        except (
            ImportError,
            ModuleNotFoundError,
            RuntimeError,
            NotImplementedError,
            TypeError,
            ValueError,
        ):
            coarse_indices_all = None
        finally:
            _profile_end(profile_start, profile_timings, "stage1")
    elif n_blocks > 0 and stage1_backend == "cutedsl_paddle_topk":
        profile_start = _profile_start(profile)
        coarse_indices_all = _coarse_topk_cutedsl_paddle(
            q_index_kernel,
            compressed_key[..., :select_dim],
            starts,
            valid_range,
            block_topk,
            ratio,
        )
        _profile_end(profile_start, profile_timings, "stage1")

    for q_start in range(0, sq, query_chunk_size):
        q_end = min(q_start + query_chunk_size, sq)
        q_chunk = q_index[:, q_start:q_end]
        chunk_len = q_end - q_start
        ranges = valid_range[:, q_start:q_end]
        positions = paddle.arange(q_start, q_end, dtype="int64").reshape(
            [1, chunk_len, 1]
        )

        if n_blocks > 0:
            selected_indices = (
                None
                if coarse_indices_all is None
                else coarse_indices_all[:, q_start:q_end]
            )
            if selected_indices is None:
                profile_start = _profile_start(profile)
                selected_indices = _coarse_topk_paddle(
                    q_chunk,
                    compressed_index_key,
                    starts,
                    ranges,
                    positions,
                    block_topk,
                    ratio,
                )
                _profile_end(profile_start, profile_timings, "stage1")
        else:
            selected_indices = paddle.empty([b, chunk_len, 0], dtype="int64")
            selected_starts = paddle.empty([b, chunk_len, 0], dtype="int64")

        use_fused_fine = fine_backend in ("tilelang", "auto") and n_blocks > 0
        fused_candidates = None
        if use_fused_fine:
            try:
                from paddlefleet.tilelang_ops.indexer.csa_fine_score import (
                    csa_fine_score_fwd,
                )

                profile_start = _profile_start(profile)
                scores, fused_candidates = csa_fine_score_fwd(
                    query[:, q_start:q_end, :, :select_dim],
                    token_key[..., :select_dim],
                    selected_indices,
                    starts,
                    ranges,
                    ratio=ratio,
                    query_offset=q_start,
                )
                _profile_end(profile_start, profile_timings, "fine_score")
                candidates = fused_candidates.cast("int64")
            except (
                ImportError,
                ModuleNotFoundError,
                RuntimeError,
                NotImplementedError,
                TypeError,
                ValueError,
            ):
                if fine_backend == "tilelang":
                    raise
                use_fused_fine = False

        if not use_fused_fine:
            # The Paddle fallback still needs the explicit candidate tensor.
            profile_start = _profile_start(profile)
            selected_valid = selected_indices >= 0
            starts_table = starts[:, None, :].expand([b, chunk_len, n_blocks])
            selected_starts = paddle.take_along_axis(
                starts_table, selected_indices, axis=-1
            )
            selected_starts = paddle.where(
                selected_valid,
                selected_starts,
                paddle.full_like(selected_starts, -1),
            )
            doc_start = ranges[:, :, 0]
            query_pos = positions.squeeze(-1).expand([b, chunk_len])
            current_start = (
                doc_start + ((query_pos - doc_start) // ratio) * ratio
            )
            duplicate = (
                (selected_starts == current_start.unsqueeze(-1)).any(axis=-1)
                if int(selected_starts.shape[-1]) > 0
                else paddle.zeros([b, chunk_len], dtype="bool")
            )
            selected_candidates = selected_starts.unsqueeze(
                -1
            ) + offsets.reshape([1, 1, 1, ratio])
            selected_candidates = paddle.where(
                selected_starts.unsqueeze(-1) >= 0,
                selected_candidates,
                paddle.full_like(selected_candidates, -1),
            ).reshape([b, chunk_len, -1])
            current_candidates = current_start.unsqueeze(-1) + offsets.reshape(
                [1, 1, ratio]
            )
            current_candidates = paddle.where(
                duplicate.unsqueeze(-1),
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
                & (candidates < sq)
            )
            candidates = paddle.where(
                candidate_valid,
                candidates,
                paddle.full_like(candidates, -1),
            )
            _profile_end(profile_start, profile_timings, "candidate_setup")
            # Fine stage reads the complete candidate buffer once and performs
            # the large top-k once. Candidate blocks are laid out consecutively,
            # which avoids repeated gather/einsum/topk launches per tile.
            profile_start = _profile_start(profile)
            gathered = _batched_gather_tokens(token_index_key, candidates)
            _profile_end(profile_start, profile_timings, "candidate_gather")
            profile_start = _profile_start(profile)
            scores = paddle.einsum("bqhd,bqnd->bqhn", q_chunk, gathered).sum(
                axis=2
            )
            _profile_end(profile_start, profile_timings, "fine_score")
            scores = paddle.where(
                candidate_valid,
                scores,
                paddle.full_like(scores, -1.0e30),
            )
        profile_start = _profile_start(profile)
        final_k = min(topk, int(scores.shape[-1]))
        final_scores, final_pos = paddle.topk(scores, final_k, axis=-1)
        final_indices = paddle.take_along_axis(candidates, final_pos, axis=-1)
        final_valid = final_scores > -1.0e20
        final_indices = paddle.where(
            final_valid,
            final_indices,
            paddle.full_like(final_indices, -1),
        )
        final_indices, _ = _pad_topk(
            final_indices,
            final_scores,
            topk,
        )
        _profile_end(profile_start, profile_timings, "fine_topk")
        outputs.append(final_indices.cast("int32"))

    profile_start = _profile_start(profile)
    result = paddle.concat(outputs, axis=1)
    _profile_end(profile_start, profile_timings, "output_concat")
    _profile_report(
        profile_timings,
        stage1_backend,
        [b, sq, heads, qdim],
    )
    return result


def csa_two_stage_topk_no_grad(*args, **kwargs):
    with paddle.no_grad():
        result = csa_two_stage_topk(*args, **kwargs)
    result.stop_gradient = True
    return result


__all__ = ["csa_two_stage_topk", "csa_two_stage_topk_no_grad"]
