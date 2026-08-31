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


def _compressed_valid_range(
    block_starts, valid_range, ratio, seq_len, query_offset=0
):
    """Map token-valid ranges to contiguous compressed-block ranges.

    ``query_offset`` is the position of query row 0 in the key timeline, so a
    short query window can sit at the end of a much longer KV cache.
    """
    _, sq = [int(x) for x in valid_range.shape[:2]]
    starts = block_starts.cast("int64")
    q_pos = (paddle.arange(sq, dtype="int64") + int(query_offset)).reshape(
        [1, sq, 1]
    )
    del seq_len
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


def _coarse_topk_tilelang(
    query,
    compressed_key,
    block_starts,
    valid_range,
    block_topk,
    ratio,
    query_offset=0,
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
        block_starts, valid_range, ratio, sq, query_offset
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
    )
    return indices.cast("int64")


def _cutedsl_topk_token_ids(scores, token_ids, topk):
    """Run CuTeDSL _TwoStageTopK over CSA score-only outputs."""
    from paddlefleet.cutedsl_ops.csa_topk import csa_topk_cutedsl

    return csa_topk_cutedsl(scores, token_ids, topk)


def _multiblock_topk_token_ids(scores, token_ids, topk):
    """Select CSA candidates with the segment-decomposed radix selector.

    The selector reports candidate *positions* while CSA candidates are
    arbitrary token ids, so one gather maps positions back to ids.  Slots the
    selector filled from sentinel columns come back with a sentinel score and
    become -1, which matches the two-stage selector's contract.
    """
    from paddlefleet.cutedsl_ops.top_kernel1 import (
        indexer_topk_prefill_multiblock,
    )

    batch, query_len, candidate_count = [int(x) for x in scores.shape]
    topk = int(topk)
    rows = batch * query_len
    flat_scores = scores.reshape([rows, candidate_count]).contiguous()
    flat_ids = token_ids.reshape([rows, candidate_count]).contiguous()
    lengths = paddle.full([rows], candidate_count, dtype="int32").contiguous()
    positions, values = indexer_topk_prefill_multiblock(
        flat_scores, lengths, topk, count_mode="input"
    )
    gathered = paddle.take_along_axis(
        flat_ids,
        positions.clip(0, candidate_count - 1).cast("int64"),
        axis=-1,
    )
    gathered = paddle.where(
        values > -1.0e20, gathered, paddle.full_like(gathered, -1)
    )
    return gathered.reshape([batch, query_len, topk])


def _fine_topk_token_ids(scores, token_ids, topk):
    """Pick the fine-stage selector.

    The segment-decomposed multi-block radix selector is the default: measured
    2.9x faster than the two-stage one on the production candidate row and it selects exactly
    the same set.  Its one precondition is that the candidate buffer is at least
    ``topk`` wide, because it skips any row it cannot fill; that is checked here
    and reported instead of being papered over.  ``cutedsl`` selects the
    two-stage selector, which has no width precondition.  An unknown value is
    rejected rather than treated as the default.
    """
    backend = (
        os.environ.get("FLEET_SLASHMLA_HCA_FINE_TOPK", "multiblock")
        .strip()
        .lower()
    )
    if backend == "multiblock":
        if int(scores.shape[-1]) < int(topk):
            raise ValueError(
                "the multi-block fine top-k selector needs a candidate buffer "
                f"at least topk wide, got {int(scores.shape[-1])} candidates "
                f"for topk={int(topk)}. Raise slashmla_hca_block_topk so that "
                "(block_topk + 1) * ratio >= topk, or set "
                "FLEET_SLASHMLA_HCA_FINE_TOPK=cutedsl."
            )
        return _multiblock_topk_token_ids(scores, token_ids, topk)
    if backend == "cutedsl":
        return _cutedsl_topk_token_ids(scores, token_ids, topk)
    raise ValueError(
        "FLEET_SLASHMLA_HCA_FINE_TOPK must be 'multiblock' or 'cutedsl', got "
        f"{backend!r}."
    )


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
    fine_backend="cutedsl",
    query_offset=0,
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
    token_len = int(token_key.shape[1])
    query_offset = int(query_offset)
    # The query window may be the tail of a longer key timeline; only the batch
    # dimension has to agree.
    if int(token_key.shape[0]) != b or token_len < sq:
        raise ValueError(
            "CSA two-stage indexer needs token_key [B,T,D] with T >= S, got "
            f"{list(token_key.shape)} against query {list(query.shape)}."
        )
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
    if stage1_backend != "tilelang":
        raise ValueError(
            "stage1_backend must be 'tilelang'; the Paddle and CuTeDSL "
            f"score-only coarse stages are unsupported. Got {stage1_backend!r}"
        )
    if fine_backend not in ("tilelang", "cutedsl"):
        raise ValueError(
            "fine_backend must be 'cutedsl' (SM90 cp.async/WGMMA score) or "
            f"'tilelang' (SM100+ fine score), got {fine_backend!r}"
        )
    if (
        select_dim <= 0
        or select_dim > qdim
        or select_dim > int(token_key.shape[-1])
    ):
        raise ValueError(
            "slash_dim must be positive and fit query and token-key widths."
        )
    if select_dim > int(compressed_key.shape[-1]):
        raise ValueError("slash_dim must fit compressed-key width.")

    # Materialize the selected tail dimensions at the indexer boundary.  Every
    # backend below must see the same compact tensors: some kernels read from
    # column zero and cannot safely consume a strided tail view of the latent
    # representation.
    query = query[..., -select_dim:].contiguous()
    token_key = token_key[..., -select_dim:].contiguous()
    compressed_key = compressed_key[..., -select_dim:].contiguous()
    starts = block_starts.cast("int64")
    valid_range = valid_range.cast("int64")
    n_blocks = int(compressed_key.shape[1])
    outputs = []
    coarse_indices_all = None
    if n_blocks > 0:
        profile_start = _profile_start(profile)
        try:
            coarse_indices_all = _coarse_topk_tilelang(
                query,
                compressed_key,
                starts,
                valid_range,
                block_topk,
                ratio,
                query_offset,
            )
        finally:
            _profile_end(profile_start, profile_timings, "stage1")

    for q_start in range(0, sq, query_chunk_size):
        q_end = min(q_start + query_chunk_size, sq)
        chunk_len = q_end - q_start
        ranges = valid_range[:, q_start:q_end]

        if n_blocks == 0:
            # No completed compressed block exists yet, so there is nothing for
            # the fine stage to score. Emit the all-invalid row the callers'
            # sparse backends already understand.
            outputs.append(paddle.full([b, chunk_len, topk], -1, dtype="int32"))
            continue
        selected_indices = coarse_indices_all[:, q_start:q_end]

        profile_start = _profile_start(profile)
        if fine_backend == "cutedsl":
            from paddlefleet.cutedsl_ops.hca_stage2 import (
                hca_stage2_score_cutedsl,
            )

            query_positions = (
                (paddle.arange(q_start, q_end, dtype="int32") + query_offset)
                .reshape([1, chunk_len])
                .expand([b, chunk_len])
            )
            scores, fused_candidates = hca_stage2_score_cutedsl(
                query[:, q_start:q_end],
                token_key,
                selected_indices,
                starts,
                ranges,
                query_positions,
                ratio=ratio,
                score_backend="cpasync",
                select_dim=select_dim,
            )
        else:
            from paddlefleet.tilelang_ops.indexer.csa_fine_score import (
                csa_fine_score_fwd,
            )

            scores, fused_candidates = csa_fine_score_fwd(
                query[:, q_start:q_end],
                token_key,
                selected_indices,
                starts,
                ranges,
                ratio=ratio,
                query_offset=q_start + query_offset,
            )
        _profile_end(profile_start, profile_timings, "fine_score")

        profile_start = _profile_start(profile)
        final_indices = _fine_topk_token_ids(scores, fused_candidates, topk)
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
