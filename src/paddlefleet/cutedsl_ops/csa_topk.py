# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CuTeDSL top-k adapter for CSA score-only outputs."""

import paddle


def csa_topk_cutedsl(scores, token_ids, topk):
    """Select CSA token IDs with ``two_stage_indexer._TwoStageTopK``."""
    from cutlass import Int32, cute

    from paddlefleet.cutedsl_ops.dlpack import paddle_to_cute_tensor
    from paddlefleet.cutedsl_ops.two_stage_indexer import (
        _THREADS,
        _TOPK_CACHE,
        _device_cc,
        _num_sms,
        _select_bucket,
        _TwoStageTopK,
    )

    if scores.ndim != 3 or token_ids.ndim != 3:
        raise ValueError("scores and token_ids must be [B, Q, N].")
    if list(scores.shape) != list(token_ids.shape):
        raise ValueError("scores and token_ids must have identical shapes.")
    batch, query_len, candidate_count = [int(x) for x in scores.shape]
    topk = int(topk)
    if candidate_count <= 0 or topk <= 0:
        raise ValueError("candidate_count and topk must be positive.")

    bucket = _select_bucket(candidate_count)
    padded_topk = 1
    while padded_topk < max(topk, _THREADS):
        padded_topk <<= 1
    if padded_topk > bucket:
        raise ValueError(f"topk={topk} does not fit CuTeDSL bucket={bucket}.")

    rows = batch * query_len
    flat_scores = scores.reshape([rows, candidate_count]).contiguous()
    if candidate_count < bucket:
        padding = paddle.full(
            [rows, bucket - candidate_count],
            -float("inf"),
            dtype="float32",
        )
        score_workspace = paddle.concat(
            [flat_scores, padding], axis=1
        ).contiguous()
    else:
        score_workspace = flat_scores
    row_visible = paddle.full(
        [rows], candidate_count, dtype="int32"
    ).contiguous()
    output = paddle.empty([rows, padded_topk], dtype="int32")

    cc_major, cc_minor = _device_cc()
    n_blocks = min(_num_sms(), rows)
    topk_key = (
        cc_major,
        cc_minor,
        rows,
        bucket,
        padded_topk,
        32,
        n_blocks,
        False,
    )
    if topk_key not in _TOPK_CACHE:
        topk_kernel = _TwoStageTopK(bucket, padded_topk, 32, n_blocks, False)
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _TOPK_CACHE[topk_key] = cute.compile(
            topk_kernel,
            paddle_to_cute_tensor(row_visible, assumed_align=4, leading_dim=0),
            paddle_to_cute_tensor(
                score_workspace, assumed_align=4, leading_dim=1
            ),
            paddle_to_cute_tensor(output, assumed_align=4, leading_dim=1),
            Int32(topk),
            fake_stream,
            options="--enable-tvm-ffi",
        )

    _TOPK_CACHE[topk_key](
        row_visible,
        score_workspace,
        output,
        Int32(topk),
    )
    positions = output[:, :topk].reshape([batch, query_len, topk])
    safe_positions = positions.clip(0, max(candidate_count - 1, 0))
    selected_scores = paddle.take_along_axis(
        scores, safe_positions.cast("int64"), axis=-1
    )
    selected_ids = paddle.take_along_axis(
        token_ids, safe_positions.cast("int64"), axis=-1
    )
    valid = (
        (positions >= 0)
        & (safe_positions < candidate_count)
        & (selected_scores > -1.0e20)
    )
    return paddle.where(valid, selected_ids, paddle.full_like(selected_ids, -1))


__all__ = ["csa_topk_cutedsl"]
