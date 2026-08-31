# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Stable HCA stage-2 score entry points.

Only the SM90 cp.async score path is exposed here.  The scalar, MMA, TMA and
cuDNN experiment branches from the source tree are intentionally not carried
into the online tree.
"""

from __future__ import annotations


def hca_stage2_score_cutedsl(
    query,
    token_key,
    selected_indices,
    block_starts,
    valid_range,
    query_positions,
    ratio=128,
    score_backend=None,
    select_dim=None,
):
    """Return HCA stage-2 candidate scores and token IDs.

    ``score_backend="cpasync"`` dispatches to the stable SM90
    ``_HcaScoreSm90CpAsync`` implementation.  Other backends are deliberately
    rejected instead of silently selecting an experimental fallback.
    """
    if selected_indices.ndim != 3 or block_starts.ndim != 2:
        raise ValueError(
            "selected_indices must be [B,Q,block_topk] and block_starts [B,C]."
        )
    if score_backend is None:
        score_backend = "cpasync"
    score_backend = str(score_backend).lower()
    if score_backend != "cpasync":
        raise ValueError(
            "online HCA stage-2 only supports score_backend='cpasync', "
            f"got {score_backend!r}."
        )

    from .hca_score_sm90 import hca_stage2_score_cpasync_sm90

    return hca_stage2_score_cpasync_sm90(
        query,
        token_key,
        selected_indices,
        block_starts,
        valid_range,
        query_positions,
        ratio=ratio,
        select_dim=select_dim,
    )


__all__ = ["hca_stage2_score_cutedsl"]
