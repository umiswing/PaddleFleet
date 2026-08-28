# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""PaddleFleet-owned SM90 DSA backward entry.

The SM90 CuTe DSL kernel is owned by ``paddlefleet.cutedsl_ops.dsa_sm90``.
This compatibility-named entry is kept for existing fusion callers, but it no
longer dispatches through the cuDNN frontend Python API.
"""


def csa_sparse_attn_bwd_cutedsl(
    q,  # (total_sq, H, D) bf16
    kv,  # (total_skv, D) bf16
    out,  # (total_sq, H, D) bf16
    dout,  # (total_sq, H, D) bf16
    lse,  # (total_sq, H) fp32
    attn_sink,  # (H,) fp32
    topk_idxs,  # (total_sq, topk) int32, global flat indices
    softmax_scale=None,
    topk_length=None,
    need_d_sink=False,
):
    """Run the local PaddleFleet SM90 DSA backward kernel."""
    assert not need_d_sink, (
        "SM90 CuTeDSL sparse-attention backward does not support d_sink; "
        "compute the sink gradient analytically in the caller."
    )
    from paddlefleet.cutedsl_ops.dsa_sm90.interface import flash_attn_bwd_sm90

    result = flash_attn_bwd_sm90(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink=attn_sink,
        topk_idxs=topk_idxs,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
    )
    if len(result) == 2:
        return result[0], result[1], None
    return result
