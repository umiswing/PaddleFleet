# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Token-sparse absorbed MLA and HCA+MLA helpers.

The original SLASH implementation was written for shared-KV VHA.  This module
keeps its token-level routing contract, but scores the absorbed MLA query
against the shared latent key and uses the absorbed-MQA sparse backend for the
actual attention.

The indexer is deliberately kept behind small function boundaries, and each
boundary picks exactly one kernel per GPU architecture -- there are no silent
Paddle fallbacks.  An unsupported device or a missing optional extension raises
instead of quietly running a different (much slower, numerically different)
implementation:

* plain token top-k: the legacy CuTeDSL two-stage indexer on SM90, the TileLang
  indexer on SM100+;
* HCA coarse stage: the TileLang CSA block indexer on every architecture;
* HCA fine stage: the SM90 cp.async/WGMMA score kernel on SM90, the TileLang
  fine-score kernel on SM100+.
"""

from __future__ import annotations

import paddle


def _device_major() -> int:
    """Compute-capability major number of the current CUDA device."""
    return int(paddle.device.cuda.get_device_capability()[0])


def _require_slashmla_flash_mla() -> None:
    """Raise unless the FlashMLA sparse forward kernel is usable.

    SlashMLA uses the local CuTeDSL backward on SM90, so it must not reuse
    ``is_dsa_available``: that helper deliberately requires SM100+ and also
    checks the cuDNN DSA backward dependency. SlashMLA only needs the FlashMLA
    sparse forward operator here.
    """
    try:
        import paddlefleet_ops

        from paddlefleet.cudnn_ops.attn import csa_sparse_attn_fwd_cudnn

        available = (
            paddlefleet_ops.is_flash_mla_available()
            and csa_sparse_attn_fwd_cudnn._flash_mla_sparse_fwd is not None
        )
    except (ImportError, RuntimeError, AttributeError) as exc:
        raise RuntimeError(
            "SlashMLA requires the FlashMLA sparse forward kernel from "
            "paddlefleet_ops, which could not be imported."
        ) from exc
    if not available:
        raise RuntimeError(
            "SlashMLA requires the FlashMLA sparse forward kernel, but it is "
            "unavailable on this device."
        )


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


def _tilelang_topk(query, key, topk, valid_range):
    """Run the TileLang token indexer used on SM100+."""
    from paddlefleet.tilelang_ops.indexer.slashmla_indexer import (
        slashmla_tilelang_topk,
    )

    return slashmla_tilelang_topk(
        query,
        key,
        min(int(topk), int(key.shape[1])),
        valid_range,
    )


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

    # One indexer per architecture, no fallback: the legacy CuTeDSL two-stage
    # kernel is Hopper-only, and the TileLang indexer is what SM100+ uses. Both
    # were verified to select the same token set, so this is a kernel choice and
    # not a semantic switch.
    major = _device_major()
    if major == 9:
        return _cutedsl_topk(query, key, topk, valid_range)
    if major >= 10:
        return _tilelang_topk(query, key, topk, valid_range)
    raise RuntimeError(
        "SlashMLA token indexer requires SM90 (CuTeDSL two-stage indexer) or "
        f"SM100+ (TileLang indexer); got compute capability major {major}."
    )


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
            # The coarse stage is TileLang on every architecture. The fine stage
            # has one kernel per architecture: the SM90 cp.async/WGMMA score
            # kernel on Hopper, the TileLang fine-score kernel on SM100+.
            stage1_backend=(
                "tilelang" if stage1_backend is None else stage1_backend
            ),
            fine_backend=(
                _default_hca_fine_backend()
                if fine_backend is None
                else fine_backend
            ),
        )
    result.stop_gradient = True
    return result


def _default_hca_fine_backend():
    """Pick the HCA stage-2 score kernel for the current architecture."""
    major = _device_major()
    if major == 9:
        return "cutedsl"
    if major >= 10:
        return "tilelang"
    raise RuntimeError(
        "SlashMLA HCA stage-2 requires SM90 (CuTeDSL cp.async score kernel) "
        f"or SM100+ (TileLang fine score); got compute capability major "
        f"{major}."
    )


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
    _require_slashmla_flash_mla()

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
    _require_slashmla_flash_mla()

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
    gate1=None,
    gate2=None,
    return_branches=False,
):
    """Return gated HCA plus SparseMLA over one shared MLA token KV.

    When ``gate1`` and ``gate2`` are provided, they are applied independently
    to the HCA and SparseMLA branches before fusion:

    ``sigmoid(gate1) * O_hca + alpha * sigmoid(gate2) * O_sparse``.

    ``return_branches`` exposes the two branch outputs so callers that own the
    gate projections can apply them without sharing one gate across branches.
    """
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
    )
    hca_output = _deabsorb_value(hca_latent, value_weight, value_dim)
    if gate1 is not None:
        hca_output = hca_output * paddle.nn.functional.sigmoid(gate1)
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
    if gate2 is not None:
        sparse_output = sparse_output * paddle.nn.functional.sigmoid(gate2)
    if return_branches:
        return hca_output, sparse_output, token_indices
    output = hca_output + float(alpha) * sparse_output
    return output, token_indices
