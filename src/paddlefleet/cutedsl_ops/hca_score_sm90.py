# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""HCA stage-2 score kernel: packed-head WGMMA over cp.async-gathered KV.

The compute structure is the one the Slash score kernel inherited from the
cuDNN SM90 indexer forward -- a padded 64-column packed-head Q tile, a
``64 x 64 x D`` WGMMA per KV tile, an FP32 accumulator reduced across the head
columns, and a width-4 butterfly to finish the sum.  Two things differ because
HCA stage 2 scores a *per-row* candidate list instead of a dense prefix:

  * one CTA owns one query row, so all 64 MMA columns belong to that row's
    heads (padded with zeros) rather than to several query tokens;
  * the KV tile origin comes from ``block_starts[selected[row, block_i]]``, so
    the producer is an indexed cp.async gather with a two-stage software
    pipeline instead of a contiguous TMA stream.

Each selected HCA block is ``ratio`` contiguous tokens, so the gather stays
coalesced: the index only picks the tile origin.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.hopper_helpers as sm90_utils
import paddle
from cutlass import BFloat16, Float32, Int32, cute
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum

from .dlpack import paddle_to_cute_tensor
from .two_stage_indexer import _sm90_acc_mn_view

_THREADS = 128
_TILE_N = 64
_KV_STAGES = 2
_MMA_HEADS = 64
_HEADS = (8, 16, 32, 64)
_DIMS = (32, 64, 128)
_NEG = -3.0e38
_CACHE: dict = {}


def _make_smem_layout(dtype, shape, stage=None):
    """Build the row-major swizzled SM90 operand layout."""
    atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, dtype, shape[1]),
        dtype,
    )
    return cute.tile_to_shape(
        atom,
        cute.append(shape, stage) if stage is not None else shape,
        order=(0, 1, 2) if stage is not None else (0, 1),
    )


@cute.jit
def _warp_reduce_add4(val: Float32) -> Float32:
    """Reduce the four lanes belonging to one WGMMA column group."""
    val = val + cute.arch.shuffle_sync_bfly(val, offset=1)
    val = val + cute.arch.shuffle_sync_bfly(val, offset=2)
    return val


def _tiled_copy_2d(dtype, major_mode_size, num_threads):
    """Async 2-D copy partition, one row group per thread quad-set."""
    import math

    num_copy_bits = math.gcd(major_mode_size, 128 // dtype.width) * dtype.width
    copy_elems = num_copy_bits // dtype.width
    atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(), dtype, num_bits_per_copy=num_copy_bits
    )
    threads_per_row = major_mode_size // copy_elems
    assert num_threads % threads_per_row == 0
    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row), order=(1, 0)
    )
    val_layout = cute.make_layout((1, copy_elems))
    return cute.make_tiled_copy_tv(atom, thr_layout, val_layout)


class _HcaScoreSm90CpAsync:
    """Score one HCA candidate list per CTA with WGMMA over gathered KV."""

    def __init__(
        self,
        heads: int,
        head_dim: int,
        block_topk: int,
        ratio: int,
        tiles_per_cta=None,
    ):
        if heads not in _HEADS:
            raise ValueError(f"unsupported SM90 head count {heads}")
        if head_dim not in _DIMS:
            raise ValueError(f"unsupported SM90 head_dim {head_dim}")
        if block_topk <= 0 or ratio <= 0:
            raise ValueError("block_topk and ratio must be positive")
        if ratio % _TILE_N != 0:
            raise ValueError(
                f"ratio must be a multiple of the {_TILE_N}-token KV tile"
            )
        self.dtype = BFloat16
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.block_topk = int(block_topk)
        self.ratio = int(ratio)
        self.candidate_count = (self.block_topk + 1) * self.ratio
        self.tiles_per_block = self.ratio // _TILE_N
        # The current block is scored as one extra candidate block.
        self.num_tiles = (self.block_topk + 1) * self.tiles_per_block
        # One CTA per (row, tile group).  Measured at 8K, walking a row's whole
        # candidate list in one CTA is the fastest point (0.95 ms, versus
        # 3.12 ms at one tile per CTA): 8192 rows already fill the machine, and
        # keeping a row together reuses its Q tile and its blocks' L2 lines.
        # Splitting only helps when the row count alone cannot fill the GPU.
        if tiles_per_cta is None:
            tiles_per_cta = self.num_tiles
        self.tiles_per_cta = max(1, min(int(tiles_per_cta), self.num_tiles))
        self.tile_groups = (
            self.num_tiles + self.tiles_per_cta - 1
        ) // self.tiles_per_cta

    def _setup_attributes(self):
        self.sQ_layout = _make_smem_layout(
            self.dtype, (_MMA_HEADS, self.head_dim)
        )
        self.sK_layout = _make_smem_layout(
            self.dtype, (_TILE_N, self.head_dim), _KV_STAGES
        )

    def _get_tiled_mma(self):
        import cutlass.utils.hopper_helpers as sm90_utils

        return sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(_TILE_N, _MMA_HEADS),
        )

    def _get_shared_storage_cls(self):
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sQ_layout)], 1024
        ]
        sK_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sK_layout)], 1024
        ]

        @cute.struct
        class SharedStorage:
            meta: cute.struct.Align[cute.struct.MemRange[Int32, 4], 16]
            sQ: sQ_struct
            sK: sK_struct

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mSelected: cute.Tensor,
        mBlockStarts: cute.Tensor,
        mValidRange: cute.Tensor,
        mQueryPosition: cute.Tensor,
        mBatchId: cute.Tensor,
        mScores: cute.Tensor,
        mTokenIds: cute.Tensor,
        token_len: Int32,
        num_blocks: Int32,
        stream: cuda.CUstream,
    ):
        self._setup_attributes()
        tiled_mma = self._get_tiled_mma()
        SharedStorage = self._get_shared_storage_cls()
        tiled_copy = _tiled_copy_2d(self.dtype, self.head_dim, _THREADS)
        self.kernel(
            mQ,
            mK,
            mSelected,
            mBlockStarts,
            mValidRange,
            mQueryPosition,
            mBatchId,
            mScores,
            mTokenIds,
            token_len,
            num_blocks,
            self.sQ_layout,
            self.sK_layout,
            tiled_mma,
            tiled_copy,
            SharedStorage,
        ).launch(
            grid=(cute.size(mScores.shape[0]), self.tile_groups, 1),
            block=[_THREADS, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mSelected: cute.Tensor,
        mBlockStarts: cute.Tensor,
        mValidRange: cute.Tensor,
        mQueryPosition: cute.Tensor,
        mBatchId: cute.Tensor,
        mScores: cute.Tensor,
        mTokenIds: cute.Tensor,
        token_len: Int32,
        num_blocks: Int32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        tiled_copy: cute.TiledCopy,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, group, _ = cute.arch.block_idx()
        row = Int32(row)
        tidx = Int32(tidx)
        tile_begin = Int32(group) * Int32(self.tiles_per_cta)
        tile_end = tile_begin + Int32(self.tiles_per_cta)
        if tile_end > Int32(self.num_tiles):
            tile_end = Int32(self.num_tiles)

        storage = cutlass.utils.SmemAllocator().allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        meta = storage.meta.get_tensor(cute.make_layout((4,), stride=(1,)))

        valid_start = Int32(mValidRange[row, 0])
        valid_end = Int32(mValidRange[row, 1])
        key_batch = Int32(mBatchId[row])
        query_position = Int32(mQueryPosition[row])
        current_start = valid_start + (
            (query_position - valid_start) // Int32(self.ratio)
        ) * Int32(self.ratio)

        # The current block may already appear among the selected blocks; the
        # scalar reference drops the duplicate, so mirror that decision once.
        if tidx == Int32(0):
            duplicate = Int32(0)
            for block_i in cutlass.range_constexpr(self.block_topk):
                block_id = Int32(mSelected[row, block_i])
                if (block_id >= Int32(0)) & (block_id < num_blocks):
                    if (
                        Int32(mBlockStarts[key_batch, block_id])
                        == current_start
                    ):
                        duplicate = Int32(1)
            meta[0] = duplicate

        # Pack the row's heads into the MMA N dimension and zero the padding so
        # the column sum in the epilogue stays exact.
        for linear in cutlass.range(tidx, _MMA_HEADS * self.head_dim, _THREADS):
            head = linear // Int32(self.head_dim)
            dim_i = linear % Int32(self.head_dim)
            value = self.dtype(0.0)
            if head < Int32(self.heads):
                value = mQ[row, head, dim_i]
            sQ[head, dim_i] = value
        cute.arch.barrier()
        cute.arch.fence_view_async_shared()

        rows = Int32(cute.size(mScores.shape[0]))
        thr_mma = tiled_mma.get_slice(tidx)
        tCrQ = thr_mma.make_fragment_B(thr_mma.partition_B(sQ))
        tCrK = thr_mma.make_fragment_A(thr_mma.partition_A(sK))
        thr_copy = tiled_copy.get_slice(tidx)
        duplicate = meta[0]
        del rows

        start, cand, ok = self._tile_info(
            tile_begin,
            mSelected,
            mBlockStarts,
            row,
            key_batch,
            current_start,
            duplicate,
            num_blocks,
            token_len,
        )
        self._issue_copy(thr_copy, mK, sK, start, Int32(0))
        cute.arch.cp_async_commit_group()

        tile = tile_begin
        while tile < tile_end:
            stage = tile % Int32(_KV_STAGES)
            start, cand, ok = self._tile_info(
                tile,
                mSelected,
                mBlockStarts,
                row,
                key_batch,
                current_start,
                duplicate,
                num_blocks,
                token_len,
            )
            next_tile = tile + Int32(1)
            if next_tile < tile_end:
                next_start, _, _ = self._tile_info(
                    next_tile,
                    mSelected,
                    mBlockStarts,
                    row,
                    key_batch,
                    current_start,
                    duplicate,
                    num_blocks,
                    token_len,
                )
                self._issue_copy(
                    thr_copy,
                    mK,
                    sK,
                    next_start,
                    next_tile % Int32(_KV_STAGES),
                )
                cute.arch.cp_async_commit_group()
                # One group still in flight is the tile we are about to read.
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            cute.arch.fence_view_async_shared()

            acc = self._gemm(tiled_mma, tCrK, tCrQ, stage)
            self._epilogue(
                acc,
                mScores,
                mTokenIds,
                tile,
                cand,
                ok,
                valid_start,
                valid_end,
                token_len,
                row,
            )
            # Hold the stage until every lane finished reading it.
            cute.arch.barrier()
            tile = tile + Int32(1)

    @cute.jit
    def _tile_info(
        self,
        tile: Int32,
        mSelected: cute.Tensor,
        mBlockStarts: cute.Tensor,
        row: Int32,
        key_batch: Int32,
        current_start: Int32,
        duplicate: Int32,
        num_blocks: Int32,
        token_len: Int32,
    ):
        """Return the gather origin, first candidate token, and block validity."""
        block_i = tile // Int32(self.tiles_per_block)
        tile_in_block = tile % Int32(self.tiles_per_block)
        block_start = current_start
        block_ok = duplicate == Int32(0)
        if block_i < Int32(self.block_topk):
            block_id = Int32(mSelected[row, block_i])
            block_ok = (block_id >= Int32(0)) & (block_id < num_blocks)
            safe_id = block_id
            if safe_id < Int32(0):
                safe_id = Int32(0)
            if safe_id >= num_blocks:
                safe_id = num_blocks - Int32(1)
            block_start = Int32(mBlockStarts[key_batch, safe_id])
        candidate_start = block_start + tile_in_block * Int32(_TILE_N)
        # Clamp the gather so it never leaves this batch's token range.  The
        # epilogue derives its candidate ids from the clamped origin, so the
        # score and token id stay consistent even when a block is bogus; the
        # validity mask then discards those slots anyway.
        if candidate_start < Int32(0):
            candidate_start = Int32(0)
        if candidate_start > token_len - Int32(_TILE_N):
            candidate_start = token_len - Int32(_TILE_N)
        return (
            key_batch * token_len + candidate_start,
            candidate_start,
            block_ok,
        )

    @cute.jit
    def _issue_copy(
        self,
        thr_copy,
        mK: cute.Tensor,
        sK: cute.Tensor,
        abs_start: Int32,
        stage: Int32,
    ):
        # ``domain_offset`` with a dynamic origin loses the alignment proof the
        # 128-bit cp.async atom needs, so rebuild the tile view on an explicitly
        # aligned pointer.  The row stride comes from the caller's tensor, which
        # lets a ``[B, T, 576]`` latent view be read in place: only the leading
        # ``head_dim`` columns are touched.  Every supported row stride is a
        # multiple of 8 bf16 elements, i.e. of 16 B, so the atom stays legal.
        row_stride = cute.assume(mK.stride[0], divby=8)
        offset = cutlass.Int64(abs_start) * cutlass.Int64(row_stride)
        ptr = cute.make_ptr(
            self.dtype,
            (mK.iterator + offset).toint(),
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        gTile = cute.make_tensor(
            ptr,
            cute.make_layout((_TILE_N, self.head_dim), stride=(row_stride, 1)),
        )
        cute.copy(
            thr_copy,
            thr_copy.partition_S(gTile),
            thr_copy.partition_D(sK[None, None, stage]),
        )

    @cute.jit
    def _gemm(
        self,
        tiled_mma: cute.TiledMma,
        tCrK: cute.Tensor,
        tCrQ: cute.Tensor,
        stage: Int32,
    ) -> cute.Tensor:
        acc = cute.make_rmem_tensor(
            tiled_mma.partition_shape_C((_TILE_N, _MMA_HEADS)), Float32
        )
        rK = tCrK[None, None, None, stage]
        warpgroup.fence()
        atom = cute.make_mma_atom(tiled_mma.op)
        atom.set(warpgroup.Field.ACCUMULATE, False)
        for k in cutlass.range_constexpr(cute.size(rK.shape[2])):
            cute.gemm(atom, acc, rK[None, None, k], tCrQ[None, None, k], acc)
            atom.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        warpgroup.wait_group(0)
        return acc

    @cute.jit
    def _epilogue(
        self,
        acc: cute.Tensor,
        mScores: cute.Tensor,
        mTokenIds: cute.Tensor,
        tile: Int32,
        candidate_start: Int32,
        block_ok,
        valid_start: Int32,
        valid_end: Int32,
        token_len: Int32,
        row: Int32,
    ):
        """Sum the padded head columns and store one KV tile of candidates."""
        acc_mn = _sm90_acc_mn_view(acc)
        n_rows = cute.size(acc_mn, mode=[0])
        n_cols = cute.size(acc_mn, mode=[1])

        lane = cute.arch.lane_idx()
        quad_lane = lane % 4
        m_base = (
            lane // 4
            + (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) * 16
        )

        sums = cute.make_rmem_tensor((n_rows,), Float32)
        for mi in cutlass.range_constexpr(n_rows):
            total = Float32(0.0)
            for ni in cutlass.range_constexpr(n_cols):
                total = total + acc_mn[mi, ni]
            sums[mi] = _warp_reduce_add4(total)

        if quad_lane == Int32(0):
            slot_base = tile * Int32(_TILE_N)
            for mi in cutlass.range_constexpr(n_rows):
                kv_m = m_base + Int32(mi * 8)
                slot = slot_base + kv_m
                candidate = candidate_start + kv_m
                valid = (
                    block_ok
                    & (candidate >= valid_start)
                    & (candidate < valid_end)
                    & (candidate < token_len)
                )
                if valid:
                    mScores[row, slot] = sums[mi]
                    mTokenIds[row, slot] = candidate
                else:
                    mScores[row, slot] = Float32(_NEG)
                    mTokenIds[row, slot] = Int32(-1)


def _device_cc():
    props = paddle.device.cuda.get_device_properties()
    return int(props.major), int(props.minor)


def hca_stage2_score_cpasync_sm90(
    query,
    token_key,
    selected_indices,
    block_starts,
    valid_range,
    query_positions,
    ratio=128,
    tiles_per_cta=None,
    select_dim=None,
):
    """Score HCA stage-2 candidates with the WGMMA + cp.async gather kernel."""
    if query.ndim != 4 or token_key.ndim != 3:
        raise ValueError("query must be [B,Q,H,D] and token_key [B,T,D].")
    if selected_indices.ndim != 3 or block_starts.ndim != 2:
        raise ValueError(
            "selected_indices must be [B,Q,block_topk] and block_starts [B,C]."
        )
    b, q_len, heads, full_dim = [int(x) for x in query.shape]
    dim = full_dim if select_dim is None else int(select_dim)
    token_full_dim = int(token_key.shape[2])
    if dim > full_dim or dim > token_full_dim:
        raise ValueError(
            f"select_dim={dim} exceeds the query/token-key widths "
            f"({full_dim}, {token_full_dim})."
        )
    token_len = int(token_key.shape[1])
    block_topk = int(selected_indices.shape[-1])
    num_blocks = int(block_starts.shape[-1])
    ratio = int(ratio)
    if heads not in _HEADS or dim not in _DIMS:
        raise ValueError(
            f"the HCA cp.async score kernel requires H in {_HEADS} and "
            f"D in {_DIMS}; got H={heads}, D={dim}."
        )
    if query.dtype != paddle.bfloat16 or token_key.dtype != paddle.bfloat16:
        raise TypeError("the HCA cp.async score kernel requires bfloat16 Q/K.")
    if token_len % _TILE_N != 0 or token_len < _TILE_N:
        raise ValueError(f"token_len must be a positive multiple of {_TILE_N}.")
    if list(valid_range.shape) != [b, q_len, 2]:
        raise ValueError("valid_range must be [B,Q,2].")
    if list(query_positions.shape) != [b, q_len]:
        raise ValueError("query_positions must be [B,Q].")
    cc_major, cc_minor = _device_cc()
    if (cc_major, cc_minor) != (9, 0):
        raise RuntimeError(
            f"the HCA cp.async score kernel requires SM90, got "
            f"SM{cc_major}{cc_minor}."
        )

    rows = b * q_len
    candidate_count = (block_topk + 1) * ratio
    # Read the latent slice in place: a contiguous ``[B,S,H,full]`` tensor
    # reshapes to ``[rows,H,full]`` without a copy, and the kernel only touches
    # the leading ``dim`` columns.  Materialising ``[..., :dim]`` instead costs
    # a 16 MB copy per call at 8K.
    q_flat = query.reshape([rows, heads, full_dim])
    k_flat = token_key.reshape([b * token_len, token_full_dim])
    selected_flat = (
        selected_indices.cast("int32").contiguous().reshape([rows, block_topk])
    )
    starts = block_starts.cast("int32").contiguous()
    ranges = valid_range.cast("int32").contiguous().reshape([rows, 2])
    positions = query_positions.cast("int32").contiguous().reshape([rows])
    batch_ids = (
        paddle.arange(b, dtype="int32")
        .reshape([b, 1])
        .expand([b, q_len])
        .reshape([rows])
        .contiguous()
    )
    scores = paddle.empty([rows, candidate_count], dtype="float32")
    token_ids = paddle.empty([rows, candidate_count], dtype="int32")

    cache_key = (
        cc_major,
        cc_minor,
        heads,
        dim,
        full_dim,
        token_full_dim,
        block_topk,
        ratio,
        rows,
        None if tiles_per_cta is None else int(tiles_per_cta),
    )
    if cache_key not in _CACHE:
        kernel = _HcaScoreSm90CpAsync(
            heads, dim, block_topk, ratio, tiles_per_cta
        )
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _CACHE[cache_key] = cute.compile(
            kernel,
            paddle_to_cute_tensor(q_flat, assumed_align=16, leading_dim=2),
            paddle_to_cute_tensor(k_flat, assumed_align=16, leading_dim=1),
            paddle_to_cute_tensor(
                selected_flat, assumed_align=4, leading_dim=1
            ),
            paddle_to_cute_tensor(starts, assumed_align=4, leading_dim=1),
            paddle_to_cute_tensor(ranges, assumed_align=4, leading_dim=1),
            paddle_to_cute_tensor(positions, assumed_align=4, leading_dim=0),
            paddle_to_cute_tensor(batch_ids, assumed_align=4, leading_dim=0),
            paddle_to_cute_tensor(scores, assumed_align=4, leading_dim=1),
            paddle_to_cute_tensor(token_ids, assumed_align=4, leading_dim=1),
            Int32(token_len),
            Int32(num_blocks),
            fake_stream,
            options="--enable-tvm-ffi",
        )
    _CACHE[cache_key](
        q_flat,
        k_flat,
        selected_flat,
        starts,
        ranges,
        positions,
        batch_ids,
        scores,
        token_ids,
        Int32(token_len),
        Int32(num_blocks),
    )
    return (
        scores.reshape([b, q_len, candidate_count]),
        token_ids.reshape([b, q_len, candidate_count]),
    )


__all__ = ["hca_stage2_score_cpasync_sm90"]
