# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

"""Two-stage Sparse VHA routing indexer for SM90.

The indexer is split into two kernels connected through a global FP32 score
workspace:

    stage 1 (score kernel)  -- reduce query heads per KV group and compute the
        grouped QK score with either warp-parallel FP32 reductions or the
        optional TMA/WGMMA backend. Both paths write ``score * sm_scale`` into
        a per-row FP32 score buffer.

    stage 2 (top-k kernel)  -- a persistent one-block-per-SM radix top-k that
        reads the FP32 scores, quantises them to 32-bit or 16-bit sortable keys
        in shared memory, and emits deterministic doc-local candidate indices.

        The top-k kernel is launched persistently (grid = SM count, each block
        grid-strides over rows) so exactly one block is resident per SM.  This
        mirrors the fused kernel, whose large shared-memory footprint already
        forces single-block occupancy; running many small top-k blocks per SM
        instead exposes a shared-memory co-residency hazard in the radix
        compaction, so we deliberately keep the proven one-block-per-SM regime.

A host driver (:func:`dsa_sparse_vha_topk_two_stage_cutedsl`) accepts BSHD
Q/KV tensors, builds per-row metadata, chunks rows so the score buffer fits in
memory, and dispatches the two kernels.

This module intentionally contains no SM100/Blackwell implementation.
"""

from __future__ import annotations

import math
from typing import Final

# ``cuda`` must be a real module-level binding, not a TYPE_CHECKING-only one:
# this file uses ``from __future__ import annotations``, so the kernel entry
# points' ``stream: cuda.CUstream`` annotations are strings that CuTeDSL
# resolves with ``inspect.signature(..., eval_str=True)`` at ``cute.compile``
# time. Behind TYPE_CHECKING that lookup raises ``NameError: name 'cuda' is
# not defined`` and every kernel in this module becomes uncompilable.
import cuda.bindings.driver as cuda  # noqa: TC002
import cutlass
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import BFloat16, Float32, Int32, Uint16, Uint32, cute
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineTmaAsync,
    PipelineUserType,
    make_pipeline_state,
    pipeline_init_arrive,
    pipeline_init_wait,
)
from cutlass.utils import LayoutEnum
from cutlass.utils.distributed import atomicAdd

from .dlpack import paddle_to_cute_tensor
from .exact_radix import ExactRadixSelector, fp32_to_sortable_uint32

_THREADS: Final = 256
_BUCKET_8K: Final = 8192
_BUCKET_32K: Final = 32768
_GROUPS: Final = 2
_SM90_MMA_TILE_K: Final = 64
_SM90_HEADS: Final = (8, 16, 32, 64)
_SM90_DIMS: Final = (32, 64)

# Cap the FP32 score workspace at ~4 GiB so 8x8192 fits a single chunk while
# large 32K launches split across several row chunks.
_SCORE_BYTES_BUDGET: Final = 4 * 1024 * 1024 * 1024

_DEFAULT_SMS: Final = 148

# Sentinel below any real grouped-QK score; padding/invalid slots sink to the end.
_NEG_SENTINEL: Final = -3.0e38

_SCORE_CACHE: dict[tuple[object, ...], object] = {}
_TOPK_CACHE: dict[tuple[object, ...], object] = {}


def _num_sms() -> int:
    """Return the device SM count (persistent grid width for the top-k stage)."""
    try:
        import paddle

        return int(
            paddle.device.cuda.get_device_properties(0).multi_processor_count
        )
    except Exception:
        return _DEFAULT_SMS


def _device_cc() -> tuple[int, int]:
    """Return the active CUDA device compute capability."""
    import paddle

    props = paddle.device.cuda.get_device_properties()
    return int(props.major), int(props.minor)


def is_cutedsl_available() -> bool:
    """Return whether Paddle CUDA and the CuTe DSL runtime are importable."""
    try:
        import paddle

        return paddle.device.is_compiled_with_cuda()
    except (ImportError, RuntimeError):
        return False


class _TwoStageRadixSelector:
    """Radix top-m threshold + stable emitter parameterised by pass count.

    This is a straight generalisation of
    :meth:`exact_radix.ExactRadixSelector.select`.  With ``passes == 4`` and a
    ``Uint32`` key cache it is bit-identical to the exact 32-bit selector; with
    ``passes == 2`` and a ``Uint16`` key cache it performs the quantised 16-bit
    selection used by the two-stage top-k kernel.  ``keys`` values are read and
    widened to ``Uint32`` so the same comparison/compaction code serves both.
    """

    RADIX = 256
    THREADS = 256

    def __init__(self, bucket_size: int, padded_topm: int, passes: int):
        if bucket_size not in (_BUCKET_8K, _BUCKET_32K):
            raise ValueError(f"unsupported radix bucket {bucket_size}")
        if padded_topm <= 0 or padded_topm > bucket_size:
            raise ValueError(f"invalid padded_topm {padded_topm}")
        if padded_topm % self.THREADS != 0:
            raise ValueError(
                f"padded_topm must be a multiple of {self.THREADS}"
            )
        if passes not in (2, 4):
            raise ValueError(f"unsupported radix pass count {passes}")
        self.bucket_size = bucket_size
        self.padded_topm = padded_topm
        self.passes = passes

    @cute.jit
    def select(
        self,
        keys: cute.Tensor,
        output: cute.Tensor,
        visible_len: Int32,
        requested_topm: Int32,
        histogram: cute.Tensor,
        flags: cute.Tensor,
        state: cute.Tensor,
    ):
        """Select the greatest ``requested_topm`` keys and emit local indices."""
        tid = Int32(cute.arch.thread_idx()[0])

        # state: [prefix, prefix_mask, rank_in_prefix, emitted_count]
        if tid < Int32(4):
            state[tid] = Int32(0)
        cute.arch.sync_threads()

        if visible_len <= requested_topm:
            for item in cutlass.range_constexpr(
                self.padded_topm // self.THREADS
            ):
                col = tid + Int32(item * self.THREADS)
                value = Int32(-1)
                if col < visible_len:
                    value = col
                output[col] = value
        else:
            # Only scan the chunks covering the visible window; positions
            # >= visible_len contribute nothing to the histogram/compaction.
            # This proportionally cuts the per-chunk block barriers (the
            # dominant stall for this kernel).
            n_scan = (visible_len + Int32(self.THREADS - 1)) // Int32(
                self.THREADS
            )
            if tid == Int32(0):
                state[2] = requested_topm
            cute.arch.sync_threads()

            for pass_id in cutlass.range_constexpr(self.passes):
                histogram[tid] = Int32(0)
                cute.arch.sync_threads()
                shift = (self.passes - 1 - pass_id) * 8
                prefix = Uint32(state[0])
                prefix_mask = Uint32(state[1])
                item = Int32(0)
                while item < n_scan:
                    col = tid + item * Int32(self.THREADS)
                    if col < visible_len:
                        key = Uint32(keys[col])
                        if (key & prefix_mask) == prefix:
                            digit = Int32((key >> shift) & Uint32(0xFF))
                            atomicAdd(histogram.iterator + digit, Int32(1))
                    item = item + Int32(1)
                cute.arch.sync_threads()

                if tid == Int32(0):
                    rank = state[2]
                    chosen = Int32(0)
                    running = Int32(0)
                    found = Int32(0)
                    for reverse_bin in cutlass.range_constexpr(self.RADIX):
                        bin_id = Int32(255 - reverse_bin)
                        count = histogram[bin_id]
                        if found == Int32(0) and rank <= running + count:
                            chosen = bin_id
                            state[2] = rank - running
                            found = Int32(1)
                        running = running + count
                    state[0] = Int32(prefix | (Uint32(chosen) << shift))
                    state[1] = Int32(prefix_mask | (Uint32(0xFF) << shift))
                cute.arch.sync_threads()

            threshold = Uint32(state[0])
            ties_needed = state[2]
            if tid == Int32(0):
                state[3] = Int32(0)
                state[2] = Int32(0)
            cute.arch.sync_threads()

            lane_idx = Int32(cute.arch.lane_idx())
            warp_idx = tid // Int32(32)
            lower_lane_mask = (Uint32(1) << lane_idx) - Uint32(1)
            chunk = Int32(0)
            while chunk < n_scan:
                col = chunk * Int32(self.THREADS) + tid
                is_greater = Int32(0)
                is_tie = Int32(0)
                if col < visible_len:
                    key = Uint32(keys[col])
                    if key > threshold:
                        is_greater = Int32(1)
                    elif key == threshold:
                        is_tie = Int32(1)

                tie_mask = Uint32(
                    cute.arch.vote_ballot_sync(is_tie != Int32(0))
                )
                tie_lane_rank = Int32(
                    cute.arch.popc(tie_mask & lower_lane_mask)
                )
                if lane_idx == Int32(0):
                    flags[warp_idx] = Int32(cute.arch.popc(tie_mask))
                cute.arch.sync_threads()

                tie_base = state[2]
                ties_before_warp = Int32(0)
                for warp in cutlass.range_constexpr(8):
                    if Int32(warp) < warp_idx:
                        ties_before_warp = ties_before_warp + flags[Int32(warp)]
                take = is_greater
                if (
                    is_tie != Int32(0)
                    and tie_base + ties_before_warp + tie_lane_rank
                    < ties_needed
                ):
                    take = Int32(1)

                take_mask = Uint32(cute.arch.vote_ballot_sync(take != Int32(0)))
                take_lane_rank = Int32(
                    cute.arch.popc(take_mask & lower_lane_mask)
                )
                if lane_idx == Int32(0):
                    flags[Int32(8) + warp_idx] = Int32(
                        cute.arch.popc(take_mask)
                    )
                cute.arch.sync_threads()

                out_base = state[3]
                selected_before_warp = Int32(0)
                selected_in_chunk = Int32(0)
                ties_in_chunk = Int32(0)
                for warp in cutlass.range_constexpr(8):
                    if Int32(warp) < warp_idx:
                        selected_before_warp = (
                            selected_before_warp + flags[Int32(8 + warp)]
                        )
                    selected_in_chunk = (
                        selected_in_chunk + flags[Int32(8 + warp)]
                    )
                    ties_in_chunk = ties_in_chunk + flags[Int32(warp)]
                if take != Int32(0):
                    output[out_base + selected_before_warp + take_lane_rank] = (
                        col
                    )
                if tid == Int32(0):
                    state[3] = out_base + selected_in_chunk
                    state[2] = tie_base + ties_in_chunk
                cute.arch.sync_threads()
                chunk = chunk + Int32(1)

            for item in cutlass.range_constexpr(
                self.padded_topm // self.THREADS
            ):
                col = tid + Int32(item * self.THREADS)
                if col >= requested_topm:
                    output[col] = Int32(-1)


@cute.jit
def _sm90_gemm_zero_init(
    tiled_mma: cute.TiledMma,
    shape_mn: cute.Shape,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    stage: Int32,
) -> cute.Tensor:
    """Issue one SM90 WGMMA tile and return its FP32 register accumulator."""
    acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C(shape_mn), Float32)
    warpgroup.fence()
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, False)
    rA = tCrA[None, None, None, stage]
    for k in cutlass.range_constexpr(cute.size(rA.shape[2])):
        cute.gemm(mma_atom, acc, rA[None, None, k], tCrB[None, None, k], acc)
        mma_atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    warpgroup.wait_group(0)
    return acc


def _sm90_acc_mn_view(acc: cute.Tensor) -> cute.Tensor:
    """Convert the native SM90 WGMMA accumulator layout to a logical M/N view."""
    col_major = cute.make_layout(acc.layout.shape)
    shape = (
        (col_major.shape[0][1], col_major.shape[1]),
        (
            col_major.shape[0][0],
            *col_major.shape[0][2:],
            col_major.shape[2],
        ),
        *col_major.shape[3:],
    )
    stride = (
        (col_major.stride[0][1], col_major.stride[1]),
        (
            col_major.stride[0][0],
            *col_major.stride[0][2:],
            col_major.stride[2],
        ),
        *col_major.stride[3:],
    )
    return cute.make_tensor(
        acc.iterator,
        cute.composition(acc.layout, cute.make_layout(shape, stride=stride)),
    )


class _TwoStageScoreSm90:
    """Stage 1 grouped-head QK score kernel for Hopper.

    The routing score only needs the sum of all per-head dot products in each
    KV group.  Summing each group's query heads first removes the otherwise
    redundant head dimension:

        sum_h dot(q_h, k_g) == dot(sum_h q_h, k_g)

    One CTA handles one query row.  Up to 128 threads cooperatively build the
    two FP32 reduced queries in shared memory, then each warp scans independent
    key positions.  Lanes cover the 32/64-wide head dimension and use a warp
    reduction before lane 0 writes the final FP32 score.
    """

    def __init__(self, bucket_size: int, heads: int, head_dim: int):
        if heads not in _SM90_HEADS:
            raise ValueError(f"unsupported SM90 head count {heads}")
        if head_dim not in _SM90_DIMS:
            raise ValueError(f"unsupported SM90 head_dim {head_dim}")
        if heads % _GROUPS != 0:
            raise ValueError(f"heads ({heads}) must be divisible by {_GROUPS}")
        self.bucket_size = bucket_size
        self.heads = heads
        self.head_dim = head_dim
        self.heads_per_group = heads // _GROUPS

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mRowQtok: cute.Tensor,
        mRowBos: cute.Tensor,
        mRowVis: cute.Tensor,
        mScores: cute.Tensor,
        requested_topk: Int32,
        sm_scale: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mQ,
            mKV,
            mRowQtok,
            mRowBos,
            mRowVis,
            mScores,
            requested_topk,
            sm_scale,
        ).launch(
            grid=(cute.size(mScores.shape[0]), 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ,
        mKV,
        mRowQtok,
        mRowBos,
        mRowVis,
        mScores,
        requested_topk: Int32,
        sm_scale: Float32,
    ):
        @cute.struct
        class SharedStorage:
            q_group: cute.struct.Align[
                cute.struct.MemRange[Float32, _GROUPS * self.head_dim], 16
            ]

        storage = cutlass.utils.SmemAllocator().allocate(SharedStorage)
        q_group = storage.q_group.get_tensor(
            cute.make_layout(
                (_GROUPS, self.head_dim), stride=(self.head_dim, 1)
            )
        )

        tid = Int32(cute.arch.thread_idx()[0])
        lane = Int32(cute.arch.lane_idx())
        warp = tid // Int32(32)
        row = Int32(cute.arch.block_idx()[0])

        if row < Int32(cute.size(mRowVis.shape[0])):
            qtok = Int32(mRowQtok[row])
            bos = Int32(mRowBos[row])
            visible_len = Int32(mRowVis[row])
            do_qk = visible_len > requested_topk

            if do_qk:
                slot = tid
                if slot < Int32(_GROUPS * self.head_dim):
                    group = slot // Int32(self.head_dim)
                    dim = slot - group * Int32(self.head_dim)
                    q_sum = Float32(0.0)
                    for h in cutlass.range_constexpr(self.heads_per_group):
                        q_sum = q_sum + Float32(
                            mQ[
                                qtok,
                                group * Int32(self.heads_per_group) + Int32(h),
                                dim,
                            ]
                        )
                    q_group[group, dim] = q_sum
                cute.arch.sync_threads()

                pos = warp
                while pos < visible_len:
                    score = Float32(0.0)
                    for group in cutlass.range_constexpr(_GROUPS):
                        partial = Float32(0.0)
                        for d_iter in cutlass.range_constexpr(
                            self.head_dim // 32
                        ):
                            dim = lane + Int32(d_iter * 32)
                            partial = (
                                partial
                                + Float32(mKV[bos + pos, group, dim])
                                * q_group[group, dim]
                            )
                        for offset in cutlass.range_constexpr(5):
                            partial = partial + cute.arch.shuffle_sync_bfly(
                                partial, offset=1 << (4 - offset)
                            )
                        score = score + partial
                    if lane == Int32(0):
                        mScores[row, pos] = score * sm_scale
                    pos = pos + Int32(_THREADS // 32)


class _TwoStageScoreSm90Mma:
    """Experimental Hopper score kernel using two-stage TMA and WGMMA."""

    def __init__(
        self,
        bucket_size: int,
        heads: int,
        head_dim: int,
        num_stages: int = 2,
    ):
        if heads not in _SM90_HEADS:
            raise ValueError(f"unsupported SM90 head count {heads}")
        if head_dim not in _SM90_DIMS:
            raise ValueError(f"unsupported SM90 head_dim {head_dim}")
        if heads % _GROUPS != 0:
            raise ValueError(f"heads ({heads}) must be divisible by {_GROUPS}")
        heads_per_group = heads // _GROUPS
        if heads_per_group < 8:
            raise ValueError(
                "SM90 MMA score backend requires at least 8 heads per KV group"
            )
        if num_stages not in (2, 3):
            raise ValueError(
                f"SM90 MMA stages must be 2 or 3, got {num_stages}"
            )
        self.bucket_size = bucket_size
        self.heads = heads
        self.head_dim = head_dim
        self.heads_per_group = heads_per_group
        self.mma_heads = max(16, heads_per_group)
        self.num_stages = num_stages
        self.mma_tiler = (
            _SM90_MMA_TILE_K,
            self.mma_heads,
            self.head_dim,
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mRowQtok: cute.Tensor,
        mRowBos: cute.Tensor,
        mRowVis: cute.Tensor,
        mScores: cute.Tensor,
        requested_topk: Int32,
        sm_scale: Float32,
        stream: cuda.CUstream,
    ):
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(_SM90_MMA_TILE_K, self.mma_heads),
        )
        sQ_layout = self._make_smem_layout(
            (self.mma_heads, self.head_dim), (_GROUPS,)
        )
        sK_layout = self._make_smem_layout(
            (_SM90_MMA_TILE_K, self.head_dim),
            (_GROUPS, self.num_stages),
        )
        kv_layout = cute.make_tensor(
            mKV.iterator,
            cute.make_layout(
                (mKV.shape[0], self.head_dim, _GROUPS),
                stride=(mKV.stride[0], mKV.stride[2], mKV.stride[1]),
            ),
        )
        tma_kv, kv_layout = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            kv_layout,
            cute.select(sK_layout, mode=[0, 1]),
            (_SM90_MMA_TILE_K, self.head_dim),
        )
        self.kernel(
            mQ,
            kv_layout,
            mRowQtok,
            mRowBos,
            mRowVis,
            mScores,
            requested_topk,
            sm_scale,
            tma_kv,
            tiled_mma,
            sQ_layout,
            sK_layout,
        ).launch(
            grid=(cute.size(mScores.shape[0]), 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    def _make_smem_layout(
        self,
        shape: tuple[int, int],
        tail_shape: tuple[int, ...],
    ) -> cute.ComposedLayout:
        layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR,
                BFloat16,
                shape[1],
            ),
            BFloat16,
        )
        full_shape = shape + tail_shape
        return cute.tile_to_shape(
            layout_atom,
            full_shape,
            order=(1, 0, *range(2, len(full_shape))),
        )

    @cute.kernel
    def kernel(
        self,
        mQ,
        mKV,
        mRowQtok,
        mRowBos,
        mRowVis,
        mScores,
        requested_topk: Int32,
        sm_scale: Float32,
        tma_kv,
        tiled_mma,
        sQ_layout,
        sK_layout,
    ):
        @cute.struct
        class SharedStorage:
            k_mbar: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sQ: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[BFloat16, cute.cosize(sK_layout)], 1024
            ]

        storage = cutlass.utils.SmemAllocator().allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)

        tid = Int32(cute.arch.thread_idx()[0])
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        row = Int32(cute.arch.block_idx()[0])

        if warp == Int32(0):
            cpasync.prefetch_descriptor(tma_kv)

        k_pipeline = PipelineTmaAsync.create(
            barrier_storage=storage.k_mbar.data_ptr(),
            num_stages=self.num_stages,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 4),
            tx_count=_GROUPS * _SM90_MMA_TILE_K * self.head_dim * 2,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
        cute.arch.sync_threads()
        pipeline_init_wait(cluster_shape_mn=(1, 1))
        if row < Int32(cute.size(mRowVis.shape[0])):
            qtok = Int32(mRowQtok[row])
            bos = Int32(mRowBos[row])
            visible_len = Int32(mRowVis[row])
            if visible_len > requested_topk:
                if tid < Int32(128):
                    self._producer(
                        mQ,
                        mKV,
                        qtok,
                        bos,
                        visible_len,
                        sQ,
                        sK,
                        tma_kv,
                        k_pipeline,
                        tid,
                        warp,
                    )
                else:
                    self._consumer(
                        tiled_mma,
                        mScores,
                        row,
                        visible_len,
                        sm_scale,
                        sQ,
                        sK,
                        k_pipeline,
                        tid,
                        warp,
                    )

    @cute.jit
    def _producer(
        self,
        mQ,
        mKV,
        qtok: Int32,
        bos: Int32,
        visible_len: Int32,
        sQ,
        sK,
        tma_kv,
        k_pipeline,
        tid: Int32,
        warp: Int32,
    ):
        num_tiles = (visible_len + Int32(_SM90_MMA_TILE_K - 1)) // Int32(
            _SM90_MMA_TILE_K
        )
        for item in cutlass.range_constexpr(
            (_GROUPS * self.mma_heads * self.head_dim) // 128
        ):
            q_slot = tid + Int32(item * 128)
            q_group = q_slot // Int32(self.mma_heads * self.head_dim)
            q_group_slot = q_slot - q_group * Int32(
                self.mma_heads * self.head_dim
            )
            q_head = q_group_slot // Int32(self.head_dim)
            q_dim = q_group_slot - q_head * Int32(self.head_dim)
            q_value = BFloat16(0.0)
            if q_head < Int32(self.heads_per_group):
                q_value = mQ[
                    qtok,
                    q_group * Int32(self.heads_per_group) + q_head,
                    q_dim,
                ]
            sQ[q_head, q_dim, q_group] = q_value
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(barrier_id=1, number_of_threads=_THREADS)

        if warp == Int32(0):
            with cute.arch.elect_one():
                producer_state = make_pipeline_state(
                    PipelineUserType.Producer, self.num_stages
                )
                mKV_doc = cute.domain_offset((bos, Int32(0), Int32(0)), mKV)
                tile_id = Int32(0)
                while tile_id < num_tiles:
                    k_pipeline.producer_acquire(producer_state)
                    tma_barrier = k_pipeline.producer_get_barrier(
                        producer_state
                    )
                    for group in cutlass.range_constexpr(_GROUPS):
                        gK = cute.local_tile(
                            mKV_doc[None, None, group],
                            (_SM90_MMA_TILE_K, self.head_dim),
                            (tile_id, 0),
                        )
                        sK_stage = sK[None, None, group, producer_state.index]
                        s_k, g_k = cpasync.tma_partition(
                            tma_kv,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(sK_stage, 0, cute.rank(sK_stage)),
                            cute.group_modes(gK, 0, cute.rank(gK)),
                        )
                        cute.copy(
                            tma_kv,
                            g_k,
                            s_k,
                            tma_bar_ptr=tma_barrier,
                        )
                    k_pipeline.producer_commit(producer_state)
                    producer_state.advance()
                    tile_id = tile_id + Int32(1)
                k_pipeline.producer_tail(producer_state)

    @cute.jit
    def _consumer(
        self,
        tiled_mma,
        mScores,
        row: Int32,
        visible_len: Int32,
        sm_scale: Float32,
        sQ,
        sK,
        k_pipeline,
        tid: Int32,
        warp: Int32,
    ):
        num_tiles = (visible_len + Int32(_SM90_MMA_TILE_K - 1)) // Int32(
            _SM90_MMA_TILE_K
        )
        consumer_state = make_pipeline_state(
            PipelineUserType.Consumer, self.num_stages
        )
        wg_tid = tid - Int32(128)
        warp_in_wg = warp - Int32(4)
        thr_mma = tiled_mma.get_slice(wg_tid)
        tCrK_g0 = thr_mma.make_fragment_A(
            thr_mma.partition_A(sK[None, None, 0, None])
        )
        tCrK_g1 = thr_mma.make_fragment_A(
            thr_mma.partition_A(sK[None, None, 1, None])
        )
        tCrQ_g0 = thr_mma.make_fragment_B(
            thr_mma.partition_B(sQ[None, None, 0])
        )
        tCrQ_g1 = thr_mma.make_fragment_B(
            thr_mma.partition_B(sQ[None, None, 1])
        )

        cute.arch.barrier(barrier_id=1, number_of_threads=_THREADS)
        tile_id = Int32(0)
        while tile_id < num_tiles:
            wait_token = k_pipeline.consumer_try_wait(consumer_state)
            k_pipeline.consumer_wait(consumer_state, wait_token)
            acc_g0 = _sm90_gemm_zero_init(
                tiled_mma,
                self.mma_tiler[:2],
                tCrK_g0,
                tCrQ_g0,
                consumer_state.index,
            )
            acc_g1 = _sm90_gemm_zero_init(
                tiled_mma,
                self.mma_tiler[:2],
                tCrK_g1,
                tCrQ_g1,
                consumer_state.index,
            )
            acc_mn_g0 = _sm90_acc_mn_view(acc_g0)
            acc_mn_g1 = _sm90_acc_mn_view(acc_g1)
            for r in cutlass.range_constexpr(cute.size(acc_mn_g0, mode=[0])):
                score = Float32(0.0)
                for c in cutlass.range_constexpr(
                    cute.size(acc_mn_g0, mode=[1])
                ):
                    score = score + acc_mn_g0[r, c] + acc_mn_g1[r, c]
                for offset in cutlass.range_constexpr(2):
                    score = score + cute.arch.shuffle_sync_bfly(
                        score, offset=1 << offset
                    )
                pos = (
                    tile_id * Int32(_SM90_MMA_TILE_K)
                    + warp_in_wg * Int32(16)
                    + Int32(r * 8)
                    + (Int32(cute.arch.lane_idx()) >> Int32(2))
                )
                if (Int32(cute.arch.lane_idx()) & Int32(3)) == Int32(
                    0
                ) and pos < visible_len:
                    mScores[row, pos] = score * sm_scale
            k_pipeline.consumer_release(consumer_state)
            consumer_state.advance()
            tile_id = tile_id + Int32(1)


class _TwoStageTopK:
    """Stage 2: persistent one-block-per-SM radix top-k over the FP32 scores.

    Each block grid-strides over the row range, processing one row at a time.
    A grid width of one block per SM keeps single-block occupancy, matching the
    fused kernel's proven execution regime.

    When ``sort_output`` is set, the candidate indices are bitonic-sorted by
    their real FP32 score (descending) in the same kernel, right after the radix
    compaction, reusing the already-resident score row.  This fuses the optional
    ordering step into the top-k launch (no extra kernel / no metadata re-read).
    The network matches ``sparse_vha.py`` (``num_iters = log2(padded_topk)``,
    ascending flag ``i & (1 << (i1 + 1))``, partner ``i ^ (1 << (i1 - i2))``);
    early-return rows keep identity order.
    """

    def __init__(
        self,
        bucket_size: int,
        padded_topk: int,
        key_bits: int,
        n_blocks: int,
        sort_output: bool = False,
    ):
        if key_bits not in (16, 32):
            raise ValueError(f"key_bits must be 16 or 32, got {key_bits}")
        self.bucket_size = bucket_size
        self.padded_topk = padded_topk
        self.key_bits = key_bits
        self.n_blocks = int(n_blocks)
        self.sort_output = bool(sort_output)
        self.num_iters = int(round(math.log2(padded_topk)))
        if key_bits == 32:
            self.selector = ExactRadixSelector(bucket_size, padded_topk)
            self.key_dtype = Uint32
        else:
            self.selector = _TwoStageRadixSelector(
                bucket_size, padded_topk, passes=2
            )
            self.key_dtype = Uint16

    @cute.jit
    def __call__(
        self,
        mRowVis: cute.Tensor,
        mScores: cute.Tensor,
        mOut: cute.Tensor,
        requested_topk: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(mRowVis, mScores, mOut, requested_topk).launch(
            grid=(self.n_blocks, 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mRowVis,
        mScores,
        mOut,
        requested_topk: Int32,
    ):
        if cutlass.const_expr(self.sort_output):

            @cute.struct
            class SharedStorage:
                keys: cute.struct.Align[
                    cute.struct.MemRange[self.key_dtype, self.bucket_size], 16
                ]
                histogram: cute.struct.MemRange[Int32, 256]
                flags: cute.struct.MemRange[Int32, 256]
                state: cute.struct.MemRange[Int32, 4]
                cand_val: cute.struct.MemRange[Float32, self.padded_topk]
                cand_idx: cute.struct.MemRange[Int32, self.padded_topk]

        else:

            @cute.struct
            class SharedStorage:
                keys: cute.struct.Align[
                    cute.struct.MemRange[self.key_dtype, self.bucket_size], 16
                ]
                histogram: cute.struct.MemRange[Int32, 256]
                flags: cute.struct.MemRange[Int32, 256]
                state: cute.struct.MemRange[Int32, 4]

        storage = cutlass.utils.SmemAllocator().allocate(SharedStorage)
        keys = storage.keys.get_tensor(
            cute.make_layout((self.bucket_size,), stride=(1,))
        )
        histogram = storage.histogram.get_tensor(
            cute.make_layout((256,), stride=(1,))
        )
        flags = storage.flags.get_tensor(cute.make_layout((256,), stride=(1,)))
        state = storage.state.get_tensor(cute.make_layout((4,), stride=(1,)))
        if cutlass.const_expr(self.sort_output):
            cand_val = storage.cand_val.get_tensor(
                cute.make_layout((self.padded_topk,), stride=(1,))
            )
            cand_idx = storage.cand_idx.get_tensor(
                cute.make_layout((self.padded_topk,), stride=(1,))
            )

        tid = Int32(cute.arch.thread_idx()[0])
        rows = Int32(mRowVis.shape[0])
        row = Int32(cute.arch.block_idx()[0])
        while row < rows:
            visible_len = Int32(mRowVis[row])
            do_qk = visible_len > requested_topk

            # Build the sortable key cache for radix rows only.  Early rows
            # (visible_len <= requested_topk) never read mScores; their scores
            # are not produced by stage 1.  The selector emits identity for them.
            n_scan = (visible_len + Int32(_THREADS - 1)) // Int32(_THREADS)
            item = Int32(0)
            while item < n_scan:
                col = tid + item * Int32(_THREADS)
                if do_qk and col < visible_len:
                    score = Float32(mScores[row, col])
                    if cutlass.const_expr(self.key_bits == 32):
                        keys[col] = fp32_to_sortable_uint32(score)
                    else:
                        keys[col] = Uint16(
                            fp32_to_sortable_uint32(score) >> Uint32(16)
                        )
                item = item + Int32(1)
            cute.arch.sync_threads()

            out_row = mOut[row, None]
            self.selector.select(
                keys,
                out_row,
                visible_len,
                requested_topk,
                histogram,
                flags,
                state,
            )
            cute.arch.sync_threads()

            if cutlass.const_expr(self.sort_output):
                # Fused bitonic sort of the selected candidates by real FP32
                # score (descending), reusing the resident score row.  Only
                # radix rows are sorted; early rows keep their identity order.
                if do_qk:
                    max_kv_i = visible_len - Int32(1)
                    items = self.padded_topk // _THREADS
                    for e in cutlass.range_constexpr(items):
                        i = tid + Int32(e * _THREADS)
                        idx = Int32(mOut[row, i])
                        val = Float32(_NEG_SENTINEL)
                        if idx >= Int32(0):
                            val = Float32(mScores[row, idx])
                        cand_idx[i] = idx
                        cand_val[i] = val
                    cute.arch.sync_threads()

                    for i1 in cutlass.range_constexpr(self.num_iters):
                        for i2 in cutlass.range_constexpr(i1 + 1):
                            asc_bit = 1 << (i1 + 1)
                            xor_bit = 1 << (i1 - i2)
                            for e in cutlass.range_constexpr(items):
                                i = tid + Int32(e * _THREADS)
                                j = i ^ Int32(xor_bit)
                                if i < j:
                                    vi = cand_val[i]
                                    vj = cand_val[j]
                                    ii = cand_idx[i]
                                    jj = cand_idx[j]
                                    do_swap = Int32(0)
                                    if (i & Int32(asc_bit)) != Int32(0):
                                        if vi > vj or (vi == vj and ii < jj):
                                            do_swap = Int32(1)
                                    else:
                                        if vi < vj or (vi == vj and ii > jj):
                                            do_swap = Int32(1)
                                    if do_swap != Int32(0):
                                        cand_val[i] = vj
                                        cand_val[j] = vi
                                        cand_idx[i] = jj
                                        cand_idx[j] = ii
                            cute.arch.sync_threads()

                    for e in cutlass.range_constexpr(items):
                        i = tid + Int32(e * _THREADS)
                        idx = cand_idx[i]
                        out_val = Int32(-1)
                        if i < requested_topk:
                            if idx >= Int32(0):
                                if idx <= max_kv_i:
                                    out_val = idx
                        mOut[row, i] = out_val
                    cute.arch.sync_threads()

            row = row + Int32(self.n_blocks)


def _select_bucket(max_doc_seq_len: int) -> int:
    if max_doc_seq_len <= 0:
        raise ValueError(
            f"max_doc_seq_len must be positive, got {max_doc_seq_len}"
        )
    if max_doc_seq_len <= _BUCKET_8K:
        return _BUCKET_8K
    if max_doc_seq_len <= _BUCKET_32K:
        return _BUCKET_32K
    raise ValueError(
        f"max_doc_seq_len must be <= {_BUCKET_32K}, got {max_doc_seq_len}"
    )


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _validate_two_stage_inputs(
    q, kv, topk, padded_topk, bucket, key_bits
) -> None:
    import paddle

    if q.dtype != paddle.bfloat16 or kv.dtype != paddle.bfloat16:
        raise TypeError("q and kv must be bfloat16")
    if q.ndim != 4:
        raise ValueError("q must have shape [B, Sq, H, D]")
    heads = int(q.shape[2])
    head_dim = int(q.shape[3])
    if heads not in _SM90_HEADS or head_dim not in _SM90_DIMS:
        raise ValueError(
            "SM90 q must have shape [B, Sq, H, D] with "
            f"H in {_SM90_HEADS} and D in {_SM90_DIMS}; got H={heads}, D={head_dim}"
        )
    if (
        kv.ndim != 4
        or int(kv.shape[2]) != _GROUPS
        or int(kv.shape[3]) != head_dim
    ):
        raise ValueError(f"kv must have shape [B, Skv, {_GROUPS}, {head_dim}]")
    if int(q.shape[0]) != int(kv.shape[0]):
        raise ValueError("q and kv must share the same batch dimension")
    if "gpu" not in str(q.place).lower() or "gpu" not in str(kv.place).lower():
        raise ValueError("q and kv must be CUDA tensors")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if not _is_power_of_two(padded_topk):
        raise ValueError(
            f"padded_topk must be a power of two, got {padded_topk}"
        )
    if padded_topk % _THREADS != 0:
        raise ValueError(
            f"padded_topk must be a multiple of {_THREADS}, got {padded_topk}"
        )
    if padded_topk < topk:
        raise ValueError(
            f"padded_topk ({padded_topk}) must be >= topk ({topk})"
        )
    if padded_topk > bucket:
        raise ValueError(
            f"padded_topk ({padded_topk}) must be <= bucket ({bucket})"
        )
    if key_bits not in (16, 32):
        raise ValueError(f"key_bits must be 16 or 32, got {key_bits}")


def _build_row_metadata(B, Sq, Skv, doc_start, doc_end):
    """Return int32 row_qtok / row_bos / row_vis of length B*Sq."""
    import paddle

    b_idx = paddle.arange(B, dtype="int32").reshape([B, 1]).tile([1, Sq])
    q_idx = paddle.arange(Sq, dtype="int32").reshape([1, Sq]).tile([B, 1])
    if doc_start is None:
        ds = paddle.zeros([B, Sq], dtype="int32")
        de = paddle.full([B, Sq], Sq, dtype="int32")
    else:
        ds = doc_start.astype("int32").reshape([B, Sq])
        if doc_end is None:
            raise ValueError("doc_end must be provided when doc_start is given")
        de = doc_end.astype("int32").reshape([B, Sq])

    qtok = (b_idx * int(Sq) + q_idx).astype("int32")
    bos = (b_idx * int(Skv) + ds).astype("int32")
    vis = paddle.minimum(q_idx - ds + 1, de - ds).astype("int32")
    vis = paddle.clip(vis, min=0)

    row_qtok = qtok.reshape([B * Sq]).contiguous()
    row_bos = bos.reshape([B * Sq]).contiguous()
    row_vis = vis.reshape([B * Sq]).contiguous()
    return row_qtok, row_bos, row_vis


def dsa_sparse_vha_topk_two_stage_cutedsl(
    q,
    kv,
    topk,
    padded_topk,
    doc_start,
    doc_end=None,
    sm_scale=None,
    query_tile=None,
    max_doc_seq_len=None,
    key_bits=16,
    sort_output=False,
    chunk_rows=None,
    sm90_score_backend="scalar",
    sm90_mma_stages=2,
):
    """Two-stage Sparse VHA top-k routing on SM90.

    Args:
        q: BSHD ``[B, Sq, H, D]`` bfloat16 query tensor with
            ``H in {8, 16, 32, 64}`` and ``D in {32, 64}``.
        kv: ``[B, Skv, 2, D]`` bfloat16 grouped key tensor.
        topk: number of routed candidates to select per row.
        padded_topk: output width; power of two, multiple of 256, in
            ``[topk, bucket]``.
        doc_start: int32 ``[B, Sq]`` doc-start key index per row, or ``None`` for
            a single full document (doc_start=0, doc_end=Sq).
        doc_end: int32 ``[B, Sq]`` doc-end (exclusive) key index per row.
        sm_scale: score multiplier; defaults to ``D ** -0.5``.
        query_tile: reserved / unused.
        max_doc_seq_len: maximum per-document length; selects the 8K/32K bucket.
        key_bits: 32 for exact FP32 keys, 16 for the quantised half-key path.
        sort_output: when True, launch a bitonic-sort kernel that reorders
            each row's candidates into descending real-FP32-score order,
            matching the TileLang two-stage top-k stage.  Early-return rows keep
            identity order.  When False, output is in radix-selection order (the
            candidate set is identical either way).
        chunk_rows: maximum rows in each score/top-k chunk.  Defaults to the
            largest value allowed by the score workspace budget.  Values above
            that capacity raise ``ValueError``.
        sm90_score_backend: ``"scalar"`` for the FP32 CUDA-core score path or
            ``"mma"`` for the experimental TMA/WGMMA score path.
        sm90_mma_stages: number of TMA KV stages for the experimental MMA path;
            must be 2 or 3.

    Returns:
        int32 ``[B, Sq, padded_topk]`` doc-local candidate indices, ``-1`` padded.
    """
    import paddle

    del query_tile  # reserved for future tiling schedules
    sort_output = bool(sort_output)

    topk = int(topk)
    padded_topk = int(padded_topk)
    key_bits = int(key_bits)
    sm90_score_backend = str(sm90_score_backend).lower()
    sm90_mma_stages = int(sm90_mma_stages)

    B = int(q.shape[0])
    Sq = int(q.shape[1])
    Skv = int(kv.shape[1])
    H = int(q.shape[2])
    D = int(q.shape[3])
    cc_major, cc_minor = _device_cc()
    if (cc_major, cc_minor) != (9, 0):
        raise RuntimeError(
            f"two-stage CuTeDSL indexer requires SM90, got SM{cc_major}{cc_minor}"
        )
    if sm90_score_backend not in ("scalar", "mma"):
        raise ValueError(
            "sm90_score_backend must be 'scalar' or 'mma', got "
            f"{sm90_score_backend!r}"
        )
    if sm90_score_backend == "mma":
        if H // _GROUPS < 8:
            raise ValueError(
                "sm90_score_backend='mma' requires at least 8 query heads "
                f"per KV group, got H={H}"
            )
        if sm90_mma_stages not in (2, 3):
            raise ValueError(
                f"sm90_mma_stages must be 2 or 3, got {sm90_mma_stages}"
            )
    if doc_start is None and Sq > Skv:
        raise ValueError(
            "doc_start=None requires Sq <= Skv so causal row metadata cannot "
            f"address past the KV sequence; got Sq={Sq}, Skv={Skv}"
        )

    if sm_scale is None:
        sm_scale = float(D) ** -0.5
    sm_scale = float(sm_scale)

    q_thd = q.reshape([B * Sq, H, D]).contiguous()
    kv_thd = kv.reshape([B * Skv, _GROUPS, D]).contiguous()

    if doc_start is not None:
        if list(doc_start.shape) != [B, Sq] or doc_end is None:
            raise ValueError(
                f"doc_start and doc_end must both have shape [{B}, {Sq}]"
            )
        if list(doc_end.shape) != [B, Sq]:
            raise ValueError(
                f"doc_start and doc_end must both have shape [{B}, {Sq}]"
            )
        if str(doc_start.place) != str(q.place) or str(doc_end.place) != str(
            q.place
        ):
            raise ValueError(
                "document metadata and q must be on the same device"
            )
        ds = doc_start.astype("int32")
        de = doc_end.astype("int32")
        q_idx = paddle.arange(Sq, dtype="int32").reshape([1, Sq])
        invalid = (
            (ds < 0) | (ds > q_idx) | (de <= q_idx) | (de < ds) | (de > Skv)
        )
        if bool(paddle.any(invalid).item()):
            raise ValueError(
                "document metadata must satisfy "
                "0 <= doc_start <= query < doc_end <= Skv"
            )

    row_qtok, row_bos, row_vis = _build_row_metadata(
        B, Sq, Skv, doc_start, doc_end
    )
    actual_max_doc_seq_len = int(paddle.max(row_vis).item())
    bucket_hint = (
        int(max_doc_seq_len)
        if max_doc_seq_len is not None
        else actual_max_doc_seq_len
    )
    if bucket_hint < actual_max_doc_seq_len:
        raise ValueError(
            f"max_doc_seq_len={bucket_hint} is smaller than the actual maximum "
            f"visible length {actual_max_doc_seq_len}"
        )
    bucket = _select_bucket(bucket_hint)
    _validate_two_stage_inputs(q, kv, topk, padded_topk, bucket, key_bits)

    rows = B * Sq

    bytes_per_row = bucket * 4
    workspace_rows = max(1, _SCORE_BYTES_BUDGET // bytes_per_row)
    if chunk_rows is None:
        max_chunk_rows = workspace_rows
    else:
        max_chunk_rows = int(chunk_rows)
        if max_chunk_rows <= 0:
            raise ValueError(
                f"chunk_rows must be positive, got {max_chunk_rows}"
            )
        if max_chunk_rows > workspace_rows:
            raise ValueError(
                f"chunk_rows ({max_chunk_rows}) exceeds score workspace capacity "
                f"({workspace_rows} rows for bucket={bucket})"
            )
    max_chunk_rows = min(rows, max_chunk_rows)
    n_chunks = (rows + max_chunk_rows - 1) // max_chunk_rows
    chunk_rows = (rows + n_chunks - 1) // n_chunks
    total_padded = n_chunks * chunk_rows

    if total_padded > rows:
        pad = total_padded - rows
        zeros = paddle.zeros([pad], dtype="int32")
        row_qtok = paddle.concat([row_qtok, zeros]).contiguous()
        row_bos = paddle.concat([row_bos, zeros]).contiguous()
        # vis == 0 -> selector emits all -1 for the padding rows.
        row_vis = paddle.concat([row_vis, zeros]).contiguous()

    output = paddle.empty([total_padded, padded_topk], dtype="int32")
    scores = paddle.empty([chunk_rows, bucket], dtype="float32")

    # Keep one persistent top-k block per SM; each block grid-strides over rows.
    n_blocks = min(_num_sms(), chunk_rows)

    score_key = (
        cc_major,
        cc_minor,
        H,
        D,
        chunk_rows,
        bucket,
        sm90_score_backend,
        sm90_mma_stages,
    )
    if score_key not in _SCORE_CACHE:
        if sm90_score_backend == "mma":
            score_kernel = _TwoStageScoreSm90Mma(bucket, H, D, sm90_mma_stages)
        else:
            score_kernel = _TwoStageScoreSm90(bucket, H, D)
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _SCORE_CACHE[score_key] = cute.compile(
            score_kernel,
            paddle_to_cute_tensor(q_thd, assumed_align=16, leading_dim=2),
            paddle_to_cute_tensor(kv_thd, assumed_align=16, leading_dim=2),
            paddle_to_cute_tensor(
                row_qtok[0:chunk_rows], assumed_align=4, leading_dim=0
            ),
            paddle_to_cute_tensor(
                row_bos[0:chunk_rows], assumed_align=4, leading_dim=0
            ),
            paddle_to_cute_tensor(
                row_vis[0:chunk_rows], assumed_align=4, leading_dim=0
            ),
            paddle_to_cute_tensor(scores, assumed_align=4, leading_dim=1),
            Int32(topk),
            Float32(sm_scale),
            fake_stream,
            options="--enable-tvm-ffi",
        )

    topk_config = (
        chunk_rows,
        bucket,
        padded_topk,
        key_bits,
        n_blocks,
        sort_output,
    )
    topk_key = (cc_major, cc_minor, *topk_config)
    if topk_key not in _TOPK_CACHE:
        topk_kernel = _TwoStageTopK(
            bucket, padded_topk, key_bits, n_blocks, sort_output
        )
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _TOPK_CACHE[topk_key] = cute.compile(
            topk_kernel,
            paddle_to_cute_tensor(
                row_vis[0:chunk_rows], assumed_align=4, leading_dim=0
            ),
            paddle_to_cute_tensor(scores, assumed_align=4, leading_dim=1),
            paddle_to_cute_tensor(
                output[0:chunk_rows], assumed_align=4, leading_dim=1
            ),
            Int32(topk),
            fake_stream,
            options="--enable-tvm-ffi",
        )

    score_compiled = _SCORE_CACHE[score_key]
    topk_compiled = _TOPK_CACHE[topk_key]

    for c in range(n_chunks):
        start = c * chunk_rows
        end = start + chunk_rows
        qtok_chunk = row_qtok[start:end].contiguous()
        bos_chunk = row_bos[start:end].contiguous()
        vis_chunk = row_vis[start:end].contiguous()
        out_chunk = output[start:end]
        score_compiled(
            q_thd,
            kv_thd,
            qtok_chunk,
            bos_chunk,
            vis_chunk,
            scores,
            Int32(topk),
            Float32(sm_scale),
        )
        topk_compiled(vis_chunk, scores, out_chunk, Int32(topk))

    return output[0:rows].reshape([B, Sq, padded_topk])


def dsa_sparse_vha_score_cutedsl(
    q,
    kv,
    row_visible,
    row_start=None,
    sm_scale=1.0,
    score_width=None,
    chunk_rows=None,
):
    """Run only the original SM90 two-stage indexer's QK score stage.

    This compact score-only entry is used by CSA's small coarse stage. Unlike
    the token indexer driver, its score bucket is the next power of two above
    ``S_k`` instead of the fixed 8K/32K token bucket.

    Args:
        q: ``[B, S_q, H, D]`` bf16, H in {8,16,32,64}, D in {32,64}.
        kv: ``[B, S_k, 2, D]`` bf16 grouped key.
        row_visible: ``[B, S_q]`` int, visible compressed-key length for each
            query row. Positions beyond the prefix are returned as ``-inf``.
        row_start: optional ``[B, S_q]`` int compressed-KV start for each row.
            The score kernel reads key positions from ``row_start`` and writes
            them into the returned row-local score buffer. ``None`` means zero.
        score_width: output score width; defaults to ``S_k``.

    Returns:
        FP32 ``[B, S_q, score_width]`` scores.
    """
    import paddle

    if q.ndim != 4 or kv.ndim != 4:
        raise ValueError("q and kv must be [B,S,H,D] and [B,S,2,D].")
    B, Sq, H, D = [int(x) for x in q.shape]
    Bk, Sk, groups, Dk = [int(x) for x in kv.shape]
    if B != Bk or groups != _GROUPS or D != Dk:
        raise ValueError("q/kv batch, group, or head-dimension mismatch.")
    if q.dtype != paddle.bfloat16 or kv.dtype != paddle.bfloat16:
        raise TypeError("q and kv must be bfloat16.")
    if H not in _SM90_HEADS or D not in _SM90_DIMS:
        raise ValueError(
            f"SM90 score stage requires H in {_SM90_HEADS}, D in "
            f"{_SM90_DIMS}; got H={H}, D={D}."
        )
    if list(row_visible.shape) != [B, Sq]:
        raise ValueError(f"row_visible must have shape [{B}, {Sq}].")
    cc_major, cc_minor = _device_cc()
    if (cc_major, cc_minor) != (9, 0):
        raise RuntimeError(
            f"CuTeDSL score stage requires SM90, got SM{cc_major}{cc_minor}."
        )

    width = Sk if score_width is None else int(score_width)
    if width <= 0 or width > Sk:
        raise ValueError(f"score_width must be in [1, {Sk}], got {width}.")
    bucket = max(32, 1 << (width - 1).bit_length())
    rows = B * Sq
    max_workspace_rows = max(1, _SCORE_BYTES_BUDGET // (bucket * 4))
    rows_per_chunk = (
        min(rows, max_workspace_rows) if chunk_rows is None else int(chunk_rows)
    )
    if rows_per_chunk <= 0 or rows_per_chunk > max_workspace_rows:
        raise ValueError(
            f"chunk_rows must be in [1, {max_workspace_rows}], "
            f"got {rows_per_chunk}."
        )

    q_thd = q.reshape([rows, H, D]).contiguous()
    kv_thd = kv.reshape([B * Sk, _GROUPS, D]).contiguous()
    row_qtok = paddle.arange(rows, dtype="int32").contiguous()
    batch_ids = paddle.arange(B, dtype="int32").reshape([B, 1])
    row_bos = (batch_ids.expand([B, Sq]) * int(Sk)).reshape([rows]).contiguous()
    if row_start is not None:
        if list(row_start.shape) != [B, Sq]:
            raise ValueError(f"row_start must have shape [{B}, {Sq}].")
        row_bos = (
            row_bos + row_start.cast("int32").reshape([rows])
        ).contiguous()
    row_vis = row_visible.cast("int32").reshape([rows]).contiguous()
    output_parts = []
    scores = paddle.empty([rows_per_chunk, bucket], dtype="float32")

    score_key = (
        cc_major,
        cc_minor,
        H,
        D,
        rows_per_chunk,
        bucket,
        "scalar_score_only",
    )
    if score_key not in _SCORE_CACHE:
        score_kernel = _TwoStageScoreSm90(bucket, H, D)
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _SCORE_CACHE[score_key] = cute.compile(
            score_kernel,
            paddle_to_cute_tensor(q_thd, assumed_align=16, leading_dim=2),
            paddle_to_cute_tensor(kv_thd, assumed_align=16, leading_dim=2),
            paddle_to_cute_tensor(
                row_qtok[:rows_per_chunk], assumed_align=4, leading_dim=0
            ),
            paddle_to_cute_tensor(
                row_bos[:rows_per_chunk], assumed_align=4, leading_dim=0
            ),
            paddle_to_cute_tensor(
                row_vis[:rows_per_chunk], assumed_align=4, leading_dim=0
            ),
            paddle_to_cute_tensor(scores, assumed_align=4, leading_dim=1),
            Int32(0),
            Float32(float(sm_scale)),
            fake_stream,
            options="--enable-tvm-ffi",
        )

    score_compiled = _SCORE_CACHE[score_key]
    columns = paddle.arange(width, dtype="int32").reshape([1, width])
    for start in range(0, rows, rows_per_chunk):
        end = min(start + rows_per_chunk, rows)
        count = end - start
        qtok_chunk = row_qtok[start:end]
        bos_chunk = row_bos[start:end]
        vis_chunk = row_vis[start:end]
        if count < rows_per_chunk:
            pad = rows_per_chunk - count
            zeros = paddle.zeros([pad], dtype="int32")
            qtok_chunk = paddle.concat([qtok_chunk, zeros]).contiguous()
            bos_chunk = paddle.concat([bos_chunk, zeros]).contiguous()
            vis_chunk = paddle.concat([vis_chunk, zeros]).contiguous()
        else:
            qtok_chunk = qtok_chunk.contiguous()
            bos_chunk = bos_chunk.contiguous()
            vis_chunk = vis_chunk.contiguous()
        score_compiled(
            q_thd,
            kv_thd,
            qtok_chunk,
            bos_chunk,
            vis_chunk,
            scores,
            Int32(0),
            Float32(float(sm_scale)),
        )
        output_parts.append(
            paddle.where(
                columns < vis_chunk[:count].reshape([count, 1]),
                scores[:count, :width],
                paddle.full([count, width], -float("inf"), dtype="float32"),
            ).contiguous()
        )
    return paddle.concat(output_parts, axis=0).reshape([B, Sq, width])


__all__ = [
    "dsa_sparse_vha_score_cutedsl",
    "dsa_sparse_vha_topk_two_stage_cutedsl",
    "is_cutedsl_available",
]
