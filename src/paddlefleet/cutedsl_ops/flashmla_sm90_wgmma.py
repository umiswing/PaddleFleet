# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone SM90 FlashMLA sparse-prefill WGMMA prototype.

This is intentionally separate from flashmla_sm90.py.  It follows the real
FlashMLA phase1 CTA shape: two 128-thread WGMMA consumers and one indexed KV
producer.  The consumers currently keep independent online-softmax state;
the cross-consumer max/sum exchange is the next optimization step.  The
ordinary-load Q path is the validated default; the experimental Q-TMA path is
kept behind ``use_tma_q`` while its barrier protocol is being aligned with
phase1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import cutlass
import cutlass.utils.hopper_helpers as sm90_utils
import paddle
from cutlass import BFloat16, Float16, Float32, Int32, cute
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum

from .dlpack import paddle_to_cute_tensor
from .dsa_sm90.copy import tma_get_copy_fn

if TYPE_CHECKING:
    import cuda.bindings.driver as cuda

_DQK: Final = 576
_DV: Final = 512
_BH: Final = 64
_BTOPK: Final = 64
_STAGES: Final = 2
_THREADS: Final = 384
_COMPILED: dict[tuple[object, ...], object] = {}


def _cutlass_dtype(dtype):
    if dtype in (paddle.bfloat16, "bfloat16"):
        return BFloat16
    if dtype in (paddle.float16, "float16"):
        return Float16
    raise TypeError(f"expected float16/bfloat16, got {dtype}")


def _transpose_view(tensor):
    shape = (tensor.shape[1], tensor.shape[0], *tensor.shape[2:])
    order = (1, 0, *range(2, cute.rank(tensor)))
    return cute.composition(
        tensor, cute.make_ordered_layout(shape, order=order)
    )


def _reshape_acc_to_frg_a(acc):
    layout = acc.layout
    if cute.rank(layout.shape[0]) == 3:
        div = 2 if layout.shape[0][2] % 2 == 0 else 1
        divided = cute.logical_divide(layout, ((None, None, div), None, None))
        converted = cute.make_layout(
            (
                (
                    divided.shape[0][0],
                    divided.shape[0][1],
                    divided.shape[0][2][0],
                ),
                divided.shape[1],
                (divided.shape[0][2][1], divided.shape[2]),
            ),
            stride=(
                (
                    divided.stride[0][0],
                    divided.stride[0][1],
                    divided.stride[0][2][0],
                ),
                divided.stride[1],
                (divided.stride[0][2][1], divided.stride[2]),
            ),
        )
    else:
        divided = cute.logical_divide(layout, (None, None, 2))
        converted = cute.make_layout(
            (
                (divided.shape[0], divided.shape[2][0]),
                divided.shape[1],
                divided.shape[2][1],
            ),
            stride=(
                (divided.stride[0], divided.stride[2][0]),
                divided.stride[1],
                divided.stride[2][1],
            ),
        )
    return cute.make_tensor(acc.iterator, converted)


class FlashMLASm90Wgmma:
    def __init__(self, dtype, topk, use_tma_q=False):
        if topk <= 0 or topk % _BTOPK:
            raise ValueError("topk must be a positive multiple of 64")
        self.dtype = dtype
        self.topk = topk
        self.num_tiles = topk // _BTOPK
        self.use_tma_q = use_tma_q

    @cute.jit
    def __call__(
        self,
        mQ,
        mK,
        mV,
        mO,
        mLSE,
        mIndices,
        mLengths,
        mSink,
        softmax_scale: Float32,
        stream: cuda.CUstream,
    ):
        tiled_qk = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(64, 64),
        )
        tiled_pv = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(64, 256),
            a_source=warpgroup.OperandSource.RMEM,
        )
        tiled_pv_remote = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(64, 256),
        )
        # [1,T,H,D] -> [H,D,T], [1,T,1,D] -> [T,D,1].
        q = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (_BH, _DQK, mQ.shape[1], 1),
                stride=(
                    mQ.stride[2],
                    mQ.stride[3],
                    mQ.stride[1],
                    mQ.stride[2] * _BH,
                ),
            ),
        )
        k_row_stride = (
            mK.stride[1]
            if isinstance(mK.stride[1], int)
            else cute.assume(mK.stride[1], divby=8)
        )
        v_row_stride = (
            mV.stride[1]
            if isinstance(mV.stride[1], int)
            else cute.assume(mV.stride[1], divby=8)
        )
        k = cute.make_tensor(
            mK.iterator,
            cute.make_layout(
                (mK.shape[1], _DQK, 1),
                stride=(k_row_stride, mK.stride[3], mK.stride[2]),
            ),
        )
        v = cute.make_tensor(
            mV.iterator,
            cute.make_layout(
                (mV.shape[1], _DV, 1),
                stride=(v_row_stride, mV.stride[3], mV.stride[2]),
            ),
        )
        o = cute.make_tensor(
            mO.iterator,
            cute.make_layout(
                (_BH, _DV, mO.shape[1]),
                stride=(mO.stride[2], mO.stride[3], mO.stride[1]),
            ),
        )
        lse = cute.make_tensor(
            mLSE.iterator,
            cute.make_layout(
                (_BH, mLSE.shape[1]), stride=(mLSE.stride[2], mLSE.stride[1])
            ),
        )

        q_atom = warpgroup.make_smem_layout_atom(
            warpgroup.SmemLayoutAtomKind.K_SW128, self.dtype
        )
        q_staged_layout = cute.tile_to_shape(
            q_atom, (_BH, _DQK, 1), order=(0, 1, 2)
        )
        s_layout = cute.tile_to_shape(q_atom, (_BH, _BTOPK), order=(0, 1))
        q_layout = cute.slice_(q_staged_layout, (None, None, 0))
        k_stage_layout = cute.tile_to_shape(
            q_atom, (_BTOPK, _DQK), order=(0, 1)
        )
        v_atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, _DV
            ),
            self.dtype,
        )
        v_layout = cute.tile_to_shape(
            v_atom, (_BTOPK, _DV, _STAGES), order=(0, 1, 2)
        )
        vt_stage_layout = cute.composition(
            k_stage_layout,
            cute.make_layout((_DQK, _BTOPK), stride=(_BTOPK, 1)),
        )
        tma_q, tq = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            q,
            q_layout,
            (_BH, _DQK),
            num_multicast=1,
        )
        async_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=128,
        )
        async_thr_copy = cute.make_tiled_copy_tv(
            async_copy_atom,
            cute.make_layout((1,)),
            cute.make_layout((8,)),
        ).get_slice(0)

        @cute.struct
        class SharedStorage:
            q_barriers: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, 2], 16
            ]
            kv_barriers: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, 16], 16
            ]
            softmax_m: cute.struct.Align[cute.struct.MemRange[Float32, 64], 16]
            softmax_l: cute.struct.Align[cute.struct.MemRange[Float32, 128], 16]
            softmax_s1: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(s_layout)], 128
            ]
            q: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(q_staged_layout)],
                128,
            ]
            k0: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(k_stage_layout)],
                128,
            ]
            k1: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(k_stage_layout)],
                128,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_q,
            tq,
            q,
            k,
            v,
            o,
            lse,
            mIndices,
            mLengths,
            mSink,
            q_staged_layout,
            q_layout,
            k_stage_layout,
            v_layout,
            vt_stage_layout,
            s_layout,
            tiled_qk,
            tiled_pv,
            tiled_pv_remote,
            async_copy_atom,
            async_thr_copy,
            softmax_scale,
        ).launch(
            grid=(q.shape[2], 1, 1),
            block=(_THREADS, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    @cute.jit
    def _copy_row_async(
        self,
        src: cute.Tensor,
        dst_row: cute.Tensor,
        selected: Int32,
        idx_in_group: Int32,
        num_tiles: int,
        copy_atom: cute.CopyAtom,
        thr_copy: cute.TiledCopy,
    ):
        """Copy one indexed 576/512-wide row in 128-bit async chunks."""
        g_row = src[selected, None, 0]
        g_chunks = cute.flat_divide(g_row, (8,))
        s_chunks = cute.flat_divide(dst_row, (8,))
        for tile_idx in cutlass.range_constexpr(num_tiles):
            chunk_idx = tile_idx * 8 + idx_in_group
            t_src = thr_copy.partition_S(g_chunks[None, chunk_idx])
            t_dst = thr_copy.partition_D(s_chunks[None, chunk_idx])
            cute.copy(copy_atom, t_src, t_dst)

    @cute.jit
    def _zero_row_async(
        self,
        dst_row: cute.Tensor,
        idx_in_group: Int32,
        num_tiles: int,
    ):
        """Zero one invalid row using the same eight-thread grouping."""
        s_chunks = cute.flat_divide(dst_row, (8,))
        for tile_idx in cutlass.range_constexpr(num_tiles):
            chunk_idx = tile_idx * 8 + idx_in_group
            s_chunks[None, chunk_idx].fill(0)

    @cute.kernel
    def kernel(
        self,
        tma_q,
        tq,
        q,
        k,
        v,
        o,
        lse,
        indices,
        lengths,
        sink,
        q_staged_layout,
        q_layout,
        k_stage_layout,
        v_layout,
        vt_stage_layout,
        s_layout,
        tiled_qk,
        tiled_pv,
        tiled_pv_remote,
        async_copy_atom,
        async_thr_copy,
        softmax_scale,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        wg = cute.arch.make_warp_uniform(tidx // 128)
        local = tidx - wg * 128
        token_idx, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        kv_barrier = storage.kv_barriers.data_ptr()
        softmax_m = storage.softmax_m.get_tensor(
            cute.make_layout((64,), stride=(1,))
        )
        softmax_l = storage.softmax_l.get_tensor(
            cute.make_layout((128,), stride=(1,))
        )

        if wg == 0 and local == 0:
            for barrier_idx in cutlass.range_constexpr(4):
                cute.arch.mbarrier_init(kv_barrier + barrier_idx, 1)
            for barrier_idx in cutlass.range_constexpr(8, 12):
                cute.arch.mbarrier_init(kv_barrier + barrier_idx, 1)
            for barrier_idx in cutlass.range_constexpr(4, 8):
                cute.arch.mbarrier_init(kv_barrier + barrier_idx, 128)
            for barrier_idx in cutlass.range_constexpr(12, 16):
                cute.arch.mbarrier_init(kv_barrier + barrier_idx, 128)
        cute.arch.sync_threads()

        sq = storage.q.get_tensor(
            q_staged_layout.outer, swizzle=q_staged_layout.inner
        )
        sq_tma = storage.q.get_tensor(q_layout.outer, swizzle=q_layout.inner)
        so = storage.q.get_tensor(cute.make_layout((_BH, _DV), stride=(_DV, 1)))
        sk0 = storage.k0.get_tensor(
            k_stage_layout.outer, swizzle=k_stage_layout.inner
        )
        sk1 = storage.k1.get_tensor(
            k_stage_layout.outer, swizzle=k_stage_layout.inner
        )
        sS0 = cute.make_tensor(
            sk0.iterator + _BTOPK * _DQK - _BH * _BTOPK,
            s_layout.outer,
        )
        sS1 = storage.softmax_s1.get_tensor(
            s_layout.outer, swizzle=s_layout.inner
        )
        svt0 = storage.k0.get_tensor(
            vt_stage_layout.outer, swizzle=vt_stage_layout.inner
        )
        svt1 = storage.k1.get_tensor(
            vt_stage_layout.outer, swizzle=vt_stage_layout.inner
        )

        if cutlass.const_expr(self.use_tma_q):
            cpasync.prefetch_descriptor(tma_q)
            q_barrier = storage.q_barriers.data_ptr()
            if wg == 0 and local == 0:
                cute.arch.mbarrier_init(q_barrier, 1)
            cute.arch.sync_threads()
            gq = tq[None, None, token_idx, 0]
            load_q, _, _ = tma_get_copy_fn(
                tma_q,
                0,
                cute.make_layout(1),
                gq,
                sq_tma,
                single_stage=True,
            )
            if wg == 0 and local < 32:
                if local == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        q_barrier,
                        cute.size_in_bytes(self.dtype, q_staged_layout),
                    )
                load_q(tma_bar_ptr=q_barrier)
            if wg < 2:
                cute.arch.mbarrier_wait(q_barrier, 0)
        else:
            if wg == 2:
                for linear in cutlass.range(local, _BH * _DQK, 128):
                    row, col = linear // _DQK, linear % _DQK
                    sq[row, col, 0] = q[row, col, token_idx, 0]
            cute.arch.barrier(barrier_id=4, number_of_threads=_THREADS)

        if wg == 2:
            cute.arch.setmaxregister_decrease(72)
            group_idx = local // 8
            idx_in_group = local % 8
            row_length = lengths[token_idx, 0]
            for tile in cutlass.range(self.num_tiles, unroll=1):
                stage = tile % _STAGES
                phase = tile // _STAGES
                sk_stage = sk0 if stage == 0 else sk1
                svt_stage = svt0 if stage == 0 else svt1
                if tile >= _STAGES:
                    for consumer_wg in cutlass.range_constexpr(2):
                        cute.arch.mbarrier_wait(
                            kv_barrier + 12 + stage * 2 + consumer_wg,
                            phase - 1,
                        )
                for local_row in cutlass.range_constexpr(4):
                    row = local_row * 16 + group_idx
                    slot = tile * _BTOPK + row
                    selected = indices[token_idx, 0, slot]
                    if (
                        slot < row_length
                        and selected >= 0
                        and selected < k.shape[0]
                    ):
                        self._copy_row_async(
                            k,
                            sk_stage[row, None],
                            selected,
                            idx_in_group,
                            _DQK // 64,
                            async_copy_atom,
                            async_thr_copy,
                        )
                    else:
                        self._zero_row_async(
                            sk_stage[row, None], idx_in_group, _DQK // 64
                        )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=7, number_of_threads=128)
                if local == 0:
                    cute.arch.mbarrier_arrive(kv_barrier + stage * 2 + stage)

                cute.arch.mbarrier_wait(
                    kv_barrier + 4 + stage * 2 + stage, phase
                )
                for local_row in cutlass.range_constexpr(4):
                    row = local_row * 16 + group_idx
                    slot = tile * _BTOPK + row
                    selected = indices[token_idx, 0, slot]
                    if (
                        slot < row_length
                        and selected >= 0
                        and selected < v.shape[0]
                    ):
                        self._copy_row_async(
                            v,
                            svt_stage[None, row],
                            selected,
                            idx_in_group,
                            _DV // 64,
                            async_copy_atom,
                            async_thr_copy,
                        )
                    else:
                        self._zero_row_async(
                            svt_stage[None, row], idx_in_group, _DV // 64
                        )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=9, number_of_threads=128)
                if local == 0:
                    for consumer_wg in cutlass.range_constexpr(2):
                        cute.arch.mbarrier_arrive(
                            kv_barrier + 8 + stage * 2 + consumer_wg
                        )

        if wg < 2:
            cute.arch.setmaxregister_increase(216)
            wg_layout = cute.make_layout(2, stride=128)
            mma_qk = tiled_qk.get_slice(wg_layout(wg))
            mma_pv = tiled_pv.get_slice(wg_layout(wg))
            mma_pv_remote = tiled_pv_remote.get_slice(wg_layout(wg))
            tQ = mma_qk.make_fragment_A(mma_qk.partition_A(sq))
            tK0 = mma_qk.make_fragment_B(mma_qk.partition_B(sk0))
            tK1 = mma_qk.make_fragment_B(mma_qk.partition_B(sk1))
            v0 = svt0
            v1 = svt1
            owner = 0 if (self.num_tiles == 1 or wg == 0) else 1
            tK_owned = tK0
            v_owned = v0
            v_remote = v1
            acc_qk = cute.make_rmem_tensor(
                mma_qk.partition_shape_C((_BH, _BTOPK)), Float32
            )
            acc_pv = cute.make_rmem_tensor(
                mma_pv.partition_shape_C((_BH, _DV // 2)), Float32
            )
            acc_pv.fill(0.0)
            p = mma_pv.make_fragment_A(mma_pv.partition_shape_A((_BH, _BTOPK)))
            row_max = cute.make_rmem_tensor((2,), Float32)
            row_sum = cute.make_rmem_tensor((2,), Float32)
            row_scale = cute.make_rmem_tensor((2,), Float32)
            row_max.fill(-1e30)
            row_sum.fill(0.0)
            row_scale.fill(1.0)
            atom_qk = cute.make_mma_atom(tiled_qk.op)
            atom_pv = cute.make_mma_atom(tiled_pv.op)
            atom_pv_remote = cute.make_mma_atom(tiled_pv_remote.op)
            tP_remote = mma_pv_remote.make_fragment_A(
                mma_pv_remote.partition_A(sS1 if wg == 0 else sS0)
            )
            smem_copy_atom_s = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    transpose=False, num_matrices=4
                ),
                self.dtype,
            )
            smem_thr_copy_s = cute.make_tiled_copy_C(
                smem_copy_atom_s, tiled_qk
            ).get_slice(local)
            row_scale_final = cute.make_rmem_tensor((2,), Float32)
            lane_group = (local % 32) // 4
            row0 = (local // 32) * 16 + lane_group
            row1 = row0 + 8

            for pair in cutlass.range((self.num_tiles + 1) // 2, unroll=1):
                tile = pair * 2 + (
                    0 if cutlass.const_expr(self.num_tiles == 1) else wg
                )
                stage = tile % _STAGES
                phase = tile // _STAGES
                cute.arch.mbarrier_wait(kv_barrier + stage * 2 + wg, phase)
                acc_qk.fill(0.0)
                warpgroup.fence()
                atom_qk.set(warpgroup.Field.ACCUMULATE, False)
                tK_owned = tK0 if owner == 0 else tK1
                for kk in cutlass.range_constexpr(cute.size(tQ.shape[2])):
                    cute.gemm(
                        atom_qk,
                        acc_qk,
                        tQ[None, None, kk, 0],
                        tK_owned[None, None, kk],
                        acc_qk,
                    )
                    atom_qk.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.commit_group()
                warpgroup.wait_group(0)
                cute.arch.mbarrier_arrive(kv_barrier + 4 + stage * 2 + wg)
                row_length = lengths[token_idx, 0]
                for i in cutlass.range_constexpr(cute.cosize(acc_qk)):
                    col = 8 * (i // 4) + (local % 4) * 2 + (i & 1)
                    slot = tile * _BTOPK + col
                    selected = indices[token_idx, 0, slot]
                    if slot >= row_length or selected < 0:
                        acc_qk[i] = -Float32.inf

                if wg == 1:
                    cute.arch.barrier(barrier_id=10, number_of_threads=256)
                for r in cutlass.range_constexpr(2):
                    cur_max = -Float32.inf
                    for i in cutlass.range_constexpr(0, cute.cosize(acc_qk), 4):
                        j = i + r * 2
                        if j + 1 < cute.cosize(acc_qk):
                            cur_max = cute.arch.fmax(
                                cur_max,
                                cute.arch.fmax(
                                    acc_qk[j] * softmax_scale,
                                    acc_qk[j + 1] * softmax_scale,
                                ),
                            )
                    cur_max = cute.arch.fmax(
                        cur_max,
                        cute.arch.shuffle_sync_bfly(cur_max, 1, 0xFFFFFFFF, 31),
                    )
                    cur_max = cute.arch.fmax(
                        cur_max,
                        cute.arch.shuffle_sync_bfly(cur_max, 2, 0xFFFFFFFF, 31),
                    )
                    new_max = cute.arch.fmax(row_max[r], cur_max)
                    if wg == 1:
                        new_max = cute.arch.fmax(
                            softmax_m[(local // 4) * 2 + r], cur_max
                        )
                    old_scale = cute.math.exp2(
                        (row_max[r] - new_max) * Float32(1.4426950409)
                    )
                    current_sum = Float32(0.0)
                    for i in cutlass.range_constexpr(0, cute.cosize(acc_qk), 4):
                        j = i + r * 2
                        if j + 1 < cute.cosize(acc_qk):
                            acc_qk[j] = cute.math.exp2(
                                (acc_qk[j] * softmax_scale - new_max)
                                * Float32(1.4426950409)
                            )
                            acc_qk[j + 1] = cute.math.exp2(
                                (acc_qk[j + 1] * softmax_scale - new_max)
                                * Float32(1.4426950409)
                            )
                            current_sum += acc_qk[j] + acc_qk[j + 1]
                    current_sum = current_sum + cute.arch.shuffle_sync_bfly(
                        current_sum, 1, 0xFFFFFFFF, 31
                    )
                    current_sum = current_sum + cute.arch.shuffle_sync_bfly(
                        current_sum, 2, 0xFFFFFFFF, 31
                    )
                    row_sum[r] = current_sum + row_sum[r] * old_scale
                    row_max[r] = new_max
                    row_scale[r] = old_scale
                if wg == 0:
                    if local % 4 == 0:
                        softmax_m[(local // 4) * 2] = row_max[0]
                        softmax_m[(local // 4) * 2 + 1] = row_max[1]
                    cute.arch.fence_view_async_shared()
                    cute.arch.barrier(barrier_id=10, number_of_threads=256)
                    cute.arch.barrier(barrier_id=11, number_of_threads=256)
                else:
                    if local % 4 == 0:
                        softmax_m[(local // 4) * 2] = row_max[0]
                        softmax_m[(local // 4) * 2 + 1] = row_max[1]
                    cute.arch.fence_view_async_shared()
                    cute.arch.barrier(barrier_id=11, number_of_threads=256)
                for r in cutlass.range_constexpr(2):
                    row_scale_final[r] = Float32(1.0)
                if wg == 0 and not cutlass.const_expr(self.num_tiles == 1):
                    for r in cutlass.range_constexpr(2):
                        final_max = softmax_m[(local // 4) * 2 + r]
                        row_scale_final[r] = cute.math.exp2(
                            (row_max[r] - final_max) * Float32(1.4426950409)
                        )
                        row_sum[r] = row_sum[r] * row_scale_final[r]
                        row_scale[r] = row_scale[r] * row_scale_final[r]
                        row_max[r] = final_max
                if not cutlass.const_expr(self.num_tiles == 1):
                    if local % 4 == 0:
                        own_l = wg * 64 + (local // 4) * 2
                        softmax_l[own_l] = row_sum[0]
                        softmax_l[own_l + 1] = row_sum[1]
                    cute.arch.fence_view_async_shared()
                    cute.arch.barrier(barrier_id=12, number_of_threads=256)
                    peer_l = (1 - wg) * 64 + (local // 4) * 2
                    row_sum[0] = row_sum[0] + softmax_l[peer_l]
                    row_sum[1] = row_sum[1] + softmax_l[peer_l + 1]
                for r in cutlass.range_constexpr(2):
                    for i in cutlass.range_constexpr(0, cute.cosize(acc_qk), 4):
                        j = i + r * 2
                        if j + 1 < cute.cosize(acc_qk):
                            if wg == 0:
                                acc_qk[j] = acc_qk[j] * row_scale_final[r]
                                acc_qk[j + 1] = (
                                    acc_qk[j + 1] * row_scale_final[r]
                                )
                p_acc = _reshape_acc_to_frg_a(acc_qk)
                p.store(p_acc.load().to(self.dtype))
                t_s = smem_thr_copy_s.partition_D(sS0 if wg == 0 else sS1)
                cute.copy(
                    smem_copy_atom_s,
                    smem_thr_copy_s.retile(p),
                    t_s,
                )
                cute.arch.fence_view_async_shared()
                cute.arch.barrier(barrier_id=13, number_of_threads=256)
                cute.arch.mbarrier_wait(kv_barrier + 8 + stage * 2 + wg, phase)
                if not cutlass.const_expr(self.num_tiles == 1):
                    cute.arch.mbarrier_wait(
                        kv_barrier + 8 + (1 - stage) * 2 + wg,
                        phase,
                    )
                v_owned = v0 if owner == 0 else v1
                v_stage = cute.local_tile(
                    v_owned, (_DV // 2, _BTOPK), coord=(wg, None)
                )
                tV = mma_pv.make_fragment_B(mma_pv.partition_B(v_stage))
                for i in cutlass.range_constexpr(cute.cosize(acc_pv)):
                    local_elem = i % 32
                    out_row = (
                        (local // 32) * 16
                        + ((local_elem % 4) // 2) * 8
                        + ((local % 32) // 4)
                    )
                    if out_row == row0:
                        acc_pv[i] = acc_pv[i] * row_scale[0]
                    elif out_row == row1:
                        acc_pv[i] = acc_pv[i] * row_scale[1]
                warpgroup.fence()
                if pair == 0:
                    atom_pv.set(warpgroup.Field.ACCUMULATE, False)
                for kk in cutlass.range_constexpr(cute.size(p.shape[2])):
                    cute.gemm(
                        atom_pv,
                        acc_pv,
                        p[None, None, kk],
                        tV[None, None, kk, 0],
                        acc_pv,
                    )
                    atom_pv.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.commit_group()
                warpgroup.wait_group(0)
                if not cutlass.const_expr(self.num_tiles == 1):
                    v_remote = v1 if owner == 0 else v0
                    peer_v_stage = cute.local_tile(
                        v_remote,
                        (_DV // 2, _BTOPK),
                        coord=(wg, None),
                    )
                    tV_remote = mma_pv_remote.make_fragment_B(
                        mma_pv_remote.partition_B(peer_v_stage)
                    )
                    warpgroup.fence()
                    atom_pv_remote.set(warpgroup.Field.ACCUMULATE, True)
                    for kk in cutlass.range_constexpr(
                        cute.size(tP_remote.shape[2])
                    ):
                        cute.gemm(
                            atom_pv_remote,
                            acc_pv,
                            tP_remote[None, None, kk],
                            tV_remote[None, None, kk, 0],
                            acc_pv,
                        )
                    warpgroup.commit_group()
                    warpgroup.wait_group(0)
                cute.arch.mbarrier_arrive(kv_barrier + 12 + stage * 2 + wg)
                if not cutlass.const_expr(self.num_tiles == 1):
                    cute.arch.mbarrier_arrive(
                        kv_barrier + 12 + (1 - stage) * 2 + wg
                    )

            sink_mass0 = cute.math.exp2(
                (sink[row0] - row_max[0]) * Float32(1.4426950409)
            )
            sink_mass1 = cute.math.exp2(
                (sink[row1] - row_max[1]) * Float32(1.4426950409)
            )
            denom0 = row_sum[0] + sink_mass0
            denom1 = row_sum[1] + sink_mass1
            scale0 = cute.arch.rcp_approx(denom0)
            scale1 = cute.arch.rcp_approx(denom1)
            if row_sum[0] == 0.0:
                scale0 = Float32(0.0)
            if row_sum[1] == 0.0:
                scale1 = Float32(0.0)
            lse[row0, token_idx] = (
                cute.math.log(denom0) + row_max[0]
                if row_sum[0] > 0.0
                else -Float32.inf
            )
            lse[row1, token_idx] = (
                cute.math.log(denom1) + row_max[1]
                if row_sum[1] > 0.0
                else -Float32.inf
            )
            for i in cutlass.range_constexpr(cute.cosize(acc_pv)):
                local_elem = i % 32
                out_row = (
                    (local // 32) * 16
                    + ((local_elem % 4) // 2) * 8
                    + ((local % 32) // 4)
                )
                if out_row == row0:
                    acc_pv[i] = acc_pv[i] * scale0
                elif out_row == row1:
                    acc_pv[i] = acc_pv[i] * scale1

            smem_copy_atom_o = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    transpose=False, num_matrices=4
                ),
                self.dtype,
            )
            smem_thr_copy_o = cute.make_tiled_copy_C(
                smem_copy_atom_o, tiled_pv
            ).get_slice(local)
            so_half = cute.local_tile(so, (_BH, _DV // 2), (0, wg))
            r_o = cute.make_rmem_tensor_like(acc_pv, self.dtype)
            r_o.store(acc_pv.load().to(self.dtype))
            t_so = smem_thr_copy_o.partition_D(so_half)
            cute.copy(
                smem_copy_atom_o,
                smem_thr_copy_o.retile(r_o),
                t_so,
            )
            t_coords_so = smem_thr_copy_o.partition_D(
                cute.make_identity_tensor((_BH, _DV // 2))
            )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier(barrier_id=5, number_of_threads=256)
            for i in cutlass.range_constexpr(cute.cosize(t_so)):
                row, col = t_coords_so[i][0], t_coords_so[i][1]
                if row < _BH and col < _DV // 2:
                    o[row, col + wg * (_DV // 2), token_idx] = t_so[i]


def flashmla_sm90_sparse_prefill_wgmma(
    q,
    kv,
    indices,
    *,
    sm_scale=None,
    topk_lengths=None,
    attn_sink=None,
    use_tma_q=False,
):
    if q.ndim != 4 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError("expected q[B,S,64,576], kv[B,Sk,576], indices[B,S,K]")
    b, s, h, dqk = q.shape
    if (h, dqk) != (_BH, _DQK) or kv.shape[0] != b or kv.shape[2] != _DQK:
        raise ValueError("expected q[B,S,64,576] and kv[B,Sk,576]")
    if indices.dtype != paddle.int32:
        raise TypeError("indices must be int32")
    topk = indices.shape[2]
    if topk <= 0 or topk % _BTOPK:
        raise ValueError("topk must be a positive multiple of 64")
    lengths = (
        topk_lengths
        if topk_lengths is not None
        else (indices >= 0).astype("int32").sum(axis=-1)
    )
    sink = (
        attn_sink
        if attn_sink is not None
        else paddle.full([_BH], -1e30, dtype="float32")
    )
    scale = _DQK**-0.5 if sm_scale is None else float(sm_scale)
    qf = q.reshape([1, b * s, _BH, _DQK]).contiguous()
    kf = kv.reshape([1, b * kv.shape[1], 1, _DQK]).contiguous()
    vf = kv[..., :_DV].reshape([1, b * kv.shape[1], 1, _DV]).contiguous()
    outf = paddle.empty([1, b * s, _BH, _DV], dtype=q.dtype)
    lsef = paddle.empty([1, b * s, _BH], dtype="float32")
    idx = indices.reshape([b * s, 1, topk]).contiguous()
    lens = lengths.reshape([b * s, 1]).contiguous()
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel = FlashMLASm90Wgmma(
        _cutlass_dtype(q.dtype), topk, use_tma_q=use_tma_q
    )
    args = {
        "mQ": paddle_to_cute_tensor(
            qf, assumed_align=16, leading_dim=3, enable_tvm_ffi=True
        ),
        "mK": paddle_to_cute_tensor(
            kf, assumed_align=16, leading_dim=3, enable_tvm_ffi=True
        ),
        "mV": paddle_to_cute_tensor(
            vf, assumed_align=16, leading_dim=3, enable_tvm_ffi=True
        ),
        "mO": paddle_to_cute_tensor(
            outf, assumed_align=16, leading_dim=3, enable_tvm_ffi=True
        ),
        "mLSE": paddle_to_cute_tensor(
            lsef, assumed_align=4, leading_dim=2, enable_tvm_ffi=True
        ),
        "mIndices": paddle_to_cute_tensor(
            idx, assumed_align=4, leading_dim=2, enable_tvm_ffi=True
        ),
        "mLengths": paddle_to_cute_tensor(
            lens, assumed_align=4, leading_dim=1, enable_tvm_ffi=True
        ),
        "mSink": paddle_to_cute_tensor(
            sink, assumed_align=4, leading_dim=0, enable_tvm_ffi=True
        ),
        "softmax_scale": Float32(scale),
        "stream": stream,
    }
    key = (
        tuple(qf.shape),
        tuple(kf.shape),
        tuple(idx.shape),
        q.dtype,
        topk,
        bool(use_tma_q),
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(kernel, **args, options="--enable-tvm-ffi")
        _COMPILED[key] = compiled
    compiled(*[args[name] for name in args if name != "stream"])
    return outf.reshape([b, s, _BH, _DV]), lsef.reshape([b, s, _BH])


__all__ = ["flashmla_sm90_sparse_prefill_wgmma"]
