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

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import cute, pipeline, utils

"""
A NSA(Native Sparse Attention) attention forward pass example for NVIDIA Ampere SM90 architecture using Cute DSL.

There are some constraints for this example:
* Only Float16 and BFloat16 are supported.
* Accumulation type is Float32.
* Supported block sizes(16, 32, 64) combined with GQA group sizes(1, 2, 4, 8, 32, 64)
"""


class HopperSelectAttentionFwd:
    def __init__(
        self,
        head_dim: int,
        value_dim: int,
        GQA_group_size: int,
        block_size: int,
        index_tiles: int,
        dtype: type[cutlass.Numeric],
        acc_dtype: type[cutlass.Numeric],
        kv_stages: int = 2,
        reuse_kv: bool = False,
        query_tiles: int = 1,
    ):
        self.dtype = dtype
        self.acc_dtype = acc_dtype
        self.atom_layout_mnk = (1, 1, 1)
        self.block_size = block_size
        self.index_tiles = index_tiles

        assert self.dtype in [cutlass.Float16, cutlass.BFloat16]
        assert self.acc_dtype in [
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float32,
        ]
        assert (
            self.block_size % 8 == 0
            and self.block_size >= 16
            and self.block_size <= 128
        ), "block_size should be a multiple of 8 and >= 16 and <= 128"

        # Two consumer warpgroups share a double-buffered indexed KV tile.
        # The producer side uses ordinary indexed stores; the two stage
        # barriers below provide the double-buffer handoff.
        self.K_stage = kv_stages
        self.V_stage = kv_stages
        self.reuse_kv = reuse_kv
        self.query_tiles = query_tiles
        self.barrier_threads = 192 if query_tiles == 1 else 256
        self.epi_stage = 3
        self.threads_per_warp_group = 128
        self.threads_per_cta = 3 * self.threads_per_warp_group
        self.threads_per_block = 32
        self.mma_warp_groups = 1  # math.prod((1, 1, 1))

        self.qk_dim = head_dim
        self.value_dim = value_dim
        self.tile_shape_mnk_QK = (16, self.block_size, self.qk_dim)
        self.tile_shape_mnk_PV = (16, self.value_dim, self.block_size)
        self.epi_tile = (16, min(32, self.tile_shape_mnk_PV[1]))
        self.GQA_group_size = GQA_group_size
        self.MMA_group_size = 16
        self.log2_e = 1.4426950408889634074

        assert self.GQA_group_size <= 64, (
            "GQA_group_size should be less than or equal to 64"
        )
        assert self.GQA_group_size == self.MMA_group_size * self.query_tiles

    @cute.jit
    def __call__(
        self,
        Q: cute.Tensor,
        K: cute.Tensor,
        V: cute.Tensor,
        O: cute.Tensor,
        L: cute.Tensor,
        M: cute.Tensor,
        block_indices: cute.Tensor,
        block_counts: cute.Tensor,
        topk_lengths: cute.Tensor,
        attn_sink: cute.Tensor,
        max_length: int,
        seq_offsets_q: cute.Tensor,
        seq_offsets_k: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        """
        Args:
            Q (cute.Tensor):
                Queries of shape `[G, K, B * T, H]`
            K (cute.Tensor):
                Keys of shape `[B*T, K, H]`
            V (cute.Tensor):
                Values of shape `[K, B*T, H]`
            O (cute.Tensor):
                Output of shape `[G, V, B * T, H]`
            L (cute.Tensor):
                Logits of shape `[G, T, H]`
            M (cute.Tensor):
                Mask of shape `[G, T, H]`
            block_indices (cute.Tensor):
                Selected block indices of shape `[B, S, H, TopK]`
            block_counts (cute.Tensor):
                Selected block counts of shape `[B, S, H]`
            topk_lengths (cute.Tensor):
                Valid token-prefix lengths of shape `[B, S, H]`
            attn_sink (cute.Tensor):
                Per-query-head sink logits of shape `[H]`
            block_size (cute.constexpr):
                Block size
            stream (cuda.CUstream):
                CUDA stream
        """
        # (1, t, h_k * h_r, d) -> (h_r, d, t, h_k)
        if cutlass.const_expr(self.query_tiles == 1):
            Q = cute.make_tensor(
                Q.iterator,
                cute.make_layout(
                    (
                        self.GQA_group_size,
                        Q.shape[3],
                        Q.shape[1] * Q.shape[0],
                        Q.shape[2] // self.GQA_group_size,
                    ),
                    stride=(
                        Q.stride[2],
                        Q.stride[3],
                        Q.stride[1],
                        Q.stride[2] * self.GQA_group_size,
                    ),
                ),
            )
        else:
            Q = cute.make_tensor(
                Q.iterator,
                cute.make_layout(
                    (
                        self.query_tiles,
                        self.MMA_group_size,
                        Q.shape[3],
                        Q.shape[1] * Q.shape[0],
                        Q.shape[2] // self.GQA_group_size,
                    ),
                    stride=(
                        Q.stride[2] * self.MMA_group_size,
                        Q.stride[2],
                        Q.stride[3],
                        Q.stride[1],
                        Q.stride[2] * self.GQA_group_size,
                    ),
                ),
            )

        # (1, t, h_k, d) -> (t, d, h_kv)
        K = cute.make_tensor(
            K.iterator,
            cute.make_layout(
                (K.shape[1] * K.shape[0], K.shape[3], K.shape[2]),
                stride=(K.stride[1], K.stride[3], K.stride[2]),
            ),
        )

        # (1, t, h_k, d) -> (t, d, h_kv)
        V = cute.make_tensor(
            V.iterator,
            cute.make_layout(
                (V.shape[1] * V.shape[0], V.shape[3], V.shape[2]),
                stride=(V.stride[1], V.stride[3], V.stride[2]),
            ),
        )

        # (1, t, h_k * h_r, d) -> (h_r, d, t, h_k)
        if cutlass.const_expr(self.query_tiles == 1):
            O = cute.make_tensor(
                O.iterator,
                cute.make_layout(
                    (
                        self.GQA_group_size,
                        O.shape[3],
                        O.shape[1] * O.shape[0],
                        O.shape[2] // self.GQA_group_size,
                    ),
                    stride=(
                        O.stride[2],
                        O.stride[3],
                        O.stride[1],
                        O.stride[2] * self.GQA_group_size,
                    ),
                ),
            )
        else:
            O = cute.make_tensor(
                O.iterator,
                cute.make_layout(
                    (
                        self.query_tiles,
                        self.MMA_group_size,
                        O.shape[3],
                        O.shape[1] * O.shape[0],
                        O.shape[2] // self.GQA_group_size,
                    ),
                    stride=(
                        O.stride[2] * self.MMA_group_size,
                        O.stride[2],
                        O.stride[3],
                        O.stride[1],
                        O.stride[2] * self.GQA_group_size,
                    ),
                ),
            )

        # (1, t, h_k * h_r) -> (h_r, t, h_k)
        if cutlass.const_expr(self.query_tiles == 1):
            L = cute.make_tensor(
                L.iterator,
                cute.make_layout(
                    (
                        self.GQA_group_size,
                        L.shape[1] * L.shape[0],
                        L.shape[2] // self.GQA_group_size,
                    ),
                    stride=(
                        L.stride[2],
                        L.stride[1],
                        L.stride[2] * self.GQA_group_size,
                    ),
                ),
            )
        else:
            L = cute.make_tensor(
                L.iterator,
                cute.make_layout(
                    (
                        self.query_tiles,
                        self.MMA_group_size,
                        L.shape[1] * L.shape[0],
                        L.shape[2] // self.GQA_group_size,
                    ),
                    stride=(
                        L.stride[2] * self.MMA_group_size,
                        L.stride[2],
                        L.stride[1],
                        L.stride[2] * self.GQA_group_size,
                    ),
                ),
            )

        # (1, t, h_k * h_r) -> (h_r, t, h_k)
        if cutlass.const_expr(self.query_tiles == 1):
            M = cute.make_tensor(
                M.iterator,
                cute.make_layout(
                    (
                        self.GQA_group_size,
                        M.shape[1] * M.shape[0],
                        M.shape[2] // self.GQA_group_size,
                    ),
                    stride=(
                        M.stride[2],
                        M.stride[1],
                        M.stride[2] * self.GQA_group_size,
                    ),
                ),
            )
        else:
            M = cute.make_tensor(
                M.iterator,
                cute.make_layout(
                    (
                        self.query_tiles,
                        self.MMA_group_size,
                        M.shape[1] * M.shape[0],
                        M.shape[2] // self.GQA_group_size,
                    ),
                    stride=(
                        M.stride[2] * self.MMA_group_size,
                        M.stride[2],
                        M.stride[1],
                        M.stride[2] * self.GQA_group_size,
                    ),
                ),
            )

        if cutlass.const_expr(self.query_tiles != 1):
            Q = cute.group_modes(Q, 0, 2)
            O = cute.group_modes(O, 0, 2)
            L = cute.group_modes(L, 0, 2)
            M = cute.group_modes(M, 0, 2)

        self.Q_layout = utils.LayoutEnum.from_tensor(Q)
        self.K_layout = utils.LayoutEnum.from_tensor(K)
        self.V_layout = utils.LayoutEnum.from_tensor(V)
        self.O_layout = utils.LayoutEnum.from_tensor(O)

        self.Q_dtype = Q.element_type
        self.K_dtype = K.element_type
        self.V_dtype = V.element_type
        self.O_dtype = O.element_type

        if cutlass.const_expr(self.Q_dtype.width != self.K_dtype.width):
            raise TypeError(
                f"Type width mismatch: {self.Q_dtype.width} != {self.K_dtype.width}"
            )

        mma_n_itr = self.block_size // 8
        tiled_mma_QK = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(
                self.Q_dtype, self.acc_dtype, (16, 8, 16)
            ),
            (1, self.threads_per_block // 32, 1),
            permutation_mnk=(
                16,
                self.threads_per_block // 32 * 8 * mma_n_itr,
                16,
            ),
        )

        tiled_mma_PV = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(
                self.Q_dtype, self.acc_dtype, (16, 8, 16)
            ),
            (1, self.threads_per_block // 32, 1),
            permutation_mnk=(16, self.threads_per_block // 32 * 8 * 8, 16),
        )

        tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()

        # qkv smem layout definition
        Q_smem_shape = (self.tile_shape_mnk_QK[0], self.tile_shape_mnk_QK[2])
        Q_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            # sm90_utils.get_smem_layout_atom(self.Q_layout, self.Q_dtype, Q_smem_shape[1]), # K-major by default
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER,
            self.Q_dtype,
        )
        assert (
            self.Q_layout.sm90_mma_major_mode()
            == cute.nvgpu.warpgroup.OperandMajorMode.K
        ), "Q_layout should be K-major"
        Q_smem_layout_staged = cute.tile_to_shape(
            Q_smem_layout_atom,
            cute.append(Q_smem_shape, self.query_tiles),
            order=(0, 1, 2),  # K-major by default
        )

        K_smem_shape = (self.block_size, self.tile_shape_mnk_QK[2])
        K_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                self.K_layout, self.K_dtype, K_smem_shape[1]
            ),  # K-major by default
            # cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER,
            self.K_dtype,
        )
        assert (
            self.K_layout.sm90_mma_major_mode()
            == cute.nvgpu.warpgroup.OperandMajorMode.K
        ), "K_layout should be K-major"
        K_smem_layout_staged = cute.tile_to_shape(
            K_smem_layout_atom,
            cute.append(K_smem_shape, self.K_stage),
            order=(0, 1, 2),  # K-major by default
        )

        V_smem_shape = (self.tile_shape_mnk_PV[2], self.tile_shape_mnk_PV[1])
        V_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                self.V_layout, self.V_dtype, V_smem_shape[1]
            ),  # K-major by default
            # cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER,
            self.V_dtype,
        )

        assert (
            self.V_layout.sm90_mma_major_mode()
            == cute.nvgpu.warpgroup.OperandMajorMode.K
        ), "V_layout should be K-major"
        V_smem_layout_staged = cute.tile_to_shape(
            V_smem_layout_atom,
            cute.append(V_smem_shape, self.V_stage),
            order=(0, 1, 2),  # K-major by default
        )

        # import pdb; pdb.set_trace()
        V_layout_atom = sm90_utils.get_smem_layout_atom(
            self.V_layout, self.V_dtype, V_smem_shape[1]
        )

        if cutlass.const_expr(
            V_layout_atom == cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128
        ):
            Vt_layout_atom = cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW128
        elif cutlass.const_expr(
            V_layout_atom == cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64
        ):
            Vt_layout_atom = cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW64
        elif cutlass.const_expr(
            V_layout_atom == cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32
        ):
            Vt_layout_atom = cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_SW32
        elif cutlass.const_expr(
            V_layout_atom == cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER
        ):
            Vt_layout_atom = cute.nvgpu.warpgroup.SmemLayoutAtomKind.MN_INTER
        else:
            raise ValueError(f"Unsupported V_layout_atom: {V_layout_atom}")

        Vt_smem_shape = (self.tile_shape_mnk_PV[1], self.tile_shape_mnk_PV[2])
        Vt_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            # sm90_utils.get_smem_layout_atom(self.Vt_layout, self.V_dtype, Vt_smem_shape[0]), # K-major by default
            Vt_layout_atom,
            self.V_dtype,
        )
        Vt_smem_layout_staged = cute.tile_to_shape(
            Vt_smem_layout_atom,
            cute.append(Vt_smem_shape, self.V_stage),
            order=(1, 0, 2),  # K-major by default
        )

        O_smem_shape = self.epi_tile
        O_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                self.O_layout, self.O_dtype, O_smem_shape[1]
            ),  # K-major by default
            self.O_dtype,
        )
        if cutlass.const_expr(self.query_tiles == 1):
            O_smem_layout_staged = cute.tile_to_shape(
                O_smem_layout_atom,
                cute.append(O_smem_shape, self.epi_stage),
                order=(1, 0, 2) if self.O_layout.is_m_major_c() else (0, 1, 2),
            )
        else:
            O_smem_layout_staged = cute.tile_to_shape(
                O_smem_layout_atom,
                cute.append(
                    cute.append(O_smem_shape, self.epi_stage),
                    self.query_tiles,
                ),
                order=(1, 0, 2, 3)
                if self.O_layout.is_m_major_c()
                else (0, 1, 2, 3),
            )

        smem_layout_Q = cute.slice_(Q_smem_layout_staged, (None, None, 0))
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_op,
            Q,
            smem_layout_Q,
            (self.tile_shape_mnk_QK[0], self.tile_shape_mnk_QK[2]),
            num_multicast=1,
        )
        smem_layout_O = (
            cute.slice_(O_smem_layout_staged, (None, None, 0))
            if cutlass.const_expr(self.query_tiles == 1)
            else cute.slice_(O_smem_layout_staged, (None, None, 0, 0))
        )
        O_cta_v_layout = cute.composition(
            cute.make_identity_layout(O.shape), self.epi_tile
        )
        tma_atom_O, tma_tensor_O = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            O,
            smem_layout_O,
            O_cta_v_layout,
        )

        L_smem_shape = (self.tile_shape_mnk_QK[0], 1)
        L_smem_layout = cute.make_layout(
            shape=L_smem_shape, stride=(1, self.tile_shape_mnk_QK[0])
        )
        M_smem_shape = (self.tile_shape_mnk_QK[0], 1)
        M_smem_layout = cute.make_layout(
            shape=M_smem_shape, stride=(1, self.tile_shape_mnk_QK[0])
        )

        BUFFER_ALIGN_BYTES = 128

        @cute.struct
        class SharedStorageShare:
            mainloop_pipeline_array_ptr: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, 4], 16
            ]
            prefetchQ_pipeline_array_ptr: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int64, 2 * self.query_tiles],
                16,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[
                    self.Q_dtype,
                    cute.cosize(Q_smem_layout_staged),
                ],
                BUFFER_ALIGN_BYTES,
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[
                    self.O_dtype,
                    cute.cosize(O_smem_layout_staged),
                ],
                BUFFER_ALIGN_BYTES,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[
                    self.K_dtype,
                    max(
                        cute.cosize(K_smem_layout_staged),
                        cute.cosize(V_smem_layout_staged),
                    )
                    if self.reuse_kv
                    else cute.cosize(K_smem_layout_staged),
                ],
                BUFFER_ALIGN_BYTES,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[
                    self.V_dtype,
                    1 if self.reuse_kv else cute.cosize(V_smem_layout_staged),
                ],
                BUFFER_ALIGN_BYTES,
            ]

        # Q/O share one allocation.  The FlashMLA port can also reuse the
        # larger K allocation for V; the explicit handoff barriers below keep
        # V from overwriting K before QK has consumed it.
        self.shared_storage = SharedStorageShare

        grid_dim = (
            max_length,
            K.layout.shape[2],
            seq_offsets_q.shape[0] - 1,
        )  # max_length, head_num_kv, batch_size
        # grid_dim = (1, 1, 1)
        block_dim = [self.threads_per_cta, 1, 1]
        cta_layout_mnk = cute.make_layout((1, 1, 1))

        self.kernel(
            tma_atom_Q,
            tma_tensor_Q,
            K,
            V,
            tma_atom_O,
            tma_tensor_O,
            L,
            L_smem_layout,
            M,
            M_smem_layout,
            seq_offsets_q,
            block_indices,
            block_counts,
            topk_lengths,
            attn_sink,
            seq_offsets_k,
            tiled_mma_QK,
            tiled_mma_PV,
            cta_layout_mnk,
            Q_smem_layout_staged,
            K_smem_layout_staged,
            V_smem_layout_staged,
            Vt_smem_layout_staged,
            O_smem_layout_staged,
            softmax_scale,
        ).launch(
            grid=grid_dim,
            block=block_dim,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    def _threadquad_reduce(
        self, val: cutlass.Float32, op: Callable, mask: int
    ) -> cutlass.Float32:
        """thread quad reduction

        :param val: register value
        :type val: cutlass.Float32
        :param op: binary operator
        :type op: Callable
        :return: reduced value
        :rtype: cutlass.Float32
        """
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(
                val, offset=2, mask=mask, mask_and_clamp=31
            ),
        )
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(
                val, offset=1, mask=mask, mask_and_clamp=31
            ),
        )
        return val

    def _threadquad_reduce_max(
        self, val: cutlass.Float32, mask: int
    ) -> cutlass.Float32:
        """thread quad reduction max

        :param val: register value
        :type val: cutlass.Float32
        :return: max value
        :rtype: cutlass.Float32
        """
        return self._threadquad_reduce(
            val, lambda x, y: cute.arch.fmax(x, y), mask
        )

    def _threadquad_reduce_sum(
        self, val: cutlass.Float32, mask: int
    ) -> cutlass.Float32:
        """thread quad reduction sum

        :param val: register value
        :type val: cutlass.Float32
        :return: sum value
        :rtype: cutlass.Float32
        """
        return self._threadquad_reduce(val, lambda x, y: x + y, mask)

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        """make acc tensor as mn layout view

        :param acc: input tensor
        :type acc: cute.Tensor
        :return: acc tensor mn layout view
        :rtype: cute.Tensor
        """
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (
                    acc_layout_col_major.shape[0][1],
                    acc_layout_col_major.shape[1],
                ),  # MMA_M
                (
                    acc_layout_col_major.shape[0][0],
                    acc_layout_col_major.shape[2],
                ),  # MMA_N
            ),
            stride=(
                (
                    acc_layout_col_major.stride[0][1],
                    acc_layout_col_major.stride[1],
                ),  # MMA_M
                (
                    acc_layout_col_major.stride[0][0],
                    acc_layout_col_major.stride[2],
                ),  # MMA_N
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)

    @cute.jit
    def _exp2f(
        self, x: cute.TensorSSA | cutlass.Float32
    ) -> cute.TensorSSA | cutlass.Float32:
        """exp2f calculation for both vector and scalar.

        :param x: input value
        :type x: cute.TensorSSA or cutlass.Float32
        :return: exp2 value
        :rtype: cute.TensorSSA or cutlass.Float32
        """
        if cutlass.const_expr(isinstance(x, cute.TensorSSA)):
            res = cute.make_rmem_tensor(x.shape, cutlass.Float32)
            res.store(x)

            for i in cutlass.range_constexpr(cute.size(x.shape)):
                res[i] = self._exp2f(res[i])

            return res.load()
        return cute.math.exp2(x, fastmath=True)

    @cute.kernel
    def kernel(
        self,
        tma_atom_Q: cute.CopyAtom,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        mO: cute.Tensor,
        mL: cute.Tensor,
        L_smem_layout: cute.Layout,
        mM: cute.Tensor,
        M_smem_layout: cute.Layout,
        seq_offsets_q: cute.Tensor,
        block_indices: cute.Tensor,
        block_counts: cute.Tensor,
        topk_lengths: cute.Tensor,
        attn_sink: cute.Tensor,
        seq_offsets_k: cute.Tensor,
        tiled_mma_QK: cute.TiledMma,
        tiled_mma_PV: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        Q_smem_layout_staged: cute.ComposedLayout,
        K_smem_layout_staged: cute.ComposedLayout,
        V_smem_layout_staged: cute.ComposedLayout,
        Vt_smem_layout_staged: cute.ComposedLayout,
        O_smem_layout_staged: cute.ComposedLayout,
        softmax_scale: cutlass.Float32,
    ):
        """
        GPU device kernel performing the batched NSA computation.
        """

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()
        warpgroup_idx = cute.arch.make_warp_uniform(
            tidx // self.threads_per_warp_group
        )
        local_tidx = tidx - warpgroup_idx * self.threads_per_warp_group
        local_warp_idx = local_tidx // self.threads_per_block
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_Q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_O)

        t, KV_head_idx, offset_idx = cute.arch.block_idx()
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        Q_smem_layout = cute.slice_(Q_smem_layout_staged, (None, None, 0))
        K_smem_layout = cute.slice_(K_smem_layout_staged, (None, None, 0))
        V_smem_layout = cute.slice_(V_smem_layout_staged, (None, None, 0))

        # WG2 performs ordinary indexed stores into shared memory.  WG0 and
        # WG1 are the two consumers of the same double-buffered KV tile.
        # In the B_H=64 phase1 mapping, each consumer owns two 16-row Q
        # tiles; all four active warps participate in the Q TMA pipeline.
        consumer_arrive_cnt = (
            2 * self.threads_per_block
            if self.query_tiles == 1
            else self.query_tiles * self.threads_per_block
        )
        Q_tma_copy_bytes = cute.size_in_bytes(self.Q_dtype, Q_smem_layout)
        prefetchQ_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        prefetchQ_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        prefetchQ_pipeline_array_ptr = (
            storage.prefetchQ_pipeline_array_ptr.data_ptr()
        )
        prefetchQ_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=prefetchQ_pipeline_array_ptr,
            num_stages=self.query_tiles,
            producer_group=prefetchQ_pipeline_producer_group,
            consumer_group=prefetchQ_pipeline_consumer_group,
            tx_count=Q_tma_copy_bytes,
        )
        q_producer, q_consumer = prefetchQ_pipeline.make_participants()

        sQ = storage.sQ.get_tensor(
            Q_smem_layout_staged.outer, swizzle=Q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner
        )
        if cutlass.const_expr(self.reuse_kv):
            sV = storage.sK.get_tensor(
                V_smem_layout_staged.outer,
                swizzle=V_smem_layout_staged.inner,
            )
            sVt = storage.sK.get_tensor(
                Vt_smem_layout_staged.outer,
                swizzle=Vt_smem_layout_staged.inner,
            )
        else:
            sV = storage.sV.get_tensor(
                V_smem_layout_staged.outer,
                swizzle=V_smem_layout_staged.inner,
            )
            sVt = storage.sV.get_tensor(
                Vt_smem_layout_staged.outer,
                swizzle=Vt_smem_layout_staged.inner,
            )
        sO = storage.sO.get_tensor(
            O_smem_layout_staged.outer,
            swizzle=O_smem_layout_staged.inner,
        )

        smem_copy_atom_Q = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=2),
            self.Q_dtype,
        )
        smem_copy_atom_K = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.K_dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.V_dtype,
        )

        smem_tiled_copy_Q = cute.make_tiled_copy_A(
            smem_copy_atom_Q, tiled_mma_QK
        )
        smem_tiled_copy_K = cute.make_tiled_copy_B(
            smem_copy_atom_K, tiled_mma_QK
        )
        smem_tiled_copy_V = cute.make_tiled_copy_B(
            smem_copy_atom_V, tiled_mma_PV
        )

        mma_tidx = local_tidx % self.threads_per_block
        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(mma_tidx)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(mma_tidx)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(mma_tidx)

        seq_len = seq_offsets_q[offset_idx + 1] - seq_offsets_q[offset_idx]
        offset = seq_offsets_q[offset_idx]
        kv_seq_len = seq_offsets_k[offset_idx + 1] - seq_offsets_k[offset_idx]
        kv_offset = seq_offsets_k[offset_idx]
        seq_len_aligned = (
            (seq_len + self.tile_shape_mnk_QK[1] - 1)
            // self.tile_shape_mnk_QK[1]
            * self.tile_shape_mnk_QK[1]
        )

        if cutlass.const_expr(self.query_tiles == 1):
            mQ_offset = cute.domain_offset((0, 0, offset, 0), mQ)
            mQ = cute.make_tensor(
                mQ_offset.iterator,
                cute.make_layout(
                    shape=(
                        mQ.shape[0],
                        mQ.shape[1],
                        seq_len_aligned,
                        mQ.shape[3],
                    ),
                    stride=mQ.stride,
                ),
            )
        else:
            mQ_offset = cute.domain_offset((0, 0, offset, 0), mQ)
            mQ = cute.make_tensor(
                mQ_offset.iterator,
                cute.make_layout(
                    shape=(
                        mQ.shape[0],
                        mQ.shape[1],
                        seq_len_aligned,
                        mQ.shape[3],
                    ),
                    stride=mQ.stride,
                ),
            )
        mK_offset = cute.domain_offset((kv_offset, 0, 0), mK)
        mK = cute.make_tensor(
            mK_offset.iterator,
            cute.make_layout(
                shape=(
                    (kv_seq_len + self.block_size - 1)
                    // self.block_size
                    * self.block_size,
                    mK.shape[1],
                    mK.shape[2],
                ),
                stride=mK.stride,
            ),
        )
        mV_offset = cute.domain_offset((kv_offset, 0, 0), mV)  # `[K, B*T, H]`
        mV = cute.make_tensor(
            mV_offset.iterator,
            cute.make_layout(
                shape=(
                    (kv_seq_len + self.block_size - 1)
                    // self.block_size
                    * self.block_size,
                    mV.shape[1],
                    mV.shape[2],
                ),
                stride=mV.stride,
            ),
        )
        if cutlass.const_expr(self.query_tiles == 1):
            mO_offset = cute.domain_offset((0, 0, offset, 0), mO)
            mO = cute.make_tensor(
                mO_offset.iterator,
                cute.make_layout(
                    shape=(
                        mO.shape[0],
                        mO.shape[1],
                        seq_len_aligned,
                        mO.shape[3],
                    ),
                    stride=mO.stride,
                ),
            )
            mL_offset = cute.domain_offset((0, offset, 0), mL)
            mL = cute.make_tensor(
                mL_offset.iterator,
                cute.make_layout(
                    shape=(mL.shape[0], seq_len_aligned, mL.shape[2]),
                    stride=mL.stride,
                ),
            )
            mM_offset = cute.domain_offset((0, offset, 0), mM)
            mM = cute.make_tensor(
                mM_offset.iterator,
                cute.make_layout(
                    shape=(mM.shape[0], seq_len_aligned, mM.shape[2]),
                    stride=mM.stride,
                ),
            )
        else:
            mO_offset = cute.domain_offset((0, 0, offset, 0), mO)
            mO = cute.make_tensor(
                mO_offset.iterator,
                cute.make_layout(
                    shape=(
                        mO.shape[0],
                        mO.shape[1],
                        seq_len_aligned,
                        mO.shape[3],
                    ),
                    stride=mO.stride,
                ),
            )
            mL_offset = cute.domain_offset((0, offset, 0), mL)
            mL = cute.make_tensor(
                mL_offset.iterator,
                cute.make_layout(
                    shape=(
                        mL.shape[0],
                        seq_len_aligned,
                        mL.shape[2],
                    ),
                    stride=mL.stride,
                ),
            )
            mM_offset = cute.domain_offset((0, offset, 0), mM)
            mM = cute.make_tensor(
                mM_offset.iterator,
                cute.make_layout(
                    shape=(
                        mM.shape[0],
                        seq_len_aligned,
                        mM.shape[2],
                    ),
                    stride=mM.stride,
                ),
            )

        # WG2 is the sole KV producer.  Each producer thread copies a
        # strided subset of both native K/V rows into the selected pipeline
        # stage.  The token index is read directly from the caller-provided
        # [B, S, topk] tensor; no gathered global workspace is involved.
        if warpgroup_idx == 2:
            if t < seq_len:
                K_elems = self.block_size * self.qk_dim
                V_elems = self.block_size * self.value_dim
                K_tile_cnt = block_counts[offset + t, KV_head_idx]
                row_topk = topk_lengths[offset + t, KV_head_idx]
                for block_idx in cutlass.range(K_tile_cnt, unroll=1):
                    stage = block_idx % self.K_stage
                    for linear in cutlass.range(
                        local_tidx, K_elems, self.threads_per_warp_group
                    ):
                        row = linear // self.qk_dim
                        col = linear - row * self.qk_dim
                        token = block_indices[
                            offset + t,
                            KV_head_idx,
                            block_idx * self.block_size + row,
                        ]
                        slot = block_idx * self.block_size + row
                        if (
                            slot < row_topk
                            and token >= 0
                            and token < kv_seq_len
                        ):
                            sK[row, col, stage] = mK[token, col, KV_head_idx]
                        else:
                            sK[row, col, stage] = self.K_dtype(0)
                    if cutlass.const_expr(self.reuse_kv):
                        cute.arch.barrier(
                            barrier_id=0, number_of_threads=self.barrier_threads
                        )
                        cute.arch.barrier(
                            barrier_id=2, number_of_threads=self.barrier_threads
                        )
                    for linear in cutlass.range(
                        local_tidx, V_elems, self.threads_per_warp_group
                    ):
                        row = linear // self.value_dim
                        col = linear - row * self.value_dim
                        token = block_indices[
                            offset + t,
                            KV_head_idx,
                            block_idx * self.block_size + row,
                        ]
                        slot = block_idx * self.block_size + row
                        if (
                            slot < row_topk
                            and token >= 0
                            and token < kv_seq_len
                        ):
                            sV[row, col, stage] = mV[token, col, KV_head_idx]
                        else:
                            sV[row, col, stage] = self.V_dtype(0)
                    if cutlass.const_expr(self.reuse_kv):
                        cute.arch.barrier(
                            barrier_id=1, number_of_threads=self.barrier_threads
                        )
                        cute.arch.barrier(
                            barrier_id=3, number_of_threads=self.barrier_threads
                        )
                    else:
                        cute.arch.barrier(
                            barrier_id=0, number_of_threads=self.barrier_threads
                        )
                        cute.arch.barrier(
                            barrier_id=1, number_of_threads=self.barrier_threads
                        )
        # The HMMA implementation below is a warp-level M=16 kernel.  Keep
        # one warp from each consumer warpgroup active for the math while the
        # pipeline still counts both consumer warps (WG0 + WG1).
        active_warps = 1 if cutlass.const_expr(self.query_tiles == 1) else 2
        if warpgroup_idx != 2 and local_warp_idx < active_warps and t < seq_len:
            query_tile = (
                0
                if cutlass.const_expr(self.query_tiles == 1)
                else warpgroup_idx * 2 + local_warp_idx
            )
            # (M, K)
            if cutlass.const_expr(self.query_tiles == 1):
                gQ = cute.local_tile(
                    mQ[None, None, t, KV_head_idx],
                    tiler=(
                        self.tile_shape_mnk_QK[0],
                        self.tile_shape_mnk_QK[2],
                    ),
                    coord=(0, None),
                )
            else:
                gQ_source = cute.domain_offset(
                    ((query_tile, 0), 0),
                    mQ[None, None, t, KV_head_idx],
                )
                gQ = cute.local_tile(
                    gQ_source,
                    tiler=(
                        self.tile_shape_mnk_QK[0],
                        self.tile_shape_mnk_QK[2],
                    ),
                    coord=(0, None),
                )
            # (n, K, loopK)
            gK = cute.local_tile(
                mK[None, None, KV_head_idx],
                tiler=(self.tile_shape_mnk_QK[1], self.tile_shape_mnk_QK[2]),
                coord=(None, 0),
            )
            # (K, n, loopK)
            gV = cute.local_tile(
                mV[None, None, KV_head_idx],
                tiler=(self.tile_shape_mnk_PV[2], self.tile_shape_mnk_PV[1]),
                coord=(None, 0),
            )
            # (M, n, loopK)
            if cutlass.const_expr(self.query_tiles == 1):
                gO = cute.local_tile(
                    mO[None, None, t, KV_head_idx],
                    tiler=(
                        self.tile_shape_mnk_PV[0],
                        self.tile_shape_mnk_PV[1],
                    ),
                    coord=(0, 0),
                )
            else:
                gO_source = cute.domain_offset(
                    ((query_tile, 0), 0),
                    mO[None, None, t, KV_head_idx],
                )
                gO = cute.local_tile(
                    gO_source,
                    tiler=(
                        self.tile_shape_mnk_PV[0],
                        self.tile_shape_mnk_PV[1],
                    ),
                    coord=(0, 0),
                )
            # (M, ) where M=64, block_size=128 not supported yet

            min_row = int(min(self.tile_shape_mnk_QK[0], self.MMA_group_size))
            if cutlass.const_expr(self.query_tiles == 1):
                gL = cute.local_tile(
                    mL[None, t, KV_head_idx],
                    tiler=(min_row,),
                    coord=(None,),
                )
                gM = cute.local_tile(
                    mM[None, t, KV_head_idx],
                    tiler=(min_row,),
                    coord=(None,),
                )
            else:
                gL_source = cute.domain_offset(
                    ((query_tile, 0),),
                    mL[None, t, KV_head_idx],
                )
                gM_source = cute.domain_offset(
                    ((query_tile, 0),),
                    mM[None, t, KV_head_idx],
                )
                gL = cute.local_tile(
                    gL_source,
                    tiler=(min_row,),
                    coord=(None,),
                )
                gM = cute.local_tile(
                    gM_source,
                    tiler=(min_row,),
                    coord=(None,),
                )

            thr_mma_QK = tiled_mma_QK.get_slice(tidx)

            q_cta_layout = cute.make_layout(
                cute.slice_(cta_layout_mnk, (0, None, 0)).shape
            )
            q_cta_crd = cluster_coord_mnk[1]
            sQ_for_tma_partition = cute.group_modes(sQ, 0, 2)
            gQ_for_tma_partition = cute.group_modes(gQ, 0, 2)

            tQsQ, tQgQ = cute.nvgpu.cpasync.tma_partition(
                tma_atom_Q,
                q_cta_crd,
                q_cta_layout,
                sQ_for_tma_partition,
                gQ_for_tma_partition,
            )

            tSrQ = thr_mma_QK.make_fragment_A(thr_mma_QK.partition_A(sQ))
            tSrK = thr_mma_QK.make_fragment_B(thr_mma_QK.partition_B(sK))
            tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
            tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
            tSsQ = smem_thr_copy_Q.partition_S(sQ)
            tSsK = smem_thr_copy_K.partition_S(sK)
            # import ipdb; ipdb.set_trace()

            acc_shape_QK = thr_mma_QK.partition_shape_C(
                (self.tile_shape_mnk_QK[0], self.tile_shape_mnk_QK[1])
            )
            acc_QK = cute.make_rmem_tensor(acc_shape_QK, self.acc_dtype)
            acc_QK.fill(0)

            # ********************
            # softmax intermediate result
            # ********************

            # shape:(mmaSahpeM * mma_m)
            row_max = cute.make_rmem_tensor(
                (acc_shape_QK[0][0] * acc_shape_QK[1]), cutlass.Float32
            )
            row_sum = cute.make_rmem_tensor(
                (acc_shape_QK[0][0] * acc_shape_QK[1]), cutlass.Float32
            )
            row_max.fill(-cutlass.Float32.inf)
            row_sum.fill(0.0)

            # ********************
            # prefetch TMA load
            # ********************
            K_tile_cnt = block_counts[offset + t, KV_head_idx]
            if warpgroup_idx == 0 and local_warp_idx == 0:
                for q_load_tile in cutlass.range_constexpr(0, self.query_tiles):
                    if cutlass.const_expr(self.query_tiles == 1):
                        tAgQ_k = tQgQ[(None, 0)]
                        tAsQ_pipe = tQsQ[(None, 0)]
                    else:
                        gQ_load = cute.local_tile(
                            cute.domain_offset(
                                (
                                    (q_load_tile, 0),
                                    0,
                                ),
                                mQ[None, None, t, KV_head_idx],
                            ),
                            tiler=(
                                self.tile_shape_mnk_QK[0],
                                self.tile_shape_mnk_QK[2],
                            ),
                            coord=(0, None),
                        )
                        gQ_load_for_tma = cute.group_modes(gQ_load, 0, 2)
                        tQsQ_load, tQgQ_load = cute.nvgpu.cpasync.tma_partition(
                            tma_atom_Q,
                            q_cta_crd,
                            q_cta_layout,
                            sQ_for_tma_partition,
                            gQ_load_for_tma,
                        )
                        tAgQ_k = tQgQ_load[(None, 0)]
                        tAsQ_pipe = tQsQ_load[(None, q_load_tile)]
                    q_handle = q_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_Q,
                        tAgQ_k,
                        tAsQ_pipe,
                        tma_bar_ptr=q_handle.barrier,
                    )
                    q_handle.commit()

            # tiled_mma_QK.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            num_K_blocks = cute.size(tSrQ, mode=[2])

            # ********************
            # mainloop
            # ********************

            thr_mma_PV = tiled_mma_PV.get_slice(mma_tidx)
            acc_shape_PV = thr_mma_PV.partition_shape_C(
                (self.tile_shape_mnk_PV[0], self.tile_shape_mnk_PV[1])
            )

            acc_PV = cute.make_rmem_tensor(acc_shape_PV, self.acc_dtype)
            acc_PV.fill(0)
            acc_QK_mn = self._make_acc_tensor_mn_view(acc_QK)
            acc_PV_mn = self._make_acc_tensor_mn_view(acc_PV)

            # predicates for GQA_group_size
            cLM = cute.make_identity_tensor((16, 1))
            cLM_thr = tiled_mma_QK.get_slice(mma_tidx).partition_C(cLM)
            cQK = cute.make_identity_tensor(
                (
                    self.tile_shape_mnk_QK[0],
                    self.tile_shape_mnk_QK[1],
                )
            )
            cQK_thr = tiled_mma_QK.get_slice(mma_tidx).partition_C(cQK)
            gL_thr = tiled_mma_QK.get_slice(mma_tidx).partition_C(gL)
            gM_thr = tiled_mma_QK.get_slice(mma_tidx).partition_C(gM)

            for q_load_tile in cutlass.range_constexpr(0, self.query_tiles):
                q_consumer_handle = q_consumer.wait_and_advance()
                if (
                    cutlass.const_expr(self.query_tiles == 1)
                    or q_load_tile == query_tile
                ):
                    for k in cutlass.range_constexpr(
                        0, cute.size(tSrQ, mode=[2])
                    ):
                        cute.copy(
                            smem_tiled_copy_Q,
                            tSsQ[None, None, k, q_consumer_handle.index],
                            tSrQ_copy_view[
                                None, None, k, q_consumer_handle.index
                            ],
                        )
                q_consumer_handle.release()

            row_topk = topk_lengths[offset + t, KV_head_idx]
            for K_tile in cutlass.range(0, K_tile_cnt, 1, unroll=1):
                cute.arch.barrier(
                    barrier_id=0, number_of_threads=self.barrier_threads
                )
                kv_stage = K_tile % self.K_stage
                acc_QK.fill(0)
                # cute.nvgpu.warpgroup.fence()
                # if tidx == 0:
                #     cute.printf("K_tile: %d, mainloop_consumer_read_state_K.index: %d\n", K_tile, mainloop_consumer_read_state_K.index)

                cute.copy(
                    smem_tiled_copy_K,
                    tSsK[None, None, 0, kv_stage],
                    tSrK_copy_view[None, None, 0, kv_stage],
                )

                for k in cutlass.range_constexpr(
                    0, cute.size(tSrQ, mode=[2]), unroll=True
                ):
                    if k < cute.size(tSrK, mode=[2]) - 1:
                        cute.copy(
                            smem_tiled_copy_K,
                            tSsK[None, None, k + 1, kv_stage],
                            tSrK_copy_view[None, None, k + 1, kv_stage],
                        )

                    cute.gemm(
                        tiled_mma_QK,
                        acc_QK,
                        tSrQ[None, None, k, 0],
                        tSrK[None, None, k, kv_stage],
                        acc_QK,
                    )

                # FlashMLA's token-level contract permits trailing -1 entries
                # and a shorter topk_length.  The producer replaces invalid
                # K/V rows with zero to avoid OOB loads; mask their logits to
                # -inf here so they contribute neither probability nor value.
                for i in cutlass.range_constexpr(cute.cosize(acc_QK)):
                    col = cQK_thr[i][1]
                    slot = K_tile * self.block_size + col
                    token = block_indices[
                        offset + t,
                        KV_head_idx,
                        slot,
                    ]
                    if slot >= row_topk or token < 0 or token >= kv_seq_len:
                        acc_QK[i] = -cutlass.Float32.inf

                if cutlass.const_expr(self.reuse_kv):
                    cute.arch.barrier(
                        barrier_id=2, number_of_threads=self.barrier_threads
                    )

                # ///////////////////////////////////////////////////////////////////////////////
                # softmax
                # ///////////////////////////////////////////////////////////////////////////////
                is_not_first_n_block = K_tile > 0
                row_max_prev = cute.make_fragment_like(row_max, cutlass.Float32)
                if is_not_first_n_block:  # not first n block
                    cute.basic_copy(row_max, row_max_prev)
                for r in cutlass.range_constexpr(cute.size(gL_thr.shape[0][1])):
                    if cute.elem_less(
                        cLM_thr[(0, r), 0, 0][0], self.MMA_group_size
                    ):
                        acc_QK_row = acc_QK_mn[r, None].load() * softmax_scale
                        row_max_cur_row = acc_QK_row.reduce(
                            cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
                        )
                        row_max_cur_row = self._threadquad_reduce_max(
                            row_max_cur_row, mask=(1 << self.MMA_group_size) - 1
                        )

                        row_max_prev_row = row_max_prev[r]
                        if is_not_first_n_block:
                            row_max_cur_row = cute.arch.fmax(
                                row_max_prev_row, row_max_cur_row
                            )

                        acc_QK_row_exp = cute.TensorSSA(  # e^{Sn-mn}
                            self._exp2f(
                                (acc_QK_row - row_max_cur_row) * self.log2_e
                            ),
                            tuple(acc_QK_row.shape),
                            cutlass.Float32,
                        )
                        acc_QK_row_sum = acc_QK_row_exp.reduce(
                            cute.ReductionOp.ADD, cutlass.Float32.zero, 0
                        )
                        acc_QK_row_sum = self._threadquad_reduce_sum(
                            acc_QK_row_sum, mask=(1 << self.MMA_group_size) - 1
                        )  # rowsum(e^{Sn-mn})
                        if is_not_first_n_block:
                            prev_minus_cur_exp = self._exp2f(
                                (row_max_prev_row - row_max_cur_row)
                                * self.log2_e
                            )  # e^{M^{(n-1)} - M^{(n)}}
                            # L^{(n)} = rowsum(e^{Sn-mn}) + L^{(n-1)} * e^{M^{(n-1)} - M^{(n)}}
                            acc_QK_row_sum = (
                                acc_QK_row_sum + row_sum[r] * prev_minus_cur_exp
                            )
                            # O^{(n-1)}' = O^{(n-1)} * e^{M^{(n-1)} - M^{(n)}}
                            acc_PV_mn[r, None] = (
                                acc_PV_mn[r, None].load() * prev_minus_cur_exp
                            )

                        row_max[r] = row_max_cur_row
                        row_sum[r] = acc_QK_row_sum
                        acc_QK_mn[r, None] = acc_QK_row_exp

                # ///////////////////////////////////////////////////////////////////////////////
                # p@V gemm calculation
                # ///////////////////////////////////////////////////////////////////////////////
                rP = cute.make_fragment_like(acc_QK, self.dtype)
                # rP.store(acc_QK.load().to(self.dtype))
                for i in cutlass.range_constexpr(cute.cosize(rP)):
                    rP[i] = self.dtype(acc_QK[i])

                if cutlass.const_expr(self.reuse_kv):
                    cute.arch.barrier(
                        barrier_id=1, number_of_threads=self.barrier_threads
                    )
                tOrVt = thr_mma_PV.make_fragment_B(thr_mma_PV.partition_B(sVt))
                tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)
                tOsVt = smem_thr_copy_V.partition_S(sVt)

                # convert rP from ((2, 2, 2*num_k_blocks_pv), 1, 1) to ((2, 2, 2), 1, 1, num_k_blocks_pv)
                num_k_blocks_pv = cute.size(tOrVt, mode=[2])
                rP_divided_dim3 = thr_mma_PV.partition_shape_A(
                    (self.tile_shape_mnk_PV[0], self.tile_shape_mnk_PV[2])
                )[0][2]

                rP_layout_divided = cute.logical_divide(
                    rP.layout, (None, None, rP_divided_dim3)
                )
                rP_mma_view = cute.make_layout(
                    (
                        (
                            rP_layout_divided.shape[0][0],
                            rP_layout_divided.shape[0][1],
                            rP_layout_divided.shape[2][0],
                        ),
                        rP_layout_divided.shape[1],
                        rP_layout_divided.shape[2][1],
                    ),
                    stride=(
                        (
                            rP_layout_divided.stride[0][0],
                            rP_layout_divided.stride[0][1],
                            rP_layout_divided.stride[2][0],
                        ),
                        rP_layout_divided.stride[1],
                        rP_layout_divided.stride[2][1],
                    ),
                )

                rP = cute.make_tensor(rP.iterator, rP_mma_view)

                cute.copy(
                    smem_tiled_copy_V,
                    tOsVt[None, None, 0, kv_stage],
                    tOrVt_copy_view[None, None, 0, kv_stage],
                )
                for k in cutlass.range_constexpr(0, cute.size(tOrVt, mode=[2])):
                    if k < cute.size(tOrVt, mode=[2]) - 1:
                        cute.copy(
                            smem_tiled_copy_V,
                            tOsVt[None, None, k + 1, kv_stage],
                            tOrVt_copy_view[None, None, k + 1, kv_stage],
                        )
                    cute.gemm(
                        tiled_mma_PV,
                        acc_PV,
                        rP[None, None, k],
                        tOrVt[None, None, k, kv_stage],
                        acc_PV,
                    )

                if cutlass.const_expr(self.reuse_kv):
                    cute.arch.barrier(
                        barrier_id=3, number_of_threads=self.barrier_threads
                    )
                else:
                    cute.arch.barrier(
                        barrier_id=1, number_of_threads=self.barrier_threads
                    )

            # write row_max and row_sum to global memory
            assert gL_thr.shape[0][1] == row_max.shape
            for row_idx in cutlass.range(gL_thr.shape[0][1]):
                if cute.elem_less(
                    cLM_thr[(0, row_idx), 0, 0][0], self.MMA_group_size
                ):
                    gM_thr[(0, row_idx), 0, 0] = row_max[row_idx]
                    sink = attn_sink[
                        KV_head_idx * self.GQA_group_size
                        + query_tile * self.MMA_group_size
                        + cLM_thr[(0, row_idx), 0, 0][0]
                    ]
                    sink_mass = self._exp2f(
                        (sink - row_max[row_idx]) * self.log2_e
                    )
                    gL_thr[(0, row_idx), 0, 0] = (
                        row_sum[row_idx] + sink_mass
                        if row_sum[row_idx] > 0.0
                        else 0.0
                    )

            # ********************
            # epilogue
            # ********************
            # softmax normalization: O^{(n)} = O^{(n)} / L^{(n)}
            for row_idx in cutlass.range(gL_thr.shape[0][1]):
                if cute.elem_less(
                    cLM_thr[(0, row_idx), 0, 0][0], self.MMA_group_size
                ):
                    acc_pv_mn_is_zero_or_nan = (
                        row_sum[row_idx] == 0.0
                        or row_sum[row_idx] != row_sum[row_idx]
                    )
                    scale = cutlass.Float32(1.0)
                    if acc_pv_mn_is_zero_or_nan:
                        scale = cutlass.Float32(1.0)
                    else:
                        sink = attn_sink[
                            KV_head_idx * self.GQA_group_size
                            + query_tile * self.MMA_group_size
                            + cLM_thr[(0, row_idx), 0, 0][0]
                        ]
                        sink_mass = self._exp2f(
                            (sink - row_max[row_idx]) * self.log2_e
                        )
                        scale = cute.arch.rcp_approx(
                            row_sum[row_idx] + sink_mass
                        )
                    acc_PV_mn[row_idx, None] = (
                        acc_PV_mn[row_idx, None].load() * scale
                    )

            tOgO_for_tma_partition = cute.zipped_divide(
                gO,
                (self.epi_tile[0], self.epi_tile[1]),
            )
            if cutlass.const_expr(self.query_tiles == 1):
                sO_query = sO
            else:
                sO_query = cute.slice_(sO, (None, None, None, query_tile))
            if (
                warpgroup_idx == 0
                if cutlass.const_expr(self.query_tiles == 1)
                else local_warp_idx < active_warps
            ):
                self.copy_reg_to_gmem(
                    self.O_layout,
                    self.O_dtype,
                    tOgO_for_tma_partition,
                    tma_atom_O,
                    tiled_mma_PV,
                    acc_PV,
                    self.acc_dtype,
                    sO_query,
                    mma_tidx,
                    warp_idx,
                )

    @cute.jit
    def copy_reg_to_gmem(
        self,
        dest_layout,
        dest_dtype,
        gmem_tensor_partition,
        tma_atom,
        tiled_mma,
        acc_tensor,
        acc_dtype,
        smem_tensor,
        tidx,
        warp_idx,
    ):
        cute.arch.fence_proxy(
            "async.shared",
            space="cta",
        )
        cute.arch.sync_warp()

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            dest_layout,
            elem_ty_d=dest_dtype,
            elem_ty_acc=acc_dtype,  # useless in sm90_get_smem_store_op
        )
        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                transpose=dest_layout.is_m_major_c(),
                num_matrices=4,
            ),
            dest_dtype,
        )
        tiled_copy_atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_atom)

        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_dv_sD = thr_copy_r2s.partition_D(smem_tensor)
        tRS_dv_rAcc = thr_copy_r2s.retile(acc_tensor)

        # Allocate D registers.
        rD_shape = cute.shape(thr_copy_r2s.partition_S(smem_tensor))
        tRS_rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_fragment_like(tRS_rD_layout, acc_dtype)
        size_tRS_rD = cute.size(tRS_rD)

        sepi_for_tma_partition = cute.group_modes(smem_tensor, 0, 2)

        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            sepi_for_tma_partition,
            gmem_tensor_partition,
        )
        epi_tile_num = cute.size(gmem_tensor_partition, mode=[1])
        epi_tile_shape = gmem_tensor_partition.shape[1]

        for epi_idx in cutlass.range(epi_tile_num, unroll=epi_tile_num):
            for epi_v in cutlass.range_constexpr(size_tRS_rD, unroll_full=True):
                tRS_rD[epi_v] = tRS_dv_rAcc[epi_idx * size_tRS_rD + epi_v]

            tRS_rD_out = cute.make_fragment_like(tRS_rD_layout, dest_dtype)

            # tRS_rD_out.store(tRS_rD.load().to(dest_dtype))
            for i in cutlass.range_constexpr(
                cute.cosize(tRS_rD_out), unroll_full=True
            ):
                tRS_rD_out[i] = dest_dtype(tRS_rD[i])

            # Copy from D registers to shared memory
            epi_buffer = epi_idx % cute.size(tRS_dv_sD, mode=[3])
            cute.copy(
                tiled_copy_r2s,
                tRS_rD_out,
                tRS_dv_sD[(None, None, None, epi_buffer)],
            )

            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            # barrier for sync
            cute.arch.sync_warp()

            epi_tile_layout = cute.make_layout(
                epi_tile_shape, stride=(epi_tile_shape[1], 1)
            )
            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
            # Copy from shared memory to global memory
            if warp_idx == 0:
                cute.copy(
                    tma_atom,
                    bSG_sD[(None, epi_buffer)],
                    bSG_gD[(None, gmem_coord)],
                )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(
                    self.epi_stage - 1, read=True
                )

            cute.arch.sync_warp()
