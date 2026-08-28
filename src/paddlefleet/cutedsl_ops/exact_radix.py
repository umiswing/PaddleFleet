# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# The radix-threshold algorithm in this file was adapted from NVIDIA's
# cuDNN Frontend indexer_top_k utility (Apache-2.0).  This implementation is
# independently integrated for PaddleFleet and does not import vendored code.

"""Exact FP32 radix selection primitives for fused Sparse VHA routing.

The selector uses all four bytes of an order-preserving uint32 encoding.  It
never performs insertion selection and its storage is O(bucket), not O(T^2).
The final emission scan is deliberately index ordered so equal-score handling
is deterministic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cutlass
from cutlass import Float32, Int32, Uint32, cute
from cutlass._mlir.dialects import llvm
from cutlass.utils.distributed import atomicAdd

if TYPE_CHECKING:
    import cuda.bindings.driver as cuda


@cute.jit
def fp32_to_sortable_uint32(value: Float32):
    """Map a finite FP32 value to an ascending, unsigned integer key."""
    bits = Uint32(llvm.bitcast(Uint32.mlir_type, value.ir_value()))
    key = bits ^ Uint32(0x80000000)
    if bits & Uint32(0x80000000):
        key = bits ^ Uint32(0xFFFFFFFF)
    return key


@cute.jit
def fp32_to_sortable_uint16(value: Float32):
    """Map a finite FP32 value to the high 16 bits of its sortable key.

    This is the quantized key used by the two-stage indexer top-k kernel: it
    halves the shared-memory radix cache and needs only two 8-bit passes at the
    cost of approximate (recall ~0.998) candidate alignment near ties.
    """
    return fp32_to_sortable_uint32(value) >> Uint32(16)


class ExactRadixSelector:
    """Four-pass, 8-bit exact radix threshold and stable index emitter.

    ``select`` is intended to be inlined into a routing top-k kernel. ``keys``
    is a CTA-local shared-memory uint32 cache. Scratch tensors must also live in
    shared memory and have lengths 256 (histogram/flags) and 4 (state).
    """

    RADIX = 256
    THREADS = 256

    def __init__(self, bucket_size: int, padded_topm: int):
        if bucket_size not in (8192, 32768):
            raise ValueError(f"unsupported radix bucket {bucket_size}")
        if padded_topm <= 0 or padded_topm > bucket_size:
            raise ValueError(f"invalid padded_topm {padded_topm}")
        if padded_topm % self.THREADS != 0:
            raise ValueError(
                f"padded_topm must be a multiple of {self.THREADS}"
            )
        self.bucket_size = bucket_size
        self.padded_topm = padded_topm

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

        # Early and radix rows are predicated and reconverge; no CTA exits before
        # a barrier. Each thread owns a compile-time fixed strided subset.
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
            if tid == Int32(0):
                state[2] = requested_topm
            cute.arch.sync_threads()

            # Select the requested_topm-th largest key one byte at a time.
            for pass_id in cutlass.range_constexpr(4):
                histogram[tid] = Int32(0)
                cute.arch.sync_threads()
                shift = 24 - pass_id * 8
                prefix = Uint32(state[0])
                prefix_mask = Uint32(state[1])
                item = Int32(0)
                while item < Int32(self.bucket_size // self.THREADS):
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

            # Stable chunked compaction preserves increasing local index.
            lane_idx = Int32(cute.arch.lane_idx())
            warp_idx = tid // Int32(32)
            lower_lane_mask = (Uint32(1) << lane_idx) - Uint32(1)
            chunk = Int32(0)
            while chunk < Int32(self.bucket_size // self.THREADS):
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


class _ExactRadixJitHarness:
    """File-backed harness used to compile-test the inlined selector."""

    def __init__(self, padded_topm: int):
        self.selector = ExactRadixSelector(8192, padded_topm)

    @cute.jit
    def __call__(
        self,
        mValues: cute.Tensor,
        mOutput: cute.Tensor,
        requested_topm: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(mValues, mOutput, requested_topm).launch(
            grid=(1, 1, 1), block=(self.selector.THREADS, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, mValues, mOutput, requested_topm: Int32):
        @cute.struct
        class SharedStorage:
            keys: cute.struct.Align[cute.struct.MemRange[Uint32, 8192], 16]
            histogram: cute.struct.MemRange[Int32, 256]
            flags: cute.struct.MemRange[Int32, 256]
            state: cute.struct.MemRange[Int32, 4]

        storage = cutlass.utils.SmemAllocator().allocate(SharedStorage)
        keys = storage.keys.get_tensor(cute.make_layout((8192,), stride=(1,)))
        histogram = storage.histogram.get_tensor(
            cute.make_layout((256,), stride=(1,))
        )
        flags = storage.flags.get_tensor(cute.make_layout((256,), stride=(1,)))
        state = storage.state.get_tensor(cute.make_layout((4,), stride=(1,)))
        tid = Int32(cute.arch.thread_idx()[0])
        for item in cutlass.range_constexpr(8192 // 256):
            col = tid + Int32(item * 256)
            keys[col] = fp32_to_sortable_uint32(Float32(mValues[col]))
        cute.arch.sync_threads()
        self.selector.select(
            keys,
            mOutput,
            Int32(8192),
            requested_topm,
            histogram,
            flags,
            state,
        )


__all__ = ["ExactRadixSelector", "fp32_to_sortable_uint32"]
