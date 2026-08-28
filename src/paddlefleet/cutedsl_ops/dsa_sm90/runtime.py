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

"""Paddle-native runtime helpers for local DSA CuTeDSL kernels."""

from functools import cache

import cuda.bindings.driver as cuda
import paddle

from ..dlpack import current_cu_stream


@cache
def device_major() -> int:
    return paddle.device.cuda.get_device_capability()[0]


def maybe_contiguous(
    x: paddle.Tensor | None,
    stream: cuda.CUstream | None = None,
) -> paddle.Tensor | None:
    if x is None or x.stride(-1) == 1:
        return x
    return x.contiguous()


def validate_q_causal_offsets(
    q_causal_offsets: paddle.Tensor | None,
    batch: int,
    device,
    stream: cuda.CUstream | None = None,
) -> paddle.Tensor | None:
    if q_causal_offsets is None:
        return None
    if q_causal_offsets.dtype != paddle.int32:
        raise ValueError("q_causal_offsets must be int32")
    if q_causal_offsets.ndim != 1 or q_causal_offsets.shape[0] != batch:
        raise ValueError(
            f"q_causal_offsets must have shape ({batch},), got {tuple(q_causal_offsets.shape)}"
        )
    if not q_causal_offsets.place.is_gpu_place():
        raise ValueError("q_causal_offsets must be a CUDA tensor")
    if q_causal_offsets.device != device:
        raise ValueError("q_causal_offsets must be on the same device as q")
    if q_causal_offsets.is_contiguous():
        return q_causal_offsets
    return q_causal_offsets.contiguous()


def resolve_stream(
    current_stream: cuda.CUstream | None = None,
) -> cuda.CUstream:
    if current_stream is not None:
        return current_stream
    return current_cu_stream()
