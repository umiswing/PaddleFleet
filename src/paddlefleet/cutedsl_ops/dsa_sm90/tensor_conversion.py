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

"""Paddle-to-CuTe tensor conversion for the local SM90 DSA kernels."""

from ..dlpack import paddle_to_cute_tensor


def _dim_order(t):
    if hasattr(t, "dim_order"):
        return t.dim_order()
    return tuple(
        i
        for i, _ in sorted(
            enumerate(tuple(t.stride())),
            key=lambda item: item[1],
            reverse=True,
        )
    )


def to_cute_tensor(
    t,
    assumed_align: int = 16,
    leading_dim: int = -1,
    fully_dynamic: bool = False,
    enable_tvm_ffi: bool = True,
    divisibility=None,
):
    """Convert a Paddle tensor to CuTe through the DLPack protocol."""
    tensor = paddle_to_cute_tensor(
        t,
        assumed_align=assumed_align,
        leading_dim=None if leading_dim == -1 else leading_dim,
    )
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    tensor = tensor.mark_layout_dynamic(leading_dim=leading_dim)
    if divisibility is not None:
        tensor = tensor.mark_compact_shape_dynamic(
            mode=leading_dim,
            stride_order=_dim_order(t),
            divisibility=divisibility,
        )
    return tensor
