# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

"""Zero-copy Paddle to CuTe DSL tensor conversion through DLPack."""

from __future__ import annotations


class _PaddleDLPackAdapter:
    """Expose Paddle's capsule API through the Python DLPack protocol."""

    def __init__(self, tensor):
        self._tensor = tensor

    def __dlpack__(self, stream=None):
        import paddle

        return paddle.utils.dlpack.to_dlpack(self._tensor)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


def current_paddle_stream_ptr() -> int:
    """Return the raw pointer of Paddle's current CUDA stream."""
    import paddle

    stream = paddle.device.current_stream()
    base = getattr(stream, "stream_base", stream)
    for name in ("cuda_stream", "raw_stream"):
        value = getattr(base, name, None)
        if value is not None:
            return int(value)
    raise RuntimeError("cannot obtain the current Paddle CUDA stream pointer")


def current_cu_stream():
    """Return Paddle's current stream as ``cuda.bindings.driver.CUstream``."""
    import cuda.bindings.driver as cuda

    return cuda.CUstream(current_paddle_stream_ptr())


def paddle_to_cute_tensor(
    tensor,
    *,
    assumed_align: int,
    leading_dim: int | None = None,
    stream_ptr: int | None = None,
    enable_tvm_ffi: bool = False,
):
    """Create a dynamic-layout CuTe tensor sharing a Paddle CUDA allocation.

    No copy is made.  Callers should use 16-byte alignment for contiguous Q/KV
    BF16 tensors and 4-byte alignment for contiguous int32 metadata/output.
    """
    from cutlass.cute.runtime import from_dlpack

    # CuTe DSL expects an object implementing the Python DLPack protocol, while
    # Paddle's explicit API returns a raw capsule. The adapter bridges those APIs
    # without copying and deliberately leaves launch-stream selection to TVM FFI.
    del stream_ptr
    result = from_dlpack(
        _PaddleDLPackAdapter(tensor),
        assumed_align=assumed_align,
        enable_tvm_ffi=enable_tvm_ffi,
    )
    if leading_dim is None:
        return result.mark_layout_dynamic()
    return result.mark_layout_dynamic(leading_dim=leading_dim)


__all__ = [
    "current_cu_stream",
    "current_paddle_stream_ptr",
    "paddle_to_cute_tensor",
]
