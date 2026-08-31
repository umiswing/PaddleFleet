# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

"""PaddleFleet-owned SM90 CuTe DSL operators."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

_LAZY_SYMBOLS = {
    "dsa_sparse_vha_score_cutedsl": ".two_stage_indexer",
    "dsa_sparse_vha_topk_two_stage_cutedsl": ".two_stage_indexer",
    "hca_stage2_score_cutedsl": ".hca_stage2",
    "native_mla_cutedsl_fwd": ".native_mla_fwd",
    "flashmla_sm90_sparse_prefill": ".flashmla_sm90",
    "flashmla_sm90_sparse_prefill_wgmma": ".flashmla_sm90_wgmma",
}

__all__ = [
    "is_cutedsl_available",
    "dsa_sparse_vha_score_cutedsl",
    "dsa_sparse_vha_topk_two_stage_cutedsl",
    "hca_stage2_score_cutedsl",
    "native_mla_cutedsl_fwd",
    "flashmla_sm90_sparse_prefill",
    "flashmla_sm90_sparse_prefill_wgmma",
]


def is_cutedsl_available() -> bool:
    """Return whether the optional CuTe DSL runtime can be loaded."""
    try:
        import paddle

        return (
            paddle.device.is_compiled_with_cuda()
            and find_spec("cuda") is not None
            and find_spec("cutlass") is not None
        )
    except (ImportError, ModuleNotFoundError, RuntimeError):
        return False


def __getattr__(name: str):
    module_name = _LAZY_SYMBOLS.get(name)
    if module_name is not None:
        module = import_module(module_name, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
