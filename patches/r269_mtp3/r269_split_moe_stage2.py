from __future__ import annotations

import os
from typing import Any

import torch

_INSTALLED = False
_HIT_LOGGED = False

# Stage-1 remains on accepted R265 common config. Only the second routed-MoE
# GEMM (M=4 verifier: 32 routed rows, K=512 -> N=2048) receives this config.
# Accepted R269 release config after full-model A/B and strict exactness gates.
_STAGE2_CONFIG: dict[str, Any] = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 4,
    "num_stages": 1,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16,
    "kpack": 2,
}


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if os.getenv("K100_R269_SPLIT_MOE_STAGE2", "0") != "1":
        return

    import vllm.model_executor.layers.fused_moe.fused_moe as fm

    if getattr(fm, "_k100_r269_split_moe_stage2", False):
        _INSTALLED = True
        return

    original = fm.dispatch_fused_moe_kernel

    def dispatch_r269(*args, **kwargs):
        global _HIT_LOGGED
        # dispatch signature starts with A, B, C. For the second routed-MoE
        # GEMM under TP1/MTP3 target verification:
        #   A: [M*topk, intermediate] = [32, 512] INT8
        #   B: [E, hidden, intermediate] = [256, 2048, 512] INT8
        # top_k passed to this second dispatch is 1. Keep every other path on
        # the exact original config.
        try:
            A = args[0] if len(args) > 0 else kwargs["A"]
            B = args[1] if len(args) > 1 else kwargs["B"]
            top_k = args[11] if len(args) > 11 else kwargs["top_k"]
            candidate = (
                A.dtype is torch.int8
                and B.dtype is torch.int8
                and A.ndim == 2
                and B.ndim == 3
                and tuple(int(v) for v in A.shape) == (32, 512)
                and tuple(int(v) for v in B.shape) == (256, 2048, 512)
                and int(top_k) == 1
            )
        except Exception:
            candidate = False

        if not candidate:
            return original(*args, **kwargs)

        # BLOCK_SIZE_M must remain 32 because sorted-token metadata was built
        # once using the R265 common BM32 before both GEMMs.
        if len(args) > 12:
            aa = list(args)
            aa[12] = _STAGE2_CONFIG
            out = original(*aa, **kwargs)
        else:
            kw = dict(kwargs)
            kw["config"] = _STAGE2_CONFIG
            out = original(*args, **kw)

        if not _HIT_LOGGED:
            _HIT_LOGGED = True
            print(
                "[K100 R269 split MoE stage2] M4 W2 config "
                "BM32/BN32/BK512/w4/waves2/kpack2/stages1 active",
                flush=True,
            )
        return out

    fm.dispatch_fused_moe_kernel = dispatch_r269
    fm._k100_r269_split_moe_stage2 = True
    _INSTALLED = True
    print(
        "[K100 R269 split MoE stage2] exact M4 second-GEMM override installed",
        flush=True,
    )
