from __future__ import annotations

import os
import torch
import triton
import triton.language as tl

_INSTALLED = False


@triton.jit
def _r224_round_int8_rocm(x):
    # Match vLLM ROCm float_to_int8_rn exactly: std::nearbyint under
    # FE_TONEAREST (ties-to-even), then convert to int8. Dynamic symmetric
    # quant guarantees the rounded range is within [-127, 127].
    return tl.extra.hip.libdevice.nearbyint(x).to(tl.int8)


@triton.jit
def _r223_dynamic_int8_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    stride_xm,
    stride_qm,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    x = tl.load(
        x_ptr + row * stride_xm + offs,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    # Match vLLM csrc/quantization/w8a8/int8/scaled_quant exactly:
    #   scale = absmax / 127.f
    #   inv_s = (absmax == 0.f) ? 0.f : 127.f / absmax
    #   nearbyint(src * inv_s)
    absmax = tl.max(tl.where(mask, tl.abs(x), 0.0))
    scale = absmax / 127.0
    inv_s = tl.where(absmax == 0.0, 0.0, 127.0 / absmax)
    q = _r224_round_int8_rocm(x * inv_s)
    tl.store(q_ptr + row * stride_qm + offs, q, mask=mask)
    tl.store(scale_ptr + row, scale)


# R224 deliberately targets only the two hot speculative-decode row counts:
# M=1 for serial draft forwards and M=4 for the target verifier. M=2/3 and
# prefill stay on the stock vLLM kernel to minimize the numerical/routing blast
# radius. Warps are conservative winners from repeated gfx928 CUDAGraph tests.
_R224_QUANT_WARPS = {
    (1, 512): 8,
    (4, 512): 2,
    (1, 1024): 2,
    (4, 1024): 4,
    (1, 2048): 1,
    (4, 2048): 1,
    (1, 4096): 2,
    (4, 4096): 8,
}


def _register_ops() -> None:
    # Functional wrappers make the dynamic INT8 quant node easy for the vLLM
    # Inductor pattern matcher to recognize. They preserve the exact stock _C
    # kernels used by the current HCU build.
    try:
        torch.ops.k100.r210_dynamic_int8_quant.default
    except (AttributeError, RuntimeError):
        @torch.library.custom_op(
            "k100::r210_dynamic_int8_quant",
            mutates_args=(),
            device_types="cuda",
        )
        def _r210_dynamic_int8_quant(
            input: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = input.contiguous()
            result = torch.empty_like(x, dtype=torch.int8)
            scale = torch.empty(
                (x.numel() // x.shape[-1], 1),
                device=x.device,
                dtype=torch.float32,
            )
            rows = int(x.numel() // x.shape[-1])
            k = int(x.shape[-1])
            warps = _R224_QUANT_WARPS.get((rows, k))
            if x.dtype == torch.bfloat16 and warps is not None:
                q2 = result.view(rows, k)
                s2 = scale.view(rows, 1)
                block = triton.next_power_of_2(k)
                _r223_dynamic_int8_kernel[(rows,)](
                    x.view(rows, k),
                    q2,
                    s2,
                    x.stride(-2) if x.dim() >= 2 else k,
                    q2.stride(0),
                    K=k,
                    BLOCK=block,
                    num_warps=warps,
                    num_stages=1,
                )
            else:
                torch.ops._C.dynamic_scaled_int8_quant(result, x, scale, None)
            return result.view_as(input), scale.view(*input.shape[:-1], 1)

        @_r210_dynamic_int8_quant.register_fake
        def _r210_dynamic_int8_quant_fake(
            input: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.empty_like(input, dtype=torch.int8),
                torch.empty(
                    (*input.shape[:-1], 1),
                    device=input.device,
                    dtype=torch.float32,
                ),
            )

    try:
        torch.ops.k100.r210_rmsnorm_int8_quant.default
    except (AttributeError, RuntimeError):
        @torch.library.custom_op(
            "k100::r210_rmsnorm_int8_quant",
            mutates_args=(),
            device_types="cuda",
        )
        def _r210_rmsnorm_int8_quant(
            input: torch.Tensor,
            weight: torch.Tensor,
            epsilon: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = input.contiguous()
            result = torch.empty_like(x, dtype=torch.int8)
            scale = torch.empty(
                (x.numel() // x.shape[-1], 1),
                device=x.device,
                dtype=torch.float32,
            )
            torch.ops._C.rms_norm_dynamic_per_token_quant(
                result,
                x,
                weight,
                scale,
                epsilon,
                None,
                None,
            )
            return result.view_as(input), scale.view(*input.shape[:-1], 1)

        @_r210_rmsnorm_int8_quant.register_fake
        def _r210_rmsnorm_int8_quant_fake(
            input: torch.Tensor,
            weight: torch.Tensor,
            epsilon: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del weight, epsilon
            return (
                torch.empty_like(input, dtype=torch.int8),
                torch.empty(
                    (*input.shape[:-1], 1),
                    device=input.device,
                    dtype=torch.float32,
                ),
            )


def _patch_dynamic_quant_wrapper() -> None:
    from vllm import _custom_ops as ops

    if getattr(ops, "_k100_r210_dynamic_int8", False):
        return
    original = ops.scaled_int8_quant

    def _scaled_int8_quant(
        input: torch.Tensor,
        scale: torch.Tensor | None = None,
        azp: torch.Tensor | None = None,
        symmetric: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if scale is None and azp is None and symmetric:
            result, result_scale = torch.ops.k100.r210_dynamic_int8_quant(input)
            return result, result_scale, None
        return original(input, scale, azp, symmetric)

    ops.scaled_int8_quant = _scaled_int8_quant
    ops._k100_r210_dynamic_int8 = True


def _patch_fusion_pass() -> None:
    import torch._inductor.pattern_matcher as pm
    from torch import fx
    from torch._inductor.pattern_matcher import PatternMatcherPass

    from vllm.compilation.passes.fusion.matcher_utils import MatcherRMSNorm
    from vllm.compilation.passes.fusion.rms_quant_fusion import RMSNormQuantFusionPass
    from vllm.compilation.passes import pass_manager as pass_manager_mod

    if getattr(RMSNormQuantFusionPass, "_k100_r210_int8", False):
        return

    class _R210RMSNormINT8Pattern:
        def __init__(self, epsilon: float, model_dtype: torch.dtype) -> None:
            self.epsilon = epsilon
            self.model_dtype = model_dtype
            self.rmsnorm_matcher = MatcherRMSNorm(epsilon)

        def register(self, pm_pass: PatternMatcherPass) -> None:
            def pattern(
                input: torch.Tensor,
                weight: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                result_rms = self.rmsnorm_matcher(input, weight)
                return torch.ops.k100.r210_dynamic_int8_quant(result_rms)

            def replacement(
                input: torch.Tensor,
                weight: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                input = input.to(dtype=self.model_dtype)
                return torch.ops.k100.r210_rmsnorm_int8_quant(
                    input, weight, self.epsilon
                )

            pm.register_replacement(
                pattern,
                replacement,
                self.rmsnorm_matcher.inputs(),
                pm.fwd_only,
                pm_pass,
            )

    original_init = RMSNormQuantFusionPass.__init__
    original_uuid = RMSNormQuantFusionPass.uuid

    def _r210_init(self, config) -> None:
        original_init(self, config)
        model_dtype = config.model_config.dtype
        for epsilon in (1e-5, 1e-6):
            _R210RMSNormINT8Pattern(epsilon, model_dtype).register(self.patterns)
        print(
            "[K100 R210 RMS+INT8] registered exact native RMSNorm->INT8 patterns",
            flush=True,
        )

    def _r210_uuid(self) -> str:
        return original_uuid(self) + "-k100-r210-native-int8-v1"

    RMSNormQuantFusionPass.__init__ = _r210_init
    RMSNormQuantFusionPass.uuid = _r210_uuid
    RMSNormQuantFusionPass._k100_r210_int8 = True

    # PassManager holds the same class object, but assign explicitly as a guard
    # against HCU builds that imported an alias during module initialization.
    pass_manager_mod.RMSNormQuantFusionPass = RMSNormQuantFusionPass

    original_configure = pass_manager_mod.PostGradPassManager.configure

    def _r210_configure(self, config) -> None:
        config.compilation_config.pass_config.fuse_norm_quant = True
        return original_configure(self, config)

    pass_manager_mod.PostGradPassManager.configure = _r210_configure


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if os.getenv("K100_RMSNORM_INT8_FUSION", "0") != "1":
        return
    _register_ops()
    _patch_dynamic_quant_wrapper()
    _patch_fusion_pass()
    _INSTALLED = True
    print(
        "[K100 R224 nearbyint dynamic INT8] R210 RMS fusion + bitwise-exact Triton quant hooks installed",
        flush=True,
    )
