"""Public runtime patch validated on Hygon K100 AI (gfx928) only.

K100 and K100 AI are different accelerator models. This code does not claim
compatibility or performance parity with K100.

K100 AI/gfx928 runtime optimization patch for Qwen3.6-35B-A3B W8A8.

This module is loaded through PYTHONPATH/sitecustomize. It keeps the stock
compressed-tensors loader and all non-MoE kernels unchanged. For decode-sized
batches it replaces only routed-expert execution with a gfx928-native INT8
AITER kernel. Prefill and unsupported layouts fall back to upstream vLLM.
"""
from __future__ import annotations

import importlib
import json
import os
import signal
from collections import Counter
from pathlib import Path
from typing import Final

import torch


_LM_HEAD_ENABLED: Final[bool] = os.getenv("K100AI_LM_HEAD_W8A8", "0") == "1"
_LM_HEAD_MAX_M: Final[int] = int(os.getenv("K100AI_LM_HEAD_W8A8_MAX_M", "2"))

if _LM_HEAD_ENABLED:
    import time as _time
    from aiter.ops.triton.gemm_a8w8 import gemm_a8w8 as _lm_gemm_a8w8
    from vllm import _custom_ops as _lm_ops
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase as _QuantizeMethodBase
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import triton_scaled_mm as _lm_stock_scaled_mm
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead as _ParallelLMHead, UnquantizedEmbeddingMethod as _UnquantizedEmbeddingMethod

    _LM_HEAD_CONFIGS = {
        1: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2,
            "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "kpack": 1,
        },
        2: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2,
            "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "kpack": 2,
        },
        3: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2,
            "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "kpack": 1,
        },
        4: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2,
            "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 1,
        },
        5: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2,
            "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "kpack": 1,
        },
    }

    @torch.library.custom_op(
        "k100ai::lm_head_w8a8", mutates_args=(), device_types="cuda"
    )
    def _k100aiai_lm_head_w8a8(
        x: torch.Tensor, weight_nk: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        original_shape = tuple(x.shape[:-1])
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        xq, xs, xzp = _lm_ops.scaled_int8_quant(
            x2, None, None, symmetric=True
        )
        assert xzp is None
        m = int(x2.shape[0])
        config = _LM_HEAD_CONFIGS.get(m) if m <= _LM_HEAD_MAX_M else None
        if config is not None:
            out = _lm_gemm_a8w8(
                xq, weight_nk, xs, weight_scale, None, torch.bfloat16,
                config=config,
            )
        else:
            out = _lm_stock_scaled_mm(
                xq, weight_nk.t(), xs, weight_scale, torch.bfloat16
            )
        return out.reshape(*original_shape, weight_nk.shape[0])

    @_k100aiai_lm_head_w8a8.register_fake
    def _k100aiai_lm_head_w8a8_fake(
        x: torch.Tensor, weight_nk: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        del weight_scale
        return x.new_empty((*x.shape[:-1], weight_nk.shape[0]), dtype=torch.bfloat16)

    class _K100AILMHeadW8A8Method(_QuantizeMethodBase):
        def create_weights(self, *args, **kwargs):
            raise RuntimeError("K100 AI lm_head method is installed after weight loading")

        def apply(
            self, layer: torch.nn.Module, x: torch.Tensor,
            bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            if bias is not None:
                return _k100aiai_lm_head_w8a8(
                    x, layer.weight, layer.k100ai_weight_scale
                ) + bias
            return _k100aiai_lm_head_w8a8(
                x, layer.weight, layer.k100ai_weight_scale
            )

    _orig_unquantized_embedding_process = (
        _UnquantizedEmbeddingMethod.process_weights_after_loading
    )

    def _k100aiai_process_unquantized_embedding(self, layer: torch.nn.Module) -> None:
        _orig_unquantized_embedding_process(self, layer)
        if not isinstance(layer, _ParallelLMHead):
            return
        if int(getattr(layer, "tp_size", 1)) != 1:
            return
        weight = layer.weight.data
        if weight.dtype not in (torch.bfloat16, torch.float16):
            return
        if weight.ndim != 2 or int(weight.shape[1]) != 2048:
            return
        started = _time.perf_counter()
        with torch.no_grad():
            scale = (
                weight.float().abs().amax(dim=1).clamp_min(1e-12).div_(127.0)
            )
            qweight = torch.round(weight.float() / scale[:, None]).clamp_(
                -127, 127
            ).to(torch.int8)
        layer.weight = torch.nn.Parameter(qweight, requires_grad=False)
        layer.register_buffer(
            "k100ai_weight_scale", scale[:, None].contiguous(), persistent=False
        )
        layer.quant_method = _K100AILMHeadW8A8Method()
        del weight, qweight, scale
        torch.cuda.empty_cache()
        print(
            "[K100 AI lm_head W8A8] enabled "
            f"shape={tuple(layer.weight.shape)} "
            f"quantize_s={_time.perf_counter() - started:.3f}",
            flush=True,
        )

    _UnquantizedEmbeddingMethod.process_weights_after_loading = (
        _k100aiai_process_unquantized_embedding
    )
    print(
        f"[K100 AI lm_head W8A8] hook installed, runtime M <= {_LM_HEAD_MAX_M}",
        flush=True,
    )

_LINEAR_ENABLED: Final[bool] = os.getenv("K100AI_SC_LINEAR_A8W8", "0") == "1"
_LINEAR_MAX_M: Final[int] = int(os.getenv("K100AI_SC_LINEAR_A8W8_MAX_M", "1"))
_FUSED_QKVZ_BA: Final[bool] = os.getenv("K100AI_FUSED_QKVZ_BA", "0") == "1"
_LOG_LINEAR_LAYOUT: Final[bool] = os.getenv("K100AI_LOG_LINEAR_LAYOUT", "0") == "1"
_TRACE_LINEAR_SHAPES: Final[bool] = os.getenv("K100AI_TRACE_LINEAR_SHAPES", "0") == "1"

# Eager-only shape probe used to recover the real M distribution produced by
# MTP verification. SIGUSR1 dumps aggregate (M,K,N) call counts to container
# logs. It is deliberately disabled whenever the optimized Linear patch is on.
if _TRACE_LINEAR_SHAPES and not _LINEAR_ENABLED:
    _trace_ct_mod = importlib.import_module(
        "vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm"
    )
    _trace_linear_mod = importlib.import_module(
        "vllm.model_executor.kernels.linear.scaled_mm.triton"
    )
    _trace_orig_scaled_mm = _trace_ct_mod.triton_scaled_mm
    _trace_counts: Counter[tuple[int, int, int]] = Counter()
    _trace_total = 0
    _trace_out = os.getenv("K100AI_TRACE_LINEAR_SHAPES_OUT", "")
    _trace_flush_every = max(
        1, int(os.getenv("K100AI_TRACE_LINEAR_SHAPES_FLUSH_EVERY", "100"))
    )

    def _trace_rows() -> list[dict[str, int]]:
        return [
            {"m": m, "k": k, "n": n, "calls": calls}
            for (m, k, n), calls in sorted(
                _trace_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]

    def _write_trace_counts() -> None:
        if not _trace_out:
            return
        path = Path(_trace_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"total_calls": _trace_total, "rows": _trace_rows()}),
            encoding="utf-8",
        )

    def _trace_scaled_mm(
        input: torch.Tensor,
        weight: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: type[torch.dtype],
        bias: torch.Tensor | None = None,
        block_size_m: int = 32,
        block_size_n: int = 32,
        block_size_k: int = 32,
        use_heuristic=True,
    ) -> torch.Tensor:
        global _trace_total
        try:
            key = (int(input.shape[0]), int(input.shape[1]), int(weight.shape[1]))
            _trace_counts[key] += 1
            _trace_total += 1
            if _trace_total % _trace_flush_every == 0:
                _write_trace_counts()
        except Exception:
            pass
        return _trace_orig_scaled_mm(
            input,
            weight,
            scale_a,
            scale_b,
            out_dtype,
            bias=bias,
            block_size_m=block_size_m,
            block_size_n=block_size_n,
            block_size_k=block_size_k,
            use_heuristic=use_heuristic,
        )

    def _dump_trace_counts(signum=None, frame=None) -> None:
        del signum, frame
        rows = _trace_rows()
        _write_trace_counts()
        print("[K100AI_LINEAR_SHAPE_TRACE] " + json.dumps(rows), flush=True)

    signal.signal(signal.SIGUSR1, _dump_trace_counts)
    _trace_ct_mod.triton_scaled_mm = _trace_scaled_mm
    _trace_linear_mod.triton_scaled_mm = _trace_scaled_mm
    print("[K100 AI Linear shape trace] enabled; send SIGUSR1 to dump counts", flush=True)

if _LINEAR_ENABLED:
    from aiter.ops.triton.gemm_a8w8 import gemm_a8w8 as _aiter_gemm_a8w8
    from vllm.triton_utils import tl as _tl, triton as _triton

    _ct_scaled_mm_mod = importlib.import_module(
        "vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm"
    )
    _linear_kernel_mod = importlib.import_module(
        "vllm.model_executor.kernels.linear.scaled_mm.triton"
    )
    _orig_triton_scaled_mm = _ct_scaled_mm_mod.triton_scaled_mm
    _scaled_mm_kernel = _ct_scaled_mm_mod.scaled_mm_kernel

    # AITER configs retained for shapes where it is still the fastest backend.
    # Keys are exact (M,K,N) runtime shapes; MTP=1 decode uses M=2.
    _LINEAR_CONFIGS_BY_M = {
        (2, 4096, 2048): {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 4,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 1,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
        },
        (2, 2048, 1024): {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 256,
            "GROUP_SIZE_M": 4,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 4,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
        },
    }
    # Explicit Triton configs fix the vendor heuristic for the dominant M=2
    # projections. These were searched on gfx928 with the real transposed
    # compressed-tensors weight layout and are bit-identical to stock output.
    _TRITON_LINEAR_CONFIGS_BY_M = {
        # Pure no-MTP target decode uses M=1 for all 40 base layers, while
        # MTP=3 also uses these shapes in the serial draft forwards. R174
        # measured 4.13x and 2.86x uplift versus the vendor heuristic on
        # gfx928, with relative_l2=0.
        (1, 2048, 12288): {
            "bm": 32, "bn": 32, "bk": 128, "warps": 8, "waves": 2,
        },
        (1, 2048, 9216): {
            "bm": 32, "bn": 32, "bk": 128, "warps": 8, "waves": 2,
        },
        # MTP=3 draft model runs three serial M=1 forwards. These exact
        # configs were missing from R171, so the draft layer fell back to the
        # vendor heuristic despite the target-model M=4 path being tuned.
        (1, 4096, 2048): {
            "bm": 32, "bn": 32, "bk": 128, "warps": 4, "waves": 1,
        },
        (1, 2048, 1024): {
            "bm": 64, "bn": 32, "bk": 256, "warps": 8, "waves": 2,
        },
        (1, 2048, 64): {
            "bm": 16, "bn": 32, "bk": 256, "warps": 4, "waves": 1,
        },
        (2, 2048, 12288): {
            "bm": 64, "bn": 32, "bk": 128, "warps": 8, "waves": 2,
        },
        (2, 2048, 9216): {
            "bm": 32, "bn": 32, "bk": 128, "warps": 8, "waves": 2,
        },
        (2, 2048, 64): {
            "bm": 16, "bn": 32, "bk": 256, "warps": 4, "waves": 2,
        },
        (3, 2048, 12288): {
            "bm": 64, "bn": 32, "bk": 128, "warps": 8, "waves": 2,
        },
        (3, 2048, 9216): {
            "bm": 64, "bn": 32, "bk": 128, "warps": 8, "waves": 2,
        },
        (3, 4096, 2048): {
            "bm": 64, "bn": 32, "bk": 128, "warps": 4, "waves": 2,
        },
        (3, 2048, 1024): {
            "bm": 64, "bn": 32, "bk": 256, "warps": 8, "waves": 2,
        },
        (3, 2048, 64): {
            "bm": 16, "bn": 32, "bk": 256, "warps": 8, "waves": 2,
        },
        (4, 2048, 12288): {
            "bm": 64, "bn": 32, "bk": 256, "warps": 8, "waves": 2,
        },
        (4, 2048, 9216): {
            "bm": 32, "bn": 32, "bk": 128, "warps": 8, "waves": 2,
        },
        (4, 4096, 2048): {
            "bm": 64, "bn": 32, "bk": 128, "warps": 4, "waves": 2,
        },
        (4, 2048, 1024): {
            "bm": 64, "bn": 32, "bk": 256, "warps": 4, "waves": 1,
        },
        (4, 2048, 64): {
            "bm": 16, "bn": 32, "bk": 256, "warps": 4, "waves": 1,
        },
        (5, 4096, 2048): {
            "bm": 64, "bn": 32, "bk": 128, "warps": 4, "waves": 2,
        },
        (5, 2048, 1024): {
            "bm": 32, "bn": 32, "bk": 256, "warps": 8, "waves": 2,
        },
        (5, 2048, 64): {
            "bm": 16, "bn": 32, "bk": 256, "warps": 4, "waves": 1,
        },
    }
    _LINEAR_PATCH_SHAPES = {
        (k, n) for (_, k, n) in (
            set(_LINEAR_CONFIGS_BY_M) | set(_TRITON_LINEAR_CONFIGS_BY_M)
        )
    }
    _LINEAR_LAYOUT_LOGGED: set[tuple[int, int, int]] = set()

    @torch.library.custom_op(
        "k100ai::moe35_sc_w8a8_linear",
        mutates_args=(),
        device_types="cuda",
    )
    def _moe35_sc_w8a8_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> torch.Tensor:
        m, k = (int(v) for v in input.shape)
        n = int(weight.shape[1])
        key = (m, k, n)
        aiter_config = _LINEAR_CONFIGS_BY_M.get(key)
        triton_config = _TRITON_LINEAR_CONFIGS_BY_M.get(key)
        if m <= _LINEAR_MAX_M and (aiter_config is not None or triton_config is not None):
            if _LOG_LINEAR_LAYOUT and key not in _LINEAR_LAYOUT_LOGGED:
                _LINEAR_LAYOUT_LOGGED.add(key)
                print(
                    "[K100AI_LINEAR_LAYOUT] "
                    f"m={m} k={k} n={n} "
                    f"weight_stride={tuple(weight.stride())} "
                    f"weight_contiguous={weight.is_contiguous()} "
                    f"weight_t_stride={tuple(weight.t().stride())} "
                    f"weight_t_contiguous={weight.t().is_contiguous()}",
                    flush=True,
                )
            if triton_config is not None:
                bm = triton_config["bm"]
                bn = triton_config["bn"]
                bk = triton_config["bk"]
                result = torch.empty(
                    (m, n), dtype=torch.bfloat16, device=input.device
                )
                grid = (
                    _triton.cdiv(m, bm) * _triton.cdiv(n, bn),
                )
                _scaled_mm_kernel[grid](
                    input,
                    weight,
                    scale_a,
                    scale_b,
                    result,
                    None,
                    m,
                    n,
                    k,
                    input.stride(0),
                    input.stride(1),
                    weight.stride(0),
                    weight.stride(1),
                    result.stride(0),
                    result.stride(1),
                    _tl.int32,
                    BLOCK_SIZE_M=bm,
                    BLOCK_SIZE_N=bn,
                    BLOCK_SIZE_K=bk,
                    BLOCK_SIZE_SCALE_A=bm,
                    BLOCK_SIZE_SCALE_B=bn,
                    num_warps=triton_config["warps"],
                    num_stages=2,
                    waves_per_eu=triton_config["waves"],
                )
                return result
            return _aiter_gemm_a8w8(
                input,
                weight.t(),
                scale_a,
                scale_b,
                None,
                torch.bfloat16,
                config=aiter_config,
            )
        return _orig_triton_scaled_mm(
            input,
            weight,
            scale_a,
            scale_b,
            torch.bfloat16,
        )

    @_moe35_sc_w8a8_linear.register_fake
    def _moe35_sc_w8a8_linear_fake(
        input: torch.Tensor,
        weight: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
    ) -> torch.Tensor:
        del scale_a, scale_b
        return input.new_empty(
            (input.shape[0], weight.shape[1]), dtype=torch.bfloat16
        )

    def _k100aiai_triton_scaled_mm(
        input: torch.Tensor,
        weight: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: type[torch.dtype],
        bias: torch.Tensor | None = None,
        block_size_m: int = 32,
        block_size_n: int = 32,
        block_size_k: int = 32,
        use_heuristic=True,
    ) -> torch.Tensor:
        shape = (int(input.shape[1]), int(weight.shape[1]))
        if (
            bias is None
            and out_dtype is torch.bfloat16
            and shape in _LINEAR_PATCH_SHAPES
        ):
            return _moe35_sc_w8a8_linear(input, weight, scale_a, scale_b)
        return _orig_triton_scaled_mm(
            input,
            weight,
            scale_a,
            scale_b,
            out_dtype,
            bias=bias,
            block_size_m=block_size_m,
            block_size_n=block_size_n,
            block_size_k=block_size_k,
            use_heuristic=use_heuristic,
        )

    _ct_scaled_mm_mod.triton_scaled_mm = _k100aiai_triton_scaled_mm
    _linear_kernel_mod.triton_scaled_mm = _k100aiai_triton_scaled_mm
    print(
        f"[K100 AI single-card W8A8 Linear] enabled, runtime M <= {_LINEAR_MAX_M}",
        flush=True,
    )

    if _FUSED_QKVZ_BA:
        from einops import rearrange as _rearrange
        from vllm import _custom_ops as _fused_qkvz_ba_ops
        from vllm.model_executor.models.qwen3_5 import (
            Qwen3_5GatedDeltaNet as _Qwen3_5GatedDeltaNet,
        )

        _FUSED_QKVZ_BA_CONFIGS = {
            1: {"bm": 64, "bn": 32, "bk": 128, "warps": 8, "waves": 2},
            4: {"bm": 32, "bn": 32, "bk": 128, "warps": 8, "waves": 2},
        }

        @_triton.jit
        def _qkvz_ba_fused_scaled_mm_kernel(
            a_ptr,
            qkvz_ptr,
            ba_ptr,
            scale_a_ptr,
            qkvz_scale_ptr,
            ba_scale_ptr,
            c_ptr,
            M,
            N_QKVZ,
            N_BA,
            K,
            stride_am,
            stride_ak,
            stride_qkvz_k,
            stride_qkvz_n,
            stride_ba_k,
            stride_ba_n,
            stride_cm,
            stride_cn,
            ACCUMULATOR_DTYPE: _tl.constexpr,
            BLOCK_SIZE_M: _tl.constexpr,
            BLOCK_SIZE_N: _tl.constexpr,
            BLOCK_SIZE_K: _tl.constexpr,
        ):
            pid = _tl.program_id(axis=0)
            total_n = N_QKVZ + N_BA
            num_pid_n = _tl.cdiv(total_n, BLOCK_SIZE_N)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            offsets_m = (
                pid_m * BLOCK_SIZE_M
                + _tl.arange(0, BLOCK_SIZE_M).to(_tl.int64)
            )
            output_n_base = pid_n * BLOCK_SIZE_N
            offsets_output_n = (
                output_n_base
                + _tl.arange(0, BLOCK_SIZE_N).to(_tl.int64)
            )
            offsets_k = _tl.arange(0, BLOCK_SIZE_K).to(_tl.int64)

            mask_m = offsets_m < M
            mask_output_n = offsets_output_n < total_n
            a_ptrs = (
                a_ptr
                + offsets_m[:, None] * stride_am
                + offsets_k[None, :] * stride_ak
            )
            accumulator = _tl.zeros(
                (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACCUMULATOR_DTYPE
            )

            # N_QKVZ is aligned to BLOCK_SIZE_N, so every program belongs
            # wholly to one weight tensor. This uniform branch avoids building
            # and masking both pointer trees for all 386 N tiles.
            if output_n_base < N_QKVZ:
                offsets_weight_n = offsets_output_n
                weight_ptrs = (
                    qkvz_ptr
                    + offsets_k[:, None] * stride_qkvz_k
                    + offsets_weight_n[None, :] * stride_qkvz_n
                )
                for _ in range(0, _tl.cdiv(K, BLOCK_SIZE_K)):
                    mask_k = offsets_k < K
                    a = _tl.load(
                        a_ptrs,
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0,
                    )
                    weight = _tl.load(
                        weight_ptrs,
                        mask=mask_k[:, None] & mask_output_n[None, :],
                        other=0,
                    )
                    accumulator = _tl.dot(
                        a,
                        weight,
                        accumulator,
                        out_dtype=ACCUMULATOR_DTYPE,
                    )
                    offsets_k += BLOCK_SIZE_K
                    a_ptrs += BLOCK_SIZE_K * stride_ak
                    weight_ptrs += BLOCK_SIZE_K * stride_qkvz_k
                scale_b = _tl.load(
                    qkvz_scale_ptr + offsets_weight_n,
                    mask=mask_output_n,
                    other=0.0,
                )[None, :]
            else:
                offsets_weight_n = offsets_output_n - N_QKVZ
                mask_weight_n = offsets_weight_n < N_BA
                weight_ptrs = (
                    ba_ptr
                    + offsets_k[:, None] * stride_ba_k
                    + offsets_weight_n[None, :] * stride_ba_n
                )
                for _ in range(0, _tl.cdiv(K, BLOCK_SIZE_K)):
                    mask_k = offsets_k < K
                    a = _tl.load(
                        a_ptrs,
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0,
                    )
                    weight = _tl.load(
                        weight_ptrs,
                        mask=mask_k[:, None] & mask_weight_n[None, :],
                        other=0,
                    )
                    accumulator = _tl.dot(
                        a,
                        weight,
                        accumulator,
                        out_dtype=ACCUMULATOR_DTYPE,
                    )
                    offsets_k += BLOCK_SIZE_K
                    a_ptrs += BLOCK_SIZE_K * stride_ak
                    weight_ptrs += BLOCK_SIZE_K * stride_ba_k
                scale_b = _tl.load(
                    ba_scale_ptr + offsets_weight_n,
                    mask=mask_weight_n,
                    other=0.0,
                )[None, :]

            scale_a = _tl.load(
                scale_a_ptr + offsets_m,
                mask=mask_m,
                other=0.0,
            )[:, None]
            result = (
                accumulator.to(_tl.float32) * scale_a * scale_b
            ).to(c_ptr.type.element_ty)

            c_ptrs = (
                c_ptr
                + offsets_m[:, None] * stride_cm
                + offsets_output_n[None, :] * stride_cn
            )
            _tl.store(
                c_ptrs,
                result,
                mask=mask_m[:, None] & mask_output_n[None, :],
            )

        @torch.library.custom_op(
            "k100ai::qkvz_ba_fused_w8a8",
            mutates_args=(),
            device_types="cuda",
        )
        def _qkvz_ba_fused_w8a8(
            x: torch.Tensor,
            qkvz_weight: torch.Tensor,
            qkvz_scale: torch.Tensor,
            ba_weight: torch.Tensor,
            ba_scale: torch.Tensor,
        ) -> torch.Tensor:
            original_shape = tuple(x.shape[:-1])
            x2 = x.reshape(-1, x.shape[-1]).contiguous()
            xq, xs, xzp = _fused_qkvz_ba_ops.scaled_int8_quant(
                x2, None, None, symmetric=True
            )
            assert xzp is None
            m = int(x2.shape[0])
            n_qkvz = int(qkvz_weight.shape[1])
            n_ba = int(ba_weight.shape[1])
            config = _FUSED_QKVZ_BA_CONFIGS.get(m)
            if (
                config is None
                or n_qkvz % config["bn"] != 0
                or int(x2.shape[1]) != 2048
                or n_qkvz != 12288
                or n_ba != 64
            ):
                qkvz = _k100aiai_triton_scaled_mm(
                    xq,
                    qkvz_weight,
                    xs,
                    qkvz_scale,
                    torch.bfloat16,
                )
                ba = _k100aiai_triton_scaled_mm(
                    xq,
                    ba_weight,
                    xs,
                    ba_scale,
                    torch.bfloat16,
                )
                result = torch.cat((qkvz, ba), dim=-1)
            else:
                result = torch.empty(
                    (m, n_qkvz + n_ba),
                    dtype=torch.bfloat16,
                    device=x.device,
                )
                grid = (
                    _triton.cdiv(m, config["bm"])
                    * _triton.cdiv(n_qkvz + n_ba, config["bn"]),
                )
                _qkvz_ba_fused_scaled_mm_kernel[grid](
                    xq,
                    qkvz_weight,
                    ba_weight,
                    xs,
                    qkvz_scale,
                    ba_scale,
                    result,
                    m,
                    n_qkvz,
                    n_ba,
                    int(x2.shape[1]),
                    xq.stride(0),
                    xq.stride(1),
                    qkvz_weight.stride(0),
                    qkvz_weight.stride(1),
                    ba_weight.stride(0),
                    ba_weight.stride(1),
                    result.stride(0),
                    result.stride(1),
                    _tl.int32,
                    BLOCK_SIZE_M=config["bm"],
                    BLOCK_SIZE_N=config["bn"],
                    BLOCK_SIZE_K=config["bk"],
                    num_warps=config["warps"],
                    num_stages=2,
                    waves_per_eu=config["waves"],
                )
            return result.reshape(
                *original_shape, n_qkvz + n_ba
            )

        @_qkvz_ba_fused_w8a8.register_fake
        def _qkvz_ba_fused_w8a8_fake(
            x: torch.Tensor,
            qkvz_weight: torch.Tensor,
            qkvz_scale: torch.Tensor,
            ba_weight: torch.Tensor,
            ba_scale: torch.Tensor,
        ) -> torch.Tensor:
            del qkvz_scale, ba_scale
            return x.new_empty(
                (
                    *x.shape[:-1],
                    qkvz_weight.shape[1] + ba_weight.shape[1],
                ),
                dtype=torch.bfloat16,
            )

        _orig_qwen35_gdn_forward = _Qwen3_5GatedDeltaNet.forward

        def _k100aiai_qwen35_gdn_forward(
            self,
            hidden_states: torch.Tensor,
            output: torch.Tensor,
        ):
            qkvz_layer = self.in_proj_qkvz
            ba_layer = self.in_proj_ba
            if (
                not hasattr(qkvz_layer, "weight_scale")
                or not hasattr(ba_layer, "weight_scale")
                or qkvz_layer.weight.dtype != torch.int8
                or ba_layer.weight.dtype != torch.int8
                or tuple(qkvz_layer.weight.shape) != (2048, 12288)
                or tuple(ba_layer.weight.shape) != (2048, 64)
            ):
                return _orig_qwen35_gdn_forward(self, hidden_states, output)

            num_tokens = hidden_states.size(0)
            projected = _qkvz_ba_fused_w8a8(
                hidden_states,
                qkvz_layer.weight,
                qkvz_layer.weight_scale,
                ba_layer.weight,
                ba_layer.weight_scale,
            )
            mixed_qkvz, ba = projected.split([12288, 64], dim=-1)
            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)
            b = b.contiguous()
            a = a.contiguous()

            core_attn_out = torch.zeros(
                (
                    num_tokens,
                    self.num_v_heads // self.tp_size,
                    self.head_v_dim,
                ),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            torch.ops.vllm.gdn_attention_core(
                mixed_qkv,
                b,
                a,
                core_attn_out,
                self.prefix,
            )

            z_shape_og = z.shape
            core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            z = z.reshape(-1, z.shape[-1])
            core_attn_out = self.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(z_shape_og)
            core_attn_out = _rearrange(core_attn_out, "... h d -> ... (h d)")
            output[:num_tokens], _ = self.out_proj(core_attn_out)

        _Qwen3_5GatedDeltaNet.forward = _k100aiai_qwen35_gdn_forward
        print(
            "[K100 AI fused QKVZ+BA W8A8] hook installed",
            flush=True,
        )
