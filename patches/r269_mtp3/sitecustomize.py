"""K100/gfx928 native INT8 MoE runtime patch for Qwen3.6-35B-A3B W8A8.

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


_LM_HEAD_ENABLED: Final[bool] = os.getenv("K100_LM_HEAD_W8A8", "0") == "1"
_LM_HEAD_MAX_M: Final[int] = int(os.getenv("K100_LM_HEAD_W8A8_MAX_M", "2"))

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
        # R258: MTP3 target verifier full-vocab lm_head, M=4. A narrow
        # gfx928 search and six-seed exact stress test found this simple
        # AITER configuration ~876.03us -> ~738.29us (1.187x) with bitwise
        # identical BF16 logits. Draft M=1 remains on the R248 top1 path.
        4: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2,
            "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2,
        },
        5: {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2,
            "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "kpack": 1,
        },
    }

    @torch.library.custom_op(
        "k100::lm_head_w8a8", mutates_args=(), device_types="cuda"
    )
    def _k100_lm_head_w8a8(
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

    @_k100_lm_head_w8a8.register_fake
    def _k100_lm_head_w8a8_fake(
        x: torch.Tensor, weight_nk: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        del weight_scale
        return x.new_empty((*x.shape[:-1], weight_nk.shape[0]), dtype=torch.bfloat16)

    class _K100LMHeadW8A8Method(_QuantizeMethodBase):
        def create_weights(self, *args, **kwargs):
            raise RuntimeError("K100 lm_head method is installed after weight loading")

        def apply(
            self, layer: torch.nn.Module, x: torch.Tensor,
            bias: torch.Tensor | None = None
        ) -> torch.Tensor:
            if bias is not None:
                return _k100_lm_head_w8a8(
                    x, layer.weight, layer.k100_weight_scale
                ) + bias
            return _k100_lm_head_w8a8(
                x, layer.weight, layer.k100_weight_scale
            )

    _orig_unquantized_embedding_process = (
        _UnquantizedEmbeddingMethod.process_weights_after_loading
    )

    def _k100_process_unquantized_embedding(self, layer: torch.nn.Module) -> None:
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
            "k100_weight_scale", scale[:, None].contiguous(), persistent=False
        )
        layer.quant_method = _K100LMHeadW8A8Method()
        del weight, qweight, scale
        torch.cuda.empty_cache()
        print(
            "[K100 lm_head W8A8] enabled "
            f"shape={tuple(layer.weight.shape)} "
            f"quantize_s={_time.perf_counter() - started:.3f}",
            flush=True,
        )

    _UnquantizedEmbeddingMethod.process_weights_after_loading = (
        _k100_process_unquantized_embedding
    )
    print(
        f"[K100 lm_head W8A8] hook installed, runtime M <= {_LM_HEAD_MAX_M}",
        flush=True,
    )

_LINEAR_ENABLED: Final[bool] = os.getenv("K100_SC_LINEAR_A8W8", "0") == "1"
_LINEAR_MAX_M: Final[int] = int(os.getenv("K100_SC_LINEAR_A8W8_MAX_M", "1"))
_FUSED_QKVZ_BA: Final[bool] = os.getenv("K100_FUSED_QKVZ_BA", "0") == "1"
_LOG_LINEAR_LAYOUT: Final[bool] = os.getenv("K100_LOG_LINEAR_LAYOUT", "0") == "1"
_TRACE_LINEAR_SHAPES: Final[bool] = os.getenv("K100_TRACE_LINEAR_SHAPES", "0") == "1"

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
    _trace_out = os.getenv("K100_TRACE_LINEAR_SHAPES_OUT", "")
    _trace_flush_every = max(
        1, int(os.getenv("K100_TRACE_LINEAR_SHAPES_FLUSH_EVERY", "100"))
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
        print("[K100_LINEAR_SHAPE_TRACE] " + json.dumps(rows), flush=True)

    signal.signal(signal.SIGUSR1, _dump_trace_counts)
    _trace_ct_mod.triton_scaled_mm = _trace_scaled_mm
    _trace_linear_mod.triton_scaled_mm = _trace_scaled_mm
    print("[K100 Linear shape trace] enabled; send SIGUSR1 to dump counts", flush=True)

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
        # R252: this is the MTP3 target-verifier hot shape used throughout the
        # 40-layer backbone. The previous BN32 layout exposed only 64 N blocks
        # for 120 K100AI CUs. A real-layout gfx928 search found this 128-block
        # configuration exact and ~50.04us -> ~28.33us (1.766x) in CUDAGraph.
        (4, 4096, 2048): {
            "bm": 16, "bn": 16, "bk": 512, "warps": 4, "waves": 2,
            "mi": 16, "kp": 2, "mlf": 1, "lat": "none",
            "hint": "local-prefetch",
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
        "k100::moe35_sc_w8a8_linear",
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
                    "[K100_LINEAR_LAYOUT] "
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
                if "kp" in triton_config:
                    # R252 advanced HCU meta is opt-in per exact shape. Keep all
                    # older configs on their byte-for-byte previous launch path.
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
                        matrix_instr_nonkdim=triton_config["mi"],
                        kpack=triton_config["kp"],
                        mmac_layout_force=triton_config["mlf"],
                        sched_latency=triton_config["lat"],
                        schedule_hint=triton_config["hint"],
                    )
                else:
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

    def _k100_triton_scaled_mm(
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

    _ct_scaled_mm_mod.triton_scaled_mm = _k100_triton_scaled_mm
    _linear_kernel_mod.triton_scaled_mm = _k100_triton_scaled_mm
    print(
        f"[K100 single-card W8A8 Linear] enabled, runtime M <= {_LINEAR_MAX_M}",
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
            # R253: MTP3 target verifier hot path. Exact CUDAGraph tuning on
            # K100AI/gfx928 reduced the fused QKVZ+BA kernel from ~76.35us to
            # ~60.51us without changing arithmetic or output bits.
            4: {"bm": 16, "bn": 16, "bk": 512, "warps": 4, "waves": 2},
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
            "k100::qkvz_ba_fused_w8a8",
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
                qkvz = _k100_triton_scaled_mm(
                    xq,
                    qkvz_weight,
                    xs,
                    qkvz_scale,
                    torch.bfloat16,
                )
                ba = _k100_triton_scaled_mm(
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

        def _k100_qwen35_gdn_forward(
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

        _Qwen3_5GatedDeltaNet.forward = _k100_qwen35_gdn_forward
        print(
            "[K100 fused QKVZ+BA W8A8] hook installed",
            flush=True,
        )

_ENABLED: Final[bool] = os.getenv("K100_NATIVE_INT8_MOE", "0") == "1"
_MAX_NATIVE_M: Final[int] = int(os.getenv("K100_NATIVE_INT8_MOE_MAX_M", "32"))
_GROUP_K: Final[int] = 128

if _ENABLED:
    import k100_channelwise_int8_moe as _native_ext
    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.fused_moe.config import (
        int8_w8a8_moe_quant_config,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
        CompressedTensorsW8A8Int8MoEMethod,
    )
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_quant_int8,
    )

    def _fallback_vllm(
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale_channel: torch.Tensor,
        w2_scale_channel: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        quant_config = int8_w8a8_moe_quant_config(
            w1_scale=w1_scale_channel,
            w2_scale=w2_scale_channel,
            a1_scale=None,
            a2_scale=None,
            per_act_token_quant=True,
        )
        return fused_experts(
            hidden_states=x,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=MoEActivation.SILU,
            global_num_experts=int(w1.shape[0]),
            quant_config=quant_config,
        )

    @torch.library.custom_op(
        "k100::native_channelwise_int8_moe",
        mutates_args=(),
        device_types="cuda",
    )
    def _native_channelwise_int8_moe(
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale_channel: torch.Tensor,
        w2_scale_channel: torch.Tensor,
        w1_scale_grouped: torch.Tensor,
        w2_scale_grouped: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        m = int(x.shape[0])
        topk = int(topk_ids.shape[1])
        num_experts = int(w1.shape[0])
        hidden = int(w1.shape[2])
        fused_intermediate = int(w1.shape[1])
        intermediate = fused_intermediate // 2

        if (
            m > _MAX_NATIVE_M
            or hidden != 2048
            or fused_intermediate != 512
            or int(w2.shape[2]) != 256
            or hidden % _GROUP_K != 0
            or intermediate % _GROUP_K != 0
            or fused_intermediate % 2 != 0
        ):
            return _fallback_vllm(
                x,
                w1,
                w2,
                w1_scale_channel,
                w2_scale_channel,
                topk_weights,
                topk_ids,
            )

        # Dynamic GPU routing is executed inside the opaque custom op so each
        # captured decode graph receives the actual expert assignments.
        sorted1, expert1, post1 = moe_align_block_size(topk_ids, 16, num_experts)
        x_q, x_scale_token = per_token_quant_int8(x.contiguous())
        x_scale = (
            x_scale_token.reshape(m, 1)
            .expand(m, hidden // _GROUP_K)
            .contiguous()
        )

        out1 = torch.empty(
            (m, topk, fused_intermediate),
            dtype=x.dtype,
            device=x.device,
        )
        _native_ext.gemm_gate_up(
            x_q,
            x_scale,
            out1,
            w1,
            w1_scale_grouped,
            sorted1,
            expert1,
            post1,
        )

        bridge = torch.empty(
            (m * topk, intermediate),
            dtype=x.dtype,
            device=x.device,
        )
        torch.ops._C.silu_and_mul(bridge, out1.view(-1, fused_intermediate))
        bridge_q, bridge_scale_token = per_token_quant_int8(bridge.contiguous())
        bridge_scale = (
            bridge_scale_token.reshape(m * topk, 1)
            .expand(m * topk, intermediate // _GROUP_K)
            .contiguous()
        )

        ids2 = topk_ids.reshape(-1, 1).contiguous()
        weights2 = topk_weights.reshape(-1, 1).contiguous()
        sorted2, expert2, post2 = moe_align_block_size(ids2, 16, num_experts)
        out2 = torch.empty(
            (m * topk, 1, hidden),
            dtype=x.dtype,
            device=x.device,
        )
        _native_ext.gemm_down(
            bridge_q,
            bridge_scale,
            out2,
            w2,
            w2_scale_grouped,
            weights2,
            sorted2,
            expert2,
            post2,
        )
        return out2.view(m, topk, hidden).sum(dim=1)

    @_native_channelwise_int8_moe.register_fake
    def _native_channelwise_int8_moe_fake(
        x: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        w1_scale_channel: torch.Tensor,
        w2_scale_channel: torch.Tensor,
        w1_scale_grouped: torch.Tensor,
        w2_scale_grouped: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        del (
            w1,
            w2,
            w1_scale_channel,
            w2_scale_channel,
            w1_scale_grouped,
            w2_scale_grouped,
            topk_weights,
            topk_ids,
        )
        return x.new_empty(x.shape)

    _orig_process = CompressedTensorsW8A8Int8MoEMethod.process_weights_after_loading
    _orig_apply = CompressedTensorsW8A8Int8MoEMethod.apply

    def _k100_process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _orig_process(self, layer)
        w1 = layer.w13_weight
        w2 = layer.w2_weight
        s1 = layer.w13_weight_scale
        s2 = layer.w2_weight_scale

        # Stock checkpoint layout is [E, N, 1] per output channel. The native
        # kernel consumes [E, N, K/128]; repeat the exact same channel scale
        # across K groups without changing the INT8 weights.
        if s1.ndim != 3 or s1.shape[-1] != 1:
            raise RuntimeError(f"Unexpected w13 scale shape: {tuple(s1.shape)}")
        if s2.ndim != 3 or s2.shape[-1] != 1:
            raise RuntimeError(f"Unexpected w2 scale shape: {tuple(s2.shape)}")
        if w1.shape[2] % _GROUP_K or w2.shape[2] % _GROUP_K:
            raise RuntimeError(
                f"K dimensions must be divisible by {_GROUP_K}: "
                f"w13={tuple(w1.shape)}, w2={tuple(w2.shape)}"
            )

        s1_grouped = (
            s1.expand(-1, -1, int(w1.shape[2]) // _GROUP_K)
            .contiguous()
        )
        s2_grouped = (
            s2.expand(-1, -1, int(w2.shape[2]) // _GROUP_K)
            .contiguous()
        )
        if hasattr(layer, "_k100_w13_scale_grouped"):
            layer._k100_w13_scale_grouped = s1_grouped
        else:
            layer.register_buffer(
                "_k100_w13_scale_grouped", s1_grouped, persistent=False
            )
        if hasattr(layer, "_k100_w2_scale_grouped"):
            layer._k100_w2_scale_grouped = s2_grouped
        else:
            layer.register_buffer(
                "_k100_w2_scale_grouped", s2_grouped, persistent=False
            )
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

    def _k100_apply(
        self,
        layer,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del shared_experts_input
        # Qwen3.6 uses SiLU gated experts, no expert map under TP, and applies
        # router weights after the second GEMM. Unsupported layouts retain the
        # upstream implementation.
        if (
            layer.activation not in (MoEActivation.SILU, "silu")
            or layer.expert_map is not None
            or layer.apply_router_weight_on_input
            or not hasattr(layer, "_k100_w13_scale_grouped")
        ):
            return _orig_apply(
                self,
                layer,
                x,
                topk_weights,
                topk_ids,
                None,
            )
        return _native_channelwise_int8_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            layer._k100_w13_scale_grouped,
            layer._k100_w2_scale_grouped,
            topk_weights,
            topk_ids,
        )

    CompressedTensorsW8A8Int8MoEMethod.process_weights_after_loading = (
        _k100_process_weights_after_loading
    )
    CompressedTensorsW8A8Int8MoEMethod.apply = _k100_apply

    print(
        f"[K100 native INT8 MoE] enabled, native decode M <= {_MAX_NATIVE_M}",
        flush=True,
    )

# R195: HCU-specific hybrid prefix-cache align fast path.
# The HCU plugin owns its own GPUModelRunner and duplicates the upstream
# synchronous accepted-token copy. Patch the class that actually executes.
_K100_HCU_ALIGN_FASTPATH: Final[bool] = os.getenv("K100_HCU_ALIGN_FASTPATH", "0") == "1"
if _K100_HCU_ALIGN_FASTPATH:
    try:
        from vllm_hcu.v1.hcu_model_runner import GPUModelRunner as _K100HcuGPUModelRunner
        from vllm.v1.worker import mamba_utils as _k100_hcu_mamba_utils

        _k100_hcu_orig_update_states = (
            _K100HcuGPUModelRunner._update_states_after_model_execute
        )

        def _k100_hcu_update_states_after_model_execute(
            self, output_token_ids: torch.Tensor, scheduler_output
        ) -> None:
            if (
                not self.speculative_config
                or not self.model_config.is_hybrid
                or self.cache_config.mamba_cache_mode != "align"
            ):
                return _k100_hcu_orig_update_states(
                    self, output_token_ids, scheduler_output
                )

            num_reqs = int(output_token_ids.size(0))
            max_accept = int(self.num_spec_tokens or 0) + 1
            margin = max(16, max_accept * 4)
            fast_ok = num_reqs > 0
            reason = "ok"
            debug_tuple = None
            try:
                block_size = getattr(self, "_k100_hcu_mamba_block_size", 0)
                if not block_size:
                    _, _mamba_spec = _k100_hcu_mamba_utils.get_mamba_groups(
                        self.kv_cache_config
                    )
                    block_size = int(_mamba_spec.block_size)
                    self._k100_hcu_mamba_block_size = block_size
                if block_size <= 0:
                    fast_ok = False
                    reason = "bad_block"
                if fast_ok:
                    req_ids = self.input_batch.req_ids[:num_reqs]
                    spec_dict = scheduler_output.scheduled_spec_decode_tokens
                    sched_dict = scheduler_output.num_scheduled_tokens
                    for req_id in req_ids:
                        if req_id is None or req_id not in self.requests:
                            fast_ok = False
                            reason = "bad_req"
                            break
                        req_state = self.requests[req_id]
                        num_sched = int(sched_dict[req_id])
                        num_draft = len(spec_dict.get(req_id, []))
                        running = (
                            int(req_state.num_computed_tokens)
                            + num_sched
                            - num_draft
                        )
                        rem = running % block_size
                        dist_next = 0 if rem == 0 else block_size - rem
                        debug_tuple = (
                            num_sched,
                            num_draft,
                            running,
                            block_size,
                            dist_next,
                        )
                        if num_sched > max_accept or num_draft > int(self.num_spec_tokens or 0):
                            fast_ok = False
                            reason = "non_decode_shape"
                            break
                        if dist_next <= margin:
                            fast_ok = False
                            reason = "near_boundary"
                            break
            except Exception as _fast_exc:
                fast_ok = False
                reason = f"exception:{type(_fast_exc).__name__}:{_fast_exc}"

            dbg = getattr(self, "_k100_hcu_align_dbg", 0)
            if dbg < 8:
                print(
                    f"[K100 R195 align decision] fast={fast_ok} reason={reason} "
                    f"vals={debug_tuple} max_accept={max_accept} margin={margin}",
                    flush=True,
                )
                self._k100_hcu_align_dbg = dbg + 1

            if not fast_ok:
                self._k100_hcu_align_sync_steps = getattr(
                    self, "_k100_hcu_align_sync_steps", 0
                ) + 1
                return _k100_hcu_orig_update_states(
                    self, output_token_ids, scheduler_output
                )

            # Exact upstream accepted-token computation.
            self.num_accepted_tokens.gpu[:num_reqs] = (
                (
                    torch.cat(
                        [
                            output_token_ids,
                            torch.full(
                                (num_reqs, 1),
                                -1,
                                device=output_token_ids.device,
                            ),
                        ],
                        dim=1,
                    )
                    == -1
                )
                .int()
                .argmax(-1)
            )
            # R199: once the direct-GPU metadata path is enabled, the next
            # speculative step consumes self.num_accepted_tokens.gpu directly.
            # Do not launch a redundant GPU->CPU copy or record an event that
            # will never be synchronized on the fast path. Boundary/fallback
            # steps still execute the untouched upstream align implementation,
            # which performs its own synchronous accepted-token transfer before
            # postprocess_mamba.
            if not globals().get("_K100_HCU_GPU_ACCEPT_FASTPATH", False):
                self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(
                    self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True
                )
                assert self.num_accepted_tokens_event is not None
                self.num_accepted_tokens_event.record()
            else:
                self._k100_r199_skipped_d2h_steps = getattr(
                    self, "_k100_r199_skipped_d2h_steps", 0
                ) + 1
            self._k100_hcu_align_fast_steps = getattr(
                self, "_k100_hcu_align_fast_steps", 0
            ) + 1

        _K100HcuGPUModelRunner._update_states_after_model_execute = (
            _k100_hcu_update_states_after_model_execute
        )
        print(
            "[K100 R195 HCU align fast-path] installed on vllm_hcu GPUModelRunner",
            flush=True,
        )
    except Exception as _k100_hcu_align_exc:
        print(
            f"[K100 R195 HCU align fast-path] disabled: {_k100_hcu_align_exc!r}",
            flush=True,
        )

# R197: When R195 took the proven-safe align fast path, the accepted-token
# counts are already correct in self.num_accepted_tokens.gpu. Avoid the next
# step's event synchronize + CPU mirror read + CPU->GPU copy. We still call the
# upstream metadata builder unchanged, but give it a temporary proxy whose
# copy_to_gpu() is a no-op and whose .gpu is the already-correct device tensor.
_K100_HCU_GPU_ACCEPT_FASTPATH: Final[bool] = os.getenv(
    "K100_HCU_GPU_ACCEPT_FASTPATH", "0"
) == "1"
if _K100_HCU_GPU_ACCEPT_FASTPATH:
    try:
        from vllm_hcu.v1.hcu_model_runner import GPUModelRunner as _K100R197Runner

        _r197_prev_update = _K100R197Runner._update_states_after_model_execute
        _r197_prev_build = _K100R197Runner._build_attention_metadata

        def _r197_update_states(self, output_token_ids, scheduler_output):
            before = getattr(self, "_k100_hcu_align_fast_steps", 0)
            result = _r197_prev_update(self, output_token_ids, scheduler_output)
            after = getattr(self, "_k100_hcu_align_fast_steps", 0)
            # True only if the R195 fast path was used for this just-finished step.
            self._k100_r197_gpu_accept_valid = after > before
            return result

        class _R197AcceptedProxy:
            __slots__ = ("gpu", "np")
            def __init__(self, original, scratch_np):
                self.gpu = original.gpu
                # Upstream writes the CPU mirror even though copy_to_gpu() is a
                # no-op here. R199 reuses one scratch array per runner instead
                # of allocating/copying a NumPy array every decode step.
                self.np = scratch_np
            def copy_to_gpu(self, *args, **kwargs):
                return None

        def _r197_build_attention_metadata(self, *args, **kwargs):
            use_spec_decode = bool(kwargs.get("use_spec_decode", False))
            if not use_spec_decode or not getattr(
                self, "_k100_r197_gpu_accept_valid", False
            ):
                return _r197_prev_build(self, *args, **kwargs)

            # Preserve padded-request semantics without launching a tiny fill
            # kernel every step. After the first initialization, only slots that
            # just became inactive need to be reset to 1. Newly-active slots are
            # overwritten by the accepted-token computation before this builder.
            try:
                if len(args) >= 2:
                    num_reqs = int(args[1])
                else:
                    num_reqs = int(kwargs["num_reqs"])
                prev_num_reqs = getattr(self, "_k100_r199_prev_num_reqs", None)
                total_slots = int(self.num_accepted_tokens.gpu.numel())
                if prev_num_reqs is None:
                    if num_reqs < total_slots:
                        self.num_accepted_tokens.gpu[num_reqs:].fill_(1)
                elif num_reqs < int(prev_num_reqs):
                    self.num_accepted_tokens.gpu[num_reqs:int(prev_num_reqs)].fill_(1)
                self._k100_r199_prev_num_reqs = num_reqs
            except Exception:
                return _r197_prev_build(self, *args, **kwargs)

            original_buf = self.num_accepted_tokens
            original_event = self.num_accepted_tokens_event
            scratch_np = getattr(self, "_k100_r199_accept_scratch_np", None)
            if scratch_np is None or scratch_np.shape != original_buf.np.shape:
                scratch_np = original_buf.np.copy()
                self._k100_r199_accept_scratch_np = scratch_np
            self.num_accepted_tokens = _R197AcceptedProxy(original_buf, scratch_np)
            self.num_accepted_tokens_event = None
            try:
                count = getattr(self, "_k100_r197_metadata_fast_steps", 0) + 1
                self._k100_r197_metadata_fast_steps = count
                if count == 1:
                    print(
                        "[K100 R197 GPU accepted-token metadata] first direct-GPU step",
                        flush=True,
                    )
                return _r197_prev_build(self, *args, **kwargs)
            finally:
                self.num_accepted_tokens = original_buf
                self.num_accepted_tokens_event = original_event

        _K100R197Runner._update_states_after_model_execute = _r197_update_states
        _K100R197Runner._build_attention_metadata = _r197_build_attention_metadata
        print(
            "[K100 R197 GPU accepted-token metadata] installed",
            flush=True,
        )
    except Exception as _r197_exc:
        print(
            f"[K100 R197 GPU accepted-token metadata] disabled: {_r197_exc!r}",
            flush=True,
        )

# R210: exact native RMSNorm -> dynamic INT8 fusion. This is deliberately
# installed after the stable R199 hooks so it can reuse the same W8A8 Linear
# kernels and only changes the norm/quant dataflow in compiled graphs.
if os.getenv("K100_RMSNORM_INT8_FUSION", "0") == "1":
    try:
        import r210_norm_int8 as _k100_r210_norm_int8
        _k100_r210_norm_int8.install()
    except Exception as _r210_exc:
        print(
            f"[K100 R210 RMS+INT8] disabled: {_r210_exc!r}",
            flush=True,
        )

# R235: bypass the redundant int32->int64 cast on the TP1 single-token
# embedding hot path. All other shapes/dtypes keep the upstream implementation.
if os.getenv("K100_R235_INT32_EMBEDDING", "0") == "1":
    try:
        import r235_int32_embedding as _k100_r235_int32_embedding
        _k100_r235_int32_embedding.install()
    except Exception as _r235_exc:
        print(
            f"[K100 R235 int32 embedding] disabled: {_r235_exc!r}",
            flush=True,
        )

# R236: the modular TritonExperts path lost the legacy sparse/naive block
# assignment optimization. Restore it only for Qwen3.6 TP1 M=1/topk=8.
if os.getenv("K100_R236_M1_NAIVE_MOE", "0") == "1":
    try:
        import r236_m1_naive_moe as _k100_r236_m1_naive_moe
        _k100_r236_m1_naive_moe.install()
    except Exception as _r236_exc:
        print(
            f"[K100 R236 M1 naive MoE] disabled: {_r236_exc!r}",
            flush=True,
        )

# R237: on TP1 multimodal runners, pure-text decode steps do not need a
# temporary embedding tensor followed by a D2D copy into inputs_embeds.gpu.
if os.getenv("K100_R237_MM_DIRECT_EMBED", "0") == "1":
    try:
        import r237_mm_direct_embed as _k100_r237_mm_direct_embed
        _k100_r237_mm_direct_embed.install()
    except Exception as _r237_exc:
        print(
            f"[K100 R237 MM direct embed] disabled: {_r237_exc!r}",
            flush=True,
        )

# R239: complete single-request MTP3 GDN metadata specialization. This
# supersedes R238 and additionally bypasses generic query-lens reductions,
# argsort/advanced indexing and temporary spec-index construction.
if os.getenv("K100_R239_GDN_FULL_SINGLE_SPEC", "0") == "1":
    try:
        import r239_gdn_full_single_spec as _k100_r239_gdn_full_single_spec
        _k100_r239_gdn_full_single_spec.install()
    except Exception as _r239_exc:
        print(
            f"[K100 R239 GDN full single-spec] disabled: {_r239_exc!r}",
            flush=True,
        )

# R241: fuse align-mode Mamba block-table index construction and gather for
# the tiny <=4-row TP1/MTP3 metadata path. R241 itself includes the R239
# single-spec specialization in the same source rewrite.
if os.getenv("K100_R241_GDN_BLOCKTABLE_FUSED", "0") == "1":
    try:
        import r241_gdn_blocktable_fused as _k100_r241_gdn_blocktable_fused
        _k100_r241_gdn_blocktable_fused.install()
    except Exception as _r241_exc:
        print(
            f"[K100 R241 GDN block-table] disabled: {_r241_exc!r}",
            flush=True,
        )

# R243: for the all-spec single-request MTP verifier path, make the existing
# fused recurrent GDN kernel write directly into core_attn_out and remove the
# temporary output tensor + D2D merge copy.
if os.getenv("K100_R243_GDN_SPEC_DIRECT_OUT", "0") == "1":
    try:
        import r243_gdn_spec_direct_out as _k100_r243_gdn_spec_direct_out
        _k100_r243_gdn_spec_direct_out.install()
    except Exception as _r243_exc:
        print(
            f"[K100 R243 GDN direct out] disabled: {_r243_exc!r}",
            flush=True,
        )

# R244 supersedes R243 on the single-spec path: it keeps R243 direct output
# and additionally lets the recurrent kernel consume strided Q/K/V split views
# without three intermediate contiguous D2D copies.
if os.getenv("K100_R244_GDN_STRIDED_QKV", "0") == "1":
    try:
        import r244_gdn_strided_qkv as _k100_r244_gdn_strided_qkv
        _k100_r244_gdn_strided_qkv.install()
    except Exception as _r244_exc:
        print(
            f"[K100 R244 GDN strided QKV] disabled: {_r244_exc!r}",
            flush=True,
        )

# R245: keep Qwen3.5 BA chunk views strided on the same single-spec path and
# let the recurrent kernel consume them directly, removing two more D2D copies.
if os.getenv("K100_R245_GDN_STRIDED_BA", "0") == "1":
    try:
        import r245_gdn_strided_ba as _k100_r245_gdn_strided_ba
        _k100_r245_gdn_strided_ba.install()
    except Exception as _r245_exc:
        print(
            f"[K100 R245 GDN strided BA] disabled: {_r245_exc!r}",
            flush=True,
        )

# R248: only the Qwen3.5 MTP draft greedy path uses a fused W8A8 lm_head
# local-top1 op. Target logits/sampling/logprobs remain on the R245 path.
if os.getenv("K100_R248_MTP_TOP1", "0") == "1":
    try:
        import r248_mtp_top1 as _k100_r248_mtp_top1
        _k100_r248_mtp_top1.install()
    except Exception as _r248_exc:
        print(
            f"[K100 R248 MTP top1] disabled: {_r248_exc!r}",
            flush=True,
        )

# R259: for the exact single-request MTP3 pure-greedy verifier hot path,
# avoid materializing the target [4,vocab] logits tensor and feed four exact
# fused W8A8 top-1 ids directly into the existing rejection-greedy kernel.
# Any feature that needs or can alter logits falls back to the original path.
if os.getenv("K100_R259_TARGET_GREEDY_TOP1", "0") == "1":
    try:
        import r259_target_greedy_top1 as _k100_r259_target_greedy_top1
        _k100_r259_target_greedy_top1.install()
    except Exception as _r259_exc:
        print(
            f"[K100 R259 target top1] disabled: {_r259_exc!r}",
            flush=True,
        )

# R269: R265's M=4 MoE config is optimal for the first expert GEMM, but the
# second GEMM has K=512 and benefits from a separate BK/launch geometry. Keep
# the shared BM32 routing metadata and override only the exact M4 W2 dispatch.
if os.getenv("K100_R269_SPLIT_MOE_STAGE2", "0") == "1":
    try:
        import r269_split_moe_stage2 as _k100_r269_split_moe_stage2
        _k100_r269_split_moe_stage2.install()
    except Exception as _r269_exc:
        print(
            f"[K100 R269 split MoE stage2] disabled: {_r269_exc!r}",
            flush=True,
        )
