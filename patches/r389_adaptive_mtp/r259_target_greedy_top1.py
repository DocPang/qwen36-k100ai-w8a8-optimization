"""R259: compact target lm_head -> greedy rejection fast path for TP1/MTP3.

For the single-request Qwen3.6 TP1 decode hot path with exactly three draft
proposals, pure greedy sampling ultimately needs only four argmax token ids
(three target checks plus one bonus token).  Avoid materializing the full
[4, vocab] BF16 logits tensor when no feature can alter or consume logits.

Safety policy: any non-greedy request, logprobs, penalties, bad words,
allowed-token mask, active min_tokens processor, prompt-logprobs, NaN-logit
instrumentation, unsupported shape/layout, or grammar output falls back to the
original full-logits implementation.
"""
from __future__ import annotations

import os
import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops

_installed = False


@triton.jit
def _local_top1_m4_kernel(
    a_ptr,
    w_ptr,
    a_scale_ptr,
    w_scale_ptr,
    local_val_ptr,
    local_idx_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    NT: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BM)
    # M=4 is intentional: duplicate the four logical rows across the MFMA M
    # tile exactly like the accepted low-M AITER family, then store only rows
    # 0..3.  No duplicate-address stores are issued.
    real_m = offs_m % 4
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = a_ptr + real_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for _ in range(0, tl.cdiv(K, BK)):
        aa = tl.load(a_ptrs)
        ww = tl.load(w_ptrs, mask=offs_n[None, :] < N, other=0)
        acc += tl.dot(aa, ww, input_precision="ieee")
        a_ptrs += BK * stride_ak
        w_ptrs += BK * stride_wk

    a_scale = tl.load(a_scale_ptr + real_m)
    w_scale = tl.load(w_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    # Match the full lm_head semantics: scale in FP32, then round/store to BF16
    # before argmax.  This preserves token identity including BF16 ties.
    vals = (acc.to(tl.float32) * a_scale[:, None] * w_scale[None, :]).to(
        tl.bfloat16
    )
    vals = tl.where(offs_n[None, :] < N, vals, -float("inf"))
    local_arg = tl.argmax(vals, axis=1, tie_break_left=True)
    local_val = tl.max(vals, axis=1)

    store_mask = offs_m < 4
    tl.store(local_val_ptr + offs_m * NT + pid_n, local_val, mask=store_mask)
    tl.store(
        local_idx_ptr + offs_m * NT + pid_n,
        (pid_n * BN + local_arg).to(tl.int32),
        mask=store_mask,
    )


@triton.jit
def _reduce_top1_chunks_kernel(
    local_val_ptr,
    local_idx_ptr,
    chunk_val_ptr,
    chunk_idx_ptr,
    NT: tl.constexpr,
    CHUNK: tl.constexpr,
    NCH: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = tl.arange(0, CHUNK)
    idx = chunk * CHUNK + offs
    vals = tl.load(
        local_val_ptr + row * NT + idx,
        mask=idx < NT,
        other=-float("inf"),
    )
    pos = tl.argmax(vals, axis=0, tie_break_left=True)
    mx = tl.max(vals, axis=0)
    global_idx = chunk * CHUNK + pos
    token = tl.load(
        local_idx_ptr + row * NT + global_idx,
        mask=global_idx < NT,
        other=0,
    )
    tl.store(chunk_val_ptr + row * NCH + chunk, mx)
    tl.store(chunk_idx_ptr + row * NCH + chunk, token)


@triton.jit
def _reduce_top1_final_kernel(
    chunk_val_ptr,
    chunk_idx_ptr,
    out_ptr,
    NCH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    vals = tl.load(
        chunk_val_ptr + row * NCH + offs,
        mask=offs < NCH,
        other=-float("inf"),
    )
    pos = tl.argmax(vals, axis=0, tie_break_left=True)
    token = tl.load(chunk_idx_ptr + row * NCH + pos)
    # RejectionSampler ultimately stores int32 token ids; returning int32 here
    # avoids an otherwise useless int64->int32 conversion.
    tl.store(out_ptr + row, token.to(tl.int32))


@torch.library.custom_op(
    "k100::target_m4_lm_head_top1_w8a8", mutates_args=(), device_types="cuda"
)
def target_m4_lm_head_top1_w8a8(
    hidden_states: torch.Tensor,
    weight_nk: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.ndim != 2 or tuple(hidden_states.shape) != (4, 2048):
        raise RuntimeError(
            f"R259 supports only target hidden [4,2048], got {tuple(hidden_states.shape)}"
        )
    if weight_nk.ndim != 2 or int(weight_nk.shape[1]) != 2048:
        raise RuntimeError(f"R259 unsupported weight shape {tuple(weight_nk.shape)}")

    x = hidden_states.contiguous()
    xq, xs, xzp = ops.scaled_int8_quant(x, None, None, symmetric=True)
    assert xzp is None

    n = int(weight_nk.shape[0])
    bn = 16
    bk = 512
    nt = triton.cdiv(n, bn)
    chunk = 1024
    nch = triton.cdiv(nt, chunk)

    local_vals = torch.empty((4, nt), dtype=torch.bfloat16, device=x.device)
    local_ids = torch.empty((4, nt), dtype=torch.int32, device=x.device)
    _local_top1_m4_kernel[(nt,)](
        xq,
        weight_nk,
        xs,
        weight_scale,
        local_vals,
        local_ids,
        N=n,
        K=2048,
        NT=nt,
        stride_am=xq.stride(0),
        stride_ak=xq.stride(1),
        stride_wn=weight_nk.stride(0),
        stride_wk=weight_nk.stride(1),
        BM=16,
        BN=bn,
        BK=bk,
        num_warps=4,
        num_stages=2,
        waves_per_eu=1,
        matrix_instr_nonkdim=16,
        kpack=2,
    )

    chunk_vals = torch.empty((4, nch), dtype=torch.bfloat16, device=x.device)
    chunk_ids = torch.empty((4, nch), dtype=torch.int32, device=x.device)
    _reduce_top1_chunks_kernel[(4, nch)](
        local_vals,
        local_ids,
        chunk_vals,
        chunk_ids,
        NT=nt,
        CHUNK=chunk,
        NCH=nch,
        num_warps=4,
    )

    out = torch.empty((4,), dtype=torch.int32, device=x.device)
    _reduce_top1_final_kernel[(4,)](
        chunk_vals,
        chunk_ids,
        out,
        NCH=nch,
        BLOCK=triton.next_power_of_2(nch),
        num_warps=1,
    )
    return out


@target_m4_lm_head_top1_w8a8.register_fake
def _target_m4_lm_head_top1_w8a8_fake(
    hidden_states: torch.Tensor,
    weight_nk: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    del weight_nk, weight_scale
    return hidden_states.new_empty((4,), dtype=torch.int32)


def _sampling_is_plain_greedy(runner) -> bool:
    try:
        from vllm import envs

        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            return False
        sm = runner.input_batch.sampling_metadata
        if not sm.all_greedy or sm.max_num_logprobs is not None:
            return False
        if not sm.no_penalties:
            return False
        if sm.allowed_token_ids_mask is not None or bool(sm.bad_words_token_ids):
            return False
        # Greedy ignores top-p/top-k in upstream code, but require the plain
        # configuration anyway so this fast path has an intentionally tiny
        # semantic surface.
        if sm.top_p is not None or sm.top_k is not None:
            return False
        # Under spec decode the only built-in non-argmax-invariant processor is
        # MinTokens.  It is safe only when its sparse active-state dict is empty.
        for proc in sm.logitsprocs.non_argmax_invariant:
            if bool(getattr(proc, "min_toks", None)):
                return False
        if bool(getattr(runner, "num_prompt_logprobs", 0)):
            return False
        if int(getattr(runner.input_batch, "num_reqs", 0)) != 1:
            return False
        # HCU _prepare_inputs writes the per-request decode draft count here
        # before target compute_logits runs.  Require the exact MTP3 hot shape.
        draft_counts = getattr(runner, "num_decode_draft_tokens", None)
        if draft_counts is None or int(draft_counts.np[0]) != 3:
            return False
        return True
    except Exception:
        return False


def install() -> None:
    global _installed
    if _installed:
        return

    from vllm_hcu.v1.hcu_model_runner import GPUModelRunner
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from vllm.v1.outputs import SamplerOutput
    from vllm.v1.sample.rejection_sampler import (
        PLACEHOLDER_TOKEN_ID,
        rejection_greedy_sample_kernel,
    )

    original_load_model = GPUModelRunner.load_model
    original_compute_logits = Qwen3_5ForConditionalGeneration.compute_logits
    original_sample = GPUModelRunner._sample
    original_sample_tokens = GPUModelRunner.sample_tokens

    def load_model_r259(self, *args, **kwargs):
        result = original_load_model(self, *args, **kwargs)
        try:
            raw_model = self.get_model()
            if isinstance(raw_model, Qwen3_5ForConditionalGeneration):
                raw_model._k100_r259_runner = self
                self._k100_r259_compact_logits_active = False
                print(
                    "[K100 R259 target top1] attached HCU runner to Qwen3.5 model",
                    flush=True,
                )
        except Exception as exc:
            print(f"[K100 R259 target top1] attach failed: {exc!r}", flush=True)
        return result

    def compute_logits_r259(self, hidden_states: torch.Tensor):
        runner = getattr(self, "_k100_r259_runner", None)
        lm_head = getattr(getattr(self, "language_model", None), "lm_head", None)
        if (
            runner is not None
            and _sampling_is_plain_greedy(runner)
            and hidden_states.ndim == 2
            and tuple(hidden_states.shape) == (4, 2048)
            and lm_head is not None
            and int(getattr(lm_head, "tp_size", 1)) == 1
            and getattr(lm_head, "weight", None) is not None
            and lm_head.weight.dtype == torch.int8
            and hasattr(lm_head, "k100_weight_scale")
            and int(lm_head.weight.shape[0]) > 0
        ):
            runner._k100_r259_compact_logits_active = True
            if not getattr(runner, "_k100_r259_first_fastpath_logged", False):
                runner._k100_r259_first_fastpath_logged = True
                print(
                    "[K100 R259 target top1] first compact M=4 verifier step",
                    flush=True,
                )
            return target_m4_lm_head_top1_w8a8(
                hidden_states,
                lm_head.weight,
                lm_head.k100_weight_scale,
            )
        if runner is not None:
            runner._k100_r259_compact_logits_active = False
        return original_compute_logits(self, hidden_states)

    def sample_r259(self, logits, spec_decode_metadata):
        if (
            getattr(self, "_k100_r259_compact_logits_active", False)
            and spec_decode_metadata is not None
            and isinstance(logits, torch.Tensor)
            and logits.ndim == 1
            and tuple(logits.shape) == (4,)
            and logits.dtype == torch.int32
            and _sampling_is_plain_greedy(self)
            and list(spec_decode_metadata.num_draft_tokens) == [3]
        ):
            output_token_ids = torch.full(
                (1, int(spec_decode_metadata.max_spec_len) + 1),
                int(PLACEHOLDER_TOKEN_ID),
                dtype=torch.int32,
                device=logits.device,
            )
            # Compact rows are exactly [target0,target1,target2,bonus].
            rejection_greedy_sample_kernel[(1,)](
                output_token_ids,
                spec_decode_metadata.cu_num_draft_tokens,
                spec_decode_metadata.draft_token_ids,
                logits[:3],
                logits[3:4],
                None,
                spec_decode_metadata.max_spec_len,
            )
            return SamplerOutput(
                sampled_token_ids=output_token_ids,
                logprobs_tensors=None,
            )
        return original_sample(self, logits, spec_decode_metadata)

    @torch.inference_mode()
    def sample_tokens_r259(self, grammar_output):
        # Grammar arrives only at sample_tokens(), after target compute_logits.
        # If it exists, reconstruct the exact original full logits from the
        # preserved sample_hidden_states and let upstream handle everything.
        if (
            grammar_output is not None
            and getattr(self, "_k100_r259_compact_logits_active", False)
            and self.execute_model_state is not None
        ):
            state = self.execute_model_state
            if (
                isinstance(state.logits, torch.Tensor)
                and state.logits.ndim == 1
                and state.logits.dtype == torch.int32
            ):
                raw_model = self.get_model()
                full_logits = original_compute_logits(raw_model, state.sample_hidden_states)
                self.execute_model_state = state._replace(logits=full_logits)
                self._k100_r259_compact_logits_active = False
                if not getattr(self, "_k100_r259_grammar_fallback_logged", False):
                    self._k100_r259_grammar_fallback_logged = True
                    print(
                        "[K100 R259 target top1] grammar fallback reconstructed full logits",
                        flush=True,
                    )
        try:
            return original_sample_tokens(self, grammar_output)
        finally:
            self._k100_r259_compact_logits_active = False

    GPUModelRunner.load_model = load_model_r259
    Qwen3_5ForConditionalGeneration.compute_logits = compute_logits_r259
    GPUModelRunner._sample = sample_r259
    GPUModelRunner.sample_tokens = sample_tokens_r259
    _installed = True
    print(
        "[K100 R259 target top1] pure-greedy TP1/MTP3 compact verifier installed",
        flush=True,
    )
