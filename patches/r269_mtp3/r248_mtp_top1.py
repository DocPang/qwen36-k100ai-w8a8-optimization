"""R248: fused W8A8 lm_head top-1 for TP1 Qwen3.5 MTP draft sampling.

Target-model logits/sampling are untouched.  Only the draft model's greedy
`get_top_tokens()` path uses this custom op for batch size 1.  The matmul keeps
an R263-retuned K100AI tile (BM16/BN16/BK1024); each vocab tile converts its
results to BF16 exactly as the full lm_head does, reduces to a local top-1, and
a second tiny kernel reduces the tile winners to the final token id.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from vllm import _custom_ops as ops

_installed = False


@triton.jit
def _local_top1_kernel(
    a_ptr, w_ptr, a_scale_ptr, w_scale_ptr, local_val_ptr, local_idx_ptr,
    N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_wn: tl.constexpr, stride_wk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GRID_MN: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    raw_pid = tl.program_id(0)
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = raw_pid % NUM_XCDS
    local_pid = raw_pid // NUM_XCDS
    if xcd < tall_xcds:
        pid_n = xcd * pids_per_xcd + local_pid
    else:
        pid_n = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # AITER's M=1/BM16 kernel repeats the same row 16 times to use MFMA.
    offs_am = tl.arange(0, BLOCK_SIZE_M) % 1
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        aa = tl.load(a_ptrs)
        ww = tl.load(w_ptrs, mask=offs_n[None, :] < N, other=0)
        acc += tl.dot(aa, ww, input_precision="ieee")
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += BLOCK_SIZE_K * stride_wk

    a_scale = tl.load(a_scale_ptr)
    w_scale = tl.load(w_scale_ptr + offs_n, mask=offs_n < N, other=0.0)
    # All BM rows are identical for M=1. Reduce the duplicates, then reproduce
    # the exact full-lm_head fp32 scale -> bf16 store semantics before argmax.
    row = (tl.max(acc, axis=0) * a_scale * w_scale).to(tl.bfloat16)
    row = tl.where(offs_n < N, row, -float("inf"))
    local_arg = tl.argmax(row, axis=0, tie_break_left=True)
    local_val = tl.max(row, axis=0)
    tl.store(local_val_ptr + pid_n, local_val)
    tl.store(
        local_idx_ptr + pid_n,
        (pid_n * BLOCK_SIZE_N + local_arg).to(tl.int32),
    )


@triton.jit
def _reduce_top1_kernel(
    local_val_ptr, local_idx_ptr, out_ptr,
    NT: tl.constexpr, BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    vals = tl.load(local_val_ptr + offs, mask=offs < NT, other=-float("inf"))
    pos = tl.argmax(vals, axis=0, tie_break_left=True)
    token = tl.load(local_idx_ptr + pos)
    tl.store(out_ptr, token.to(tl.int64))


@torch.library.custom_op(
    "k100::mtp_lm_head_top1_w8a8", mutates_args=(), device_types="cuda"
)
def mtp_lm_head_top1_w8a8(
    x: torch.Tensor, weight_nk: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    if x.ndim != 2 or int(x.shape[0]) != 1 or int(x.shape[1]) != 2048:
        raise RuntimeError(f"R248 supports only [1,2048], got {tuple(x.shape)}")
    x2 = x.contiguous()
    xq, xs, xzp = ops.scaled_int8_quant(x2, None, None, symmetric=True)
    assert xzp is None
    n = int(weight_nk.shape[0])
    nt = triton.cdiv(n, 16)
    local_vals = torch.empty((nt,), dtype=torch.bfloat16, device=x.device)
    local_ids = torch.empty((nt,), dtype=torch.int32, device=x.device)
    out = torch.empty((1,), dtype=torch.int64, device=x.device)
    _local_top1_kernel[(nt,)](
        xq, weight_nk, xs, weight_scale, local_vals, local_ids,
        N=n, K=2048,
        stride_am=xq.stride(0), stride_ak=xq.stride(1),
        stride_wn=weight_nk.stride(0), stride_wk=weight_nk.stride(1),
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=1024,
        GRID_MN=nt, NUM_XCDS=8,
        num_warps=4, num_stages=2, waves_per_eu=2,
        matrix_instr_nonkdim=16, kpack=2,
    )
    _reduce_top1_kernel[(1,)](
        local_vals, local_ids, out,
        NT=nt, BLOCK=triton.next_power_of_2(nt), num_warps=4,
    )
    return out


@mtp_lm_head_top1_w8a8.register_fake
def _mtp_lm_head_top1_w8a8_fake(
    x: torch.Tensor, weight_nk: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    del weight_nk, weight_scale
    return x.new_empty((x.shape[0],), dtype=torch.int64)


def install() -> None:
    global _installed
    if _installed:
        return
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        lm_head = self.lm_head
        if (
            hidden_states.ndim == 2
            and int(hidden_states.shape[0]) == 1
            and int(hidden_states.shape[1]) == 2048
            and int(getattr(lm_head, "tp_size", 1)) == 1
            and getattr(lm_head, "weight", None) is not None
            and lm_head.weight.dtype == torch.int8
            and hasattr(lm_head, "k100_weight_scale")
            and int(lm_head.weight.shape[0]) == int(self.config.vocab_size)
        ):
            return mtp_lm_head_top1_w8a8(
                hidden_states, lm_head.weight, lm_head.k100_weight_scale
            )
        # Safe fallback for multi-request batches and any unsupported layout.
        logits = self.compute_logits(hidden_states)
        assert logits is not None
        return logits.argmax(dim=-1)

    Qwen3_5MTP.get_top_tokens = get_top_tokens
    _installed = True
    print(
        "[K100 R263 draft top1] TP1 batch1 fused W8A8 lm_head BN16/BK1024 installed",
        flush=True,
    )
