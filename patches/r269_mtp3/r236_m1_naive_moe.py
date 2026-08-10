"""R236: restore vLLM's sparse/naive MoE block assignment for the hot M=1 path.

The legacy fused_experts_impl already uses naive block assignment when routing is
sufficiently sparse.  The newer modular TritonExperts.apply path unconditionally
calls moe_align_block_size instead.  For Qwen3.6-35B-A3B TP1 decode, M=1/topk=8
with 256 experts is exactly the sparse case.  Use the legacy metadata contract
only for that shape; M=4 verifier and every other path remain upstream.
"""
from __future__ import annotations

import torch

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return

    import vllm.model_executor.layers.fused_moe.fused_moe as fm

    original = fm.moe_align_block_size

    def moe_align_block_size_r236(
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        expert_map: torch.Tensor | None = None,
        pad_sorted_ids: bool = False,
        ignore_invalid_experts: bool = False,
    ):
        if (
            expert_map is None
            and not pad_sorted_ids
            and not ignore_invalid_experts
            and topk_ids.ndim == 2
            and topk_ids.shape[0] == 1
            and topk_ids.shape[1] == 8
            and num_experts == 256
            and block_size == 32
        ):
            # Exact legacy fused_experts_impl sparse assignment contract.
            # Each routed token/expert pair gets one BLOCK_SIZE_M block; the
            # fused_moe kernel's naive_block_assignment path maps pid_m directly
            # to that route and masks the remaining rows in the block.
            expert_ids = topk_ids.view(-1)
            num_tokens_post_padded = torch.empty(
                (1), dtype=torch.int32, device=topk_ids.device
            )
            num_tokens_post_padded.fill_(topk_ids.numel() * block_size)
            return None, expert_ids, num_tokens_post_padded

        return original(
            topk_ids,
            block_size,
            num_experts,
            expert_map,
            pad_sorted_ids,
            ignore_invalid_experts,
        )

    fm.moe_align_block_size = moe_align_block_size_r236
    _installed = True
    print(
        "[K100 R236 M1 naive MoE] modular TritonExperts sparse assignment installed",
        flush=True,
    )
