"""R235: remove redundant int32->int64 cast for the hot TP1 single-token embedding path.

vLLM VocabParallelEmbedding.forward_native unconditionally calls masked_input.long().
In the Qwen3.6 MTP3 hot path the token id is already int32 and F.embedding accepts
int32 indices on this ROCm/PyTorch build with bitwise-identical BF16 output.  Keep
all non-TP1, non-int32, and non-single-token cases on the upstream implementation.
"""
from __future__ import annotations

import torch

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return

    import vllm.model_executor.layers.vocab_parallel_embedding as vpe

    cls = vpe.VocabParallelEmbedding
    original = cls.forward_native

    def forward_native_r235(self, input_):
        if (
            self.tp_size == 1
            and input_.dtype == torch.int32
            and input_.numel() == 1
        ):
            # TP=1 means no masking/sharding transform is required.  Preserve the
            # upstream all-reduce call so the method contract remains identical.
            output_parallel = self.quant_method.embedding(self, input_)
            return vpe.tensor_model_parallel_all_reduce(output_parallel)
        return original(self, input_)

    cls.forward_native = forward_native_r235
    _installed = True
    print(
        "[K100 R235 int32 embedding] TP1 single-token .long() cast bypass installed",
        flush=True,
    )
