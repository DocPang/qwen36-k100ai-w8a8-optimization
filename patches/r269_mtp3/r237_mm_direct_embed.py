"""R237: write pure-text multimodal-runner embeddings directly into its persistent buffer.

The HCU multimodal runner always materializes text embeddings into a temporary tensor
and then copies them into `inputs_embeds.gpu`, even on decode steps with no active
image/video embeddings.  For TP1 pure-text steps, write the embedding rows directly
into that persistent buffer.  Multimodal steps fall back to the original model method.
"""
from __future__ import annotations

import torch

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return

    from vllm_hcu.v1.hcu_model_runner import GPUModelRunner
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    original_load_model = GPUModelRunner.load_model
    original_embed_input_ids = Qwen3_5ForConditionalGeneration.embed_input_ids

    def load_model_r237(self, *args, **kwargs):
        result = original_load_model(self, *args, **kwargs)
        try:
            if self.supports_mm_inputs and self.parallel_config.tensor_parallel_size == 1:
                raw_model = self.get_model()
                if isinstance(raw_model, Qwen3_5ForConditionalGeneration):
                    raw_model._k100_r237_inputs_embeds_buffer = self.inputs_embeds.gpu
                    raw_model._k100_r237_direct_embed_enabled = True
                    print(
                        "[K100 R237 MM direct embed] attached runner inputs_embeds buffer",
                        flush=True,
                    )
        except Exception as exc:
            print(f"[K100 R237 MM direct embed] attach failed: {exc!r}", flush=True)
        return result

    def embed_input_ids_r237(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        buf = getattr(self, "_k100_r237_inputs_embeds_buffer", None)
        # The Python list is empty exactly when no image/video embedding overlaps
        # this scheduling step. Avoid GPU `.any().item()` checks and therefore any
        # new synchronization. Only TP1 runners receive the attached buffer.
        no_mm = multimodal_embeddings is None or len(multimodal_embeddings) == 0
        if (
            buf is not None
            and no_mm
            and input_ids.ndim == 1
            and input_ids.numel() > 0
            and input_ids.numel() <= 4
            and input_ids.dtype in (torch.int32, torch.int64)
        ):
            out = buf[: input_ids.numel()]
            # TP1 VocabParallelEmbedding is a direct BF16 table lookup.  index_select
            # with `out=` preserves the exact rows while avoiding a temporary tensor.
            weight = self.language_model.model.embed_tokens.weight
            torch.index_select(weight, 0, input_ids, out=out)
            return out

        return original_embed_input_ids(
            self,
            input_ids,
            multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    GPUModelRunner.load_model = load_model_r237
    Qwen3_5ForConditionalGeneration.embed_input_ids = embed_input_ids_r237
    _installed = True
    print(
        "[K100 R237 MM direct embed] TP1 pure-text decode fast path installed",
        flush=True,
    )
