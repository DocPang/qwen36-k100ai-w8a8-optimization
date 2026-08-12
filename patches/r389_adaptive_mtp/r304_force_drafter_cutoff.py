"""R304: force a real effective MTP drafter cutoff for HCU GPUModelRunner.

vLLM's SpeculativeConfig.max_model_len is intended to allow sequences to skip
speculation, but the Qwen3.5/Qwen3.6 built-in MTP path can leave
``draft_model_config`` unset. HCU GPUModelRunner then falls back to the target
``max_model_len`` (262K), so speculation never actually stops.

This patch changes only the runner's *effective drafter max length*. Target
model max length, KV cache, sampling, and target-model math are untouched.
Above the cutoff the existing upstream fallback path is used.
"""
from __future__ import annotations

import os

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    cutoff = int(os.getenv("K100_R304_DRAFTER_MAX_MODEL_LEN", "0"))
    if cutoff <= 0:
        return

    from vllm_hcu.v1.hcu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_k100_r304_installed", False):
        _INSTALLED = True
        return

    original_init = GPUModelRunner.__init__
    original_update_max_model_len = GPUModelRunner.update_max_model_len

    def _force_cutoff(self) -> None:
        spec = getattr(self, "speculative_config", None)
        if spec is None:
            return
        # Do not enlarge a genuinely smaller draft-model limit.
        target_max = int(getattr(self, "max_model_len", cutoff))
        forced = min(cutoff, target_max)
        old = int(getattr(self, "effective_drafter_max_model_len", target_max))
        self.effective_drafter_max_model_len = forced
        if not getattr(self, "_k100_r304_logged", False):
            self._k100_r304_logged = True
            print(
                f"[K100 R304 adaptive MTP] effective drafter max_model_len "
                f"{old}->{forced}; target max_model_len={target_max}",
                flush=True,
            )

    def init_r304(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _force_cutoff(self)

    def update_max_model_len_r304(self, max_model_len: int) -> None:
        original_update_max_model_len(self, max_model_len)
        _force_cutoff(self)

    GPUModelRunner.__init__ = init_r304
    GPUModelRunner.update_max_model_len = update_max_model_len_r304
    GPUModelRunner._k100_r304_installed = True
    _INSTALLED = True
    print(
        f"[K100 R304 adaptive MTP] runner cutoff hook installed cutoff={cutoff}",
        flush=True,
    )
