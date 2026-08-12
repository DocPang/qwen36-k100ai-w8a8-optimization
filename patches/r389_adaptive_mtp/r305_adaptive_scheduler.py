"""R305: stop scheduling speculative placeholders above a context cutoff.

HCU/vLLM async scheduling always seeds ``request.spec_token_ids`` with a
fixed-length placeholder list after every decode step whenever speculative
decoding is configured.  When the MTP drafter is skipped because the sequence
is too long, these placeholders still make the next target step verify M=4,
which defeats the intended long-context fallback.

This patch preserves the normal async output placeholder bookkeeping for the
current step, but clears *future* speculative placeholders once the request's
computed sequence length reaches the configured cutoff.  The next step then
schedules a normal single-token target decode.
"""
from __future__ import annotations

import os

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    cutoff = int(os.getenv("K100_R305_SPEC_CUTOFF", "0"))
    if cutoff <= 0:
        return

    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    if getattr(AsyncScheduler, "_k100_r305_installed", False):
        _INSTALLED = True
        return

    original = AsyncScheduler._update_after_schedule
    hits = {"n": 0}

    def update_after_schedule_r305(self, scheduler_output) -> None:
        original(self, scheduler_output)

        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(req_id)
            if request is None or request.is_finished() or request.is_prefill_chunk:
                continue

            # num_computed_tokens is already advanced by Scheduler's base
            # _update_after_schedule.  Clearing spec_token_ids here affects only
            # what will be scheduled on the *next* step; current-step output
            # placeholder accounting remains untouched.
            if int(request.num_computed_tokens) >= cutoff:
                if request.spec_token_ids:
                    request.spec_token_ids = []
                    hits["n"] += 1
                    if hits["n"] <= 3:
                        print(
                            f"[K100 R305 adaptive scheduler] req={req_id} "
                            f"seq={int(request.num_computed_tokens)} >= {cutoff}; "
                            "next step speculative placeholders cleared",
                            flush=True,
                        )

    AsyncScheduler._update_after_schedule = update_after_schedule_r305
    AsyncScheduler._k100_r305_installed = True
    _INSTALLED = True
    print(
        f"[K100 R305 adaptive scheduler] installed cutoff={cutoff}",
        flush=True,
    )
