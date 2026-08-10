# Changelog

This file records accepted public releases. New optimization work should be added here and to the top-level Release History in `README.md`; historical documentation should not be replaced.

## R269 — 2026-08-10

Current accepted single-GPU TP1 + MTP3 champion for Qwen3.6-35B-A3B W8A8 on Hygon K100AI.

- Base: accepted R265 cumulative exact kernel/runtime stack.
- New delta: dedicated M=4 routed-MoE stage-2 config (`BM32/BN32/BK512`, 4 warps, waves2, kpack2, stages1).
- GPU1 formal 12-run result: 106.3198 tok/s mean, 106.2957 tok/s median.
- GPU7 six-run result: 107.4435 tok/s mean, 107.4592 tok/s median.
- Same-GPU gain over R265: +5.79% mean / +5.72% median.
- Correctness: fixed-512 SHA identical; greedy/logprob/multimodal/MTP acceptance gates passed.
- Reproduce: `scripts/quickstart_r269.sh` + `scripts/benchmark_fixed_512.py`.
- Details: `docs/R269_RELEASE_NOTES.md`.

## R199 / R202 — 2026-08-08

Long-running Agent profile.

- 262K context.
- Prefix Cache.
- multimodal and Tool Calling.
- HCU/Mamba-align runtime fast path and GPU-side speculative metadata.
- `max_num_batched_tokens=4096`.
- Fixed-512 GPU0 median: 85.21 tok/s.

## R184 — 2026-08-07

MTP3 hot-path milestone.

- small-M W8A8 Linear tuning.
- INT8 `lm_head`.
- GDN QKVZ+BA fusion.
- fixed-512 median: 85.29 tok/s.
- `llm_speedtest` workload average: 96.84 tok/s; observed workload peak: 107.44 tok/s.

## R180 — 2026-08-07

No-MTP kernel milestone.

- fixed-512: 54.61 tok/s.
- demonstrated that exact small-M kernel tuning alone could move the single-GPU path from ~19.9 tok/s to the mid-50s.

## Earlier baseline

- normalized stock/hot no-MTP reference: 19.873 tok/s.
- historical June report: 12.55 tok/s (preserved for provenance; not used as the normalized speedup denominator).
