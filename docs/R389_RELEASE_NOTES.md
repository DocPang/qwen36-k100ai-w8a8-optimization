# R389 Release Notes — Adaptive MTP for the Full 512→32K Curve

Date: 2026-08-12

R389 supersedes R269 as the **recommended deployment profile** for Qwen3.6-35B-A3B W8A8 on Hygon K100AI.

R269 remains reproducible as a historical short-context / fixed-512 peak result, but fixed MTP3 is no longer treated as the globally optimal deployment policy.

## What changed

- Added context-aware adaptive MTP scheduling.
- MTP3 is used below 6,144 computed tokens.
- At/above 6,144 tokens, the drafter is skipped and future speculative placeholders are cleared so target decoding returns to true single-token decode.
- Target `max_model_len` remains 262,144; the model is loaded only once.
- Prefix Cache remains enabled.
- Historical R237 multimodal direct-embedding shortcut is disabled by default because later arbitrary-length chunked-prefill-tail testing found semantic risk.
- Existing accepted R269 small-M W8A8 / lm_head / GDN / MoE / verifier optimizations remain in the cumulative patch.

## Why this release exists

The previous public release optimized and reported a fixed-512 MTP3 peak of roughly 106–107 tok/s. That number remains a valid local benchmark result, but a 512→32K sweep showed that a fixed speculative depth is not globally optimal: long-context draft + verify overhead eventually costs more than it saves.

R389 therefore optimizes the **performance envelope**, not one benchmark point.

## 10-point validation curve

| Prompt | Decode tok/s |
|---:|---:|
| 512 | 88.33 |
| 1,024 | 67.31 |
| 2,048 | 69.81 |
| 3,072 | 70.27 |
| 4,096 | 56.66 |
| 6,144 | 40.08 |
| 8,192 | 40.03 |
| 12,288 | 39.19 |
| 16,384 | 39.11 |
| 32,768 | 37.68 |

Mean Decode: **53.29 tok/s**.

The 6K→32K region forms a stable no-MTP plateau instead of continuing to pay fixed MTP3 overhead.

## Quality gates

- exact string: PASS
- arithmetic (`137*29`): PASS
- code extraction: PASS
- deterministic repeat, 3 runs: identical SHA256
- natural ~32K needle retrieval: 3/3 PASS

## Reproduce

```bash
git clone https://github.com/DocPang/qwen36-k100ai-w8a8-optimization.git
cd qwen36-k100ai-w8a8-optimization
python3 -m pip install -U modelscope

MODEL_DIR="$HOME/models/Qwen3.6-35B-A3B-W8A8-DCU" \
GPU_ID=0 PORT=8000 \
bash scripts/quickstart.sh
```

The validated cutoff is 6,144 for this exact hardware/model/runtime profile. It can be overridden for research with `MTP_CUTOFF=...`, but users should profile their own hardware rather than assuming 6,144 is universal.

## Scope

R389 is validated here across **512→32K**. The target service still supports `max_model_len=262144`, but this release does not claim R389 is the formal 262K full-range champion until the same adaptive profile passes the 257,901-token acceptance gate.

For the research story and implementation rationale, see [`ADAPTIVE_MTP_SCHEDULING.md`](ADAPTIVE_MTP_SCHEDULING.md).
