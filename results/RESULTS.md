# Benchmark results — Qwen3.6-35B-A3B W8A8 / K100AI

## Scope

All headline numbers below use:

- Hygon **K100AI / gfx928**
- one GPU, TP=1
- `metax-tech/Qwen3.6-35B-A3B-W8A8`
- DCU-adapted config supplied by this repository
- vLLM 0.18.1 / DTK 26.04
- single-request decode
- hot runtime after initial compile/JIT

**K100 and K100AI are different products.** These results do not claim the same performance on K100.

## Current accepted champion: R269 + MTP3

R269 is the current exact single-GPU champion in this repository.

### GPU7 fixed-512 steady runs

```text
107.4490
107.4696
107.4034
107.3989
107.4706
107.4694 tok/s
```

- mean: **107.4435 tok/s**
- median: **107.4592 tok/s**

### Formal same-GPU comparison on GPU1

Immediately before R269, the same GPU1 ran the accepted R265 base:

- R265 mean: **100.4988 tok/s**
- R269 mean: **106.3198 tok/s**
- R269 median: **106.2957 tok/s**
- same-card mean gain: **+5.79%**

Fixed-512 output SHA256:

```text
80c82006a973ecc78fa3fb7a8483b76bc311693bdf277cb296365be0db6c7e00
```

Quality gates against the reference production path:

- 5/5 greedy text identical
- 5/5 logprob text identical
- 5/5 logprob token sequence identical
- max logprob absolute difference: **0.0**
- historical multimodal response identical
- MTP draft/accepted-token counters identical on the validation workload

R269 therefore improves the execution path rather than obtaining speed by changing speculative acceptance or generated output.

## Comparable historical results

| Build | MTP | Profile | Decode throughput |
|---|---:|---|---:|
| normalized stock hot baseline | off | TP1 | **19.873 tok/s** |
| R180 | off | optimized W8A8 | **54.61 tok/s fixed-512** |
| R184 | MTP3 | older 32K text-only path | **85.29 tok/s median** |
| R202 | MTP3 | 262K + Prefix Cache + multimodal | **85.21 tok/s median** |
| **R269** | **MTP3** | 262K + Prefix Cache + multimodal | **106.30 tok/s formal GPU1 / 107.46 tok/s GPU7 median** |

The old R184 `107.44 tok/s` figure was a prompt-dependent high observation from the earlier branch. R269 is different: the **106–107 tok/s range is now the accepted fixed-512 steady result** on the validated R269 stack.

## Why a generic webpage benchmark can show lower numbers

MTP throughput depends strongly on speculative-token acceptance. The repository's fixed benchmark keeps the prompt and sampling settings constant so versions can be compared fairly.

A generic benchmark tool that generates different prompts may show substantially different decode throughput even on the same R269 server. This is expected and is not evidence that the patch failed.

Prefix Cache is a separate concern. It mainly changes repeated-prefix prefill/TTFT. When validating decode, check the server's `Prefix cache hit rate` rather than assuming a later run is faster because it was cached.

## Reproduce the fixed-512 benchmark

After starting R269 with `scripts/serve_r269_mtp3.sh`:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/benchmark_fixed_512.py \
  --base http://127.0.0.1:8000 \
  --model qwen36-35b-a3b-w8a8-k100ai \
  --rounds 6 \
  --max-tokens 512 \
  --out results/my_r269_fixed512.json
```

The script performs a short warmup first, then prints every run, median/mean throughput and output SHA256.

Do not include the initial Triton/compile request in the steady-state result.

## R269 local MoE change

During MTP3 target verification, M=4 routed MoE uses different shapes in W1 and W2. R269 keeps the accepted stage-1 configuration but gives stage 2 its own K=512 configuration.

Production-like full `fused_experts` microbenchmark:

- previous common config: **241.9239 us**
- R269 stage-2 split config: **199.5232 us**
- local speedup: **1.2125x**

See [`docs/R269_RELEASE_NOTES.md`](../docs/R269_RELEASE_NOTES.md) for the exact geometry and validation details.
