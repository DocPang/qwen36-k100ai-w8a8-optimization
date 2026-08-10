# Benchmark results

Hardware: **Hygon K100AI (gfx928)**, one GPU, TP=1, single concurrency.

## Comparable hot-path results

| Build | Speculative decoding | Decode throughput | Measurement |
|---|---|---:|---|
| Stock runtime | off | 19.873 tok/s | normalized warm baseline |
| R180 | off | 53.46 tok/s average | 4 prompt lengths, 512 output tokens |
| R180 | off | 54.61 tok/s | repeated fixed 512-token run |
| R184 | MTP3 | 85.29 tok/s median | 6 repeated fixed 512-token runs |
| R184 | MTP3 | 96.84 tok/s average | 4 prompt lengths, workload-dependent MTP acceptance |
| R184 | MTP3 | 107.44 tok/s peak | one high-acceptance prompt length |
| R202 Agent profile | MTP3 | **85.21 tok/s median** | 4 fixed 512-token GPU0 runs; 262K + Prefix Cache + multimodal |
| R265 | MTP3 | **~100.55 tok/s median** | GPU1 same-card pre-R269 reference |
| **R269** | MTP3 | **106.30 tok/s GPU1 median / 107.46 tok/s GPU7 median** | current accepted single-GPU champion |

## R269 release — 2026-08-10

R269 is the current cumulative TP1 + MTP3 release. It retains the previously accepted exact kernel/runtime stack and changes only the M=4 routed-MoE second GEMM to use a stage-2-specific configuration.

Formal GPU1 arbitration:

- R265 12-run mean: **100.4988 tok/s**;
- R265 12-run median: **100.5479 tok/s**;
- R269 12-run mean: **106.3198 tok/s**;
- R269 12-run median: **106.2957 tok/s**;
- same-card gain: **+5.79% mean / +5.72% median**.

GPU7 exploratory six-run steady result:

- 107.4490 / 107.4696 / 107.4034 / 107.3989 / 107.4706 / 107.4694 tok/s;
- mean **107.4435 tok/s**;
- median **107.4592 tok/s**.

All fixed-512 R269 outputs used SHA256 `80c82006a973ecc78fa3fb7a8483b76bc311693bdf277cb296365be0db6c7e00`. Greedy text, logprob token sequence/values, historical multimodal output and MTP acceptance counters matched the reference validation. Full notes: [`../docs/R269_RELEASE_NOTES.md`](../docs/R269_RELEASE_NOTES.md).

### R202, long-context Agent profile

R202 keeps the features required by a long-running Agent service while recovering most of the Decode throughput lost when Prefix Cache is enabled for the hybrid GDN/Mamba path.

Configuration highlights:

- R199 HCU/Mamba-align runtime fastpath;
- accepted-token metadata stays on GPU for the common decode path;
- `max_num_batched_tokens=4096`;
- MTP3;
- `max_model_len=262144`;
- Prefix Cache enabled;
- multimodal enabled;
- Tool Calling enabled.

Final GPU0 fixed-512 hot runs:

| Round | Decode tok/s |
|---:|---:|
| 1 | 85.08 |
| 2 | 84.81 |
| 3 | 85.33 |
| 4 | 85.49 |

Median: **85.21 tok/s**. All four outputs had the same SHA256.

A standard ~55.8K-token repeated-prefix validation on the equivalent R202 profile showed roughly **73.8 s cold -> 5.8 s hot**, demonstrating that the recovered Decode speed does not remove the long-prefix cache benefit.

The previous full-feature Agent baseline was about 73.5-73.8 tok/s, so the R199 + 4096 path recovers roughly **15%+** Decode throughput while keeping long context, Prefix Cache and multimodal capability.

### R180, no MTP

| Input tokens | Prefill tok/s | Decode tok/s | TTFT ms |
|---:|---:|---:|---:|
| 128 | 71.94 | 53.43 | 1794.04 |
| 256 | 1230.97 | 53.46 | 229.31 |
| 512 | 2536.54 | 53.49 | 224.09 |
| 1024 | 3476.67 | 53.45 | 314.87 |

### R184, MTP3

| Input tokens | Prefill tok/s | Decode tok/s | TTFT ms |
|---:|---:|---:|---:|
| 128 | 69.76 | 89.25 | 1846.50 |
| 256 | 1158.03 | 87.25 | 233.48 |
| 512 | 2317.75 | 107.44 | 235.61 |
| 1024 | 2766.12 | 103.42 | 386.11 |

The MTP streaming path can emit multiple accepted tokens in one stream chunk. Chunk-to-chunk latency is therefore not the same thing as token-to-token latency; throughput above is based on generated token count divided by measured decode duration.

## Historical June report

An older June report recorded 12.55 tok/s for single-card/no-MTP. We preserve that number for provenance, but do **not** use it as the normalized speedup denominator because the early report included cold-path and measurement differences.

## Correctness gates

- R184 was compared with its same-mode MTP3 baseline using fixed prompts: text, token sequence, and logprobs were identical in the validation set.
- R180's two dominant `M=1` projection kernels were numerically identical to the reference kernel in microbenchmarks (`relative-L2 = 0`).
- No-MTP and MTP3 are separate inference modes and are **not** claimed to produce byte-identical generations to each other.
- R202 fixed-512 GPU0 validation produced identical output SHA256 across all four hot runs.
- The 4096/8192 compile-range experiments showed that deterministic `temperature=0` text can still diverge on near-tie tokens when the compiled numeric path changes; therefore structured output, Tool Calling and task-level gates remain part of deployment validation.
