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
