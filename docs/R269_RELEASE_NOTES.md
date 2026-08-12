# R269 release notes

> [!WARNING]
> **Superseded for deployment by R389 (2026-08-12).** R269 remains a valid historical fixed-512 / fixed-MTP3 short-context local-peak result, but it should not be interpreted as the globally optimal one-load policy across context lengths. New users should use `scripts/quickstart.sh` and read `docs/R389_RELEASE_NOTES.md`.

R269 is the historical validated single-GPU TP1 + fixed-MTP3 short-context performance release for Qwen3.6-35B-A3B W8A8 on Hygon K100AI / gfx928.

## What changed

R269 keeps the previously accepted K100AI runtime optimizations and fixes a specific inefficiency in the CompressedTensors INT8 routed-MoE path for MTP3 target verification.

For the M=4 verifier path, the two routed-MoE stages have very different shapes:

- stage 1 / W1: input `[4, 2048]`, 32 routed rows, weight `[256, 1024, 2048]`
- stage 2 / W2: input `[32, 512]`, weight `[256, 2048, 512]`

The old path forced both stages to share one configuration. R269 keeps the accepted stage-1 config and overrides only M=4 stage 2 with:

```text
BLOCK_SIZE_M=32
BLOCK_SIZE_N=32
BLOCK_SIZE_K=512
num_warps=4
waves_per_eu=2
kpack=2
num_stages=1
```

In the production-like full `fused_experts` microbenchmark, this reduced the local routed-MoE latency from 241.9239 us to 199.5232 us, about a 1.2125x speedup for that operation.

## End-to-end validation

GPU7 exploratory hot fixed-512 runs:

```text
107.4490
107.4696
107.4034
107.3989
107.4706
107.4694 tok/s
```

Mean: **107.4435 tok/s**  
Median: **107.4592 tok/s**

Formal same-GPU arbitration on GPU1:

- R265 baseline mean: **100.4988 tok/s**
- R269 mean: **106.3198 tok/s**
- R269 median: **106.2957 tok/s**
- mean gain: **+5.79%**

The fixed benchmark output SHA256 was:

```text
80c82006a973ecc78fa3fb7a8483b76bc311693bdf277cb296365be0db6c7e00
```

## Quality gates

R269 was checked against the reference production path:

- 5/5 greedy text identical
- 5/5 logprob text identical
- 5/5 logprob token sequences identical
- maximum logprob absolute difference: **0.0**
- historical multimodal response identical
- MTP draft/accepted-token counters identical for the validation workload
- warm KV capacity unchanged at about 19.95 GiB / 236,912 tokens

The speedup therefore did not come from reducing model work by lowering speculative acceptance or changing the sampled output.

## Reproduction entry points

```text
patches/r269_mtp3/
configs/E=256,N=512,device_name=K100_AI.r269.json
scripts/quickstart_r269.sh
scripts/serve_r269_mtp3.sh
scripts/benchmark_fixed_512.py
```

Use the exact vLLM 0.18.1 / DTK 26.04 image and the specified DCU-adapted W8A8 checkpoint. Different prompts can produce different MTP acceptance rates, so real application throughput can be lower or higher than the fixed benchmark.
