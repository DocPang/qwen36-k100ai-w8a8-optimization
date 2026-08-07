# Qwen3.6-35B-A3B W8A8 inference optimization for Hygon K100AI

**English** | [简体中文](README.zh-CN.md)

Single-GPU inference tuning for **Qwen3.6-35B-A3B W8A8** on **Hygon K100AI (gfx928)** using the Hygon community vLLM 0.18.1 / DTK 26.04 environment.

> [!IMPORTANT]
> **K100 and K100AI are different accelerator models.**
> This repository was developed and validated on **K100AI** only. It does **not** claim that the same kernels, tile choices, performance, or even compatibility apply to K100.

This repository contains the runtime patches and configuration that produced our best stable single-concurrency results. It intentionally excludes private infrastructure details, model weights, Docker image contents, and experimental branches that did not enter the final configuration.

## Results

All numbers below are single-GPU, tensor-parallel size 1, single-concurrency decode results on K100AI.

| Configuration | MTP | Decode throughput | Notes |
|---|---:|---:|---|
| Upstream hot baseline | off | **19.873 tok/s** | Stock vLLM path under a normalized warm/graph-enabled test |
| R180 | off | **53.46 tok/s avg** | `llm_speedtest`, 128/256/512/1024-token prompts, 512-token output |
| R180 fixed 512-token run | off | **54.61 tok/s** | Repeated fixed prompt |
| R184 fixed 512-token run | MTP3 | **85.29 tok/s median** | Conservative stable figure |
| R184 `llm_speedtest` | MTP3 | **96.84 tok/s avg** | Workload-dependent speculative acceptance |
| R184 peak observed | MTP3 | **107.44 tok/s** | High-acceptance prompt; do not treat as universal throughput |

An older June test report recorded **12.55 tok/s** for the original single-card/no-MTP path, but that result included earlier cold-path and measurement differences. We therefore use **19.873 tok/s** as the normalized stock baseline when discussing optimization uplift.

MTP throughput is inherently prompt-dependent because accepted speculative tokens vary. For production planning, the fixed-prompt **~85 tok/s** result is a safer expectation than the 107 tok/s peak.

See [`results/RESULTS.md`](results/RESULTS.md) for the benchmark table and methodology notes.

## What was optimized

### 1. Shape-aware W8A8 Linear kernels

Decode on this model is dominated by very small-M matrix multiplications. The vendor/general heuristic was not ideal for K100AI/gfx928, so exact `(M, K, N)` decode shapes were mapped to tuned Triton configurations.

Two important no-MTP `M=1` projections improved in microbenchmarks from:

- `2048 -> 12288`: ~371.99 us -> ~90.03 us
- `2048 -> 9216`: ~221.75 us -> ~77.53 us

Both were validated with zero numerical difference against the reference kernel in the tuning harness.

### 2. W8A8 `lm_head`

The 248,320-token vocabulary makes the output projection unusually expensive at decode time. The runtime patch installs a decode-size W8A8 path for the large `lm_head` instead of leaving it on the slower default route.

### 3. K100AI-specific MoE configuration

Qwen3.6-35B-A3B uses 256 routed experts and activates 8 per token. We provide the tuned vLLM FusedMoE configuration:

`configs/E=256,N=512,device_name=K100_AI.json`

The filename follows the device naming expected by this Hygon vLLM build.

### 4. Gated DeltaNet QKVZ + BA fusion (R184)

R184 fuses the QKVZ and BA W8A8 projections in the Gated DeltaNet path, avoiding redundant launch/quantization overhead without the large KV-cache penalty we observed with full weight prepacking experiments.

### 5. `torch.compile` and CUDAGraph capture for the real decode shapes

The final service uses:

- compile mode 3
- combo-kernel benchmark disabled
- CUDAGraph capture size `1` for no-MTP
- CUDAGraph capture sizes `1 4` for MTP3

### 6. Native Qwen MTP head, 3 speculative tokens

The best stable speculative configuration uses the model's own MTP head:

```text
method = qwen3_next_mtp
num_speculative_tokens = 3
quantization = compressed-tensors
```

The target model still verifies drafted tokens. MTP3 speeds up decode when future tokens are accepted, but acceptance rate varies by workload.

## Upstream environment and model sources

This project does **not** redistribute model weights or the vendor Docker image.

### Hygon community vLLM environment

The tested runtime follows the Hygon/SourceFind Qwen3.6 ModelZoo instructions:

- vLLM: `0.18.1+das.fa71803.dtk2604`
- DTK: `26.04`
- Triton: `3.6.0+gitc73250c4.staging`
- Torch: `2.10.0+das.opt1.dtk2604.20260325.g6b060a`
- recommended image tag: `harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`
- exact image digest used for the published results: `sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811`

K100AI support tag (v1.1, “新增K100AI支持，及FP8数据类型”):

<https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/tags>

Environment revision used to pin the vLLM 0.18.1 image/version information:

<https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/commit/8a54c4e3df888c81165e5106657d17865bac3644>

Current ModelZoo page:

<https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/blob/main/README.md>

The Qwen3.6 ModelZoo v1.1 tag explicitly added **K100AI** support. The public launch scripts in this repository also keep `--disable-custom-all-reduce`, which is the K100AI launch requirement used by the tested Hygon vLLM runtime family.

### W8A8 model used in this project

The **weight shards** used for these results came from ModelScope:

**`metax-tech/Qwen3.6-35B-A3B-W8A8`**

<https://www.modelscope.cn/models/metax-tech/Qwen3.6-35B-A3B-W8A8>

However, the published K100AI results did **not** use that repository completely unchanged. The upstream `config.json` has no `quantization_config`. During the original June deployment we created a local **DCU-adapted checkpoint directory** named `Qwen3.6-35B-A3B-W8A8-DCU` and replaced only `config.json` with the compressed-tensors metadata required by the tested Hygon vLLM 0.18.1 stack. The eight safetensors weight shards remained the ModelScope files.

This repository includes the exact validated DCU config:

`configs/Qwen3.6-35B-A3B-W8A8-DCU.config.json`

Validated hashes:

- original ModelScope `config.json`: `ba62ca6d8a773ab4c15407acf0653761198c4bcb74d7e8d82edc88132c4ba6a6`
- DCU-adapted `config.json`: `b550b28342afd4c61841e2684b06da15f3a0ec3c807ceb22259b0074be9975ae`

Download directly into a `-DCU` local directory, then apply the validated config:

```bash
pip install modelscope
modelscope download \
  --model metax-tech/Qwen3.6-35B-A3B-W8A8 \
  --local_dir /path/to/Qwen3.6-35B-A3B-W8A8-DCU

python3 scripts/apply_dcu_config.py \
  --model-dir /path/to/Qwen3.6-35B-A3B-W8A8-DCU
```

The adapter adds the `compressed-tensors` W8A8 description used by our service: static INT8 per-channel weights, dynamic INT8 per-token activations, and the validated ignore list for non-quantized modules.

The original unquantized Qwen model is:

<https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B>

## Quick start

Clone this repository on the K100AI host, download the ModelScope weight shards, and **apply the DCU config before starting vLLM**. Using the raw upstream `config.json` is not the configuration that produced the published results.

### No MTP / R180

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8-DCU
export GPU_ID=0
export PORT=8000
bash scripts/serve_nomtp.sh
```

### MTP3 / R184

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8-DCU
export GPU_ID=0
export PORT=8000
bash scripts/serve_mtp3.sh
```

The scripts default to the validated tag+digest combination, so a later tag update cannot silently change the runtime. Registry authentication may be required by the upstream community service.

## Benchmark

After the service is ready:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/benchmark_openai.py \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model qwen36-35b-a3b-w8a8-k100ai \
  --max-tokens 512 \
  --rounds 6
```

The benchmark prints per-run decode throughput and output SHA256. The first request may trigger runtime compilation; report it separately rather than silently mixing it into steady-state throughput.

## Reproducing with Codex

A sanitized, infrastructure-independent Codex procedure is provided in [`docs/REPRODUCE_WITH_CODEX.md`](docs/REPRODUCE_WITH_CODEX.md).

## Repository layout

```text
configs/                 K100AI FusedMoE config used by vLLM
patches/r180_nomtp/      best no-MTP runtime patch
patches/r184_mtp3/       best MTP3 runtime patch
scripts/serve_nomtp.sh   generic single-GPU R180 launcher
scripts/serve_mtp3.sh    generic single-GPU R184 launcher
scripts/benchmark_openai.py
results/                 sanitized benchmark summary
```

## Scope and limitations

- Validated on **Hygon K100AI / gfx928 only**.
- K100 was **not** used for these results.
- Tuned for single-concurrency / low-concurrency decode. High-concurrency throughput needs separate tuning.
- MTP performance depends on speculative-token acceptance rate.
- The exact Docker image and model weights are external dependencies and are not redistributed here.
- Kernel microbenchmarks are not accepted as final wins unless full-model A/B also improves.

## License

The code in this repository is released under Apache License 2.0. Model weights and the vendor runtime remain under their respective upstream licenses and terms. See [`NOTICE.md`](NOTICE.md).
