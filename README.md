# Qwen3.6-35B-A3B W8A8 on Hygon K100AI

[**中文**](#zh-cn) · [**English**](#english)

> [!IMPORTANT]
> **K100 和 K100AI 是两个不同型号。** 本项目只在 **Hygon K100AI / gfx928** 上开发和验证，本文所有性能数据均来自 K100AI。

---

<a id="zh-cn"></a>

# 中文

本项目针对 **Qwen3.6-35B-A3B W8A8** 在 **海光 K100AI** 上的单卡、低并发 Decode 性能进行优化，并提供可直接复现的运行补丁、MoE 配置、DCU 模型配置、启动脚本和测速脚本。

## 1. 做了哪些优化

最终保留的优化主要有以下几项。

### 1.1 Decode 小 M W8A8 Linear 专用配置

Qwen3.6-35B-A3B 在单并发 Decode 时大量 GEMM 的 `M` 很小。通用 Triton heuristic 在 K100AI/gfx928 上并不理想，因此我们按真实 `(M, K, N)` 形状为 `M=1~4` 选择专用配置。

其中无 MTP 路径两个主要投影的微基准变化为：

| 投影 | 原路径 | 优化后 | 提升 |
|---|---:|---:|---:|
| `2048 -> 12288` | 371.99 μs | 90.03 μs | 4.13× |
| `2048 -> 9216` | 221.75 μs | 77.53 μs | 2.86× |

这部分代码位于：

```text
patches/r180_nomtp/sitecustomize.py
patches/r184_mtp3/sitecustomize.py
```

### 1.2 W8A8 `lm_head`

模型词表为 248,320，Decode 阶段 `lm_head` 开销较大。最终补丁为小 M Decode 路径启用了专门的 W8A8 `lm_head` 实现。

### 1.3 K100AI MoE 配置调优

模型使用 256 个 routed experts，每 token 激活 8 个专家。仓库提供最终使用的 vLLM FusedMoE 配置：

```text
configs/E=256,N=512,device_name=K100_AI.json
```

### 1.4 Gated DeltaNet QKVZ + BA 融合

R184 在 Gated DeltaNet 路径中融合了 QKVZ 与 BA 两组 W8A8 投影，减少重复 kernel launch 和量化/写回开销。

### 1.5 `torch.compile` + CUDAGraph

最终运行参数针对单卡 Decode 路径固定为：

- `torch.compile` mode 3
- 关闭无收益的 combo-kernel benchmark
- R180 无 MTP：CUDAGraph capture size `1`
- R184 MTP3：CUDAGraph capture size `1 4`

### 1.6 模型原生 MTP3

R184 使用 Qwen 模型自带 MTP head：

```text
method = qwen3_next_mtp
num_speculative_tokens = 3
quantization = compressed-tensors
```

MTP 的最终 token 仍由目标模型验证。实际速度取决于 speculative token 接受率，因此不同提示词的速度会有明显变化。

---

## 2. 性能结果

测试条件：

- GPU：**Hygon K100AI / gfx928**
- 单卡，TP=1
- 单并发
- vLLM 0.18.1 / DTK 26.04
- 模型：本 README 第 3 节所述 W8A8 + DCU 配置

| 方案 | MTP | Decode 性能 |
|---|---:|---:|
| 原版热态基线 | 关闭 | **19.873 tok/s** |
| R180 | 关闭 | **53.46 tok/s 平均** |
| R180 固定 512-token | 关闭 | **54.61 tok/s** |
| R184 固定 512-token | MTP3 | **85.29 tok/s 中位数** |
| R184 `llm_speedtest` | MTP3 | **96.84 tok/s 平均** |
| R184 最高实测 | MTP3 | **107.44 tok/s** |

MTP 速度与提示词接受率有关。部署时建议把 **约 85 tok/s** 视为更保守的稳定参考，而不是把 107 tok/s 当作所有请求的固定速度。

详细结果见：

```text
results/RESULTS.md
results/benchmark_summary.json
```

---

## 3. 模型来源与 DCU 适配

### 3.1 使用的 W8A8 权重

权重来自 ModelScope：

```text
metax-tech/Qwen3.6-35B-A3B-W8A8
```

模型页：

https://www.modelscope.cn/models/metax-tech/Qwen3.6-35B-A3B-W8A8

这套权重本身已经是 W8A8，不需要重新量化。模型自带的 `llmc_qconfig.yaml` 表明其量化方式为：

- LLMC / SmoothQuant
- weight：INT8、per-channel、symmetric
- activation：INT8、per-token、symmetric
- `use_mtp: true`
- `save_vllm: true`

### 3.2 `W8A8-DCU` 是什么

`Qwen3.6-35B-A3B-W8A8-DCU` **不是另一个公开的 ModelScope model ID**。

本项目中的 `-DCU` 表示：

> 下载 `metax-tech/Qwen3.6-35B-A3B-W8A8` 的 W8A8 权重后，应用本仓库提供的 DCU/vLLM 配置，使海光 vLLM 0.18.1 通过 `compressed-tensors` 正确加载这套已经量化好的权重。

**不会重新量化或修改 safetensors 权重。**

上游 `config.json` SHA256：

```text
ba62ca6d8a773ab4c15407acf0653761198c4bcb74d7e8d82edc88132c4ba6a6
```

本项目实际验证的 DCU `config.json` SHA256：

```text
b550b28342afd4c61841e2684b06da15f3a0ec3c807ceb22259b0074be9975ae
```

DCU 配置文件：

```text
configs/Qwen3.6-35B-A3B-W8A8-DCU.config.json
```

应用脚本：

```text
scripts/apply_dcu_config.py
```

启动脚本会强制校验 DCU config SHA256。如果未完成适配，会直接拒绝启动，避免使用错误配置得到不可比结果。

---

## 4. 运行环境

本项目使用海光/光合开发者社区 Qwen3.6 对应的 vLLM 0.18 环境。

参考：

- Qwen3.6 ModelZoo：https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/blob/main/README.md
- Tags：https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/tags

验证环境：

```text
GPU:    Hygon K100AI / gfx928
vLLM:  0.18.1
DTK:   26.04
Torch: 2.10.0
Triton: 3.6.x Hygon build
```

Docker image：

```text
harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423
```

本项目测试时使用的 image digest：

```text
sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811
```

公开启动脚本默认锁定该 tag + digest。

---

## 5. 完整复现步骤

### 5.1 克隆本仓库

```bash
git clone https://github.com/DocPang/qwen36-k100ai-w8a8-optimization.git
cd qwen36-k100ai-w8a8-optimization
```

### 5.2 下载 W8A8 模型

```bash
pip install modelscope

modelscope download \
  --model metax-tech/Qwen3.6-35B-A3B-W8A8 \
  --local_dir /path/to/Qwen3.6-35B-A3B-W8A8-DCU
```

目录名可以自行修改，但建议保留 `-DCU`，便于区分已经完成 DCU 配置适配的部署目录。

### 5.3 应用 DCU 配置

```bash
python3 scripts/apply_dcu_config.py \
  --model-dir /path/to/Qwen3.6-35B-A3B-W8A8-DCU
```

成功时应看到最终 config SHA256：

```text
b550b28342afd4c61841e2684b06da15f3a0ec3c807ceb22259b0074be9975ae
```

脚本会保留原始配置备份：

```text
config.json.upstream.bak
```

### 5.4 启动 R180：无 MTP

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8-DCU
export GPU_ID=0
export PORT=8000

bash scripts/serve_nomtp.sh
```

对应的核心 `vllm serve` 参数如下，供手工部署和排障时参考。该命令假设已经按 `scripts/serve_nomtp.sh` 挂载 R180 runtime patch 和 K100AI MoE 配置：

```bash
vllm serve /models/qwen36-w8a8 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --quantization compressed-tensors \
  --served-model-name qwen36-35b-a3b-w8a8-k100ai \
  --language-model-only \
  --generation-config vllm \
  --disable-custom-all-reduce \
  -cc.mode=3 \
  -cc.inductor_compile_config '{"combo_kernels": false, "benchmark_combo_kernel": false}' \
  --cudagraph-capture-sizes 1 \
  --max-num-seqs 32
```

预期稳定 Decode：

```text
约 53~55 tok/s
```

### 5.5 启动 R184：MTP3

先停止上一测试容器，或指定新的容器名/端口，然后：

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8-DCU
export GPU_ID=0
export PORT=8000

bash scripts/serve_mtp3.sh
```

对应的核心 `vllm serve` 参数如下。R184 除了加载自己的 runtime patch，还开启 Qwen 原生 MTP3：

```bash
vllm serve /models/qwen36-w8a8 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --quantization compressed-tensors \
  --served-model-name qwen36-35b-a3b-w8a8-k100ai \
  --language-model-only \
  --generation-config vllm \
  --disable-custom-all-reduce \
  -cc.mode=3 \
  -cc.inductor_compile_config '{"combo_kernels": false, "benchmark_combo_kernel": false}' \
  --cudagraph-capture-sizes 1 4 \
  --max-num-seqs 32 \
  --speculative-config '{"model":"/models/qwen36-w8a8","method":"qwen3_next_mtp","num_speculative_tokens":3,"quantization":"compressed-tensors"}'
```

保守稳定参考：

```text
约 85 tok/s
```

高 speculative acceptance 的请求可能达到 100+ tok/s。

---

## 6. 测速

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

服务 ready 后：

```bash
python3 scripts/benchmark_openai.py \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model qwen36-35b-a3b-w8a8-k100ai \
  --max-tokens 512 \
  --rounds 6
```

首次请求可能触发 Triton/compile 编译，不应把第一次冷启动耗时与稳定 Decode 混为一谈。

建议至少记录：

- completion tokens
- wall time
- tokens/s
- 输出 SHA256

---

## 7. 仓库结构

```text
configs/
  E=256,N=512,device_name=K100_AI.json
  Qwen3.6-35B-A3B-W8A8-DCU.config.json

patches/
  r180_nomtp/sitecustomize.py
  r184_mtp3/sitecustomize.py

scripts/
  apply_dcu_config.py
  serve_nomtp.sh
  serve_mtp3.sh
  benchmark_openai.py

results/
  RESULTS.md
  benchmark_summary.json
```

---

## 8. 适用范围

- 仅在 **Hygon K100AI / gfx928** 验证。
- **K100 与 K100AI 不是同一型号。** 本项目不宣称可直接用于 K100。
- 主要针对单并发/低并发 Decode 延迟优化。
- 高并发吞吐需要重新评估。
- MTP 性能取决于 speculative token 接受率。
- 模型权重与海光 Docker image 均由各自上游提供，本仓库不重新分发。

---

<a id="english"></a>

# English

This project optimizes **Qwen3.6-35B-A3B W8A8** for single-GPU, low-concurrency decoding on **Hygon K100AI / gfx928**. It provides the final runtime patches, tuned MoE config, validated DCU model config, launch scripts, and benchmark scripts required to reproduce the results.

> [!IMPORTANT]
> **K100 and K100AI are different accelerator models.** All results in this repository were measured on **K100AI** only.

## 1. Optimizations

### 1.1 Shape-aware W8A8 Linear tuning

Single-concurrency decode is dominated by very small-M GEMMs. The generic Triton heuristic was not optimal for K100AI/gfx928, so exact `(M, K, N)` decode shapes for `M=1~4` were mapped to tuned configurations.

Two important no-MTP projections improved from:

| Projection | Before | After | Speedup |
|---|---:|---:|---:|
| `2048 -> 12288` | 371.99 μs | 90.03 μs | 4.13× |
| `2048 -> 9216` | 221.75 μs | 77.53 μs | 2.86× |

Implemented in:

```text
patches/r180_nomtp/sitecustomize.py
patches/r184_mtp3/sitecustomize.py
```

### 1.2 W8A8 `lm_head`

The model has a 248,320-token vocabulary. A small-M W8A8 path is used for the large output projection during decode.

### 1.3 K100AI MoE tuning

The model contains 256 routed experts with top-8 routing. The tuned vLLM FusedMoE configuration is:

```text
configs/E=256,N=512,device_name=K100_AI.json
```

### 1.4 Gated DeltaNet QKVZ + BA fusion

R184 fuses the QKVZ and BA W8A8 projections in the Gated DeltaNet path to reduce launch and quantization/writeback overhead.

### 1.5 `torch.compile` + CUDAGraph

Final settings:

- compile mode 3
- combo-kernel benchmark disabled
- R180 no-MTP: CUDAGraph capture size `1`
- R184 MTP3: CUDAGraph capture sizes `1 4`

### 1.6 Native Qwen MTP3

R184 uses the model's own MTP head:

```text
method = qwen3_next_mtp
num_speculative_tokens = 3
quantization = compressed-tensors
```

Speculative tokens are still verified by the target model. Throughput therefore depends on acceptance rate.

---

## 2. Results

Test conditions:

- Hygon K100AI / gfx928
- one GPU, TP=1
- single concurrency
- vLLM 0.18.1 / DTK 26.04

| Configuration | MTP | Decode throughput |
|---|---:|---:|
| normalized stock baseline | off | **19.873 tok/s** |
| R180 | off | **53.46 tok/s average** |
| R180 fixed 512-token | off | **54.61 tok/s** |
| R184 fixed 512-token | MTP3 | **85.29 tok/s median** |
| R184 `llm_speedtest` | MTP3 | **96.84 tok/s average** |
| R184 peak observed | MTP3 | **107.44 tok/s** |

For production planning, ~85 tok/s is a more conservative MTP3 reference than the 107 tok/s peak.

See:

```text
results/RESULTS.md
results/benchmark_summary.json
```

---

## 3. Model and DCU configuration

### 3.1 W8A8 weights

The weights come from ModelScope:

```text
metax-tech/Qwen3.6-35B-A3B-W8A8
```

https://www.modelscope.cn/models/metax-tech/Qwen3.6-35B-A3B-W8A8

The checkpoint is already quantized. Its `llmc_qconfig.yaml` describes:

- LLMC / SmoothQuant
- INT8 per-channel symmetric weights
- INT8 per-token symmetric activations
- `use_mtp: true`
- `save_vllm: true`

No weight requantization is required.

### 3.2 What `W8A8-DCU` means

`Qwen3.6-35B-A3B-W8A8-DCU` is **not a separate public ModelScope model ID**.

In this repository it means:

> the public `metax-tech/Qwen3.6-35B-A3B-W8A8` weights plus the validated DCU/vLLM `config.json` used by Hygon vLLM 0.18.1 to load the already-quantized weights through `compressed-tensors`.

The safetensors weights are not modified or requantized.

Upstream config SHA256:

```text
ba62ca6d8a773ab4c15407acf0653761198c4bcb74d7e8d82edc88132c4ba6a6
```

Validated DCU config SHA256:

```text
b550b28342afd4c61841e2684b06da15f3a0ec3c807ceb22259b0074be9975ae
```

Files:

```text
configs/Qwen3.6-35B-A3B-W8A8-DCU.config.json
scripts/apply_dcu_config.py
```

The launch scripts verify the DCU config hash before starting.

---

## 4. Runtime

Validated environment:

```text
GPU:    Hygon K100AI / gfx928
vLLM:  0.18.1
DTK:   26.04
Torch: 2.10.0
Triton: 3.6.x Hygon build
```

Hygon/SourceFind Qwen3.6 ModelZoo:

https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/blob/main/README.md

Docker image:

```text
harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423
```

Validated digest:

```text
sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811
```

---

## 5. Reproduce

### 5.1 Clone

```bash
git clone https://github.com/DocPang/qwen36-k100ai-w8a8-optimization.git
cd qwen36-k100ai-w8a8-optimization
```

### 5.2 Download the W8A8 checkpoint

```bash
pip install modelscope

modelscope download \
  --model metax-tech/Qwen3.6-35B-A3B-W8A8 \
  --local_dir /path/to/Qwen3.6-35B-A3B-W8A8-DCU
```

### 5.3 Apply the validated DCU config

```bash
python3 scripts/apply_dcu_config.py \
  --model-dir /path/to/Qwen3.6-35B-A3B-W8A8-DCU
```

Expected final config SHA256:

```text
b550b28342afd4c61841e2684b06da15f3a0ec3c807ceb22259b0074be9975ae
```

### 5.4 R180, no MTP

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8-DCU
export GPU_ID=0
export PORT=8000
bash scripts/serve_nomtp.sh
```

Core `vllm serve` arguments for manual deployment/debugging are shown below. This assumes the R180 runtime patch and K100AI MoE config are mounted exactly as done by `scripts/serve_nomtp.sh`:

```bash
vllm serve /models/qwen36-w8a8 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --quantization compressed-tensors \
  --served-model-name qwen36-35b-a3b-w8a8-k100ai \
  --language-model-only \
  --generation-config vllm \
  --disable-custom-all-reduce \
  -cc.mode=3 \
  -cc.inductor_compile_config '{"combo_kernels": false, "benchmark_combo_kernel": false}' \
  --cudagraph-capture-sizes 1 \
  --max-num-seqs 32
```

Expected steady-state decode:

```text
~53-55 tok/s
```

### 5.5 R184, MTP3

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8-DCU
export GPU_ID=0
export PORT=8000
bash scripts/serve_mtp3.sh
```

Core `vllm serve` arguments are below. R184 also enables the model-native MTP3 speculative path:

```bash
vllm serve /models/qwen36-w8a8 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --quantization compressed-tensors \
  --served-model-name qwen36-35b-a3b-w8a8-k100ai \
  --language-model-only \
  --generation-config vllm \
  --disable-custom-all-reduce \
  -cc.mode=3 \
  -cc.inductor_compile_config '{"combo_kernels": false, "benchmark_combo_kernel": false}' \
  --cudagraph-capture-sizes 1 4 \
  --max-num-seqs 32 \
  --speculative-config '{"model":"/models/qwen36-w8a8","method":"qwen3_next_mtp","num_speculative_tokens":3,"quantization":"compressed-tensors"}'
```

Conservative expected decode:

```text
~85 tok/s
```

High-acceptance workloads may exceed 100 tok/s.

---

## 6. Benchmark

```bash
python3 -m pip install -r requirements.txt

python3 scripts/benchmark_openai.py \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model qwen36-35b-a3b-w8a8-k100ai \
  --max-tokens 512 \
  --rounds 6
```

The first request may trigger Triton/compile work. Do not mix cold-start latency into steady-state decode results.

---

## 7. Repository layout

```text
configs/
patches/r180_nomtp/
patches/r184_mtp3/
scripts/apply_dcu_config.py
scripts/serve_nomtp.sh
scripts/serve_mtp3.sh
scripts/benchmark_openai.py
results/
```

---

## 8. Scope

- Validated on **Hygon K100AI / gfx928 only**.
- **K100 is a different accelerator model.**
- Tuned for single-concurrency / low-concurrency decode.
- High-concurrency throughput requires separate tuning.
- MTP throughput depends on speculative-token acceptance rate.
- Model weights and the Hygon Docker image are external dependencies and are not redistributed here.

## License

Repository code is released under Apache License 2.0. Upstream model weights and vendor runtime remain under their own licenses and terms.
