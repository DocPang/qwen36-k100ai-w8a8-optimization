# Qwen3.6-35B-A3B W8A8 on Hygon K100AI

[中文](#中文) · [English](#english)

> [!IMPORTANT]
> **K100 和 K100AI 是不同型号。** 本仓库只在 **Hygon K100AI / gfx928** 上开发和验证。
>
> 当前推荐版本是 **R269 + MTP3**。在相同软件栈和固定 512-token 负载下，正式同卡复测约 **106.3 tok/s**，GPU7 稳态中位数 **107.46 tok/s**。

---

# 中文

本仓库提供 **Qwen3.6-35B-A3B W8A8** 在单张海光 K100AI 上的可复现优化部署。目标不是只公布一个速度数字，而是把别人真正需要的东西一起交付：

- 指定模型来源；
- 已验证的 DCU `config.json`；
- 固定 Docker image + digest；
- K100AI MoE 配置；
- 当前冠军 **R269 runtime patch**；
- 一键准备模型和启动脚本；
- 与验收相同口径的 fixed-512 测速脚本；
- 质量、SHA、MTP acceptance 和多模态验证说明。

## 1. 最快复现：从零到 R269

前提：Linux 服务器已经安装 Docker，K100AI 驱动工作正常，并且能够访问 GitHub、ModelScope 和本文使用的海光 Docker 镜像仓库。

### 1.1 克隆仓库

```bash
git clone https://github.com/DocPang/qwen36-k100ai-w8a8-optimization.git
cd qwen36-k100ai-w8a8-optimization
```

### 1.2 安装 ModelScope CLI

```bash
python3 -m pip install -U modelscope
```

### 1.3 一键下载模型、应用 DCU 配置并启动 R269

```bash
MODEL_DIR="$HOME/models/Qwen3.6-35B-A3B-W8A8-DCU" \
GPU_ID=0 \
PORT=8000 \
bash scripts/quickstart_r269.sh
```

脚本会依次完成：

```text
下载 metax-tech/Qwen3.6-35B-A3B-W8A8
        ↓
应用仓库内已验证的 DCU/vLLM config
        ↓
校验 config SHA256
        ↓
挂载 R269 runtime patch + K100AI MoE config
        ↓
启动 vLLM 0.18.1 + MTP3
```

模型已经下载过时不会重复下载；只会重新校验/应用配置并启动服务。

### 1.4 查看启动日志

默认容器名：

```text
qwen36-35b-k100ai-r269-mtp3
```

查看日志：

```bash
docker logs -f qwen36-35b-k100ai-r269-mtp3
```

R269 正常加载时应能看到类似：

```text
[K100 R269 split MoE stage2] exact M4 second-GEMM override installed
```

检查 OpenAI 兼容接口：

```bash
curl http://127.0.0.1:8000/v1/models
```

---

## 2. 当前性能

验证环境：

```text
GPU:     Hygon K100AI / gfx928
GPU 数:  1
TP:      1
vLLM:    0.18.1
DTK:     26.04
Torch:   2.10.0
Triton:  3.6.x Hygon build
模型:    metax-tech/Qwen3.6-35B-A3B-W8A8
量化:    W8A8 / compressed-tensors
MTP:     qwen3_next_mtp, 3 draft tokens
```

### R269 fixed-512

GPU7 六轮稳态：

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

同一张 GPU1 上的正式 A/B：

- R265 mean: **100.4988 tok/s**
- R269 mean: **106.3198 tok/s**
- R269 median: **106.2957 tok/s**
- same-GPU mean gain: **+5.79%**

固定 512 输出 SHA256：

```text
80c82006a973ecc78fa3fb7a8483b76bc311693bdf277cb296365be0db6c7e00
```

历史参考：

| 方案 | MTP | Decode |
|---|---:|---:|
| 原版热态 TP1 | 关闭 | **19.873 tok/s** |
| R180 | 关闭 | **54.61 tok/s fixed-512** |
| R184 | MTP3 | **85.29 tok/s median** |
| R202 长上下文 Agent | MTP3 | **85.21 tok/s median** |
| **R269 当前冠军** | **MTP3** | **106.30 tok/s formal / 107.46 tok/s GPU7 median** |

详细结果见 [`results/RESULTS.md`](results/RESULTS.md) 和 [`docs/R269_RELEASE_NOTES.md`](docs/R269_RELEASE_NOTES.md)。

---

## 3. 用相同口径测速

不要只用随机网页 prompt 去判断是否复现成功。MTP 的速度会随着 speculative token 接受率变化，不同问题可以差很多。

安装测速依赖：

```bash
python3 -m pip install -r requirements.txt
```

然后使用与 R269 验收一致的 fixed-512 prompt：

```bash
python3 scripts/benchmark_fixed_512.py \
  --base http://127.0.0.1:8000 \
  --model qwen36-35b-a3b-w8a8-k100ai \
  --rounds 6 \
  --max-tokens 512 \
  --out results/my_r269_fixed512.json
```

测速脚本会：

1. 先做一个很短的 warmup；
2. 固定 prompt；
3. `temperature=0`；
4. 固定生成 512 tokens；
5. 输出每轮 wall time、tok/s、SHA256；
6. 最后计算 mean / median。

**第一次 Triton/compile/JIT 请求不要计入稳态性能。**

### 网页测速为什么可能只有 80~90 tok/s？

这是正常的。网页工具会生成不同 prompt，MTP3 的接受率随输出轨迹改变，因此同一 R269 服务可能出现明显波动。

另外，Prefix Cache 主要影响重复前缀的 Prefill / TTFT。判断缓存有没有介入，应查看 vLLM 日志中的：

```text
Prefix cache hit rate
```

不要因为第二次测速更快就自动认为是缓存作弊。

---

## 4. R269 做了什么

R269 不是通过降低精度、减少专家或改变 MTP 接受率来换速度。

Qwen3.6-35B-A3B 在 MTP3 的 M=4 target verifier 中，routed MoE 两个阶段形状不同：

```text
W1: input [4, 2048], routed rows 32, weight [256, 1024, 2048]
W2: input [32, 512],             weight [256, 2048, 512]
```

旧路径让两个阶段共用一套配置。R269 保留已验证的 W1 配置，只给 W2 使用更合适的 K=512 配置：

```text
BM32 / BN32 / BK512 / warps4 / waves2 / kpack2 / stages1
```

完整 `fused_experts` 本地微基准：

```text
241.9239 us -> 199.5232 us
```

约 **1.2125x**。

R269 同时继承此前已经验收的：

- 小 M W8A8 Linear shape tuning；
- W8A8 `lm_head`；
- Gated DeltaNet QKVZ + BA 融合；
- RMSNorm -> INT8 fusion；
- embedding / metadata / GDN decode fast paths；
- MTP target greedy/top1 快路径；
- Prefix Cache、262K 上下文、Tool Calling、多模态长期服务配置。

完整代码在：

```text
patches/r269_mtp3/
```

---

## 5. 质量门禁

R269 对参考生产路径做过完整验证：

- 5/5 greedy text identical；
- 5/5 logprob text identical；
- 5/5 logprob token sequence identical；
- max logprob absolute diff = **0.0**；
- 历史多模态图片回答 identical；
- fixed-512 的 MTP draft / accepted counters identical；
- warm KV 可用容量约 **19.95 GiB / 236,912 tokens**，无持久显存回退。

因此 R269 的 +5.79% 同卡收益来自执行路径优化，不是通过让 MTP 少算、改变输出或降低验证要求获得。

---

## 6. 模型和 DCU 配置

模型来源：

```text
metax-tech/Qwen3.6-35B-A3B-W8A8
```

ModelScope：

https://www.modelscope.cn/models/metax-tech/Qwen3.6-35B-A3B-W8A8

`Qwen3.6-35B-A3B-W8A8-DCU` 不是另一个公开模型 ID。`-DCU` 只是本项目建议的本地目录名，表示已经应用本仓库的海光/vLLM 配置。

**不会重新量化或修改 safetensors 权重。**

验证过的 DCU `config.json` SHA256：

```text
b550b28342afd4c61841e2684b06da15f3a0ec3c807ceb22259b0074be9975ae
```

配置文件：

```text
configs/Qwen3.6-35B-A3B-W8A8-DCU.config.json
```

手工应用：

```bash
python3 scripts/apply_dcu_config.py \
  --model-dir /path/to/Qwen3.6-35B-A3B-W8A8-DCU
```

---

## 7. Docker 环境

验证镜像：

```text
harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423
```

验证 digest：

```text
sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811
```

启动脚本默认锁定 tag + digest。为了最大化复现概率，不建议先换成其它 vLLM/DTK 镜像再比较速度。

---

## 8. 手工启动 R269

已经准备好模型后：

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8-DCU
export GPU_ID=0
export PORT=8000

bash scripts/serve_r269_mtp3.sh
```

可覆盖：

```text
IMAGE
GPU_ID
PORT
MAX_MODEL_LEN
GPU_MEMORY_UTILIZATION
MAX_NUM_BATCHED_TOKENS
CONTAINER_NAME
CACHE_DIR
```

默认服务配置：

- 262144 max context；
- Prefix Cache；
- Tool Calling；
- multimodal；
- `max_num_batched_tokens=4096`；
- CUDAGraph capture sizes `1 4`；
- Qwen native MTP3；
- `use_local_argmax_reduction=true`。

如果只想研究无 MTP，旧的 `scripts/serve_nomtp.sh` 仍保留用于对照。

---

## 9. 仓库结构

```text
configs/
  Qwen3.6-35B-A3B-W8A8-DCU.config.json
  E=256,N=512,device_name=K100_AI.json       # historical R180/R184/R202 config
  E=256,N=512,device_name=K100_AI.r269.json  # R269 config

patches/
  r269_mtp3/          # 当前冠军
  r180_nomtp/         # 历史 no-MTP 对照
  r184_mtp3/          # 历史 MTP3 对照
  r199_agent/         # 历史 Agent runtime 对照

scripts/
  quickstart_r269.sh
  prepare_model.sh
  serve_r269_mtp3.sh
  benchmark_fixed_512.py
  apply_dcu_config.py
  check_release.py

results/
  RESULTS.md
  benchmark_summary.json

docs/
  R269_RELEASE_NOTES.md
  REPRODUCE_WITH_CODEX.md
  面向国产AI加速卡的大模型推理专项优化方法.md
  200+轮实验：研究路线、失败尝试与结论.md
```

---

## 10. 适用范围

- 仅验证 **Hygon K100AI / gfx928**；
- 目标是单并发/低并发 Decode；
- 高并发吞吐需要重新调优；
- 不保证其它 K100AI 个体、驱动频率、温度、NUMA/PCIe 环境完全复现同一小数点速度；
- MTP 性能与 prompt/acceptance 有关；
- 模型和 Docker image 属于其各自上游，本仓库不重新分发权重或镜像。

---

<a id="english"></a>

# English

This repository provides a reproducible single-GPU optimization package for **Qwen3.6-35B-A3B W8A8** on **Hygon K100AI / gfx928**.

The current recommended release is **R269 + MTP3**:

- formal same-GPU GPU1 mean: **106.3198 tok/s**
- formal same-GPU GPU1 median: **106.2957 tok/s**
- GPU7 hot fixed-512 median: **107.4592 tok/s**
- TP=1, single request, vLLM 0.18.1 / DTK 26.04

> **K100 and K100AI are different products.** This repository was validated on K100AI only.

## Quick start

```bash
git clone https://github.com/DocPang/qwen36-k100ai-w8a8-optimization.git
cd qwen36-k100ai-w8a8-optimization
python3 -m pip install -U modelscope

MODEL_DIR="$HOME/models/Qwen3.6-35B-A3B-W8A8-DCU" \
GPU_ID=0 PORT=8000 \
bash scripts/quickstart_r269.sh
```

The script downloads `metax-tech/Qwen3.6-35B-A3B-W8A8` when needed, applies the validated DCU config, mounts the R269 runtime patch and tuned K100AI MoE config, and starts the pinned vLLM 0.18.1 image with native Qwen MTP3.

Check the API:

```bash
curl http://127.0.0.1:8000/v1/models
```

Expected R269 log marker:

```text
[K100 R269 split MoE stage2] exact M4 second-GEMM override installed
```

## Reproduce the fixed-512 result

```bash
python3 -m pip install -r requirements.txt
python3 scripts/benchmark_fixed_512.py \
  --base http://127.0.0.1:8000 \
  --model qwen36-35b-a3b-w8a8-k100ai \
  --rounds 6 \
  --max-tokens 512 \
  --out results/my_r269_fixed512.json
```

The benchmark uses the same fixed prompt family and deterministic settings used for R269 acceptance. The initial compile/JIT request is excluded by a warmup.

R269 GPU7 steady runs:

```text
107.4490 / 107.4696 / 107.4034 / 107.3989 / 107.4706 / 107.4694 tok/s
```

Formal same-GPU GPU1 comparison:

```text
R265 mean: 100.4988 tok/s
R269 mean: 106.3198 tok/s
Gain:      +5.79%
```

## What R269 changes

MTP3 target verification uses M=4 routed-MoE work. W1 and W2 have very different shapes, but the legacy path used one shared configuration. R269 keeps the accepted W1 config and overrides only the M=4 W2 path with a K=512-specific configuration.

The production-like full `fused_experts` local latency changed from **241.9239 us** to **199.5232 us** (~1.2125x).

R269 also includes the previously validated K100AI small-M W8A8, `lm_head`, GDN, RMSNorm/INT8, metadata and MTP runtime fast paths. See [`docs/R269_RELEASE_NOTES.md`](docs/R269_RELEASE_NOTES.md).

## Quality validation

Against the reference production path:

- 5/5 greedy text identical
- 5/5 logprob text identical
- 5/5 logprob token sequences identical
- max logprob absolute difference: **0.0**
- historical multimodal response identical
- MTP acceptance counters identical on the validation workload

Therefore the R269 gain is an execution-path optimization, not a reduced-work or changed-output shortcut.

## Model and environment

ModelScope checkpoint:

```text
metax-tech/Qwen3.6-35B-A3B-W8A8
```

Validated DCU config SHA256:

```text
b550b28342afd4c61841e2684b06da15f3a0ec3c807ceb22259b0074be9975ae
```

Pinned container:

```text
harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423@sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811
```

## Benchmark interpretation

MTP throughput is prompt- and acceptance-rate-dependent. A generic webpage benchmark can show lower or higher numbers than the fixed-512 release benchmark because it generates different prompts.

Prefix Cache mainly affects repeated-prefix prefill/TTFT. When diagnosing cache effects, inspect vLLM's `Prefix cache hit rate` instead of assuming a faster second run used cached decode.

See [`results/RESULTS.md`](results/RESULTS.md) for full result interpretation and historical comparisons.
