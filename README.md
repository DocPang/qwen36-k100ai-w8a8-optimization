# Qwen3.6-35B-A3B W8A8 on Hygon K100AI

[**中文**](#zh-cn) · [**English**](#english)

> [!IMPORTANT]
> **K100 和 K100AI 是两个不同型号。** 本项目只在 **Hygon K100AI / gfx928** 上开发和验证，本文所有性能数据均来自 K100AI。

---

<a id="zh-cn"></a>

# 中文

本项目针对 **Qwen3.6-35B-A3B W8A8** 在 **海光 K100AI** 上的单卡、低并发 Decode 性能进行优化，并提供可直接复现的运行补丁、MoE 配置、DCU 模型配置、启动脚本和测速脚本。

### 这项专项优化对其他用户和厂商有什么意义？

这里的最终参数当然是为 **一个指定模型 × 一个指定计算卡 × 一个指定场景** 找到的，不能把 K100AI 上的 tile、wave 或 chunked-prefill 参数原样复制到另一张卡上。但 200+ 轮实验真正沉淀下来的，是一套可迁移的方法：如何从真实 shape 出发定位瓶颈，如何区分 kernel 微基准和完整模型收益，如何优化 MoE / INT8 / Runtime / MTP / Prefix Cache，以及如何用质量门禁避免“跑得快但模型行为已经变了”。

对于其他海光卡、其他国产AI加速卡和其他模型，**应该复用这套实验方法，而不是照抄最终参数**；对于芯片/软件栈厂商，则可以进一步把真实LLM shape数据库、小M kernel、量化融合、runtime同步审计和自动A/B流程产品化。

- [《Context-Aware Adaptive MTP Scheduling：从局部峰值到完整性能包络》](docs/ADAPTIVE_MTP_SCHEDULING.md)
- [《面向国产AI加速卡的大模型推理专项优化方法》](docs/面向国产AI加速卡的大模型推理专项优化方法.md)
- [《200+轮实验：研究路线、失败尝试与结论》](docs/200+轮实验：研究路线、失败尝试与结论.md)

## 最新版本 / Release History

### 2026-08-12 · R389 — 当前推荐：上下文感知自适应 MTP

R389 修正了此前发布方法中的一个关键局限：**fixed-512 / 固定 MTP3 的局部峰值不能代表一次加载后的完整上下文性能。** R269 的 106–107 tok/s 仍是有效的短上下文局部峰值，但 512→32K 扫描表明，长上下文继续维持 MTP3 会让 draft + verify 的额外成本超过收益。

R389 把它改成服务内部自动调度：

- `< 6144 computed tokens`：MTP3；
- `>= 6144`：停止 drafter，并清除下一步 speculative placeholders，使 target 回到真正的单 token decode；
- target `max_model_len=262144` 保持不变；
- **同一个容器、同一次模型加载自动完成切换，不需要中途改参数或重启。**

10 点 512→32K Decode 曲线：**88.33 / 67.31 / 69.81 / 70.27 / 56.66 / 40.08 / 40.03 / 39.19 / 39.11 / 37.68 tok/s**，平均 **53.29 tok/s**。约 32K natural needle 3/3 PASS，确定性重复输出 SHA256 一致。

同时，后来发现 R237 multimodal direct-embedding shortcut 对 arbitrary-length chunked-prefill tail 存在语义风险，因此 R389 默认显式关闭 R237。

**新用户默认复现 R389：**

```bash
git clone https://github.com/DocPang/qwen36-k100ai-w8a8-optimization.git
cd qwen36-k100ai-w8a8-optimization
python3 -m pip install -U modelscope

MODEL_DIR="$HOME/models/Qwen3.6-35B-A3B-W8A8-DCU" \
GPU_ID=0 PORT=8000 \
bash scripts/quickstart.sh
```

完整方法、曲线和实现说明见 [`docs/ADAPTIVE_MTP_SCHEDULING.md`](docs/ADAPTIVE_MTP_SCHEDULING.md) 与 [`docs/R389_RELEASE_NOTES.md`](docs/R389_RELEASE_NOTES.md)。

> **R269 已降级为历史短上下文 benchmark 版本。** 它的 fixed-512 106–107 tok/s 数据保留用于研究和回归，但不再作为默认长期部署入口。

### 之前的重要发布节点

| 版本 | 主要阶段 | 代表性能 |
|---|---|---:|
| Stock / 早期基线 | 原始 W8A8 单卡路径 | 19.873 tok/s |
| R180 | 无 MTP，小 M W8A8 kernel 系统优化 | 54.61 tok/s fixed512 |
| R184 | MTP3 + INT8 `lm_head` + GDN 等累计优化 | 85.29 tok/s fixed512 median；网页负载最高 107.44 |
| R199/R202 | 262K + Prefix Cache + 多模态 + Agent runtime | 85.21 tok/s median |
| R265 | 后续 exact kernel/runtime 累计版本 | ~100.5 tok/s（GPU1） |
| R269 | 固定 MTP3 的短上下文局部峰值（历史 benchmark） | **106.30 tok/s（GPU1） / 107.46 tok/s（GPU7）** |
| **R389** | **一次加载，自适应 MTP3→no-MTP；512→32K** | **53.29 tok/s 10点均值；6K→32K 约 40→37.68 tok/s 平台** |

> fixed-512 峰值和完整上下文性能包络是两个不同指标。R389 之后，默认部署版本以**完整 token-length 曲线 + 质量门禁**仲裁，不再用单一短 prompt 峰值代表全局最优。

---

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

### 1.6 历史固定策略：模型原生 MTP3

R184/R269 使用 Qwen 模型自带 MTP head，并在整个请求生命周期固定 speculative depth：

```text
method = qwen3_next_mtp
num_speculative_tokens = 3
quantization = compressed-tensors
```

MTP 的最终 token 仍由目标模型验证。实际速度取决于 speculative token 接受率，因此不同提示词的速度会有明显变化。后续完整 token-length 曲线进一步证明：**固定 MTP3 也会随上下文长度跨过收益交叉点，因此不能作为全范围默认策略。**

### 1.7 R199/R202：Prefix Cache / Agent Runtime 快路径

为了把纯 benchmark 版本变成长期 Agent 服务，后续又加入了：

- 262K 上下文；
- Prefix Cache；
- Tool Calling；
- 多模态；
- `max_num_batched_tokens=4096`；
- HCU/Mamba-align 常规 Decode 快路径；
- accepted-token metadata 直接留在 GPU，删除不再消费的 D2H copy / event / 重复 scratch 开销。

这条路线解决了混合 GDN 模型开启 Prefix Cache 后 Decode 从约 85 tok/s 降到约 73 tok/s 的问题。R202 在固定短上下文口径下重新回到 **约 85 tok/s**，但它仍然采用固定 MTP3，因此现在只作为历史 Agent 路径保留。

### 1.8 R389：上下文感知自适应 MTP

R389 在启动时同时加载 R304/R305 调度补丁。默认 `<6144` 使用 MTP3，达到 cutoff 后让 drafter 退出并清掉后续 speculative placeholders，使 target 自动回到单 token decode。整个过程不重启、不换权重、不修改外部启动参数。

当前 512→32K 验收表明，这种两段式策略可以同时保留短上下文 MTP 收益，并让 6K→32K Decode 稳定在约 **40→37.68 tok/s**。详细原理见 [`docs/ADAPTIVE_MTP_SCHEDULING.md`](docs/ADAPTIVE_MTP_SCHEDULING.md)。

当前默认公开复现入口：

```text
patches/r389_adaptive_mtp/
scripts/quickstart.sh
scripts/serve_r389_adaptive_mtp.sh
```

历史 R199/R202 复现入口：

```text
patches/r199_agent/sitecustomize.py
scripts/serve_agent_mtp3.sh
```

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
| R202 长期 Agent（262K + Prefix + 多模态） | MTP3 | **85.21 tok/s 中位数（GPU0，4轮固定512）** |
| R265 后续累计 exact 优化 | MTP3 | **约 100.5 tok/s（GPU1 同卡参考）** |
| R269 历史局部峰值 | 固定 MTP3 | **106.30 tok/s GPU1 同卡中位；107.46 tok/s GPU7 六轮中位** |
| **R389 当前推荐部署** | **自适应：MTP3 <6144；之后 no-MTP** | **512→32K 10点均值 53.29 tok/s；6K→32K 40.08→37.68 tok/s** |

R269 的 107 tok/s 不能解释为所有上下文长度的固定速度。R389 之后，仓库将 fixed-512 峰值视为**局部性能点**，把完整 token-length 曲线作为默认部署仲裁依据。

R202 与 R184 的历史测试目标不同；它们的 fixed-512 数据仍保留用于回归。R389 则明确面向一次加载后的 512→32K 性能包络。

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

### 5.6 启动 R202：长期 Agent / 262K / Prefix Cache / 多模态

如果目标不是短上下文 benchmark，而是长期 Agent 服务，使用：

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8-DCU
export GPU_ID=0
export PORT=8000

bash scripts/serve_agent_mtp3.sh
```

默认配置包括：

- R199 runtime fastpath；
- MTP3；
- 262144 最大上下文；
- Prefix Cache；
- Tool Calling；
- 多模态；
- `max_num_batched_tokens=4096`；
- `restart=unless-stopped`。

该配置现在作为**历史固定 MTP3 Agent 路径**保留，不再是默认推荐。新部署应使用下面的 R389 自适应入口；只有做历史回归时才建议使用 R202/R184。

### 5.7 启动 R389：一次加载，自适应 MTP3 → no-MTP

推荐部署：

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8-DCU
export GPU_ID=0
export PORT=8000

bash scripts/quickstart.sh
```

默认 `MTP_CUTOFF=6144`。服务启动后不需要再改任何参数；请求的 computed token 数达到 cutoff 后会在服务内部自动退出 MTP。研究者可以在重新 profile 后通过启动时环境变量覆盖 cutoff，例如 `MTP_CUTOFF=8192`，但 **6144 才是当前公开验收值**。

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
  r199_agent/sitecustomize.py
  r269_mtp3/                  # 历史 fixed-MTP3 峰值
  r389_adaptive_mtp/          # 当前默认：R304/R305 自适应调度

scripts/
  apply_dcu_config.py
  quickstart.sh               # 当前默认入口 -> R389
  serve_r389_adaptive_mtp.sh
  serve_nomtp.sh
  serve_mtp3.sh               # 历史入口
  serve_agent_mtp3.sh         # 历史入口
  benchmark_openai.py

docs/
  ADAPTIVE_MTP_SCHEDULING.md
  R389_RELEASE_NOTES.md
  面向国产AI加速卡的大模型推理专项优化方法.md
  200+轮实验：研究路线、失败尝试与结论.md

results/
  RESULTS.md
  benchmark_summary.json
  r389_adaptive_curve.json
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

The exact kernel parameters are specific to this model/GPU/workload combination, but the project also documents the general optimization methodology and the lessons from 200+ experiments. Other accelerator users should reuse the **profiling, shape-inventory, A/B, quality-gating, runtime and workload methodology**, not blindly copy K100AI-specific tiles.

- [Context-Aware Adaptive MTP Scheduling: from local peak to full performance envelope](docs/ADAPTIVE_MTP_SCHEDULING.md)
- [General methodology (Chinese): 面向国产AI加速卡的大模型推理专项优化方法](docs/面向国产AI加速卡的大模型推理专项优化方法.md)
- [200+ experiment review (Chinese): 研究路线、失败尝试与结论](docs/200+轮实验：研究路线、失败尝试与结论.md)

> [!IMPORTANT]
> **K100 and K100AI are different accelerator models.** All results in this repository were measured on **K100AI** only.

## Latest release / Release history

### 2026-08-12 · R389 — recommended deployment: context-aware adaptive MTP

R389 corrects a key limitation in the previous public methodology: a **fixed-512 / fixed-MTP3 local peak is not the same thing as the best one-load deployment policy across context lengths**. R269's 106–107 tok/s result remains valid as a short-context local peak, but the 512→32K sweep showed that keeping MTP3 enabled at long context eventually costs more draft + verification work than it saves.

R389 turns that parameter choice into an in-server scheduling policy:

- `< 6144 computed tokens`: MTP3;
- `>= 6144`: stop the drafter and clear future speculative placeholders so target decoding returns to true single-token decode;
- target `max_model_len=262144` is unchanged;
- **one container, one model load, no mid-run parameter rewrite or restart.**

The validated 10-point 512→32K Decode curve is **88.33 / 67.31 / 69.81 / 70.27 / 56.66 / 40.08 / 40.03 / 39.19 / 39.11 / 37.68 tok/s**, with a **53.29 tok/s** mean. Natural ~32K needle retrieval passed 3/3, and deterministic repeated outputs had identical SHA256.

R389 also disables the historical R237 multimodal direct-embedding shortcut by default because later arbitrary-length chunked-prefill-tail testing exposed semantic risk.

Recommended reproduction for new users:

```bash
git clone https://github.com/DocPang/qwen36-k100ai-w8a8-optimization.git
cd qwen36-k100ai-w8a8-optimization
python3 -m pip install -U modelscope
MODEL_DIR="$HOME/models/Qwen3.6-35B-A3B-W8A8-DCU" GPU_ID=0 PORT=8000 \
  bash scripts/quickstart.sh
```

See [`docs/ADAPTIVE_MTP_SCHEDULING.md`](docs/ADAPTIVE_MTP_SCHEDULING.md) and [`docs/R389_RELEASE_NOTES.md`](docs/R389_RELEASE_NOTES.md) for the implementation and research path.

> **R269 is now a historical short-context benchmark release.** Its fixed-512 106–107 tok/s numbers are retained for research and regression, but it is no longer the default deployment entry point.

| Release | Stage | Representative decode |
|---|---|---:|
| Stock / early baseline | original single-GPU W8A8 path | 19.873 tok/s |
| R180 | no-MTP small-M kernel optimization | 54.61 tok/s fixed512 |
| R184 | cumulative MTP3 hot path | 85.29 tok/s fixed512 median; workload peak 107.44 |
| R199/R202 | 262K + Prefix Cache + multimodal Agent profile | 85.21 tok/s median |
| R265 | later exact kernel/runtime stack | ~100.5 tok/s on GPU1 |
| R269 | fixed-MTP3 short-context local peak (historical benchmark) | **106.30 GPU1 / 107.46 GPU7 median** |
| **R389** | **single-load adaptive MTP3→no-MTP, 512→32K** | **53.29 tok/s 10-point mean; 6K→32K plateau 40.08→37.68** |

---

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

### 1.6 Historical fixed policy: native Qwen MTP3

R184/R269 use the model's own MTP head with a fixed speculative depth for the whole request:

```text
method = qwen3_next_mtp
num_speculative_tokens = 3
quantization = compressed-tensors
```

Speculative tokens are still verified by the target model. Throughput therefore depends on acceptance rate. The later full token-length sweep also showed that fixed MTP3 crosses a context-length profitability boundary, so it should not be treated as the default full-range policy.

### 1.7 R199/R202 Agent runtime path

The long-context Agent profile adds 262K context, Prefix Cache, Tool Calling and multimodal support. R199 removes redundant D2H/event work from the common hybrid-GDN/Mamba-align decode path and keeps accepted-token metadata on GPU. R202 combines that runtime fastpath with `max_num_batched_tokens=4096` and MTP3.

This recovered the historical Prefix-Cache Agent path from roughly 73 tok/s back to roughly 85 tok/s on the fixed short-context workload, but it still uses fixed MTP3 and is now retained as a regression profile.

### 1.8 R389 context-aware adaptive MTP

R389 loads the R304/R305 scheduling patches once at startup. Below 6,144 computed tokens it uses MTP3; at the cutoff the drafter exits and future speculative placeholders are cleared, returning the target to normal single-token decode without a restart or parameter rewrite.

The validated 512→32K sweep preserves short-context MTP acceleration while keeping the 6K→32K region near **40→37.68 tok/s**. See [`docs/ADAPTIVE_MTP_SCHEDULING.md`](docs/ADAPTIVE_MTP_SCHEDULING.md).

Current public entry points:

```text
patches/r389_adaptive_mtp/
scripts/quickstart.sh
scripts/serve_r389_adaptive_mtp.sh
```

Historical R199/R202 entry points:

```text
patches/r199_agent/sitecustomize.py
scripts/serve_agent_mtp3.sh
```

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
| R202 long-context Agent profile (262K + prefix cache + multimodal) | MTP3 | **85.21 tok/s median, 4 fixed-512 runs on GPU0** |
| R265 later cumulative exact stack | MTP3 | **~100.5 tok/s on GPU1** |
| R269 historical local peak | fixed MTP3 | **106.30 tok/s GPU1 median; 107.46 tok/s GPU7 median** |
| **R389 recommended deployment** | **adaptive: MTP3 <6144; no-MTP afterwards** | **53.29 tok/s 10-point 512→32K mean; 6K→32K 40.08→37.68 tok/s** |

The 107 tok/s R269 result should not be interpreted as a constant speed across context lengths. Starting with R389, fixed-512 peaks are treated as local performance points; the default deployment decision is based on the full token-length curve plus quality gates.

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

### 5.6 R202 long-context Agent profile

For a long-running Agent rather than a short-context benchmark:

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8-DCU
export GPU_ID=0
export PORT=8000
bash scripts/serve_agent_mtp3.sh
```

This historical profile enables R199 runtime fastpaths, fixed MTP3, 262K context, Prefix Cache, multimodal support, Tool Calling, and `max_num_batched_tokens=4096`. It is retained for regression, not as the default deployment recommendation.

### 5.7 R389 single-load adaptive MTP3 -> no-MTP

Recommended deployment:

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8-DCU
export GPU_ID=0
export PORT=8000
bash scripts/quickstart.sh
```

The validated default is `MTP_CUTOFF=6144`. No runtime parameter change is needed after startup: once a request reaches the cutoff, the in-server scheduler automatically exits speculation for subsequent decode steps. Researchers may override the cutoff at startup after profiling their own hardware, but **6144 is the currently validated public value**.

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
patches/r199_agent/
scripts/apply_dcu_config.py
scripts/serve_nomtp.sh
scripts/serve_mtp3.sh
scripts/serve_agent_mtp3.sh
scripts/benchmark_openai.py
docs/面向国产AI加速卡的大模型推理专项优化方法.md
docs/200+轮实验：研究路线、失败尝试与结论.md
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
