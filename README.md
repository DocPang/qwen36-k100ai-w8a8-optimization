# Qwen3.6-35B-A3B W8A8 on Hygon K100AI

<div align="center">

[**中文**](#zh-cn) · [**English**](#english)

</div>

---

<a id="zh-cn"></a>

# 中文说明

本项目公开了 **Qwen3.6-35B-A3B W8A8 在海光 K100AI（gfx928）上的单卡推理优化方案**，目标是让别人能够从公开上游资源出发，稳定复现我们最终的无 MTP 与 MTP3 推理结果。

> [!IMPORTANT]
> **K100 和 K100AI 是两个不同型号的加速卡。**
> 本项目的代码、Triton tile、MoE 配置和性能数据均只在 **K100AI / gfx928** 上验证，不能把结果直接写成“K100”，也不宣称这些配置在 K100 上具有相同兼容性或性能。

## 1. 最终结果

测试条件：单张 K100AI、TP=1、单并发、Qwen3.6-35B-A3B W8A8。

| 配置 | MTP | Decode 速度 | 说明 |
|---|---:|---:|---|
| 原版热态基线 | 关闭 | **19.873 tok/s** | 原版 vLLM 路径，统一热态/图捕获口径 |
| R180 | 关闭 | **53.46 tok/s 平均** | `llm_speedtest` 四档输入、输出 512 tokens |
| R180 | 关闭 | **54.61 tok/s** | 固定提示词重复 512-token 测试 |
| R184 | MTP3 | **85.29 tok/s 中位数** | 固定提示词保守稳定值 |
| R184 | MTP3 | **96.84 tok/s 平均** | `llm_speedtest` 四档平均，受 MTP 接受率影响 |
| R184 | MTP3 | **107.44 tok/s 峰值** | 高接受率负载，不应视为所有请求的固定速度 |

6 月份早期报告曾记录原始单卡无 MTP 为 **12.55 tok/s**。该数据包含更早期的冷路径和不同测速条件，因此本项目保留它作为历史记录，但计算优化提升时使用统一热态基线 **19.873 tok/s**。

完整数据见 [`results/RESULTS.md`](results/RESULTS.md)。

## 2. 最容易误解的地方：什么是这里的 `W8A8-DCU`？

这是复现本项目最关键的一点。

### 2.1 服务器上曾经同时存在两套独立下载的 W8A8

这两套模型不是同一个目录的复制品，也不是“普通 W8A8 转成 DCU 版”的关系。

历史记录能够明确区分它们：

- **Eco-Tech 版本**：`Eco-Tech/Qwen3.6-35B-A3B-w8a8`
  - ModelScope 下载锁时间：2026-06-10 14:19:55 (+08:00)
  - shell history 仍保留了明确的 `modelscope download --model Eco-Tech/Qwen3.6-35B-A3B-w8a8 ...` 命令
  - 本地文件为 10 个 `quant_model_weights-*.safetensors` 分片
  - 模型卡写明精度测试平台是 Ascend 800T A3 / vllm-ascend
  - **这不是本项目最终 K100AI 优化所使用的模型**

- **本项目主线版本**：`metax-tech/Qwen3.6-35B-A3B-W8A8`
  - ModelScope 下载锁时间：2026-06-10 15:49:23 (+08:00)
  - 1 秒后本地 `Qwen3.6-35B-A3B-W8A8-DCU` 目录开始写入文件
  - 下载得到 8 个 `model-*.safetensors` 分片
  - 目录中的 ModelScope `.msc` revision 与 metax-tech 仓库真实上传 commit 一一对应
  - 抽查的本地权重 SHA256 与该仓库 Git LFS `oid sha256` 完全一致

因此可以确定：**`W8A8-DCU` 主线是第二次独立从 ModelScope 下载的 metax-tech 模型，不是由 Eco-Tech 普通 W8A8 目录复制、转换或重新量化得到的。**

### 2.2 上游公开 model ID 不带 `-DCU`

我们实际使用的 8 个 `safetensors` W8A8 权重来自魔搭：

**`metax-tech/Qwen3.6-35B-A3B-W8A8`**

<https://www.modelscope.cn/models/metax-tech/Qwen3.6-35B-A3B-W8A8>

我们重新验证过魔搭 Git 仓库：

```text
metax-tech/Qwen3.6-35B-A3B-W8A8       -> 存在
metax-tech/Qwen3.6-35B-A3B-W8A8-DCU   -> 不存在
```

所以仓库里的 `Qwen3.6-35B-A3B-W8A8-DCU` **不是另一个公开模型 ID**，而是我们给“已经完成 DCU 配置适配的本地模型目录”使用的名称。

### 2.3 真正的 DCU 适配发生在 `config.json`

魔搭原始模型的权重本身就是 W8A8，但原始 `config.json` 没有我们在海光 vLLM 0.18.1 环境中实际使用的 `compressed-tensors` 量化描述。

我们 6 月部署时保留了原始配置备份，因此可以精确验证：

```text
魔搭原始 config.json SHA256
ba62ca6d8a773ab4c15407acf0653761198c4bcb74d7e8d82edc88132c4ba6a6

最终实际运行的 DCU config.json SHA256
b550b28342afd4c61841e2684b06da15f3a0ec3c807ceb22259b0074be9975ae
```

原始 config 与服务器最早的备份哈希完全一致。随后 DCU 适配加入了完整的 `compressed-tensors` W8A8 元数据，包括：

- `quant_method = compressed-tensors`
- `format = int-quantized`
- Linear 权重：静态 INT8、per-channel、symmetric
- 激活：动态 INT8、per-token、symmetric
- 与真实 checkpoint 对应的完整 ignore 列表（194 项）

**权重文件不需要重新量化。DCU 适配步骤只替换模型的 `config.json`。**

本仓库直接提供了我们最终实际运行的精确配置：

```text
configs/Qwen3.6-35B-A3B-W8A8-DCU.config.json
```

以及安全应用脚本：

```text
scripts/apply_dcu_config.py
```

脚本会：

1. 检查当前模型是不是预期的魔搭原始 config；
2. 校验原始 SHA256；
3. 备份为 `config.json.upstream.bak`；
4. 替换为本项目实际验证的 DCU config；
5. 再校验最终 SHA256。

如果模型已经适配过，则脚本是幂等的，不会重复破坏配置。

### 2.4 为什么启动脚本会强制检查 DCU config？

为了防止“下载完魔搭模型直接启动”导致复现环境与本项目实际测试环境不一致，`serve_nomtp.sh` 和 `serve_mtp3.sh` 都会检查：

```text
config.json SHA256 == b550b28342afd4c61841e2684b06da15f3a0ec3c807ceb22259b0074be9975ae
```

不一致就拒绝启动，并提示先运行 `apply_dcu_config.py`。

因此，本项目所说的“基础模型”准确地讲是：

> **魔搭 `metax-tech/Qwen3.6-35B-A3B-W8A8` 的公开 W8A8 权重 + 本仓库提供的、实际验证过的 DCU `config.json`。**

## 3. 上游运行环境

我们使用的是海光/光合开发者社区 Qwen3.6 ModelZoo 对应的 vLLM 0.18 环境。

相关上游页面：

- Qwen3.6 ModelZoo：<https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/blob/main/README.md>
- Tags：<https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/tags>
- 用于固定 vLLM 0.18.1 环境信息的 revision：<https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/commit/8a54c4e3df888c81165e5106657d17865bac3644>

实际验证环境：

```text
GPU:    Hygon K100AI / gfx928
vLLM:  0.18.1
DTK:   26.04
Torch: 2.10.0
Triton: 3.6.x (Hygon build)
```

Docker 镜像：

```text
harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423
```

为避免未来同名 tag 被更新，本项目启动脚本默认锁定我们实际跑出结果的 image digest：

```text
sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811
```

光合 Qwen3.6 ModelZoo 的 v1.1 标签明确记录了 **K100AI 支持**。本项目启动参数也保留了 K100AI 实际环境使用的：

```text
--disable-custom-all-reduce
```

## 4. 从零开始复现

下面的步骤不依赖本项目开发时的任何私有 IP、用户名、服务器目录或生产端口。

### 4.1 克隆本仓库

```bash
git clone https://github.com/DocPang/qwen36-k100ai-w8a8-optimization.git
cd qwen36-k100ai-w8a8-optimization
```

### 4.2 下载魔搭 W8A8 权重

推荐直接把本地目录命名成 `-DCU`，用于提醒自己这是将要进行 DCU 适配的副本：

```bash
python3 -m pip install modelscope

modelscope download \
  --model metax-tech/Qwen3.6-35B-A3B-W8A8 \
  --local_dir /path/to/Qwen3.6-35B-A3B-W8A8-DCU
```

这里的 `-DCU` **只是本地目录名**，下载来源仍然是上面的公开魔搭 model ID。

### 4.3 应用经过验证的 DCU 配置

```bash
python3 scripts/apply_dcu_config.py \
  --model-dir /path/to/Qwen3.6-35B-A3B-W8A8-DCU
```

成功时会打印类似：

```text
sha256: b550b28342afd4c61841e2684b06da15f3a0ec3c807ceb22259b0074be9975ae
quantization: compressed-tensors / int-quantized / ignore=194 entries
```

不要手工删除 `config.json.upstream.bak`，它便于后续核验来源和回滚。

### 4.4 启动 R180：无 MTP

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8-DCU
export GPU_ID=0
export PORT=8000

bash scripts/serve_nomtp.sh
```

该配置是纯目标模型路径，不启用 speculative decoding。

期望单并发 Decode：

```text
约 53～55 tok/s
```

### 4.5 启动 R184：MTP3

确保同一张测试卡上没有同时运行另一份模型服务，然后：

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8-DCU
export GPU_ID=0
export PORT=8000

bash scripts/serve_mtp3.sh
```

核心配置：

```text
method = qwen3_next_mtp
num_speculative_tokens = 3
quantization = compressed-tensors
CUDAGraph capture sizes = 1, 4
```

保守固定负载期望：

```text
约 85 tok/s
```

高接受率提示词可以达到 100+ tok/s，但 MTP 的速度依赖 speculative token 接受率，因此不要把 107.44 tok/s 峰值写成普遍稳定速度。

### 4.6 验证 API

服务启动和图捕获完成后：

```bash
curl http://127.0.0.1:8000/v1/models
```

再安装本项目测速依赖：

```bash
python3 -m pip install -r requirements.txt
```

固定请求测试：

```bash
python3 scripts/benchmark_openai.py \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model qwen36-35b-a3b-w8a8-k100ai \
  --max-tokens 512 \
  --rounds 6
```

第一次真实请求可能触发 Triton runtime compile。应单独标记首次编译请求，不要把它与稳定段混在一起计算，也不要为了展示更高速度而删除后续较慢的有效结果。

## 5. 我们到底优化了什么？

### 5.1 针对 K100AI/gfx928 的 small-M W8A8 Linear

单并发 Decode 主要是 `M=1～4` 的 skinny GEMM。通用 heuristic 对 K100AI/gfx928 并不理想，因此我们针对模型真实 `(M,K,N)` 形状搜索并固定 Triton tile。

无 MTP 中两个最重要的 `M=1` 大投影：

| 投影 | 原通用路径 | K100AI tile | 微基准提升 |
|---|---:|---:|---:|
| `2048 -> 12288` | ~371.99 μs | ~90.03 μs | ~4.13× |
| `2048 -> 9216` | ~221.75 μs | ~77.53 μs | ~2.86× |

调优时 numerical check 为零差异。

### 5.2 W8A8 `lm_head`

Qwen3.6-35B-A3B 的词表为 248,320，Decode 时输出投影本身成本很高。补丁为低 M 场景安装了专用 W8A8 `lm_head` 路径。

### 5.3 K100AI FusedMoE 配置

模型有 256 个 routed experts，每个 token 激活 8 个专家。仓库提供实际使用的 vLLM FusedMoE 配置：

```text
configs/E=256,N=512,device_name=K100_AI.json
```

这里文件名里的 `K100_AI` 是该海光 vLLM 运行时识别的设备配置命名。

### 5.4 R184：Gated DeltaNet QKVZ + BA 融合

R184 将 Gated DeltaNet 路径中原本分离的 QKVZ 与 BA W8A8 投影合并为双权重指针融合 kernel，减少重复量化、kernel launch 和中间开销，同时避免预拼接完整权重带来的 KV cache 损失。

### 5.5 `torch.compile` + CUDAGraph

最终服务按真实 Decode shape 配置：

```text
compile mode = 3
combo kernel benchmark = off
R180 CUDAGraph = 1
R184 CUDAGraph = 1, 4
```

### 5.6 模型原生 MTP3

R184 使用 Qwen 自带 MTP head：

```text
method = qwen3_next_mtp
num_speculative_tokens = 3
```

MTP draft token 最终仍由目标模型验证；实际收益取决于接受率。

## 6. 正确性与性能口径

我们不接受“裸 kernel 微基准更快”就直接宣称模型优化成功。

最终候选必须通过完整模型测试。

已完成的主要门禁包括：

- R184 与同模式 MTP3 基线：固定提示词文本一致、token 一致、logprob 最大差为 0；
- R180 关键大投影：微基准 `relative-L2 = 0`；
- R189/R192 等实验虽然局部 kernel 更快，但端到端变慢，因此被淘汰；
- 无 MTP 和 MTP3 是两种独立推理模式，不宣称二者生成字节级完全一致。

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
  check_release.py

results/
  RESULTS.md
  benchmark_summary.json

docs/
  SOURCES.md
  REPRODUCE_WITH_CODEX.md
```

## 8. 用 Codex 自动复现

如果服务器已经具备 Docker、K100AI 和访问上游资源的能力，可直接把：

[`docs/REPRODUCE_WITH_CODEX.md`](docs/REPRODUCE_WITH_CODEX.md)

交给 Codex 执行。

它明确要求：

1. 验证硬件确实是 K100AI；
2. 下载指定魔搭 W8A8 权重；
3. 应用并校验 DCU config；
4. 分别启动 R180 / R184；
5. 预热后再测速；
6. 保存吞吐、输出 SHA256 与正确性记录；
7. 不触碰无关 GPU 和已有服务。

## 9. 范围与限制

- **仅验证 K100AI / gfx928。**
- K100 是不同型号，本项目未在 K100 上验证。
- 主要目标是单并发/低并发 Decode latency，而不是高并发总吞吐。
- MTP 性能高度依赖 speculative acceptance rate。
- 模型权重和海光 Docker 镜像均由各自上游提供，本仓库不重新分发。
- DCU config 是复现本项目 W8A8 路径的必要组成部分，但它不修改模型权重本身。

## 10. License

本仓库代码采用 Apache License 2.0。模型权重、海光运行时和其他上游组件遵循各自的许可证与使用条款，详见 [`NOTICE.md`](NOTICE.md)。

---

<a id="english"></a>

# English

This repository publishes the **single-GPU inference optimizations for Qwen3.6-35B-A3B W8A8 on Hygon K100AI (gfx928)**, together with a reproducible path from public upstream artifacts to the exact DCU-adapted model configuration used for the reported results.

> [!IMPORTANT]
> **K100 and K100AI are different accelerator models.**
> Everything in this repository was developed and validated on **K100AI / gfx928 only**. The kernel choices, MoE configuration, compatibility and performance must not be presented as K100 results.

## 1. Results

Single K100AI, TP=1, single-concurrency decode:

| Configuration | MTP | Decode throughput | Notes |
|---|---:|---:|---|
| Stock hot baseline | off | **19.873 tok/s** | normalized warm/graph-enabled stock vLLM path |
| R180 | off | **53.46 tok/s avg** | four prompt lengths, 512 output tokens |
| R180 fixed run | off | **54.61 tok/s** | repeated fixed 512-token request |
| R184 fixed run | MTP3 | **85.29 tok/s median** | conservative stable figure |
| R184 benchmark average | MTP3 | **96.84 tok/s avg** | workload-dependent speculative acceptance |
| R184 observed peak | MTP3 | **107.44 tok/s** | high-acceptance workload, not a universal rate |

An older June report recorded **12.55 tok/s** for the original no-MTP single-card path. Because that measurement included earlier cold-path and methodology differences, the normalized stock baseline used for speedup comparisons is **19.873 tok/s**.

See [`results/RESULTS.md`](results/RESULTS.md).

## 2. What does `W8A8-DCU` mean here?

This is the most important reproducibility detail.

### 2.1 Two independently downloaded W8A8 checkpoints existed on the server

These were not copies of the same directory, and the DCU deployment was not produced by converting the ordinary W8A8 directory.

The historical evidence distinguishes them clearly:

- **Eco-Tech checkpoint**: `Eco-Tech/Qwen3.6-35B-A3B-w8a8`
  - ModelScope download-lock timestamp: 2026-06-10 14:19:55 (+08:00)
  - shell history still contains the explicit `modelscope download --model Eco-Tech/Qwen3.6-35B-A3B-w8a8 ...` command
  - stored as 10 `quant_model_weights-*.safetensors` shards
  - its model card targets Ascend 800T A3 / vllm-ascend
  - **this is not the checkpoint used for the final K100AI optimization results**

- **Checkpoint used by this project**: `metax-tech/Qwen3.6-35B-A3B-W8A8`
  - ModelScope download-lock timestamp: 2026-06-10 15:49:23 (+08:00)
  - files began appearing in the local `Qwen3.6-35B-A3B-W8A8-DCU` directory one second later
  - stored as 8 `model-*.safetensors` shards
  - the ModelScope `.msc` revisions match real upload commits in the metax-tech repository
  - representative local shard SHA256 values exactly match the repository Git LFS `oid sha256` values

Therefore the evidence shows that **the DCU mainline was a second, independent ModelScope download of the metax-tech checkpoint. It was not copied, converted, or requantized from the Eco-Tech W8A8 directory.**

### 2.2 The public upstream model ID does not include `-DCU`

The eight W8A8 `safetensors` shards used by the validated deployment came from:

**`metax-tech/Qwen3.6-35B-A3B-W8A8`**

<https://www.modelscope.cn/models/metax-tech/Qwen3.6-35B-A3B-W8A8>

We verified that the public repository without the suffix exists, while the guessed public repository `metax-tech/Qwen3.6-35B-A3B-W8A8-DCU` does not.

Therefore `Qwen3.6-35B-A3B-W8A8-DCU` is **not another public model ID**. It is the local name we use for the checkpoint after applying the validated DCU configuration.

### 2.3 The DCU adaptation is an exact `config.json` replacement

The original ModelScope W8A8 checkpoint config does not contain the `compressed-tensors` quantization metadata used by the validated Hygon vLLM 0.18.1 deployment.

The original config from ModelScope matches the earliest backup in the test environment exactly:

```text
upstream config.json SHA256
ba62ca6d8a773ab4c15407acf0653761198c4bcb74d7e8d82edc88132c4ba6a6
```

The final DCU-adapted config actually used for the benchmarks is:

```text
DCU config.json SHA256
b550b28342afd4c61841e2684b06da15f3a0ec3c807ceb22259b0074be9975ae
```

It supplies the full compressed-tensors W8A8 description used by the runtime:

- `quant_method = compressed-tensors`
- `format = int-quantized`
- static symmetric INT8 per-channel weights
- dynamic symmetric INT8 per-token activations
- the full 194-entry ignore list matching the checkpoint

**The DCU adaptation does not requantize or rewrite the safetensors weights. It replaces only `config.json`.**

This repository contains both the exact validated config and a guarded application script:

```text
configs/Qwen3.6-35B-A3B-W8A8-DCU.config.json
scripts/apply_dcu_config.py
```

The script verifies the upstream hash, makes a backup, applies the DCU config, and verifies the final hash.

The public service launchers also refuse to start if `MODEL_DIR/config.json` does not match the validated DCU hash. This prevents accidentally benchmarking a different model configuration.

In precise terms, the model used by this project is:

> **The public W8A8 weights from `metax-tech/Qwen3.6-35B-A3B-W8A8` plus the exact DCU `config.json` published in this repository.**

## 3. Upstream runtime

The deployment uses the Hygon/SourceFind Qwen3.6 vLLM 0.18 environment.

Sources:

- ModelZoo: <https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/blob/main/README.md>
- Tags: <https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/tags>
- Environment revision: <https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/commit/8a54c4e3df888c81165e5106657d17865bac3644>

Validated runtime:

```text
GPU:    Hygon K100AI / gfx928
vLLM:  0.18.1
DTK:   26.04
Torch: 2.10.0
Triton: 3.6.x Hygon build
```

Image tag:

```text
harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423
```

Validated image digest used by the launch scripts:

```text
sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811
```

The Hygon Qwen3.6 v1.1 tag records K100AI support. The tested launch path also keeps:

```text
--disable-custom-all-reduce
```

## 4. Reproduce from scratch

### 4.1 Clone this repository

```bash
git clone https://github.com/DocPang/qwen36-k100ai-w8a8-optimization.git
cd qwen36-k100ai-w8a8-optimization
```

### 4.2 Download the upstream W8A8 weights

```bash
python3 -m pip install modelscope

modelscope download \
  --model metax-tech/Qwen3.6-35B-A3B-W8A8 \
  --local_dir /path/to/Qwen3.6-35B-A3B-W8A8-DCU
```

The `-DCU` suffix above is only a local directory name. The actual upstream model ID is the ModelScope repository without that suffix.

### 4.3 Apply the validated DCU config

```bash
python3 scripts/apply_dcu_config.py \
  --model-dir /path/to/Qwen3.6-35B-A3B-W8A8-DCU
```

Expected final hash:

```text
b550b28342afd4c61841e2684b06da15f3a0ec3c807ceb22259b0074be9975ae
```

The original config is backed up as `config.json.upstream.bak`.

### 4.4 Start R180 without MTP

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8-DCU
export GPU_ID=0
export PORT=8000
bash scripts/serve_nomtp.sh
```

Expected steady-state single-concurrency decode: **about 53–55 tok/s**.

### 4.5 Start R184 with MTP3

Run only one model service on the selected benchmark GPU, then:

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8-DCU
export GPU_ID=0
export PORT=8000
bash scripts/serve_mtp3.sh
```

Core speculative configuration:

```text
method = qwen3_next_mtp
num_speculative_tokens = 3
quantization = compressed-tensors
CUDAGraph capture sizes = 1, 4
```

A conservative fixed-workload expectation is **around 85 tok/s**. Higher-acceptance prompts can exceed 100 tok/s, but the 107.44 tok/s peak is not a universal rate.

### 4.6 Benchmark

After the service is ready:

```bash
curl http://127.0.0.1:8000/v1/models
python3 -m pip install -r requirements.txt

python3 scripts/benchmark_openai.py \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model qwen36-35b-a3b-w8a8-k100ai \
  --max-tokens 512 \
  --rounds 6
```

The first real request may trigger Triton runtime compilation. Record it separately from steady-state results.

## 5. What was optimized?

### 5.1 Shape-aware small-M W8A8 Linear

Single-concurrency decode is dominated by skinny `M=1–4` GEMMs. We tuned exact runtime `(M,K,N)` shapes for K100AI/gfx928 rather than relying entirely on the general heuristic.

Two important no-MTP M=1 projections:

| Projection | General path | K100AI tuned | Microbenchmark uplift |
|---|---:|---:|---:|
| `2048 -> 12288` | ~371.99 μs | ~90.03 μs | ~4.13× |
| `2048 -> 9216` | ~221.75 μs | ~77.53 μs | ~2.86× |

### 5.2 W8A8 `lm_head`

The 248,320-token vocabulary makes the output projection expensive during decode. The runtime patch installs a low-M W8A8 path for the large `lm_head`.

### 5.3 K100AI FusedMoE configuration

The model has 256 routed experts and activates 8 per token. The tested vLLM FusedMoE config is:

```text
configs/E=256,N=512,device_name=K100_AI.json
```

### 5.4 R184 Gated DeltaNet QKVZ + BA fusion

R184 fuses the QKVZ and BA W8A8 projections with a dual-weight-pointer kernel, reducing duplicate quantization/launch overhead without the KV-cache penalty observed in full-weight prepacking experiments.

### 5.5 `torch.compile` and CUDAGraph

```text
compile mode = 3
combo-kernel benchmark = off
R180 capture sizes = 1
R184 capture sizes = 1, 4
```

### 5.6 Native Qwen MTP3

```text
method = qwen3_next_mtp
num_speculative_tokens = 3
```

Draft tokens remain verified by the target model; speedup depends on acceptance rate.

## 6. Correctness and benchmarking policy

A faster isolated kernel is not accepted as a model-level win by itself.

Important validation gates include:

- R184 vs its same-mode MTP3 baseline: identical fixed-prompt text/token output and max logprob difference 0 in the validation set;
- R180 dominant projection microbenchmarks: `relative-L2 = 0`;
- R189/R192 had faster isolated kernels but slower full-model throughput and were rejected;
- no-MTP and MTP3 are separate inference modes and are not claimed to generate byte-identical output to each other.

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
docs/
```

## 8. Reproduce with Codex

See [`docs/REPRODUCE_WITH_CODEX.md`](docs/REPRODUCE_WITH_CODEX.md). The procedure explicitly verifies K100AI hardware, downloads the upstream weights, applies the exact DCU config, starts R180/R184 independently, warms up before benchmarking, and records throughput and correctness data.

## 9. Scope and limitations

- Validated on **K100AI / gfx928 only**.
- K100 is a different accelerator and was not used for these results.
- Optimized primarily for single-/low-concurrency decode latency.
- MTP throughput depends strongly on speculative-token acceptance rate.
- Model weights and the Hygon Docker image remain external upstream dependencies.
- The DCU config is required to reproduce this W8A8 deployment, but it does not modify the model weights themselves.

## 10. License

Repository code is released under Apache License 2.0. Model weights, Hygon runtime components, and other upstream artifacts remain under their respective licenses and terms. See [`NOTICE.md`](NOTICE.md).
