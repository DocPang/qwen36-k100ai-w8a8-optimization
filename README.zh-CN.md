# 面向海光 K100AI 的 Qwen3.6-35B-A3B W8A8 推理优化

[English](README.md) | **简体中文**

本项目针对 **海光 K100AI（gfx928）**，在海光社区 vLLM 0.18.1 / DTK 26.04 环境中，对 **Qwen3.6-35B-A3B W8A8** 进行单卡推理优化。

> [!IMPORTANT]
> **K100 和 K100AI 是两个不同型号的加速卡。**
> 本仓库的开发、调优与性能验证全部基于 **K100AI**，不代表这些 kernel、tile 参数、性能结果或兼容性可以直接套用到 K100。

本仓库只保留最终稳定、可复现的优化路径，包括运行时补丁、MoE 配置、启动脚本、测试脚本和结果说明。个人基础设施信息、模型权重、Docker 镜像本体以及未进入最终方案的实验分支均未公开。

## 性能结果

以下均为 **K100AI 单卡、TP=1、单并发 Decode** 结果。

| 配置 | MTP | Decode 吞吐 | 说明 |
|---|---:|---:|---|
| 原版热态基线 | 关闭 | **19.873 tok/s** | 统一热态、正常 compile/CUDAGraph 条件下的原版 vLLM 路径 |
| R180 | 关闭 | **平均 53.46 tok/s** | `llm_speedtest`：输入 128/256/512/1024 tokens，输出 512 tokens |
| R180 固定 512-token | 关闭 | **54.61 tok/s** | 固定提示词重复测试 |
| R184 固定 512-token | MTP3 | **中位约 85.29 tok/s** | 更适合作为稳定保守值 |
| R184 `llm_speedtest` | MTP3 | **平均 96.84 tok/s** | 受推测 token 接受率影响 |
| R184 实测峰值 | MTP3 | **107.44 tok/s** | 高接受率提示词，不应视为所有负载的固定速度 |

6 月份的一份早期测试报告记录过原版单卡、无 MTP 为 **12.55 tok/s**，但该数据包含早期冷路径和测速口径差异。因此，本项目讨论优化倍率时统一采用 **19.873 tok/s** 作为可比较的原版热态基线。

MTP 的速度取决于推测 token 接受率。实际部署时，固定提示词下约 **85 tok/s** 比 107 tok/s 峰值更适合作为保守预期。

更完整的结果见 [`results/RESULTS.md`](results/RESULTS.md)。

## 做了哪些优化

### 1. 针对真实 Decode 形状的 W8A8 Linear 调优

这个模型在单并发 Decode 时存在大量非常小的 `M` 矩阵乘。通用 heuristic 在 K100AI/gfx928 上并不理想，因此我们按真实 `(M, K, N)` 形状为常见 Decode 路径选择专门的 Triton 配置。

两个重要的无 MTP、`M=1` 大投影微基准从：

- `2048 -> 12288`：约 371.99 us → **90.03 us**
- `2048 -> 9216`：约 221.75 us → **77.53 us**

在调优测试中，这两条路径与参考 kernel 的数值结果一致。

### 2. W8A8 `lm_head`

Qwen3.6-35B-A3B 的词表大小为 248,320，Decode 时超大的输出投影开销不可忽略。运行时补丁为低 batch 的 `lm_head` 增加 W8A8 快速路径，避免继续走较慢的默认实现。

### 3. K100AI 专用 MoE 配置

Qwen3.6-35B-A3B 包含 256 个 routed experts，每个 token 激活 8 个专家。本仓库提供实际使用的 vLLM FusedMoE 配置：

`configs/E=256,N=512,device_name=K100_AI.json`

这里的文件名沿用海光 vLLM 运行时识别的设备命名。

### 4. R184：融合 Gated DeltaNet 的 QKVZ + BA 投影

R184 将 Gated DeltaNet 路径中原本分离的 QKVZ 与 BA W8A8 投影合并处理，减少重复的 kernel launch、量化与中间数据开销。

我们也测试过完整权重预拼接方案，但它会明显占用更多显存、降低 KV Cache 容量，因此最终没有采用。R184 保留了更好的显存效率。

### 5. `torch.compile` 与 CUDAGraph

最终配置针对实际 Decode 形状使用：

- compile mode 3
- 关闭无收益的 combo-kernel benchmark
- 无 MTP：CUDAGraph capture size `1`
- MTP3：CUDAGraph capture size `1 4`

也就是说，主模型单 token Decode 和 MTP3 的 4-token 校验路径都进入了图捕获。

### 6. 使用 Qwen 原生 MTP 头，推测 3 个 token

当前最稳定的推测解码配置使用模型自带 MTP 头：

```text
method = qwen3_next_mtp
num_speculative_tokens = 3
quantization = compressed-tensors
```

MTP 只负责提出未来 token 草稿，最终仍由目标模型验证。提示词越容易连续预测，接受率越高，Decode 提升越明显；复杂、分叉较多的生成则会接近 85～90 tok/s 的保守区间。

## 上游环境与模型来源

本项目**不重新分发模型权重，也不重新分发海光 Docker 镜像**。

### 海光社区 vLLM 环境

本次测试采用海光/光合开发者社区 Qwen3.6 ModelZoo 对应环境：

- vLLM：`0.18.1+das.fa71803.dtk2604`
- DTK：`26.04`
- Triton：`3.6.0+gitc73250c4.staging`
- Torch：`2.10.0+das.opt1.dtk2604.20260325.g6b060a`
- 推荐镜像 tag：`harbor.sourcefind.cn:5443/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423`
- 本项目实测所用镜像 digest：`sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811`

Qwen3.6 ModelZoo v1.1 标签明确写有“新增 K100AI 支持，及 FP8 数据类型”：

<https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/tags>

用于固定 vLLM 0.18.1 镜像与版本信息的环境修订：

<https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/commit/8a54c4e3df888c81165e5106657d17865bac3644>

当前 Qwen3.6 ModelZoo 页面：

<https://developer.sourcefind.cn/codes/modelzoo/qwen3.6/-/blob/main/README.md>

公开启动脚本继续保留 `--disable-custom-all-reduce`，与本次 K100AI 实际验证环境保持一致。

### 本项目使用的 W8A8 模型

本次结果使用的 W8A8 权重来自魔搭社区：

**`metax-tech/Qwen3.6-35B-A3B-W8A8`**

<https://www.modelscope.cn/models/metax-tech/Qwen3.6-35B-A3B-W8A8>

下载示例：

```bash
pip install modelscope
modelscope download \
  --model metax-tech/Qwen3.6-35B-A3B-W8A8 \
  --local_dir /path/to/Qwen3.6-35B-A3B-W8A8
```

该 checkpoint 通过 vLLM 的 `compressed-tensors` W8A8 路径加载：

- 权重：静态 INT8、per-channel
- 激活：动态 INT8、per-token
- 相关输出路径保持 BF16

原始未量化 Qwen 模型：

<https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B>

## 快速开始

在 K100AI 服务器上克隆本仓库，并先下载魔搭模型。

### 无 MTP / R180

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8
export GPU_ID=0
export PORT=8000
bash scripts/serve_nomtp.sh
```

### MTP3 / R184

```bash
export MODEL_DIR=/path/to/Qwen3.6-35B-A3B-W8A8
export GPU_ID=0
export PORT=8000
bash scripts/serve_mtp3.sh
```

启动脚本默认使用已经验证过的镜像 tag + digest 组合，避免未来社区覆盖同名 tag 后环境发生变化。上游镜像仓库可能要求登录权限。

## 测速

服务 ready 后：

```bash
python3 -m pip install -r requirements.txt
python3 scripts/benchmark_openai.py \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model qwen36-35b-a3b-w8a8-k100ai \
  --max-tokens 512 \
  --rounds 6
```

测速脚本会打印每轮 Decode 吞吐和输出 SHA256。首次请求可能触发 Triton/runtime 编译，应单独记录，不要与后续稳定请求混在一起计算稳态速度。

## 用 Codex 复现

已经提供一份不包含个人基础设施信息的 Codex 复现流程：

[`docs/REPRODUCE_WITH_CODEX.md`](docs/REPRODUCE_WITH_CODEX.md)

推荐让 Codex 严格按仓库中的启动脚本、同卡预热、固定输出门禁和完整模型 A/B 执行，不要只根据裸 kernel 微基准判断优化是否成功。

## 仓库结构

```text
configs/                 K100AI 的 vLLM FusedMoE 配置
patches/r180_nomtp/      当前最佳无 MTP runtime patch
patches/r184_mtp3/       当前最佳 MTP3 runtime patch
scripts/serve_nomtp.sh   通用单卡 R180 启动脚本
scripts/serve_mtp3.sh    通用单卡 R184 启动脚本
scripts/benchmark_openai.py
results/                 已清洗的性能结果
```

## 适用范围与限制

- 仅在 **海光 K100AI / gfx928** 上完成验证。
- **没有使用 K100** 得出本仓库任何性能结果。
- 当前调优目标是单并发/低并发 Decode，不能直接外推到高并发吞吐。
- MTP 性能高度依赖推测 token 接受率。
- Docker 镜像和模型权重属于外部依赖，本仓库不重新分发。
- 裸 kernel 微基准再快，如果完整模型同卡 A/B 没有收益，也不会被列为最终优化方案。

## License

本仓库代码以 Apache License 2.0 发布。模型权重和海光运行环境分别遵循各自上游许可与使用条款，详见 [`NOTICE.md`](NOTICE.md)。
