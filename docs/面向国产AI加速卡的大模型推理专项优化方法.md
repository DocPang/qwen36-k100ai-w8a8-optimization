# 面向国产AI加速卡的大模型推理专项优化方法

> 本文不是一份“照抄参数即可提速”的调参清单，而是一套从真实负载出发，针对 **模型 × 加速卡 × 推理框架 × 业务场景** 联合优化的方法论。
>
> 本项目的完整案例是：Qwen3.6-35B-A3B W8A8 × Hygon K100AI/gfx928 × vLLM × 单卡低并发/Agent。具体 tile、wave、split-K 等参数只对该组合有效；本文总结的分析流程、验证方法和优化边界则可迁移到其他海光卡、其他国产AI加速卡和其他大模型。

## 1. 为什么“专项优化”仍然具有通用价值

很多性能项目容易陷入两个极端：

1. 认为厂商通用库已经足够好，应用侧几乎没有优化空间；
2. 把某张卡上搜出的 `BM/BN/BK` 当成普适答案，换卡、换模型继续照抄。

两者都不准确。

大模型推理性能由四个维度共同决定：

- **模型结构**：Dense/MoE、Attention/Linear Attention、hidden size、expert 数、top-k、词表大小；
- **硬件结构**：CU 数量、wave 大小、矩阵指令、LDS/VGPR、显存带宽、拓扑；
- **框架实现**：量化路径、kernel heuristic、CUDAGraph、调度器、KV/状态缓存、CPU/GPU metadata；
- **真实负载**：单并发还是高并发、短上下文还是长上下文、普通聊天还是Agent、是否使用多模态和Prefix Cache。

因此，真正的优化对象不是“Qwen”或“K100AI”中的任意一个，而是：

> **某个模型在某个硬件上，以某种框架实现，在某种真实负载下的执行图。**

这也是专项优化可以迁移的原因：具体参数不通用，但“怎样找到执行图中真正昂贵的部分”是通用的。

---

## 2. 第一原则：先固定真实目标，再谈性能

在开始 profiler 之前，必须先写清楚目标负载。

例如本项目最终关注的是：

- 单卡 TP=1；
- 单并发/低并发；
- Decode 延迟优先；
- W8A8；
- Agent 会重复携带大量 System Prompt、工具 schema 和历史上下文；
- 最终需要 Tool Calling、262K上下文、Prefix Cache 和多模态。

如果目标换成高并发离线吞吐，最优方案可能完全相反。

典型冲突包括：

- MTP 可以提高单请求速度，但高并发下额外草稿/验证成本未必划算；
- Prefix Cache 可能让纯 Decode benchmark 变慢，却大幅降低Agent重复长前缀的整轮延迟；
- 更大的 `max_num_batched_tokens` 可能提高某些 Decode 图效率，却恶化超长冷 Prefill；
- TP2/TP4能提高总吞吐，但通信开销可能让单请求延迟不升反降。

所以不能只问：

> “tok/s最高是多少？”

应该问：

> “哪个配置让目标业务的端到端时间最短，同时满足质量、显存、上下文和功能要求？”

---

## 3. 建立可信基线：所有优化的地基

基线必须做到可重复、可对比、可回滚。

建议至少固定：

- 模型权重与配置 SHA256；
- 容器 image/tag/digest；
- vLLM、Torch、Triton、驱动/运行时版本；
- GPU 型号与数量；
- TP、上下文长度、显存利用率；
- CUDAGraph/compile 参数；
- prompt、输出 token 数、temperature/seed；
- 是否 MTP、Prefix Cache、多模态；
- 冷启动与热态分别记录。

测试时至少记录：

- completion tokens；
- wall time；
- tok/s；
- TTFT；
- Prefill tok/s；
- 输出 SHA256；
- 请求成功率；
- speculative acceptance；
- KV/cache 容量。

### 3.1 不要混用“历史数字”和“当前统一口径”

早期结果可能受到冷启动、不同 prompt、不同上下文长度、不同流式计时方式影响。

它们可以用于追溯历史，但不能在没有统一口径时直接做百分比宣传。

### 3.2 同卡A/B优先于跨卡A/B

即使是同型号GPU，也可能存在：

- 频率状态差异；
- 温度差异；
- 其他进程占用；
- cache/JIT状态不同。

因此关键候选最好使用同一张卡顺序测试，或交替A/B，而不是仅比较两张卡上的一次结果。

---

## 4. 不要优化“Linear”，要优化真实 Shape

这是本项目收益最大的经验之一。

大模型单并发 Decode 时，很多 GEMM 实际上是极小 M：

```text
M = 1 / 2 / 3 / 4
K = hidden/intermediate dimension
N = projection dimension
```

通用 GEMM heuristic 通常需要兼顾大量 batch/shape，不一定适合这种极端 skinny GEMM。

在本项目中，两个真实 M=1 大投影：

| Shape | 通用路径 | K100AI专项配置 | 微基准提升 |
|---|---:|---:|---:|
| `M=1, 2048 -> 12288` | 371.99 μs | 90.03 μs | 4.13× |
| `M=1, 2048 -> 9216` | 221.75 μs | 77.53 μs | 2.86× |

把这两个高频shape补齐后，无MTP端到端从约36 tok/s跃升到54 tok/s级别。

这里真正可迁移的结论不是 `90.03 μs`，而是：

1. **先采集真实 `(M,K,N,dtype,layout)` 分布；**
2. 统计每种shape每token调用次数和累计时间；
3. 对高频shape单独搜索；
4. 检查真实权重stride/layout，不能用错误的微基准布局；
5. 再把最优配置加入运行时精确dispatch。

换成另一张海光卡时，shape可能仍然相同，但最优：

- tile；
- wave；
- split-K；
- block-per-CU；
- LDS策略；
- 权重预排布；

都应该重新搜索。

---

## 5. Microbenchmark只是筛选器，不是最终答案

本项目多次出现：

> 裸kernel看起来非常漂亮，但完整模型没有收益，甚至变慢。

原因可能包括：

- custom-op调用开销；
- CUDAGraph形状改变；
- register/LDS资源竞争；
- 前后kernel无法很好重叠；
- 额外量化/写回抵消了GEMM收益；
- MTP接受率发生变化；
- 运行时metadata和CPU同步成为新瓶颈。

例如某次QKVZ+BA实验，微基准接近1.8×，完整模型却没有获得对应收益，最终被淘汰。

因此推荐使用三级门禁：

### Gate A：算子正确性

- relative-L2；
- max abs diff；
- eager/compile一致性；
- CUDAGraph可运行。

### Gate B：固定完整模型A/B

- 同卡；
- 相同prompt；
- 相同输出长度；
- 多轮热态；
- 输出hash/logprob对照。

### Gate C：目标业务A/B

例如Agent应额外测试：

- 长System Prompt；
- Tool Calling；
- JSON/结构化输出；
- Prefix Cache冷热；
- 多模态；
- 跨Mamba/GDN缓存边界；
- 长时间稳定运行。

只有通过Gate C的优化才有资格进入长期服务。

---

## 6. 每次大优化以后必须重新 Profile

性能优化会改变瓶颈排序。

早期：

> Linear是绝对第一热点。

大投影和小M Linear优化后：

> MoE重新成为第一热点。

继续优化后：

> dynamic INT8 quant、elementwise、runtime metadata、CPU/GPU同步开始变得值得关注。

因此不能长期拿最早的一份profile指导后面几十轮优化。

推荐循环：

```text
Baseline
  -> Profile
  -> 优化最大热点
  -> A/B + Quality Gate
  -> 重新Profile
  -> 再优化新的最大热点
```

这本质上是一个不断变化的 Pareto frontier。

---

## 7. MoE优化：不要只盯两次GEMM

对于MoE模型，一个token的MoE路径通常还包括：

- router；
- top-k；
- token/expert排序与对齐；
- stage1 gate/up GEMM；
- activation；
- 动态量化；
- stage2 down GEMM；
- router weight；
- reduce/sum；
- shared expert。

本项目的profile中，仅routed expert两段GEMM就曾占约20%，加上top-k、量化、归约与shared expert后，完整MoE块是更大的优化对象。

对于单token、固定top-k的场景，通用FusedMoE往往承担了为大batch设计的排序、padding和通用分发成本。

可迁移的专用路线是：

1. router直接得到固定数量的expert id/weight；
2. 对命中专家直接分配工作，不走通用大batch token sort；
3. stage1 epilogue融合 activation；
4. activation后直接量化供stage2使用；
5. stage2直接乘router weight；
6. 尾部直接reduce；
7. 有条件时把shared expert一起累加。

这类优化对DeepSeek、Qwen MoE、其他稀疏模型同样有意义，只是expert数量、top-k和intermediate shape不同。

---

## 8. 量化优化的重点不只是“换成INT8”

W8A8模型中，真正的成本包括：

- 权重读取；
- activation动态量化；
- scale计算；
- 中间结果写回；
- 相邻算子再次读取。

因此，在已经使用INT8之后，仍可能有明显空间。

典型融合边界：

```text
RMSNorm/residual
    -> dynamic per-token INT8 quant
    -> Linear
```

以及：

```text
Gate/Up GEMM
    -> SiLU * Up
    -> dynamic INT8 quant
    -> Down GEMM
```

如果同一个hidden同时送往多个投影，还应优先考虑：

> **量化一次，多路Linear共享 xq/xscale。**

本项目当前使用的compressed-tensors W8A8 INT8路径没有自动获得vLLM主要面向FP8/FP4的融合，因此“扩展INT8融合支持”是后续很有价值的方向。

---

## 9. 静态权重应该考虑预重排，而不是每token重复整理

推理与训练不同，大多数权重在部署期间是静态的。

因此可以在模型加载时一次性完成：

- preshuffle；
- tile-friendly layout；
- scale重排；
- 合并连续投影；
- 针对矩阵指令的内存布局。

运行时换取：

- 更合并的global load；
- 更少寄存器shuffle；
- 更少临时buffer；
- 更好的LDS流水。

但必须同时看显存成本。本项目尝试过连续预打包QKVZ+BA，虽然局部更快，却因为额外权重/AOT保留显著降低KV容量，最终没有采用。

因此正确目标不是“kernel最快”，而是：

> **在服务需要的上下文和并发约束下，端到端最快。**

---

## 10. Runtime同步可能和GEMM一样值得优化

当主要GEMM已经提速后，CPU/GPU之间的小同步会变得非常显眼。

本项目在混合GDN + Prefix Cache + speculative decoding路径中发现：

- accepted-token metadata已经在GPU上；
- 运行时仍然每个decode step发起GPU→CPU copy；
- record event；
- 后续再次依赖CPU镜像构造metadata。

通过让常规快路径直接消费GPU上的accepted-token计数，删除不再使用的D2H/event和重复scratch分配，固定512 Decode从约73 tok/s提升到约80 tok/s。

这类经验非常适合厂商框架团队：

> profiler不能只盯大kernel，也要看每token之间的空隙、event、memcpy、`.item()`、synchronize、Python/Numpy metadata构造。

对于低batch Decode，几十微秒的小同步乘以每token几十/几百次，会形成非常可观的总成本。

---

## 11. Speculative Decoding要寻找甜点位，而不是越多越好

模型原生MTP可以让一次目标模型验证推进多个token。

但第n个草稿token的接受率通常逐级下降。

本项目曾观察到类似：

```text
第1枚：高
第2枚：明显下降
第3枚：继续下降
第4枚：额外收益不足以覆盖草稿成本
```

因此：

- MTP1不是永远最快；
- MTP2不是永远比MTP3稳妥；
- MTP4/MTP5也绝不是“预测更多就更快”。

正确方法是记录：

- 每个draft position的接受率；
- 每次target forward平均推进token数；
- draft head成本；
- verification成本；
- 完整tok/s。

在本项目当前Agent负载上，MTP3仍是甜点位；MTP2整体更慢，历史MTP4也曾明显退化。

---

## 12. Agent优化不能只看Decode tok/s

普通benchmark容易忽略Agent的真实结构：

```text
System Prompt
+ Tool schemas
+ Skills
+ 历史对话
+ 当前问题
+ Tool result
+ 再次推理
```

大量前缀会在每一轮重复。

因此Prefix Cache可能比再提高5 tok/s更重要。

本项目中，同一约55K-token前缀：

- 首次处理：几十秒；
- 缓存命中后：约6秒。

即使Prefix Cache与混合GDN的缓存布局一度让纯Decode从85 tok/s级下降到73 tok/s级，真实Agent重复任务的整轮延迟仍大幅降低。

后续通过runtime fastpath和4096 chunked-prefill配置，又把**保留Prefix Cache**的长期版Decode重新拉回约85 tok/s。

这个案例说明：

> **业务指标应当是Agent整轮延迟，而不是单独的Decode benchmark。**

---

## 13. Chunked Prefill需要同时看Prefill与Decode

`max_num_batched_tokens`这类参数看起来像“Prefill参数”，实际可能改变：

- compile range；
- graph形状；
- kernel选择；
- MTP数值路径和接受率；
- 激活峰值与KV容量。

本项目出现了典型结果：

- 4096：Decode明显更快，KV容量几乎不受影响，综合最佳；
- 8192：Decode再快约1%，但55K冷Prefill恶化到100秒以上，因此淘汰。

这说明参数调优必须使用**多目标指标**：

```text
Decode
Prefill
TTFT
KV容量
MTP acceptance
功能正确性
```

而不是只取一个数字最大化。

---

## 14. 多模态优化必须区分“显存成本”和“文本速度”

开启视觉编码器后：

- 常驻权重/工作区会占用更多显存；
- 可用KV容量下降；
- 但纯文本请求不一定会跑视觉encoder，因此Decode速度未必下降。

本项目实测就是这种情况：

- 多模态开启后KV容量下降；
- 纯文本固定512 Decode基本保持相同水平；
- 图片理解、Tool Calling、Prefix Cache可以同时正常工作。

所以不要根据“多加载了一个视觉模块”就假定文本推理必然变慢，应该实际测。

---

## 15. 质量门禁：性能优化绝不能只看“能生成文字”

建议把正确性分为四层。

### 15.1 算子数值一致性

- relative-L2；
- max abs diff；
- NaN/Inf；
- eager/compile/CUDAGraph。

### 15.2 完整模型确定性

- temperature=0；
- fixed seed；
- token序列；
- logprob；
- 输出SHA256。

### 15.3 任务级语义质量

- 数学；
- 代码；
- 中文长文本；
- JSON；
- Tool Calling；
- 多模态；
- 长上下文。

### 15.4 部署门禁

对于无法解释的输出分叉：

- 可以继续作为研究候选；
- 不能因为“看起来还能回答”就直接替换生产。

本项目曾经有一个候选端到端快30%以上，但5个固定prompt中4个出现贪心token分叉。即使回答仍然连贯，也没有直接进入长期服务。

这类失败实验非常值得保留，因为它能防止团队再次走同一条路。

---

## 16. 哪些东西可以直接借鉴，哪些必须重做

### 可以直接借鉴的方法

- 真实shape采集；
- 高频shape排序；
- skinny GEMM / split-K思路；
- static weight preshuffle；
- 单token MoE专用路径；
- Norm/Activation + Quant融合；
- CPU/GPU同步审计；
- MTP position acceptance分析；
- Prefix Cache冷热A/B；
- 同卡端到端门禁；
- 失败实验记录方式。

### 必须按硬件重做的参数

- tile size；
- wave/warp数量；
- split-K；
- fixed grid；
- block-per-CU；
- LDS/VGPR预算；
- preshuffle layout；
- CUDAGraph尺寸；
- batch/prefill甜点位。

因此其他海光卡用户最正确的学习方式不是复制K100AI参数，而是：

> **复制实验流程，再为自己的卡重新搜索。**

---

## 17. 对芯片/软件栈厂商的意义

从厂商角度，这类专项优化最值得产品化的不是某一个patch，而是四类能力。

### 17.1 建立真实LLM Shape数据库

按以下维度索引：

```text
GPU架构
+ dtype/quant scheme
+ M/K/N
+ weight layout
+ batch regime
+ epilogue
+ 最优kernel配置
```

框架在加载模型后自动识别高频shape并匹配已调优配置。

### 17.2 为小batch Decode提供独立kernel族

厂商库不能只围绕大矩阵峰值FLOPS设计。

交互式LLM大量工作是：

```text
M=1~4 的 skinny GEMM/GEMV
```

应该单独维护：

- small-M INT8；
- split-K；
- preshuffle；
- single-token MoE；
- small-M lm_head。

### 17.3 把融合扩展到真实量化方案

不仅支持FP8/FP4，也要覆盖：

- compressed-tensors INT8；
- dynamic per-token activation quant；
- Norm→Quant；
- Activation→Quant；
- MoE中间量化融合。

### 17.4 Runtime也要进入性能工程范围

应持续检查：

- D2H/H2D小拷贝；
- event/synchronize；
- `.item()`；
- scheduler metadata；
- cache update；
- speculative bookkeeping；
- Python层小对象/数组分配。

大模型优化不能被限定为“GEMM团队的工作”。

---

## 18. 如何把这套流程自动化

可以进一步把专项优化做成“模型-加速卡性能体检系统”。

输入：

```text
模型路径
GPU型号
推理框架
量化方式
目标业务负载
```

自动执行：

1. 建立统一baseline；
2. profile真实请求；
3. 生成shape inventory；
4. 统计每token调用次数与累计耗时；
5. 对高频shape自动autotune；
6. 检查量化/融合机会；
7. 检查CPU/GPU同步；
8. 生成候选patch；
9. 同卡A/B；
10. 跑数值、JSON、Tool Calling、多模态质量门禁；
11. 重新profile；
12. 输出针对“该模型 × 该卡 × 该业务”的recipe。

Codex/Agent可以承担外层实验编排，而kernel profiler、benchmark和质量门禁保持程序化、可重复。

这样新模型到来时，工程师不必从零开始重复数百轮人工排查。

---

## 19. 推荐的标准优化流程

可以把本文压缩成下面这张清单：

```text
[1] 定义目标业务
        ↓
[2] 锁定模型/环境/测量口径
        ↓
[3] 建立稳定热态baseline
        ↓
[4] Profile + Shape Inventory
        ↓
[5] 优化最大热点
        ↓
[6] 算子数值门禁
        ↓
[7] 同卡完整模型A/B
        ↓
[8] 真实业务门禁
        ↓
[9] 重新Profile
        ↓
[10] Runtime / Fusion / Cache / MTP联合优化
        ↓
[11] 记录失败路线和适用边界
        ↓
[12] 固化可复现脚本与自动化recipe
```

最重要的三个原则是：

1. **优化真实shape，不优化抽象算子名称；**
2. **完整模型和真实业务优先于微基准；**
3. **每次大优化以后重新profile，不要拿旧瓶颈指导新系统。**

---

## 20. 结语

“某个模型在某张国产卡上的专项优化”并不是只能服务一个项目。

只要完整记录：

- 基线；
- 热点；
- shape；
- kernel搜索；
- runtime路径；
- 失败实验；
- 数值门禁；
- 真实业务A/B；

它就可以从一次性的工程经验，升级为：

> **可迁移的国产AI加速卡大模型推理优化方法。**

K100AI + Qwen3.6只是本文的验证案例。真正应该被复制到其他卡、其他模型和厂商软件栈中的，是这套寻找瓶颈、验证收益、控制风险和持续迭代的流程。
