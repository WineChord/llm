# 实验方法

最小实验不是缩小项目规模，而是隔离一个机制并保留能证伪它的观察。质量、正确性、性能和资源需要分别测量，再解释它们之间的关系。

## 实验卡

每次实验固定：

| 项 | 最小记录 |
| --- | --- |
| 问题 | 可证伪假设、预期机制、成功与失败条件 |
| 对照 | baseline、唯一主要变量、保持不变的条件 |
| 输入 | 数据快照、slice、tokenizer、模板与采样方式 |
| 模型 | checkpoint、adapter、精度、上下文与解码参数 |
| 代码 | commit、依赖、kernel、编译与运行时版本 |
| 系统 | 硬件、拓扑、驱动、并行布局和资源限制 |
| 统计 | 指标、样本数、重复、seed、停止规则与区间 |
| 产物 | 原始输出、日志、trace、失败样本与结论边界 |

“其余配置相同”不够具体。至少保存实际解析后的配置和派生量，例如 global batch tokens、有效 loss tokens、KV bytes 与请求长度分布。

## 四类实验

### 语义 reference

用手算、FP64 小张量或确定性状态机固定目标。入口见[手撕实现](minimal-implementations.md)。

### 等价性

比较优化前后 forward、backward、状态转移和随机分布。先找第一个分叉点，不用最终 logits 容差掩盖中间错误。

### 消融

一次只改变能解释因果的问题变量。若同时换数据、tokenizer、训练时长和模型结构，结果只能说明整套 recipe 不同。

### 压力与故障

主动构造极端长度、空 mask、零方差、OOM、worker 退出、工具超时、注入和 stale cache。happy path 不能代表可靠性。

## 实现路径

| 路线 | 入口 |
| --- | --- |
| 输入协议 | [Tokenizer](tokenizers.md)、[序列构造](../data/sequence-construction.md) |
| 模型数学 | [张量原语](tensor-primitives.md)、[Decoder-only Transformer](transformer-from-scratch.md) |
| 替代结构 | [递推与记忆](sequence-models.md)、[多模态原语](multimodal.md) |
| 学习目标 | [训练目标](training-objectives.md) |
| 集群语义 | [分布式与容错](distributed-systems.md) |
| 服务状态 | [推理引擎](inference-engine.md) |
| 外部系统 | [检索与智能体](retrieval-agents.md) |
| 额外计算 | [推理时计算](test-time-compute.md) |
| 结论可信度 | [评测工具](evaluation-tooling.md) |

这些 reference 不包含 CLI、配置和部署脚手架。需要生产系统时读对应机制页与成熟实现；需要解释一个结果时，先让最小 reference 通过。

## 正确性矩阵

每个算子或状态机至少覆盖：

- **identity**：退化为已知实现或不改变输入；
- **mask**：被屏蔽内容不影响输出和梯度；
- **degenerate**：空、单元素、零方差、全相同、容量耗尽；
- **partition**：切 block、microbatch、rank 或 chunk 后语义不变；
- **resume**：中断恢复与连续执行一致；
- **version**：不兼容 tokenizer、layout、schema 或 checkpoint 被拒绝；
- **failure**：错误类别显式返回，不被吞成普通零值。

## 性能实验

1. 用[性能模型](../systems/performance-model.md)先算 FLOPs、bytes、通信与峰值显存；
2. 固定硬件功耗、时钟、NUMA 与网络拓扑；
3. 预热 kernel、allocator、编译图和 cache；
4. 使用真实 shape 与长度分布，不只测整齐方阵；
5. 区分 host、queue、kernel、通信与同步时间；
6. 同时报吞吐、p50/p95/p99、显存、能耗与质量；
7. 给出置信区间和运行间方差；
8. 保存 profiler trace，并解释瓶颈为何出现。

只报峰值 utilization 会遗漏有效工作比例；只报 tokens/s 会遗漏 SLO 违约。推理服务应优先看 [goodput](../inference/scheduling-goodput.md)。

## 复现层级

| 层级 | 能说明什么 |
| --- | --- |
| 接口可运行 | 环境与基本调用兼容 |
| reference 对齐 | 数学或状态语义一致 |
| 指标接近 | 在给定数据和协议上接近 |
| 消融重现 | 机制方向得到支持 |
| 跨规模/硬件重现 | 外推边界扩大 |

一个层级通过不能自动推出下一层。作者报告、开放代码、开放权重与独立复现也应分开记录。

## 失败记录

失败实验保留：

- 首个异常 step 与最后一个正常 checkpoint；
- 输入、随机状态、数据 cursor 和并行布局；
- 最小复现与完整运行的差异；
- 被排除的假设及证据；
- 修复后的回归样例；
- 结论仍未知的部分。

按症状定位的顺序见[调试手册](debugging.md)，评测中的配对、bootstrap 与缺失值见[手撕评测工具](evaluation-tooling.md)。
