# 知识架构

知识库按稳定对象组织：训练目标、数据分布、张量状态、计算图、环境状态和测量协议。模型名称、框架和榜单会变化，这些对象仍能连接新的证据。

## 十二个问题域

| 问题域 | 稳定对象 | 入口 |
| --- | --- | --- |
| 基础 | token、概率、目标、上下文学习、缩放 | [基础知识地图](../foundations/index.md) |
| 数据 | 来源、谱系、过滤、混合、序列、反馈 | [数据工程](../data/index.md) |
| 模型结构 | attention、位置、MLP、MoE、递推、记忆 | [模型结构](../architecture/index.md) |
| 多模态 | encoder、投影、模态 token、生成与动作 | [多模态](../multimodal/index.md) |
| 训练与对齐 | optimizer、SFT、蒸馏、PEFT、checkpoint | [训练与对齐](../training/index.md) |
| 系统 | dtype、kernel、通信、placement、checkpoint | [系统与基础设施](../systems/index.md) |
| 推理与服务 | 解码、KV、缓存、调度、量化、集群 | [推理与服务](../inference/index.md) |
| 检索与智能体 | 索引、证据、工具、记忆、规划、权限 | [检索、工具与智能体](../applications/index.md) |
| 推理时计算 | 候选、搜索状态、验证器、预算 | [推理与推理时计算](../reasoning/index.md) |
| 强化学习 | 序贯决策、策略、价值、reward、环境与反馈 | [强化学习](../reinforcement-learning/index.md) |
| 评测与可靠性 | estimand、分母、judge、威胁、SLO | [评测与可靠性](../evaluation/index.md) |
| 谱系与证据 | 历史转折、关键工作、版本、来源、未知 | [技术谱系](../landscape/index.md) |

## 依赖关系

```text
目标与数据
  -> 模型结构
  -> 训练算法
  -> checkpoint
  -> 推理系统
  -> 检索 / 工具 / 环境
  -> 端到端行为
  -> 评测、门禁与反馈
```

系统层贯穿训练和推理；多模态改变输入、结构和评测；推理时计算可以只发生在部署，也可以通过蒸馏、偏好或 RL 回到训练。技术谱系解释这些对象之间的历史转折，不替代机制定义。

### 数据与目标

[来源谱系](../data/sources-provenance.md)确定训练对象能否回溯，[数据混合](../data/mixtures-curricula.md)决定真实 token share，[序列构造](../data/sequence-construction.md)将文档变成 attention、position 和 loss mask。[概率与损失](../foundations/probability-objectives.md)定义这些张量如何产生梯度。

### 结构与系统

[Decoder Block](../architecture/decoder-block.md)、[注意力](../architecture/attention-variants.md)、[MoE](../architecture/moe.md)和[记忆架构](../architecture/memory-architectures.md)定义计算图；[性能成本模型](../systems/performance-model.md)、[Attention Kernel](../systems/attention-kernels.md)和[模型并行](../systems/model-parallelism.md)解释它如何映射到硬件。

### 训练与推理

[预训练](../training/pretraining.md)形成基础分布，[SFT](../training/supervised-finetuning.md)提供示范，[蒸馏](../training/distillation.md)与 [PEFT](../training/peft.md)改变知识和适配成本；[强化学习](../reinforcement-learning/index.md)则在策略会改变采样分布、反馈可能延迟时讨论行为改进。部署侧由[解码](../inference/decoding.md)、[KV Cache](../inference/kv-cache.md)、[调度](../inference/scheduling-goodput.md)和[量化](../inference/quantization.md)决定可交付能力与成本。

### 知识、行动与反馈

[RAG](../applications/rag.md)连接外部证据，[工具调用](../applications/tool-use.md)连接外部动作，[智能体运行时](../applications/agent-runtime.md)维护状态。[搜索与验证](../reasoning/search-verification.md)产生推理时反馈；需要参数更新时进入[推理后训练](../training/reasoning-posttraining.md)，涉及环境转移与长时信用时继续进入 [Agentic RL](../agentic-rl/index.md)。

### 测量

[语言模型评测协议](../evaluation/language-model-evaluation.md)冻结模型、数据和 harness；[统计推断](../evaluation/statistical-inference.md)给出 effect 与区间；[校准](../evaluation/calibration-uncertainty.md)、[事实性](../evaluation/hallucination.md)、[Agent 终态](../evaluation/agent-tool-evaluation.md)和[安全评测](../evaluation/safety-evaluation.md)定义不同的失败边界。

## 页面角色

### 总览

总览回答该问题域包含哪些对象、它依赖什么、下一步读哪里。它保留稳定 URL，不复制专题中的全部推导。

### 机制

机制页围绕一个可独立验证的问题，通常包含：

```text
motivation and scope
variables, shapes or state
derivation / mechanism
minimal implementation contract
system cost and trade-offs
failure / attack surface
evaluation and primary evidence
```

公式说明变量和归一化；性能结论绑定 dtype、shape、硬件与软件；训练结论绑定数据、预算和评测。

### 实践

[实践页](../practice/index.md)用短实现固定关键语义，而不是复刻完整框架。每个实现以 reference、断言、退化输入和适用边界为核心，再与优化实现比较。

### 技术谱系与工作深读

[谱系页](../landscape/index.md)解释一个瓶颈如何推动下一项机制，并允许路线并行、分叉和重新汇合。它不复制 canonical 机制页的全部推导。

工作深读围绕一项或一组紧密相关的原始工作，连接前序矛盾、关键公式、最小可执行 reference、实验支持范围、官方实现边界与后续影响。它不是论文摘要或逐节复述。

### 家族案例

案例用于检验通用框架，明确模型、论文、权重、API 与产品版本。一个模型采用某机制，不会使该案例页成为机制的唯一解释。

### 索引

[术语表](../glossary.md)、[精选文献](../references.md)、[覆盖地图](coverage.md)和[更新日志](../changelog.md)负责发现与回溯；正文中的技术结论仍落在 canonical mechanism page。

## Canonical 边界

- 一个概念保留一个主要解释页，其他页面只补充上下文并链接。
- 旧的宽主题 URL 可保留为稳定总览，例如[参数高效训练与压缩](../training/peft-compression.md)与[后训练总览](../training/post-training.md)。
- 架构机制与实现优化分开：attention 定义在模型结构，kernel 在系统，KV 生命周期在推理。
- 离线偏好、在线 RL、推理时搜索和 Agentic RL 在同一强化学习主线中保持独立页面，因为数据分布、状态和目标不同。
- 模型发布与普遍规律分开；时效性事实进入谱系或证据卡。

## 交叉链接 {#cross-links}

一条链接应表达关系：

- 目标函数对应的 mask 和最小实现；
- 模型结构改变的 FLOPs、bytes 或 KV；
- 训练算法需要的数据、reference 和 verifier；
- 系统优化保留的数值与状态不变量；
- 评测结论使用的分母、预算和污染边界。

只写“相关内容”或堆叠论文列表，不会形成知识网络。完整主干见[覆盖地图](coverage.md)。

## 完整性标准

一项成熟专题至少能回答：

1. 对象、输入输出和适用范围是什么；
2. 公式或状态转移怎样得到；
3. 最小 reference 如何验证；
4. 真实系统的计算、内存、通信和时延怎样变化；
5. 数据、指标、分母和预算如何固定；
6. 哪些退化输入、分布偏移或攻击会破坏它；
7. 结论来自理论、原论文、官方实现还是仍待独立验证。

阅读纪律见[阅读方法](method.md)，事实与时效口径见[证据与研究方法](evidence.md)。

## Reference {#reference}

- [Diátaxis documentation framework](https://diataxis.fr/)
- [MkDocs Material navigation](https://squidfunk.github.io/mkdocs-material/setup/setting-up-navigation/)
