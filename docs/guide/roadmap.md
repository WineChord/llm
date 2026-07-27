# 学习路线

LLM 横跨统计学习、模型结构、分布式系统和交互环境。先建立一条共同因果链，再沿问题深入；论文、框架和模型发布都放回这条链中理解。

## 共同主线

1. **表示与目标**：从[语言建模](../foundations/language-modeling.md)、[分词](../foundations/tokenization.md)和[概率、损失与梯度](../foundations/probability-objectives.md)理解 token 条件概率与训练目标。
2. **基本计算图**：推导 [Transformer](../architecture/transformer.md)、[Decoder Block](../architecture/decoder-block.md)、[位置编码](../architecture/position-encoding.md)和[注意力家族](../architecture/attention-variants.md)。
3. **数据到梯度**：连接[来源谱系](../data/sources-provenance.md)、[过滤去重](../data/filtering-dedup.md)、[混合课程](../data/mixtures-curricula.md)与[序列构造](../data/sequence-construction.md)。
4. **训练到行为**：依次阅读[预训练](../training/pretraining.md)、[SFT](../training/supervised-finetuning.md)、[蒸馏](../training/distillation.md)、[奖励建模](../training/reward-modeling.md)和[偏好优化](../training/offline-preference.md)。
5. **计算到系统**：用[性能成本模型](../systems/performance-model.md)连接[数值精度](../systems/precision-numerics.md)、[GPU 执行](../systems/gpu-execution.md)、[集合通信](../systems/collectives-sharding.md)与[模型并行](../systems/model-parallelism.md)。
6. **checkpoint 到服务**：从[解码](../inference/decoding.md)、[KV Cache](../inference/kv-cache.md)进入[运行时](../inference/runtime.md)、[调度与 Goodput](../inference/scheduling-goodput.md)、[量化](../inference/quantization.md)和 [P/D 分离](../inference/disaggregation.md)。
7. **外部知识与行动**：学习[索引与召回](../applications/retrieval-indexing.md)、[重排与上下文](../applications/reranking-context.md)、[证据约束生成](../applications/grounded-generation.md)、[工具调用](../applications/tool-use.md)和[智能体运行时](../applications/agent-runtime.md)。
8. **推理与反馈**：区分[推理时计算](../reasoning/test-time-compute.md)、[搜索与验证](../reasoning/search-verification.md)、[推理后训练](../training/reasoning-posttraining.md)与 [Agentic RL](../agentic-rl/index.md)。
9. **评测与边界**：最后用[评测协议](../evaluation/language-model-evaluation.md)、[统计推断](../evaluation/statistical-inference.md)、[校准](../evaluation/calibration-uncertainty.md)、[安全评测](../evaluation/safety-evaluation.md)验证整条链。

共同主线的最低实践是完成[手撕 Decoder-only Transformer](../practice/transformer-from-scratch.md)、[训练目标](../practice/training-objectives.md)与[评测工具](../practice/evaluation-tooling.md)中的核心不变量。

## 深入路径

### 模型结构与算法

阅读顺序：

1. [注意力家族](../architecture/attention-variants.md)与[位置编码](../architecture/position-encoding.md)；
2. [长上下文](../architecture/long-context.md)与[记忆架构](../architecture/memory-architectures.md)；
3. [MoE](../architecture/moe.md)与[状态空间、线性注意力](../architecture/state-space-linear-attention.md)；
4. [缩放与实验设计](../training/scaling-experiment-design.md)；
5. [张量原语](../practice/tensor-primitives.md)、[递推与记忆](../practice/sequence-models.md)。

每种结构都回答：它改变了什么状态与归纳偏置，计算图和复杂度怎样变化，收益在哪些数据、长度与硬件条件下成立。

### 训练系统

从[性能成本模型](../systems/performance-model.md)建立 FLOPs、bytes、通信与峰值显存账本，再读：

1. [数值精度与低精度计算](../systems/precision-numerics.md)；
2. [GPU 执行模型](../systems/gpu-execution.md)与 [Attention Kernel](../systems/attention-kernels.md)；
3. [集合通信与状态分片](../systems/collectives-sharding.md)；
4. [TP、PP、CP 与 EP](../systems/model-parallelism.md)和 [MoE 系统](../systems/moe-systems.md)；
5. [检查点与容错](../systems/checkpointing.md)及[系统韧性](../systems/resilience-observability.md)；
6. [分布式与容错实现](../practice/distributed-systems.md)。

目标是能从 tensor shape、dtype、placement 和 topology 推导通信量、临时状态与失败边界，而不是记并行缩写。

### 推理与 AI Infra

先区分 prefill 与 decode 的 shape 和瓶颈，再沿：

1. [KV Cache](../inference/kv-cache.md)与[缓存复用](../inference/cache-reuse.md)；
2. [推理运行时](../inference/runtime.md)与[调度、Goodput](../inference/scheduling-goodput.md)；
3. [量化](../inference/quantization.md)与[推测解码](../inference/speculative-decoding.md)；
4. [Prefill–Decode 分离](../inference/disaggregation.md)；
5. [推理基准与可靠性](../inference/benchmarking-reliability.md)和[推理引擎实现](../practice/inference-engine.md)。

评价方案时同时报告 TTFT、TPOT、尾延迟、Goodput、显存、质量、成本与失败状态。

### 应用与智能体

先把[检索增强](../applications/rag.md)拆成[索引召回](../applications/retrieval-indexing.md)、[重排上下文](../applications/reranking-context.md)与[证据约束生成](../applications/grounded-generation.md)，再进入：

1. [工具调用](../applications/tool-use.md)；
2. [记忆与规划](../applications/memory-planning.md)；
3. [智能体运行时](../applications/agent-runtime.md)与[安全](../applications/agent-security.md)；
4. [Coding Agent](../applications/coding-agents.md)；
5. [Agent 与工具评测](../evaluation/agent-tool-evaluation.md)；
6. [Agentic RL 轨迹契约](../agentic-rl/trajectory-contract.md)和[训练系统](../agentic-rl/training-systems.md)。

始终分开模型输出、外部证据、工具权限、环境终态和未授权副作用。

### 多模态

从[视觉语言模型](../multimodal/vision-language.md)和[融合训练](../multimodal/architecture-training.md)进入，再按任务分流：

- 理解与生成：[统一建模](../multimodal/unified-understanding-generation.md)与[生成模型](../multimodal/generative-modeling.md)；
- 高分辨率与交互：[文档、图表、GUI 与 Grounding](../multimodal/document-gui-grounding.md)；
- 时序信号：[音频与语音](../multimodal/audio-language-models.md)、[视频与世界模型](../multimodal/video-world-models.md)；
- 动作：[具身智能](../multimodal/embodied-agents.md)；
- 验证：[多模态原语](../practice/multimodal.md)与[多模态评测](../evaluation/multimodal-evaluation.md)。

### 评测与可靠性

从[指标与 estimand](../evaluation/metrics.md)开始，依次阅读：

1. [Benchmark 注册表](../evaluation/benchmark-registry.md)与[评测污染](../evaluation/contamination.md)；
2. [统计推断](../evaluation/statistical-inference.md)与[校准、不确定性](../evaluation/calibration-uncertainty.md)；
3. [生成式 Judge](../evaluation/generative-judges.md)与[幻觉、事实性](../evaluation/hallucination.md)；
4. [指令遵循](../evaluation/instruction-following.md)与[安全评测](../evaluation/safety-evaluation.md)；
5. [生产可靠性](../evaluation/production-reliability.md)。

目标是把错误定位到模型、数据、检索、工具、环境、评分器或服务层，并用明确分母、effect 和置信区间表达结论。

## 学会的标准

每个专题至少完成一次：

1. 定义输入、输出、状态和不变量；
2. 手算一个最小例子；
3. 阅读原始论文和一个公开实现；
4. 用 reference 与优化实现做对照；
5. 构造退化、缺失和攻击样例；
6. 写清数据、版本、预算、分母、effect 与未知。

[覆盖地图](coverage.md)用于检查是否只掌握了名词，[证据与研究方法](evidence.md)用于区分论文结论、实现状态与外推。
