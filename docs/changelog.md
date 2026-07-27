# 更新日志

## 2026-07-27

- 为 162 篇知识页统一可深链的 `Reference` 书目，补齐原论文、规范、官方文档与实现入口，并在[阅读方法](guide/method.md#page-reference)中固定页级引用规则；索引、导航、术语表、更新日志与[全站文献](references.md)按页面职责避免机械重复。
- 建立带有历史脉络的[技术谱系](landscape/index.md)：从[计数模型到可学习状态](landscape/lineages/counts-to-learned-state.md)、[递归到注意力](landscape/lineages/transduction-to-attention.md)、[预训练目标](landscape/lineages/pretraining-objectives.md)延伸到[后训练与对齐](landscape/lineages/training-alignment.md)、[分布式训练](landscape/lineages/distributed-training-systems.md)、[推理服务](landscape/lineages/inference-serving.md)、[多模态](landscape/lineages/multimodal-generation.md)、[检索与智能体](landscape/lineages/retrieval-agents.md)及[评测](landscape/lineages/evaluation.md)，并新增 22 篇带公式、来源、实现边界与可执行代码的关键工作深读。
- 新增 [SAO 与 CompactionRL](landscape/works/sao-compactionrl.md)：沿时间轴解释 single-rollout 异步更新、DIS 与 critic，沿空间轴解释学习式摘要、token-level normalization 与跨 segment 信用。
- 统一入口层：[首页](index.md)提供主题地图，[学习路线](guide/roadmap.md)提供递进路径，[知识架构](guide/architecture.md)定义章节边界，[覆盖地图](guide/coverage.md)连接概念、机制、实现与验证。
- 扩展基础链路：[上下文学习](foundations/in-context-learning.md)、[概率与目标函数](foundations/probability-objectives.md)、[数据来源与谱系](data/sources-provenance.md)、[过滤与去重](data/filtering-dedup.md)、[数据混合与课程](data/mixtures-curricula.md)、[合成数据](data/synthetic-data.md)和[记忆与隐私](data/memorization-privacy.md)。
- 完善模型结构：[Decoder Block](architecture/decoder-block.md)、[注意力变体](architecture/attention-variants.md)、[位置编码](architecture/position-encoding.md)、[长上下文](architecture/long-context.md)、[MoE](architecture/moe.md)、[状态空间与线性注意力](architecture/state-space-linear-attention.md)和[记忆架构](architecture/memory-architectures.md)。
- 重构多模态主线：[架构与训练](multimodal/architecture-training.md)、[统一理解与生成](multimodal/unified-understanding-generation.md)、[原生生成](multimodal/native-generation.md)、[文档与 GUI Grounding](multimodal/document-gui-grounding.md)、[音频语言模型](multimodal/audio-language-models.md)、[视频与世界模型](multimodal/video-world-models.md)和[具身智能体](multimodal/embodied-agents.md)。
- 展开训练全周期：[预训练](training/pretraining.md)、[缩放实验设计](training/scaling-experiment-design.md)、[蒸馏](training/distillation.md)、[参数高效适配](training/peft.md)、[奖励建模](training/reward-modeling.md)、[离线偏好优化](training/offline-preference.md)、[在线 RL](training/online-rl.md)和[推理后训练](training/reasoning-posttraining.md)。
- 深化训练系统：[性能模型](systems/performance-model.md)、[精度与数值](systems/precision-numerics.md)、[GPU 执行](systems/gpu-execution.md)、[并行策略](systems/model-parallelism.md)、[Attention Kernel](systems/attention-kernels.md)、[MoE 系统](systems/moe-systems.md)及[容错与可观测性](systems/resilience-observability.md)。
- 完善推理服务：[调度与 Goodput](inference/scheduling-goodput.md)、[KV Cache](inference/kv-cache.md)、[缓存复用](inference/cache-reuse.md)、[量化](inference/quantization.md)、[推测解码](inference/speculative-decoding.md)、[P/D 分离](inference/disaggregation.md)和[基准与可靠性](inference/benchmarking-reliability.md)。
- 贯通知识与行动：[检索与索引](applications/retrieval-indexing.md)、[重排与上下文](applications/reranking-context.md)、[Grounded Generation](applications/grounded-generation.md)、[工具使用](applications/tool-use.md)、[记忆与规划](applications/memory-planning.md)、[Agent 运行时](applications/agent-runtime.md)和[Agent 安全](applications/agent-security.md)。
- 建立推理与反馈主线：[推理总览](reasoning/index.md)、[测试时计算](reasoning/test-time-compute.md)、[搜索与验证](reasoning/search-verification.md)，并由 [Agentic RL](agentic-rl/index.md)连接[轨迹契约](agentic-rl/trajectory-contract.md)、[数据与环境](agentic-rl/data-environments.md)、[训练系统](agentic-rl/training-systems.md)和[长时任务](agentic-rl/long-horizon.md)。
- 扩展评测体系：[基准注册表](evaluation/benchmark-registry.md)、[统计推断](evaluation/statistical-inference.md)、[校准与不确定性](evaluation/calibration-uncertainty.md)、[生成式评审](evaluation/generative-judges.md)、[Agent 与工具评测](evaluation/agent-tool-evaluation.md)、[安全评测](evaluation/safety-evaluation.md)、[数据污染](evaluation/contamination.md)和[多模态评测](evaluation/multimodal-evaluation.md)。
- 新增可运行的最小实现：[张量原语](practice/tensor-primitives.md)、[Tokenizer](practice/tokenizers.md)、[训练目标](practice/training-objectives.md)、[从零实现 Transformer](practice/transformer-from-scratch.md)、[训练系统](practice/distributed-systems.md)、[推理引擎](practice/inference-engine.md)、[检索与智能体](practice/retrieval-agents.md)、[序列模型](practice/sequence-models.md)、[多模态](practice/multimodal.md)、[测试时计算](practice/test-time-compute.md)与[评测工具](practice/evaluation-tooling.md)。
- 更新索引层：[术语表](glossary.md)统一跨章节概念，[文献](references.md)整理原始论文与官方实现，[证据方法](guide/evidence.md)约束时效性结论、推断与未知的表达。

## 2026-07-24

- 建立从[基础](foundations/index.md)、[数据](data/index.md)、[模型结构](architecture/index.md)到[训练](training/index.md)、[系统](systems/index.md)、[推理](inference/index.md)、[应用](applications/index.md)与[评测](evaluation/index.md)的首版知识架构。
- 发布[语言建模](foundations/language-modeling.md)、[分词](foundations/tokenization.md)、[缩放规律](foundations/scaling.md)、[数据质量治理](data/quality-governance.md)、[Transformer](architecture/transformer.md)、[注意力与位置](architecture/attention-position.md)、[预训练](training/pretraining.md)、[后训练](training/post-training.md)、[分布式训练](systems/parallelism.md)、[在线服务](inference/serving.md)、[RAG](applications/rag.md)与[智能体](applications/agents.md)等核心章节。
- 加入[公式与 Markdown 规范](guide/method.md)，统一数学分隔符、代码块与链接格式，并固定 MathJax 渲染版本。
- 采用 [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) 的原生布局，支持浅色/深色主题、全文搜索与响应式阅读。
- 建立[术语表](glossary.md)、[文献索引](references.md)和[更新日志](changelog.md)，为后续专题扩展提供稳定入口。
