# 覆盖地图

覆盖不是页面数量，而是一个问题能否从定义走到验证。下表连接 canonical mechanism、最小实现以及系统或评测边界；模型案例和最新论文不替代这条链。

## 覆盖维度

| 维度 | 需要回答的问题 |
| --- | --- |
| 定义 | 对象、变量、输入输出和适用范围是什么 |
| 推导 | 目标、复杂度和关键公式怎样得到 |
| 实现 | 最短可审计代码怎样固定语义 |
| 系统 | FLOPs、bytes、通信、状态和延迟怎样变化 |
| 评测 | 数据、指标、统计与预算怎样固定 |
| 失效 | 哪些退化输入、分布偏移或攻击会破坏结论 |
| 证据 | 结论来自理论、作者实验、开放实现还是独立复现 |
| 脉络 | 它承接什么限制，又把瓶颈移到哪里 |

机制页不必平均展开八项，但不能只列名词与论文。

技术发展的因果关系由[技术谱系](../landscape/index.md)维护；关键工作页再把原论文、最小实现与今天的机制连接起来。谱系不是第八种孤立内容，而是帮助读者判断每一项机制为何存在。

## 主干矩阵

| 主题 | 原理与机制 | 最小实现 | 系统与评测 |
| --- | --- | --- | --- |
| 表示与目标 | [语言建模](../foundations/language-modeling.md)、[分词](../foundations/tokenization.md)、[概率与损失](../foundations/probability-objectives.md)、[上下文学习](../foundations/in-context-learning.md) | [Tokenizer](../practice/tokenizers.md)、[张量原语](../practice/tensor-primitives.md) | [缩放](../foundations/scaling.md)、[语言模型评测](../evaluation/language-model-evaluation.md) |
| 数据 | [来源谱系](../data/sources-provenance.md)、[过滤去重](../data/filtering-dedup.md)、[混合课程](../data/mixtures-curricula.md)、[合成数据](../data/synthetic-data.md) | [序列构造](../data/sequence-construction.md) | [质量治理](../data/quality-governance.md)、[记忆与隐私](../data/memorization-privacy.md)、[评测污染](../evaluation/contamination.md) |
| Transformer | [Decoder Block](../architecture/decoder-block.md)、[注意力](../architecture/attention-variants.md)、[位置](../architecture/position-encoding.md) | [完整 Transformer](../practice/transformer-from-scratch.md) | [Attention Kernel](../systems/attention-kernels.md)、[KV Cache](../inference/kv-cache.md) |
| 稀疏与递推 | [MoE](../architecture/moe.md)、[状态空间与线性注意力](../architecture/state-space-linear-attention.md)、[记忆架构](../architecture/memory-architectures.md) | [递推与记忆](../practice/sequence-models.md)、[MoE dispatch](../practice/distributed-systems.md) | [MoE 系统](../systems/moe-systems.md)、[长上下文](../architecture/long-context.md) |
| 多模态共同基础 | [信号与 Token](../multimodal/foundations/signals-tokenization.md)、[对齐与融合](../multimodal/foundations/alignment-fusion.md)、[位置与 Mask](../multimodal/foundations/position-time-masks.md) | [多模态原语](../practice/multimodal.md) | [数据、训练与系统](../multimodal/foundations/data-training-systems.md)、[多模态评测](../evaluation/multimodal-evaluation.md) |
| 视觉理解与空间 | [视觉表示与 Grounding](../multimodal/vision/representation-grounding.md)、[视觉语言](../multimodal/vision-language.md)、[空间与三维](../multimodal/vision/spatial-3d.md) | [多模态原语](../practice/multimodal.md) | [文档与 GUI](../multimodal/document-gui-grounding.md)、[多模态评测](../evaluation/multimodal-evaluation.md) |
| 媒体生成与交互 | [图像生成](../multimodal/generative-modeling.md)、[音频生成](../multimodal/audio/generation-streaming.md)、[视频生成](../multimodal/video/generation.md)、[Any-to-Any](../multimodal/omni/any-to-any.md) | [多模态原语](../practice/multimodal.md) | [控制、编辑与评测](../multimodal/image-generation/control-editing-evaluation.md)、[连续媒体协议](../multimodal/audio-video.md) |
| 世界模型 | [世界模型总览](../world-models/index.md)、[潜在动力学与规划](../world-models/dynamics-planning.md)、[预测与生成式世界](../world-models/predictive-generative-worlds.md) | [模型、规划与层级决策](../reinforcement-learning/models-planning-hierarchy.md) | [闭环规划与不确定性](../embodied/planning-evaluation-safety.md) |
| 具身智能 | [具身总览](../embodied/index.md)、[状态、动作与策略](../embodied/state-action-policies.md)、[VLA 与跨本体](../embodied/vla-data-lineage.md) | [模仿与 Offline RL](../reinforcement-learning/offline-imitation.md) | [规划、闭环评测与安全](../embodied/planning-evaluation-safety.md) |
| 训练与对齐 | [预训练](../training/pretraining.md)、[SFT](../training/supervised-finetuning.md)、[蒸馏](../training/distillation.md)、[PEFT](../training/peft.md) | [训练目标](../practice/training-objectives.md) | [规模实验](../training/scaling-experiment-design.md)、[优化稳定性](../training/optimization.md)、[训练系统](../systems/index.md) |
| 策略优化主线 | [序贯决策](../reinforcement-learning/decision-processes.md)、[价值与 Bellman](../reinforcement-learning/values-bellman.md)、[MC 与 TD](../reinforcement-learning/prediction-control.md)、[Policy Gradient](../reinforcement-learning/policy-gradient.md)、[Actor–Critic](../reinforcement-learning/actor-critic.md)、[GAE](../reinforcement-learning/advantage-estimation-gae.md)、[TRPO](../reinforcement-learning/trust-region.md)、[PPO](../reinforcement-learning/trust-region-ppo.md) | [强化学习](../practice/reinforcement-learning.md) | [函数逼近](../reinforcement-learning/function-approximation.md)、[探索](../reinforcement-learning/exploration-entropy.md)、[实验诊断](../reinforcement-learning/evaluation-debugging.md) |
| LLM 强化学习 | [配方地图](../reinforcement-learning/reasoning-rl-recipes.md)、[GRPO](../reinforcement-learning/grpo.md)、[Ratio 与 Gate](../reinforcement-learning/ratio-clipping-gating.md)、[训推分布](../reinforcement-learning/training-inference-discrepancy.md)、[在线 RL](../training/online-rl.md) | [LLM 策略优化](../practice/llm-policy-optimization.md) | [DAPO](../landscape/works/dapo.md)、[VAPO](../landscape/works/vapo.md)、[信用分配](../reinforcement-learning/credit-assignment.md)、[RLVR](../reinforcement-learning/rlvr.md) |
| 训练系统 | [性能模型](../systems/performance-model.md)、[数值精度](../systems/precision-numerics.md)、[GPU 执行](../systems/gpu-execution.md) | [分布式与容错](../practice/distributed-systems.md) | [状态分片](../systems/collectives-sharding.md)、[模型并行](../systems/model-parallelism.md)、[系统韧性](../systems/resilience-observability.md) |
| 推理时计算 | [预算与搜索](../reasoning/test-time-compute.md)、[搜索验证](../reasoning/search-verification.md) | [推理时计算](../practice/test-time-compute.md) | [推理后训练](../training/reasoning-posttraining.md)、[Agentic RL](../agentic-rl/index.md) |
| 推理服务 | [解码](../inference/decoding.md)、[运行时](../inference/runtime.md)、[调度](../inference/scheduling-goodput.md) | [推理引擎](../practice/inference-engine.md) | [缓存复用](../inference/cache-reuse.md)、[量化](../inference/quantization.md)、[推测解码](../inference/speculative-decoding.md)、[推理可靠性](../inference/benchmarking-reliability.md) |
| 检索与生成 | [索引召回](../applications/retrieval-indexing.md)、[重排上下文](../applications/reranking-context.md)、[RAG](../applications/rag.md) | [检索与智能体](../practice/retrieval-agents.md) | [证据约束生成](../applications/grounded-generation.md)、[事实性](../evaluation/hallucination.md) |
| 工具与智能体 | [工具调用](../applications/tool-use.md)、[记忆规划](../applications/memory-planning.md)、[运行时](../applications/agent-runtime.md) | [检索与智能体](../practice/retrieval-agents.md) | [智能体安全](../applications/agent-security.md)、[Agent 评测](../evaluation/agent-tool-evaluation.md) |
| Agentic RL | [从经典 RL 到语言 Agent](../agentic-rl/rl-foundations.md)、[算法决策](../agentic-rl/math-algorithms.md)、[数据环境](../agentic-rl/data-environments.md) | [LLM 策略优化](../practice/llm-policy-optimization.md)、[推理时计算](../practice/test-time-compute.md) | [轨迹契约](../agentic-rl/trajectory-contract.md)、[训练系统](../agentic-rl/training-systems.md)、[长时任务](../agentic-rl/long-horizon.md)、[SAO 与 CompactionRL](../landscape/works/sao-compactionrl.md) |
| DeepSeek 家族 | [家族总览](../landscape/families/deepseek.md)、[演化时间线](../landscape/deepseek-timeline.md)、[V4 深读](../landscape/works/deepseek-v4.md) | [CSA / HCA](../landscape/works/deepseek-compressed-attention.md)、[mHC](../landscape/works/manifold-hyper-connections.md)、[全词表 OPD](../landscape/works/on-policy-distillation.md) 的页内 reference | [多模态分支](../multimodal/deepseek.md)、[TileLang、MegaMoE 与 DSec](../landscape/works/tilelang-mega-moe.md)、[103 项引用图谱](../landscape/deepseek-v4-reference-map.md)与 canonical 机制页 |
| Kimi 家族 | [家族总览](../landscape/families/kimi.md)、[技术谱系](../landscape/kimi-timeline.md)、[K3 深读](../landscape/works/kimi-k3.md) | [KDA](../landscape/works/kimi-k3.md#kda-recurrence)、[AttnRes](../landscape/works/attention-residuals.md#attnres-online-merge)、[SiTU-GLU](../landscape/works/kimi-k3.md#situ-glu)、[QB](../landscape/works/kimi-k3.md#quantile-balancing)、[MOPD](../landscape/works/kimi-k3.md#mopd)、[KCP](../landscape/works/kimi-linear-flashkda.md#kcp-affine-scan) 与 [hybrid cache](../landscape/works/kimi-k3.md#hybrid-prefix-cache) | [多模态分支](../multimodal/kimi.md)、[150 项引用图谱](../landscape/kimi-k3-reference-map.md)、canonical 机制页与评测证据边界 |
| GLM 家族 | [家族总览](../landscape/families/glm.md)、[演化时间线](../landscape/glm-timeline.md)、[GLM-5 深读](../landscape/works/glm-5.md) | [MLA / DSA / MTP](../landscape/works/glm-5-architecture.md)、[IndexShare](../landscape/works/indexcache.md)、[TITO / direct IS](../landscape/works/slime-async-agentic-rl.md) | [多模态分支](../multimodal/glm.md)、[Agentic Engineering](../landscape/works/glm-agentic-engineering.md)、[63 项引用图谱](../landscape/glm-5-reference-map.md)与 canonical 机制页 |
| 评测与可靠性 | [评测协议](../evaluation/language-model-evaluation.md)、[指标](../evaluation/metrics.md)、[校准](../evaluation/calibration-uncertainty.md) | [评测工具](../practice/evaluation-tooling.md) | [Judge](../evaluation/generative-judges.md)、[安全](../evaluation/safety-evaluation.md)、[生产可靠性](../evaluation/production-reliability.md) |

这张表用于发现断链。例如量化只有论文摘要，却没有运行时 dtype、实际 kernel、质量 slice 和 reference 对照，就尚未形成完整专题。

## 稳定性层级

### 稳定基础

定义、公式或系统约束已经跨多项工作复用，例如交叉熵、causal attention、AdamW、collective 语义和 KV 容量。正文直接讲机制，但仍标明假设。

### 条件化方法

有原始论文、官方实现或多个公开系统采用，收益依赖数据、shape、硬件、预算或训练配方。正文说明适用条件，不使用无条件“更优”。

### 研究前沿

近期论文、单一团队报告或快速演进的软件需要记录：

- 首次公开与最近核验日期；
- 论文、代码、权重和复现是否可用；
- 已验证规模、任务与硬件；
- 尚未证明的外推；
- 与稳定 baseline 的差异。

模型发布数字与机制结论分开；前者进入[技术谱系](../landscape/index.md)，后者只有在抽象后才进入机制页。

## 版本与证据

持续演进的对象需要版本化：

- 论文绑定具体版本、实验表与假设；
- 软件绑定 release、commit 或带日期的官方文档；
- 模型绑定明确 checkpoint 与 chat template；
- benchmark 绑定数据 revision、harness commit 与污染时间窗；
- 协议绑定规范版本；
- 硬件结论绑定 device、dtype、shape 和软件栈。

某个链接仍可访问，不代表其中结论仍是当前状态。核验方法见[证据与研究方法](evidence.md)。

## 阅读检查

读者可以用同一组问题审查任何页面：

1. 公式里的量是否有定义；
2. 理论复杂度是否与真实执行分开；
3. 最大能力是否与有效能力分开；
4. 作者报告是否与独立证据分开；
5. 最小实现是否包含 mask、退化和断言；
6. 指标是否给出分母、预算和缺失值；
7. 前沿结论是否有日期与外推边界；
8. 链接是否指向概念的 canonical page。

页面边界见[知识架构](architecture.md)，实验验收见[实验方法](../practice/index.md)，来源判断见[证据与研究方法](evidence.md)。

## Reference {#reference}

- [ACM Artifact Review and Badging](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
- [NIST AI Risk Management Framework 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10)
