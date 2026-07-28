# 学习路线

LLM 横跨统计学习、模型结构、分布式系统和交互环境。先建立一条共同因果链，再沿问题深入；论文、框架和模型发布都放回这条链中理解。

## 共同主线

1. **表示与目标**：从[语言建模](../foundations/language-modeling.md)、[分词](../foundations/tokenization.md)和[概率、损失与梯度](../foundations/probability-objectives.md)理解 token 条件概率与训练目标。
2. **基本计算图**：推导 [Transformer](../architecture/transformer.md)、[Decoder Block](../architecture/decoder-block.md)、[位置编码](../architecture/position-encoding.md)和[注意力家族](../architecture/attention-variants.md)。
3. **数据到梯度**：连接[来源谱系](../data/sources-provenance.md)、[过滤去重](../data/filtering-dedup.md)、[混合课程](../data/mixtures-curricula.md)与[序列构造](../data/sequence-construction.md)。
4. **训练到行为**：依次阅读[预训练](../training/pretraining.md)、[SFT](../training/supervised-finetuning.md)和[蒸馏](../training/distillation.md)，再从[强化学习](../reinforcement-learning/index.md)理解反馈怎样改变策略分布。
5. **计算到系统**：用[性能成本模型](../systems/performance-model.md)连接[数值精度](../systems/precision-numerics.md)、[GPU 执行](../systems/gpu-execution.md)、[集合通信](../systems/collectives-sharding.md)与[模型并行](../systems/model-parallelism.md)。
6. **checkpoint 到服务**：从[解码](../inference/decoding.md)、[KV Cache](../inference/kv-cache.md)进入[运行时](../inference/runtime.md)、[调度与 Goodput](../inference/scheduling-goodput.md)、[量化](../inference/quantization.md)和 [P/D 分离](../inference/disaggregation.md)。
7. **外部知识与行动**：学习[索引与召回](../applications/retrieval-indexing.md)、[重排与上下文](../applications/reranking-context.md)、[证据约束生成](../applications/grounded-generation.md)、[工具调用](../applications/tool-use.md)和[智能体运行时](../applications/agent-runtime.md)。
8. **推理与反馈**：区分[推理时计算](../reasoning/test-time-compute.md)、[搜索与验证](../reasoning/search-verification.md)、[RLHF / RLAIF / RLVR](../reinforcement-learning/feedback-regimes.md)、[推理后训练](../training/reasoning-posttraining.md)与 [Agentic RL](../agentic-rl/index.md)。
9. **评测与边界**：最后用[评测协议](../evaluation/language-model-evaluation.md)、[统计推断](../evaluation/statistical-inference.md)、[校准](../evaluation/calibration-uncertainty.md)、[安全评测](../evaluation/safety-evaluation.md)验证整条链。

共同主线的最低实践是完成[手撕 Decoder-only Transformer](../practice/transformer-from-scratch.md)、[训练目标](../practice/training-objectives.md)与[评测工具](../practice/evaluation-tooling.md)中的核心不变量。

## 沿历史脉络建立直觉

共同主线告诉你今天的系统由哪些对象组成，[技术谱系](../landscape/index.md)解释这些对象为什么会变成今天的样子。第一次阅读可以穿插五次转折：

1. [计数、连续表示与可学习状态](../landscape/lineages/counts-to-learned-state.md)；
2. [固定向量、神经对齐与 self-attention](../landscape/lineages/transduction-to-attention.md)；
3. [预训练目标](../landscape/lineages/pretraining-objectives.md)、[规模规律与上下文适应](../landscape/lineages/scaling-and-context.md)；
4. [指令、偏好与在线学习](../landscape/lineages/training-alignment.md)、[推理与可验证搜索](../landscape/lineages/reasoning-verification.md)；
5. [分布式训练](../landscape/lineages/distributed-training-systems.md)与[推理运行时](../landscape/lineages/inference-serving.md)怎样把算法变成可运行系统。

每条谱系都连接关键工作深读和现代机制页。先看矛盾怎样移动，再阅读论文细节，会比孤立记忆模型名称更容易判断新工作究竟改变了什么。

若要练习“完整读一份现代技术报告”，可从 [GLM 演化](../landscape/glm-timeline.md)确定版本边界，再沿 [GLM-5 总深读](../landscape/works/glm-5.md)进入[架构](../landscape/works/glm-5-architecture.md)、[异步 RL 系统](../landscape/works/slime-async-agentic-rl.md)和[可执行环境](../landscape/works/glm-agentic-engineering.md)，最后用[引用图谱](../landscape/glm-5-reference-map.md)反查每个机制从哪里来、报告证据最远能支持什么。

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

### 强化学习与 LLM 后训练

先把“反馈是什么”“数据从哪里来”和“怎样更新策略”拆开，再沿六步深入：

1. 从 [MDP、POMDP 与回报](../reinforcement-learning/decision-processes.md)进入[价值与 Bellman](../reinforcement-learning/values-bellman.md)、[Monte Carlo 与 TD](../reinforcement-learning/prediction-control.md)；
2. 用 [Policy Gradient](../reinforcement-learning/policy-gradient.md)、[Actor–Critic](../reinforcement-learning/actor-critic.md)、[GAE](../reinforcement-learning/advantage-estimation-gae.md)、[TRPO](../reinforcement-learning/trust-region.md)和 [PPO](../reinforcement-learning/trust-region-ppo.md)建立策略优化主线；
3. 将语言模型重新写成策略，阅读[动作尺度](../reinforcement-learning/language-model-policy.md)、[KL 正则化控制](../reinforcement-learning/kl-regularized-control.md)、[反馈制度](../reinforcement-learning/feedback-regimes.md)与 [RLHF 数据闭环](../reinforcement-learning/rlhf-pipeline.md)；
4. 通过[推理 RL 配方地图](../reinforcement-learning/reasoning-rl-recipes.md)比较[无 critic baseline](../reinforcement-learning/critic-free-baselines.md)、[GRPO](../reinforcement-learning/grpo.md)、[DAPO](../landscape/works/dapo.md)与 [VAPO](../landscape/works/vapo.md)；
5. 再拆开[Ratio、Clipping 与 Gate](../reinforcement-learning/ratio-clipping-gating.md)、[训推分布与策略滞后](../reinforcement-learning/training-inference-discrepancy.md)和[在线 RL](../training/online-rl.md)；
6. 最后处理[语言模型信用分配](../reinforcement-learning/credit-assignment.md)、[异步 off-policy 校正](../reinforcement-learning/off-policy-correction.md)、[Agentic RL](../agentic-rl/index.md)和[实验诊断](../reinforcement-learning/evaluation-debugging.md)。

配套的[手撕强化学习](../practice/reinforcement-learning.md)固定经典 return、TD、trace、policy gradient 与 off-policy 语义；[手撕 LLM 策略优化](../practice/llm-policy-optimization.md)固定 packed GAE、PPO、GRPO、DAPO、VAPO、GSPO、SAPO 与 DIS。算法名称相同而 action mask、归一化分母、behavior policy 或终止规则不同，不能视为同一实验。

### 多模态

先把原始信号怎样进入模型讲清，再沿理解、生成和实时交互分流：

1. 从[信号、表示与 Token 化](../multimodal/foundations/signals-tokenization.md)进入[对齐、桥接与融合](../multimodal/foundations/alignment-fusion.md)，同时固定[空间、时间、位置与 Mask](../multimodal/foundations/position-time-masks.md)；
2. 视觉侧依次阅读[视觉表示、感知与 Grounding](../multimodal/vision/representation-grounding.md)、[视觉语言模型](../multimodal/vision-language.md)和[空间智能与三维表示](../multimodal/vision/spatial-3d.md)；
3. 生成侧从[图像生成总览](../multimodal/generative-modeling.md)进入[视觉 Tokenizer](../multimodal/image-generation/autoencoders-tokenizers.md)、[Diffusion 与 Score](../multimodal/image-generation/diffusion-score.md)、[DiT 与 Flow](../multimodal/image-generation/latent-dit-flow.md)；
4. 连续媒体分别阅读[音频表示与理解](../multimodal/audio/representations-understanding.md)、[音频生成与流式](../multimodal/audio/generation-streaming.md)、[视频理解与长程记忆](../multimodal/video/understanding-long-context.md)和[视频生成](../multimodal/video/generation.md)；
5. 最后用[统一理解与生成](../multimodal/unified-understanding-generation.md)与[Any-to-Any 系统](../multimodal/omni/any-to-any.md)连接多输入、多输出和实时状态。

实现与验证沿[多模态原语](../practice/multimodal.md)和[多模态评测](../evaluation/multimodal-evaluation.md)回查。

### 世界模型与具身智能

这条路线把媒体预测推进到动作条件与闭环决策：

1. 先用[世界模型总览](../world-models/index.md)区分 latent dynamics、JEPA、视频生成器和交互环境；
2. 在[潜在动力学、想象与规划](../world-models/dynamics-planning.md)中连接 RSSM、Dreamer、MuZero、MPC 与 CEM；
3. 在[表示预测与生成式世界](../world-models/predictive-generative-worlds.md)中比较 feature prediction、latent action 与可控视频；
4. 进入[具身智能总览](../embodied/index.md)，再读[状态、动作与策略](../embodied/state-action-policies.md)；
5. 沿[VLA、数据与跨本体学习](../embodied/vla-data-lineage.md)理解互联网知识、机器人轨迹和 embodiment gap；
6. 最后用[规划、闭环评测与安全](../embodied/planning-evaluation-safety.md)检查控制频率、恢复、运行时监督和物理边界。

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

## Reference {#reference}

- [Deep Learning](https://www.deeplearningbook.org/)
- [Dive into Deep Learning](https://d2l.ai/)
