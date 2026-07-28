# 技术谱系

一个领域真正变得可理解，不是在所有论文旁边标上年份，而是看见问题怎样移动：固定窗口解决了可估计性，却截断历史；循环状态延长记忆，却把一切压进有限向量；attention 打开全局寻址，又把代价转成平方关系和 KV 状态。后来的工作不是凭空出现，它们总在接住前一代留下的某个矛盾。

这里把知识库中的稳定机制与具体工作连接起来。谱系页讲“为什么下一步会发生”，工作深读讲“一项工作到底改变了什么”，模型结构、训练、系统和评测页则保留完整定义与实现边界。

## 不是一条单线年表

技术路线会并行、分叉，也会在多年后重新汇合。RNN 与连续词表示早于 2003 年前馈神经语言模型并行发展；BERT、GPT 与 T5 是不同信息流选择，不是谁简单取代谁；线性 attention 与 SSM 从不同方向走向有限状态。阅读时应允许多个问题同时存在，而不是把论文排成版本号。

### 从计数到寻址

[从计数到可学习状态](lineages/counts-to-learned-state.md)沿着 n-gram、连续表示、RNN 与 LSTM 观察“怎样共享统计、怎样保存历史”。[从固定向量到内容寻址](lineages/transduction-to-attention.md)接着解释 seq2seq 的 context bottleneck 如何推动可微对齐，attention 又怎样从 encoder–decoder 接口进入每一层。

这一转折的三个近距离切面是 [LSTM](works/lstm.md)、[Seq2Seq 与神经对齐](works/seq2seq-and-neural-alignment.md)和 [Attention Is All You Need](works/attention-is-all-you-need.md)。它们分别改变梯度路径、条件生成接口和序列内部的信息路径。

### 从预训练目标到基础模型

[预训练目标与信息流](lineages/pretraining-objectives.md)不把 GPT、BERT、T5 排成继任关系，而是比较 causal、masked 与 span-corruption 三种可见性和输出接口。[从规模规律到上下文内适应](lineages/scaling-and-context.md)把经验幂律、compute-optimal 训练与 in-context learning 放回各自的证据层。

对应工作深读：

- [GPT-1](works/generative-pretraining-gpt.md)：causal pretraining 与任务迁移怎样共享一个接口；
- [BERT](works/bert.md)：双向表示为何需要 corruption，以及 15% / 80-10-10 的真实语义；
- [T5](works/t5.md)：span corruption 与 text-to-text 如何共同形成配方；
- [Scaling Laws 与 Chinchilla](works/scaling-laws-chinchilla.md)：固定算力下的拟合、资源分配和外推边界。

[从可下载权重到可研究系统](lineages/open-model-ecosystem.md)则补上另一条历史：公开 API、权重、代码、数据、中间 checkpoint 与许可证逐层扩大了研究者真正能够检验的对象。

### GLM：从统一目标到 Agentic Engineering

[GLM 演化](glm-timeline.md)从 2021 年的 General Language Model 出发，经过 GLM-130B、ChatGLM、GLM-4 的 All Tools 与 GLM-4.5 的 Agentic MoE，走到 GLM-5 / 5.2。它不是简单的模型尺寸年表，而是四条逐渐汇合的支线：双向与自回归的统一预训练、开放双语模型、工具与推理后训练、面向长程 Agent 的训练—推理系统。

- [GLM-5 总深读](works/glm-5.md)逐项覆盖 40 页报告的 13 幅图、13 张表、5 个编号公式、4 段 listing 与附录，并保留正文和配置之间的冲突；
- [GLM-5 架构](works/glm-5-architecture.md)连接 MoE、MLA-256、Muon Split、Shared MTP 与 DSA；
- [IndexCache 与 IndexShare](works/indexcache.md)解释为什么稀疏索引可以跨层共享，以及它和 KV / prefix cache 的边界；
- [slime 与异步 Agentic RL](works/slime-async-agentic-rl.md)拆开 TITO、policy lag、直接重要性比、版本过滤和 KV-locality routing；
- [GLM Agentic Engineering](works/glm-agentic-engineering.md)从 issue–PR 环境、terminal/search 数据走到上下文管理、slide reward 与运行式评测；
- [GLM-5 引用图谱](glm-5-reference-map.md)按报告实际使用的 63 项引用追踪方法来源、系统实现、数据与 benchmark。

### 当一条模型家族汇合多条技术路线

[Kimi 技术谱系](kimi-timeline.md)没有把同名模型排成发布列表，而是沿长推理、稀疏宽度、线性状态、深度寻址、原生视觉和 agent system 六条支线解释 K1.5、Kimi-VL、K2、Kimi Linear、K2.5、Attention Residuals 与 K3 怎样前后衔接。页面同时区分 paper、weights、code、API、license 与 release date，避免把“公开报告”“开放权重”和“完整系统可复现”写成同一件事。

K3 是观察这种汇流的一个切面：

- [Kimi Linear 与 FlashKDA](works/kimi-linear-flashkda.md)沿 fast weights、DeltaNet、KDA、chunkwise kernel 与 KCP 追踪有限状态如何落到 GPU；
- [Kimi k1.5](works/kimi-k1-5.md)、[Kimi K2](works/kimi-k2.md)与[Kimi K2.5](works/kimi-k2-5.md)分别展开长程 RL、1T MoE / MuonClip，以及原生多模态与 Agent Swarm 的三次转折；
- [Attention Residuals](works/attention-residuals.md)从 PreNorm dilution 推到 Full/Block AttnRes、online softmax 与 pipeline cache；
- [Stable LatentMoE 与 Quantile Balancing](works/latentmoe-quantile-balancing.md)区分原始 LatentMoE 与 K3 的数值、激活和路由增量；
- [K3 工作深读](works/kimi-k3.md)把模型结构、预训练、长上下文、post-training、系统和评测还原成相互制约的闭环；
- [MoonEP](works/moonep.md)解释动态冗余 expert 如何在不改变路由语义时固定 rank 负载；
- [K3 引用图谱](kimi-k3-reference-map.md)逐项解释技术报告 150 项文献在论证链中的角色，并区分直接来源、技术前身、并行工作、benchmark 与比较基线；
- [Kimi-VL 深读](../multimodal/kimi-vl.md)解释原生分辨率视觉塔、稀疏语言主干与长视觉上下文；[Kimi 多模态分支](../multimodal/kimi.md)再梳理它与 K2.5、MoonViT-V2、Kimi-Audio 的关系，避免把家族经验误写成同一 checkpoint 的能力。

### 百万 token 怎样变成端到端能力 {#deepseek-v4-system}

[DeepSeek 演化](deepseek-timeline.md)从 DeepSeekMoE、MLA、无辅助损失路由和 Multi-Token Prediction 走到 V3、R1、V3.2 与 V4。V4 的关键变化并不是把 context length 单独调大，而是让表示压缩、残差拓扑、优化器、kernel、缓存、后训练和故障恢复共同适配长轨迹：

- [DeepSeek-V4 总深读](works/deepseek-v4.md)把 15 幅图、14 张正式表、29 个编号公式、Algorithm 1 与附录 A–B 还原为一条可检查的因果链；
- [CSA 与 HCA](works/deepseek-compressed-attention.md)解释时间压缩、Lightning Indexer、shared-KV MQA、inverse RoPE 与局部窗口如何共同保持因果性；
- [mHC](works/manifold-hyper-connections.md)从 Hyper-Connections 走到 Birkhoff polytope 与 Sinkhorn projection，区分局部矩阵约束和完整网络稳定性；
- [On-Policy Distillation](works/on-policy-distillation.md)分开轨迹分布、KL 方向、词表估计器和多教师调度；
- [TileLang、MegaMoE 与 DSec](works/tilelang-mega-moe.md)沿 wave pipeline、batch invariance、异构 cache、token WAL 和 sandbox 恢复连接训练与部署；
- [V4 引用图谱](deepseek-v4-reference-map.md)逐项标注 103 项正文引用在家族前身、方法来源、系统实现和 benchmark 中承担的角色。

### 当稠密 attention 不再是唯一答案

[容量与激活计算怎样分开](lineages/conditional-compute.md)追踪 sparse gating 如何把参数容量与每 token 计算部分解耦，又把瓶颈转移到负载、overflow 与 all-to-all。[从显式寻址到有限状态](lineages/linear-time-sequence-models.md)区分 kernelized attention、fast weights、S4 与 selective SSM，解释它们在哪些代数结构上汇合、又在哪里保持不同。

两篇工作深读把公式落到执行语义：

- [Sparse MoE](works/sparse-moe.md)从 top-$k$ 路由约定走到 capacity、dispatch 和 expert parallel；
- [S4 到 Mamba](works/s4-mamba.md)从 recurrence–convolution duality 走到 input-dependent selective scan。

### 从续写到偏好、搜索与验证

[从续写到偏好与在线学习](lineages/training-alignment.md)沿监督来源、数据分布和 optimizer 三条坐标连接 instruction tuning、reward model、PPO、DPO、RLOO、GRPO 与 RLVR。[从外显推理到可验证搜索](lineages/reasoning-verification.md)连接 CoT、自一致采样、verifier、过程监督、搜索和 search-to-training；[推理策略优化](lineages/reasoning-policy-optimization.md)则沿 baseline、ratio、归约与系统约束追踪 PPO、GRPO、DAPO、VAPO 直到异步 Agentic RL。

关键工作不按算法热度排列，而按它改变的闭环环节阅读：

- [InstructGPT](works/instructgpt.md)连接 demonstration、pairwise reward 与 online PPO；
- [DPO](works/dpo.md)从 KL-regularized policy 推出离线 pair objective，却不提供在线探索；
- [DeepSeek-R1](works/deepseek-r1.md)把规则奖励 RL、cold start、rejection sampling、二次对齐与蒸馏放进多阶段流程。
- [DAPO](works/dapo.md)把 Clip-Higher、mixed-group sampling、global token reduction 与 overlong handling 组织成开放 RLVR 配方；
- [VAPO](works/vapo.md)重新设计长推理中的 critic、GAE 与稀疏正样本利用；
- [SAO 与 CompactionRL](works/sao-compactionrl.md)把长程 Agentic RL 的 group barrier 与 context exhaustion 拆成时间轴、空间轴，再连接 token-level correction、critic 与跨段信用。

### 从单卡计算到分布式状态

[分布式训练系统](lineages/distributed-training-systems.md)从参数复制、collective、tensor/pipeline parallel 讲到 ZeRO/FSDP、IO-aware kernel 与 durable checkpoint。重点不是并行缩写，而是一次创新减少了哪份状态或哪段等待，又把通信和恢复压力移到哪里。

[推理运行时与服务](lineages/inference-serving.md)从静态 batch、Orca iteration scheduling 走到 PagedAttention、prefix sharing、推测解码、chunked prefill 和 Prefill–Decode 分离。模型生成仍然逐 token，系统进步来自更准确地管理动态状态和不同 SLO。

对应深读：

- [Megatron-LM 与 ZeRO](works/megatron-zero.md)：层内切分与状态分片为何解决不同瓶颈；
- [FlashAttention](works/flashattention.md)：不引入 attention approximation，怎样通过 online softmax 计算同一 dense softmax attention 并改写 HBM traffic；
- [vLLM 与 PagedAttention](works/vllm-pagedattention.md)：KV Cache 如何从连续 tensor 变成带所有权的分页状态。

### 当模型开始看见、检索与行动

[从“看懂”到“生成”](lineages/multimodal-generation.md)把视觉语言对齐与生成建模视作两股汇流，而不把“支持图片”当成统一技术标签：

- [CLIP](works/clip.md)让自然语言成为开放视觉接口；
- [Flamingo、BLIP-2 与 LLaVA](works/visual-language-bridges.md)展示冻结模型之间三种不同的桥；
- [从 DDPM 到 DiT 与 Flow](works/diffusion-dit-flow.md)拆开表示、backbone 与概率路径。

[从参数记忆到可行动系统](lineages/retrieval-agents.md)则沿外部记忆、工具调用和环境轨迹扩展模型边界：

- [RAG](works/rag.md)把检索文档写成生成中的隐变量；
- [ReAct 与 Toolformer](works/react-toolformer.md)分别从推理轨迹和训练数据回答“何时调用工具”。

### 评测对象为何越来越大

[从困惑度到运行中的评测](lineages/evaluation.md)解释测量对象怎样从 token likelihood 扩展到多任务协议、生成式裁判、人类偏好和 Agent 终态。[HELM 与 Chatbot Arena](works/helm-arena.md)展示两次关键转折：先把 scenario、adaptation、metric 和输出记录组成透明协议，再让开放式交互进入动态成对比较。

评测没有位于历史的末尾。每当模型新增上下文、检索、工具或搜索预算，协议都必须记录新的自由度，否则系统改动会伪装成 checkpoint 能力。

## 工作深读怎样使用

一篇工作页不会逐节复述论文。它固定五个问题：

1. 前一阶段留下了什么具体矛盾；
2. 这项工作改变的是数据、目标、计算图、optimizer、状态还是测量协议；
3. 关键公式怎样映射到最小可运行代码；
4. 原论文实验最窄能支持什么结论；
5. 后续工作为何仍有出现的必要。

每段 reference 都保持短小并带断言；它用于冻结机制语义，不冒充论文的完整训练系统。更完整的张量、训练与系统实现仍在[手撕实现](../practice/index.md)中维护。

## 一项“发布”常有多个日期

论文首次公开、会议版本、权重、代码、API、产品与许可证变更可能发生在不同日期。模型家族记录至少区分：

```text
paper / revision
checkpoint / weights
training and inference code
dataset and recipe
API / product
license
```

同名产品也可能更换 checkpoint；一篇论文则可能在权重发布后继续修订。训练规模的可比口径见[训练 token](training-tokens.md)，怎样拆分一条复杂家族可参考[DeepSeek 演化案例](deepseek-timeline.md)。

## 三种阅读路径

- **沿问题走**：从本页选择一条谱系，先理解矛盾怎样移动，再进入工作深读。
- **沿机制走**：从[知识架构](../guide/architecture.md)进入 canonical 机制页，再回到谱系确认它在历史上的位置。
- **沿实现走**：先运行[最小实现](../practice/minimal-implementations.md)，再阅读对应工作为什么需要这些 mask、状态、归一化或调度语义。

技术史的价值不是替今天的方案排祖先，而是让设计选择重新获得上下文：什么问题已经解决，什么只是换了成本位置，什么至今仍没有可靠答案。
