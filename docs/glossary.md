# 术语表

术语采用本知识库中的主要含义；同一缩写在不同论文或系统中可能有不同定义，引用时应以原文为准。

## A–F

**Abstention**

模型在证据或置信度不足时拒绝作答、请求澄清或转交人工，而不是强行给出结论。

**[Activation checkpointing](systems/memory-numerics-hardware.md)**

只保存部分前向激活，反向时重算其余激活，以额外计算换取显存。

**Action mask**

标记轨迹中哪些 token 或 span 由策略选择并进入 policy loss；observation、prompt 与 padding 通常不属于动作。

**Agent**

在目标、状态和约束下循环观察、决策并作用于环境的系统；模型只是其中的策略或推理组件之一。

**[Agentic RL](agentic-rl/index.md)**

让策略在多步环境中通过观察、行动、工具结果与奖励学习的方法族，不是单一算法。

**Alignment**

让模型行为更符合任务、偏好、原则或安全边界的一组训练与系统方法。

**All-reduce**

在多个 rank 上归约张量并把结果返回所有参与者，常用于同步数据并行梯度。

**[Arithmetic intensity](systems/performance-model.md)**

算术运算量与数据搬运字节数之比，用于判断算子更可能受计算还是带宽限制。

**[Autoregressive model](foundations/language-modeling.md)**

按先前 token 条件化，将序列联合概率分解为逐 token 条件概率的模型。

**Behavior policy**

实际产生 rollout 的策略；异步训练中它可能落后于正在更新的 learner policy。

**BF16**

具有 8-bit exponent 和 7-bit fraction 的 16-bit 浮点格式，动态范围接近 FP32。

**[BPE](foundations/tokenization.md)**

Byte Pair Encoding，迭代合并高频相邻符号的子词算法。

**Brier score**

概率预测与二值结果之间的均方误差；越低越好，同时受校准与区分能力影响。

**[Calibration](evaluation/calibration-uncertainty.md)**

预测置信度与长期经验正确率的一致程度；高准确率不自动意味着校准良好。

**Checkpoint**

用于保存或恢复模型参数、优化器、随机数与数据进度等状态的持久化快照。

**Cluster bootstrap**

以题目、用户或会话等独立簇为重采样单位估计不确定性，避免把相关样本误当独立样本。

**[Contamination](evaluation/contamination.md)**

评测样本或其近似形式进入训练、调参、提示设计或模型选择过程，导致结果高估。

**Context parallelism, CP**

沿序列维分片 attention 上下文，并交换 K/V block 或归一化统计的并行方式。

**Context window**

一次模型计算允许读取的 token 范围；声明长度不等于所有位置都能被可靠利用。

**[Continuous batching](inference/serving.md)**

在生成迭代边界动态加入新请求并移除已完成请求的服务调度方式。

**Coverage**

在评测中指系统选择回答的样本比例；常与剩余样本上的 risk 一起报告。

**Credit assignment**

把延迟出现的结果或奖励归因到先前状态与动作的问题。

**Data lineage**

记录数据来源、许可、处理、版本、混合与消费关系的可追溯链。

**[Data parallelism, DP](systems/model-parallelism.md)**

多个副本处理不同数据，并同步梯度或分片训练状态。

**Decode**

自回归推理中利用已有 KV Cache 逐步生成新 token 的阶段。

**Decontamination**

按精确、近似、语义或谱系关系识别并隔离可能污染评测的数据。

**[Disaggregated serving](inference/disaggregation.md)**

将 prefill 与 decode 放在不同 worker 或资源池，并在两者之间传输 KV Cache 的服务架构。

**[Distillation](training/distillation.md)**

让学生模型学习教师的 logits、分布、表征、轨迹或偏好，以转移能力或压缩模型。

**[DPO](training/offline-preference.md)**

Direct Preference Optimization，直接用 chosen/rejected 对与参考策略优化偏好的离线方法。

**ECE**

Expected Calibration Error，将预测按置信度分桶后汇总置信度与经验准确率之差；结果依赖分桶方案。

**Environment**

Agent 交互的外部状态与转移规则，可包含工具、模拟器、代码仓库、服务或现实系统。

**Estimand**

评测真正想估计的目标量，例如某输入分布上的平均成功率；指标只有在对应明确 estimand 时才可解释。

**Expert parallelism, EP**

把 MoE 专家分布到不同设备，并通过 all-to-all 路由 token 的并行方式。

**Flow matching**

通过回归概率路径上的向量场训练 Continuous Normalizing Flow 的生成建模方法。

**FP8**

8-bit 浮点格式族，常见 E4M3 与 E5M2 在精度和动态范围之间取不同权衡。

**FSDP**

Fully Sharded Data Parallel，按 rank 分片参数、梯度与 optimizer state 的数据并行路线。

## G–P

**[Goodput](inference/scheduling-goodput.md)**

满足延迟、正确性或其他 SLO 的有效吞吐，而不是所有完成工作的总吞吐。

**[Grounding](applications/grounded-generation.md)**

让输出中的主张能够对应到给定证据、工具结果或可验证环境状态。

**[GQA](architecture/attention-variants.md)**

Grouped-Query Attention，多组 query head 分别共享较少的 K/V head。

**GRPO**

Group Relative Policy Optimization，使用同一输入的成组样本和组内相对信号估计优势的策略优化方法。

**Gradient accumulation**

多次 microbatch 反向后再更新参数，用时间换取更大的有效 batch。

**Hallucination**

输出缺乏可靠依据、与事实或给定证据冲突的现象；边界必须按任务定义。

**Importance ratio**

新旧策略对同一动作概率的比值，常用于离策略校正或限制策略更新幅度。

**[In-context learning](foundations/in-context-learning.md)**

不更新权重，仅通过当前上下文中的说明、示例或中间结果改变模型行为。

**Instruction following**

在优先级、作用域、格式、条件和排除约束下完成任务的能力。

**ITL / TPOT**

Inter-token latency / time per output token，描述流式生成中相邻 token 的延迟。

**Judge / LLM-as-a-judge**

用模型对候选输出进行评分、比较或批注；必须检查位置、长度、风格与自偏好等偏差。

**[KV Cache](inference/kv-cache.md)**

自回归推理中缓存历史 token 的 attention key/value，以避免重复计算。

**Log-sum-exp**

稳定计算 $\log \sum_i e^{z_i}$ 的基本操作，通常先减去最大 logit。

**[LoRA](training/peft.md)**

Low-Rank Adaptation，用低秩增量更新冻结权重的参数高效微调方法。

**Loss scaling**

放大 loss 或 gradient 以减轻 FP16 下溢，再在参数更新前缩回。

**LSH**

Locality-Sensitive Hashing，让相似对象更可能落入同一桶，常用于近似去重和近邻检索。

**MFU**

Model FLOPs Utilization，用模型理论有效 FLOPs 与硬件峰值比较的利用率口径；计算定义必须明确。

**MinHash**

用集合的最小哈希签名近似 Jaccard 相似度，常与 LSH 配合做大规模近重复检测。

**MLA**

Multi-head Latent Attention，将 K/V 压缩为潜变量以降低缓存和带宽的一类注意力结构。

**MLLM**

Multimodal Large Language Model，能联合处理文本与一个或多个其他模态的大模型。

**Model lineage**

按架构、数据、训练、能力、发布对象与时间记录模型继承和分支关系的方法。

**[MoE](architecture/moe.md)**

Mixture of Experts，按输入动态选择部分专家计算的稀疏架构。

**MQA**

Multi-Query Attention，多个 query head 共享同一组 K/V。

**Offline preference learning**

从固定偏好数据学习策略，不在训练中持续向当前策略采样新轨迹的方法族。

**Online RL**

用当前或近期策略持续采样，再依据奖励或验证信号更新策略的训练方式。

**[PagedAttention](inference/runtime.md)**

以固定物理 block 管理非连续 KV Cache 的服务内存方法。

**Pass@k**

从同一任务独立采样 $k$ 个候选，至少一个通过验证的概率或其估计量；不能与单样本准确率混用。

**Pass$^k$**

同一任务的 $k$ 个候选全部通过的概率或估计量，用于衡量重复执行的一致可靠性。

**[PEFT](training/peft-compression.md)**

Parameter-Efficient Fine-Tuning，只训练少量新增或选定参数的适配方法族。

**Pipeline parallelism, PP**

把不同层放到不同 stage，并以 microbatch 流水执行。

**Policy lag**

生成 rollout 的 behavior policy 与 learner 当前 policy 之间的版本或分布差异。

**Prefill**

推理中并行处理输入 prompt、构建 KV Cache 并产生首个输出位置的阶段。

**[PPO](training/online-rl.md)**

Proximal Policy Optimization，使用裁剪 surrogate objective 限制更新幅度的策略梯度算法。

**Process reward model, PRM**

对中间步骤或状态提供过程评分的模型；分数是否代表最终可达性需要单独验证。

**Prompt injection**

不可信内容试图改变 Agent 的指令层级、工具调用或数据边界的攻击方式。

## Q–Z

**[Quantization](inference/quantization.md)**

用更低精度表示权重、激活或 KV，以降低内存、带宽或计算成本。

**[RAG](applications/rag.md)**

Retrieval-Augmented Generation，生成前检索外部证据并注入上下文。

**Reference policy**

偏好优化或在线 RL 中用于定义 KL 约束或概率比的基准策略，通常是训练开始时的冻结策略。

**Reward hacking**

策略利用奖励函数、环境或验证器漏洞取得高分，却没有完成真实目标的行为。

**Reward model**

把输入、输出或轨迹映射为偏好分数或奖励信号的模型。

**Risk–coverage curve**

随拒答阈值变化，联合展示已回答样本比例与其错误风险的曲线。

**RLAIF**

Reinforcement Learning from AI Feedback，主要使用 AI 评审信号构造偏好或奖励的路线。

**RLHF**

Reinforcement Learning from Human Feedback，使用人类反馈训练奖励或策略的流程统称。

**RLOO**

REINFORCE Leave-One-Out，用同一输入的其他样本奖励构造 baseline 的策略梯度估计方法。

**[RMSNorm](architecture/decoder-block.md)**

按均方根缩放表示、不减去均值的归一化方法。

**[RoPE](architecture/position-encoding.md)**

Rotary Position Embedding，用位置相关旋转把相对位置信息注入 Q/K。

**Rollout**

策略在任务或环境中从初始状态到终止或截断产生的一次交互序列。

**Semantic uncertainty**

先按语义等价关系聚合多个生成，再衡量不同意义之间的概率分散程度。

**Sequence parallelism, SP**

沿 token 维分片部分逐 token activation 与算子的并行方式，常与 tensor parallel 组合。

**SFT**

Supervised Fine-Tuning，在指令—答案或任务示范上做监督微调。

**SLO**

Service Level Objective，对延迟、可用性、正确性或成本等服务指标设定的目标边界。

**[Speculative decoding](inference/speculative-decoding.md)**

由较快模型提出候选 token，再由目标模型并行验证的解码加速方法。

**SSM**

State Space Model，以状态递推表示序列历史的一类模型。

**Tensor parallelism, TP**

把单层张量运算沿 hidden、head 或其他维度分布到多个设备。

**[Test-time compute](reasoning/test-time-compute.md)**

在权重固定后，通过更长生成、并行采样、搜索、验证或工具交互增加单题计算。

**Token share**

某数据源在实际训练 token 流中的占比；它不等于原始文档数或存储字节占比。

**Tokenizer**

文本与 token ID 序列之间的编码系统，包括 normalizer、pre-tokenizer、词表和特殊 token。

**Tool schema**

工具名称、参数类型、返回结构、错误与副作用的机器可读契约。

**Trajectory**

一个 episode 中观察、动作、工具结果、奖励与终止状态组成的序列。

**TTFT**

Time to first token，从请求到首个输出 token 的时间。

**Unauthorized side effect**

系统在没有相应权限或意图确认时产生的外部状态变化，即使最终答案看似正确也属于失败。

**V-trace**

用截断 importance weight 修正 behavior policy 与 target policy 差异的离策略回报估计方法。

**Verifier**

判断候选答案、步骤或轨迹是否满足可检查条件的程序或模型。

**VLM**

Vision-Language Model，联合处理视觉与语言输入或输出的模型。

**World model**

预测环境状态、观测或结果的模型，可用于规划、仿真、表示学习或生成。

**YaRN**

通过频率分段缩放与 attention temperature 调整扩展 RoPE 上下文的方法。

**ZeRO**

Zero Redundancy Optimizer，在数据并行 rank 间分片 optimizer state、gradient 和 parameter 的方法族。
