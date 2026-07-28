# 术语表

术语采用本知识库中的主要含义；同一缩写在不同论文或系统中可能有不同定义，引用时应以原文为准。

## A–F

**Abstention**

模型在证据或置信度不足时拒绝作答、请求澄清或转交人工，而不是强行给出结论。

**[Activation checkpointing](systems/memory-numerics-hardware.md)**

只保存部分前向激活，反向时重算其余激活，以额外计算换取显存。

**Action mask**

标记轨迹中哪些 token 或 span 由策略选择并进入 policy loss；observation、prompt 与 padding 通常不属于动作。

**[Action chunk](embodied/state-action-policies.md)**

策略一次预测的连续动作序列；chunk horizon、实际执行步数、控制频率与模型前向频率必须分开。

**[Advantage](reinforcement-learning/actor-critic.md)**

$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$，表示动作相对该状态下策略平均水平好多少；它不等于原始 reward。

**[Attention Residuals, AttnRes](landscape/works/kimi-k3.md#attention-residuals)**

把固定 residual accumulation 改成对 embedding、历史 layer 或 block representation 的 depth-wise softmax 选择。

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

**[Any-to-Any](multimodal/omni/any-to-any.md)**

接受多种模态输入并产生多种模态输出的系统目标；它不保证 encoder、主干、decoder、训练或 runtime 全部共享。

**[Behavior policy](reinforcement-learning/training-inference-discrepancy.md)**

实际产生 rollout 的分布；它由 checkpoint、推理引擎和 sampling processor 共同定义，未必等于训练侧 old policy。

**[Batch invariance](landscape/works/tilelang-mega-moe.md#batch-invariant-attention)**

同一 token 在 batch 顺序、切分或并行布局改变时仍得到逐位一致结果的性质；它比固定配置下重复运行的 determinism 更强。

**[CISPO](reinforcement-learning/ratio-clipping-gating.md#cispo)**

Clipped IS-weight Policy Optimization，把裁剪后的 importance ratio 作为 detached policy-gradient 系数；越界后权重饱和而非梯度归零。

**[DAPO](landscape/works/dapo.md)**

Decoupled Clip and Dynamic sAmpling Policy Optimization，组合 Clip-Higher、动态采样、global token loss 与 overlong handling 的 RLVR 配方。

**[DIS](reinforcement-learning/ratio-clipping-gating.md#dis)**

Direct Double-Sided Importance Sampling，SAO 中以 current/rollout direct ratio 做双侧接受门的分布校正组件。

**[DSec](landscape/works/tilelang-mega-moe.md#dsec)**

面向长程 Agent rollout 的可恢复 sandbox：把环境状态、overlay 文件系统和可重放 I/O 一起纳入 checkpoint，而不只保存模型 token。

**[DSA](architecture/attention-variants.md#glm-dsa)**

DeepSeek Sparse Attention，用轻量 indexer 为每个 query 从历史位置选择 top-$k$ 候选，再执行高维核心注意力；索引成本、稀疏 attention 成本和近似误差需要分开。

**[DSpark](landscape/works/dspark.md)**

以深并行骨干、轻量顺序头、校准的接受置信度和硬件感知前缀调度组成的无损推测解码
框架；不是新的基础模型，也不是 DeepSeek Sparse Attention。

**[Bellman equation](reinforcement-learning/values-bellman.md)**

把当前价值写成即时 reward 与下一状态价值期望的递推等式，是动态规划、TD 与 critic 的共同接口。

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

**[CSA](landscape/works/deepseek-compressed-attention.md#token-compressor)**

Compressed Sparse Attention，先把历史 token 压成重叠 block representation，再由轻量 indexer 为每个 query 选择少量压缩项。

**[Contextual bandit](reinforcement-learning/decision-processes.md)**

每轮观察 context、选择动作并立即得到 reward，但动作不影响下一轮状态的决策模型；许多单轮 response-level 后训练可作此近似。

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

**[Diffusion model](multimodal/image-generation/diffusion-score.md)**

定义数据到噪声的前向扰动，并学习反向去噪、score 或等价参数化以生成样本的模型族；prediction type、schedule 与 sampler 必须配套。

**[DiT](multimodal/image-generation/latent-dit-flow.md)**

Diffusion Transformer，以 Transformer 作为去噪或向量场 backbone；它描述网络结构，不等同于某一种扩散时间参数化或 sampler。

**Discount factor**

$\gamma\in[0,1]$，控制未来 reward 在 return 中的相对权重；它也可表达任务时域，而不只是数值稳定技巧。

**[DPO](training/offline-preference.md)**

Direct Preference Optimization，直接用 chosen/rejected 对与参考策略优化偏好的离线方法。

**ECE**

Expected Calibration Error，将预测按置信度分桶后汇总置信度与经验准确率之差；结果依赖分桶方案。

**Environment**

Agent 交互的外部状态与转移规则，可包含工具、模拟器、代码仓库、服务或现实系统。

**[Entropy regularization](reinforcement-learning/exploration-entropy.md)**

在策略目标中奖励分布熵，鼓励保留随机性；它能改变探索强度，但不保证产生语义上不同或有效的行为。

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

**[GAE](reinforcement-learning/advantage-estimation-gae.md)**

Generalized Advantage Estimation，用 $\lambda$ 加权多步 TD residual，在 critic 偏差与采样方差之间折中。

**[Goodput](inference/scheduling-goodput.md)**

满足延迟、正确性或其他 SLO 的有效吞吐，而不是所有完成工作的总吞吐。

**[Grounding](applications/grounded-generation.md)**

让输出中的主张能够对应到给定证据、工具结果或可验证环境状态。

**[GQA](architecture/attention-variants.md)**

Grouped-Query Attention，多组 query head 分别共享较少的 K/V head。

**[GRPO](reinforcement-learning/grpo.md)**

Group Relative Policy Optimization，使用同一输入的成组样本和组内相对信号估计优势的策略优化方法。

**[GSPO](reinforcement-learning/ratio-clipping-gating.md#gspo)**

Group Sequence Policy Optimization，使用长度归一化的 sequence ratio，让同一 response 共用 clipping 决策。

**Gradient accumulation**

多次 microbatch 反向后再更新参数，用时间换取更大的有效 batch。

**Hallucination**

输出缺乏可靠依据、与事实或给定证据冲突的现象；边界必须按任务定义。

**[HCA](landscape/works/deepseek-compressed-attention.md)**

Heavily Compressed Attention，以更高时间压缩率保存全局历史寻址，并与短滑窗局部路径互补。

**[Importance ratio](reinforcement-learning/training-inference-discrepancy.md)**

两个分布对同一动作概率的比值；必须说明分子、分母、token/sequence 粒度与 sampling processor。

**[JEPA](world-models/predictive-generative-worlds.md)**

Joint-Embedding Predictive Architecture，在表示空间预测目标区域或未来状态，而不是要求逐像素重建；feature distance 是否适合决策仍需下游验证。

**[KDA](landscape/works/kimi-k3.md#kda-recurrence)**

Kimi Delta Attention，先以逐 key-channel retention 衰减 fast-weight state，再按 delta rule 擦写当前 key 关联。

**[KCP](landscape/works/kimi-k3.md#flashkda-kcp)**

KDA Context Parallelism，把每段序列表示为可结合的仿射 state transition，并用跨设备 prefix scan 恢复各段初态。

**[IcePop](reinforcement-learning/training-inference-discrepancy.md#icepop)**

对训练引擎与 rollout 引擎的 importance ratio 先做区间内校正、再拒绝双侧尾部的 mismatch 处理方法。

**[In-context learning](foundations/in-context-learning.md)**

不更新权重，仅通过当前上下文中的说明、示例或中间结果改变模型行为。

**[IndexShare / IndexCache](landscape/works/indexcache.md#indexshare)**

让少数层运行稀疏注意力 indexer，后续相邻层复用 top-$k$ 位置索引，以减少沿网络深度重复选择候选的计算；它缓存的不是 KV 向量。

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

**[Markov property](reinforcement-learning/decision-processes.md)**

给定当前状态后，未来转移不再依赖更早历史。语言 Agent 的 context 通常只是对隐藏环境状态的部分观察。

**[MegaMoE](landscape/works/tilelang-mega-moe.md#wave-pipeline)**

把 expert activation 切成 wave，让 pull、两段 GEMM 与回传在细粒度流水中重叠的 MoE 执行路径。

**[mHC](landscape/works/manifold-hyper-connections.md#birkhoff-polytope)**

Manifold-Constrained Hyper-Connections，用多路 residual stream 扩展层间表示，并把动态混合矩阵约束在 Birkhoff polytope 上。

**[LatentMoE](landscape/works/kimi-k3.md#latent-path)**

shared path 保持完整模型宽度，而 routed experts 在较窄 latent space 中计算的 MoE 结构。

**LSH**

Locality-Sensitive Hashing，让相似对象更可能落入同一桶，常用于近似去重和近邻检索。

**MFU**

Model FLOPs Utilization，用模型理论有效 FLOPs 与硬件峰值比较的利用率口径；计算定义必须明确。

**MinHash**

用集合的最小哈希签名近似 Jaccard 相似度，常与 LSH 配合做大规模近重复检测。

**[MLA](architecture/attention-variants.md)**

Multi-head Latent Attention，将 K/V 压缩为潜变量以降低缓存和带宽的一类注意力结构。

**[Muon Split](landscape/works/glm-5-architecture.md#muon-split)**

在 Q/K/V up-projection 上先按 attention head 切分更新矩阵，再分别执行 Muon 正交化；它改变的是正交化的矩阵边界，不只是实现分块。

**[MOPD](landscape/works/kimi-k3.md#mopd)**

Multi-Teacher On-Policy Distillation，让 student 从自身策略采样，再由与 domain/effort 对应的 teacher 提供逐 token signal。

**[MoonEP](systems/moe-systems.md)**

以动态 redundant expert placement、固定 dispatch buffer 与通信—计算重叠追求精确 rank 负载的 expert-parallel 系统。

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

**Occupancy measure**

策略在时间上访问状态或状态—动作对的折扣频率分布；策略更新会改变该分布，因此在线 RL 的训练数据并非固定。

**Online RL**

用当前或近期策略持续采样，再依据奖励或验证信号更新策略的训练方式。

**[On-Policy Distillation, OPD](landscape/works/on-policy-distillation.md#reverse-kl)**

让学生从自身策略访问前缀，再由教师在这些状态上提供分布监督；轨迹是否 on-policy、KL 方向和词表估计器是彼此独立的设计轴。

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

**[Policy lag](reinforcement-learning/training-inference-discrepancy.md)**

生成 rollout 的 behavior policy 与 learner 当前 policy 之间的版本或分布差异。

**Prefill**

推理中并行处理输入 prompt、构建 KV Cache 并产生首个输出位置的阶段。

**[PPO](reinforcement-learning/trust-region-ppo.md)**

Proximal Policy Optimization，使用裁剪 surrogate objective 限制更新幅度的策略梯度算法。

**[Quantile Balancing, QB](landscape/works/kimi-k3.md#quantile-balancing)**

把 balanced expert assignment 的对偶坐标更新写成 margin quantile，以只参与 top-$k$ 选择的 bias 调节下一步负载。

**Process reward model, PRM**

对中间步骤或状态提供过程评分的模型；分数是否代表最终可达性需要单独验证。

**Prompt injection**

不可信内容试图改变 Agent 的指令层级、工具调用或数据边界的攻击方式。

## Q–Z

**[Residual Vector Quantization, RVQ](multimodal/audio-language-models.md#residual-vector-quantization)**

用多个码本依次量化上一层残差，并把各层码向量相加重建表示；码本顺序不自动保证语义—声学分层。

**[Tubelet](multimodal/video/understanding-long-context.md)**

把连续帧中的局部时空立方体映射为一个视频 token；时间跨度、空间 patch 与原始帧率共同决定信息和 token 数。

**[Vision-Language-Action model, VLA](embodied/vla-data-lineage.md)**

把视觉、语言和机器人状态映射为动作的模型族；VLM planner、VLA policy、运行时监督和低层控制器是不同责任层。

**[World model](world-models/index.md)**

表示环境状态并预测其在动作条件下怎样转移、从而支持规划或策略学习的模型；无动作的视频生成器或未来帧预测器不自动满足这一含义。

**[R3](reinforcement-learning/training-inference-discrepancy.md#r3)**

Rollout Routing Replay，为 MoE 强化学习记录并重放 rollout 时的专家路由，使训练侧 log-prob 在相同 route 条件下重算。

**[Quantization](inference/quantization.md)**

用更低精度表示权重、激活或 KV，以降低内存、带宽或计算成本。

**[RAG](applications/rag.md)**

Retrieval-Augmented Generation，生成前检索外部证据并注入上下文。

**[Reference policy](reinforcement-learning/training-inference-discrepancy.md)**

偏好优化或在线 RL 中用于定义 KL 约束或概率比的基准策略，通常是训练开始时的冻结策略。

**[Return](reinforcement-learning/decision-processes.md)**

从某一时刻开始的折扣累计 reward，常写为 $G_t=\sum_{k\ge0}\gamma^kR_{t+k+1}$。

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

**[RLVR](reinforcement-learning/rlvr.md)**

Reinforcement Learning with Verifiable Rewards，主要依赖程序、形式规则或可查询环境结果提供可重复 reward；它描述反馈接口，不指定优化器。

**[RLOO](reinforcement-learning/critic-free-baselines.md#rloo)**

REINFORCE Leave-One-Out，用同一输入的其他样本奖励构造 baseline 的策略梯度估计方法。

**[RMSNorm](architecture/decoder-block.md)**

按均方根缩放表示、不减去均值的归一化方法。

**[RoPE](architecture/position-encoding.md)**

Rotary Position Embedding，用位置相关旋转把相对位置信息注入 Q/K。

**Rollout**

策略在任务或环境中从初始状态到终止或截断产生的一次交互序列。

**[SAPO](reinforcement-learning/ratio-clipping-gating.md#sapo)**

Soft Adaptive Policy Optimization，以 sigmoid surrogate 让远离 old policy 的 token 梯度平滑衰减。

**Semantic uncertainty**

先按语义等价关系聚合多个生成，再衡量不同意义之间的概率分散程度。

**Sequence parallelism, SP**

沿 token 维分片部分逐 token activation 与算子的并行方式，常与 tensor parallel 组合。

**SFT**

Supervised Fine-Tuning，在指令—答案或任务示范上做监督微调。

**[SMDP](reinforcement-learning/models-planning-hierarchy.md)**

Semi-Markov Decision Process，动作可持续不等时长并在结束后转移，适合描述 option、tool call 与长程子任务。

**SLO**

Service Level Objective，对延迟、可用性、正确性或成本等服务指标设定的目标边界。

**[Speculative decoding](inference/speculative-decoding.md)**

由较快模型提出候选 token，再由目标模型并行验证的解码加速方法。

**[Shared MTP](landscape/works/glm-5-architecture.md#shared-mtp)**

用同一组 Multi-Token Prediction 参数递归预测多个未来位置，使额外训练目标和 speculative draft path 共享权重。

**[SiTU-GLU](landscape/works/kimi-k3.md#situ-glu)**

用 scaled tanh 平滑限制 gate 的线性因子与 up branch，在原点附近保持 SwiGLU 的一阶行为并给出有限输出上界。

**SSM**

State Space Model，以状态递推表示序列历史的一类模型。

**Tensor parallelism, TP**

把单层张量运算沿 hidden、head 或其他维度分布到多个设备。

**[TD error](reinforcement-learning/prediction-control.md)**

$\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)$，用一步 bootstrap 目标衡量当前价值预测误差。

**[TIS](reinforcement-learning/training-inference-discrepancy.md#tis)**

Truncated Importance Sampling，对同一 checkpoint 的训练分布与 rollout 分布之比做上截断，以有限方差校正 engine mismatch。

**[TITO](landscape/works/slime-async-agentic-rl.md#tito)**

Token-in-Token-out，让 learner 直接消费 rollout engine 实际产生的 token IDs、log-probability 与边界元数据，避免文本 round-trip 的二次分词错位。

**[Test-time compute](reasoning/test-time-compute.md)**

在权重固定后，通过更长生成、并行采样、搜索、验证或工具交互增加单题计算。

**[TileLang](landscape/works/tilelang-mega-moe.md#host-codegen)**

以 tile-level 程序表达高性能算子的 DSL 与编译栈；性能不仅取决于设备 kernel，也取决于 host code generation、静态分析与浮点语义。

**Token share**

某数据源在实际训练 token 流中的占比；它不等于原始文档数或存储字节占比。

**[Trust region](reinforcement-learning/trust-region.md)**

限制新旧策略分布变化的更新思想；PPO clipping 是可计算的局部 surrogate，并不等价于严格满足 KL 约束。

**[TRPO](reinforcement-learning/trust-region.md#trpo)**

Trust Region Policy Optimization，用 Fisher 近似、共轭梯度与 line search 近似求解带平均 KL 约束的策略更新。

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

**Verifier**

判断候选答案、步骤或轨迹是否满足可检查条件的程序或模型。

**VLM**

Vision-Language Model，联合处理视觉与语言输入或输出的模型。

**[V-trace](reinforcement-learning/off-policy-correction.md)**

使用截断 importance weight 构造多步 value target 的 off-policy 校正方法；截断降低方差，也引入偏差。

**[VAPO](landscape/works/vapo.md)**

Value-model-based PPO 长推理配方，组合 value warmup、decoupled/adaptive GAE、Clip-Higher、token loss、正样本 NLL 与 group sampling。

**XTML**

eXtensible Token Markup Language，用 special token 统一表示 role、thinking、response、typed tool call 与 option 生命周期的 chat template；完整结构见 [Kimi K3 附录](landscape/works/kimi-k3.md#appendices)。

**[Value function](reinforcement-learning/values-bellman.md)**

$V^\pi(s)$ 或 $Q^\pi(s,a)$，表示在给定策略下从状态或状态—动作对出发的期望 return。

**World model**

预测环境状态、观测或结果的模型，可用于规划、仿真、表示学习或生成。

**YaRN**

通过频率分段缩放与 attention temperature 调整扩展 RoPE 上下文的方法。

**ZeRO**

Zero Redundancy Optimizer，在数据并行 rank 间分片 optimizer state、gradient 和 parameter 的方法族。
