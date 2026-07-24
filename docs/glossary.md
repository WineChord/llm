# 术语表

## A–F

**Activation checkpointing**

只保存部分前向激活，反向时重算其余激活，以计算换显存。

**Alignment**

让模型行为更符合任务、偏好、原则或安全边界的一组训练与系统方法；不等于单一算法。

**All-reduce**

在多个 rank 上归约数据并把结果发回所有参与者，常用于同步数据并行梯度。

**Arithmetic intensity**

FLOPs 与数据搬运字节数之比，用于判断算子更可能受计算还是带宽限制。

**Autoregressive model**

按先前 token 条件化，逐 token 分解联合概率的模型。

**BF16**

具有 8-bit exponent 和 7-bit fraction 的 16-bit 浮点格式，动态范围接近 FP32。

**BPE**

Byte Pair Encoding，迭代合并高频相邻符号的子词算法。

**Checkpoint**

用于保存或恢复模型及训练状态的持久化快照。

**Context window**

一次模型计算可读取的 token 范围；声明长度不等于所有位置都能可靠利用。

**Continuous batching**

在生成迭代边界动态加入新请求和移除已完成请求的服务调度。

**Data parallelism, DP**

多个副本处理不同数据，并同步梯度或分片状态。

**Decode**

自回归推理中逐步生成新 token 的阶段。

**DPO**

Direct Preference Optimization，直接用 chosen/rejected 对与参考策略优化偏好的方法。

**Expert parallelism, EP**

把 MoE 专家分布到不同设备，并路由 token 的并行方式。

**FSDP**

Fully Sharded Data Parallel，按 rank 分片参数、梯度与 optimizer state 的数据并行路线。

## G–P

**Goodput**

满足延迟或其他 SLO 的有效吞吐，而非所有完成工作的总吞吐。

**GQA**

Grouped-Query Attention，多组 query head 分别共享较少 K/V head。

**Gradient accumulation**

多次 microbatch 反向后再更新参数，用时间换更大的有效 batch。

**Hallucination**

输出缺乏可靠依据、与事实或给定证据冲突的现象；需要按任务定义。

**In-context learning**

不更新权重，仅通过当前上下文中的说明或示例改变行为。

**ITL / TPOT**

Inter-token latency / time per output token，描述流式生成相邻 token 的延迟。

**KV Cache**

自回归推理中缓存历史 token 的 attention key/value，避免重复计算。

**LoRA**

Low-Rank Adaptation，用低秩增量更新冻结权重的参数高效微调方法。

**Loss scaling**

放大 loss/gradient 以减轻 FP16 下溢，再在更新前缩回。

**MFU**

Model FLOPs Utilization，用模型理论有效 FLOPs 与硬件峰值比较的利用率口径；计算定义必须明确。

**MLLM**

Multimodal Large Language Model，能处理文本以外一个或多个模态的大模型。

**MoE**

Mixture of Experts，按输入动态选择部分专家计算的稀疏架构。

**MQA**

Multi-Query Attention，多个 query head 共享同一组 K/V。

**PagedAttention**

以固定物理 block 管理非连续 KV Cache 的服务内存方法。

**Pipeline parallelism, PP**

把不同层放到不同 stage，并以 microbatch 流水执行。

**Prefill**

推理中并行处理输入 prompt、构建 KV Cache 并产生首 token 的阶段。

**PPO**

Proximal Policy Optimization，带裁剪目标的策略梯度算法，常用于 RLHF。

## Q–Z

**Quantization**

用更低精度表示权重、激活或 KV，以降低内存、带宽或计算成本。

**RAG**

Retrieval-Augmented Generation，生成前检索外部证据并注入上下文。

**RLAIF**

Reinforcement Learning from AI Feedback，主要使用 AI 评审信号的偏好学习路线。

**RLHF**

Reinforcement Learning from Human Feedback，使用人类偏好训练奖励或策略的流程统称。

**RoPE**

Rotary Position Embedding，用位置相关旋转把相对位置信息注入 Q/K。

**SFT**

Supervised Fine-Tuning，在指令—答案或任务示范上做监督微调。

**Speculative decoding**

由较快模型提出候选 token，再由目标模型并行验证的解码加速。

**SSM**

State Space Model，以状态递推表示序列历史的一类模型。

**Tensor parallelism, TP**

把单层张量运算沿 hidden、head 或其他维度分布到多个设备。

**Tokenizer**

文本与 token ID 序列之间的编码系统，包括 normalizer、pre-tokenizer、词表和特殊 token。

**TTFT**

Time to first token，从请求到首个输出 token 的时间。

**VLM**

Vision-Language Model，联合处理视觉与语言的模型。

**ZeRO**

Zero Redundancy Optimizer，通过分片训练状态降低数据并行冗余。
