# 并行训练

并行策略决定模型状态和计算沿哪些维度切分。实际系统通常组合多种并行，目标是在显存可容纳的前提下，让高带宽通信留在节点内、低频通信跨节点。

本页保留为选择总览。collective、DDP、ZeRO 与 FSDP 的数据布局见[集合通信与状态分片](collectives-sharding.md)；column/row TP、pipeline schedule、Ring Attention 与 expert parallel 的计算路径见[模型并行](model-parallelism.md)。

## Data Parallel

每个 rank 持有模型副本并处理不同数据，随后同步梯度。Distributed Data Parallel 通常 all-reduce 梯度；ZeRO/FSDP 进一步分片 optimizer state、梯度与参数。

[ZeRO](https://arxiv.org/abs/1910.02054) 将冗余状态分阶段消除。分片降低单卡显存，却增加 all-gather、reduce-scatter 和 checkpoint 复杂度。

## Tensor Parallel

张量并行把单层矩阵沿维度分片。列并行与行并行线性层通过 all-gather、all-reduce 或 reduce-scatter 连接。它能切分大层，但每层都可能通信，因此通常优先放在高速节点内互联。

[Megatron-LM](https://arxiv.org/abs/1909.08053) 给出 Transformer 张量并行的经典实现。现代实现还组合 sequence parallel、context parallel 与专家并行。

## Pipeline Parallel

流水线并行把层分为 stage，通过 microbatch 填充流水线。若 stage 数为 $p$，microbatch 数为 $m$，简单 GPipe 调度的空泡比例约为：

$$
\frac{p-1}{m+p-1}
$$

增加 $m$ 可减小空泡，但会改变激活存储与调度。[GPipe](https://arxiv.org/abs/1811.06965) 是代表性方案；1F1B 等调度进一步控制峰值激活。

## Context、Sequence 与 Expert Parallel

- **Sequence parallel** 分片部分逐 token 激活与 norm/dropout 计算。
- **Context parallel** 沿长序列分片 attention，需要交换 K/V 或中间统计。
- **Expert parallel** 把专家分布到不同 rank，token 通过 all-to-all 路由。

## 拓扑映射

并行维度不能只看乘积等于 GPU 数。应把通信最频繁、消息最小且时延敏感的维度映射到最快互联，并避免多个 collective 在同一链路同时拥塞。

## 选择顺序

1. 先用激活重计算和分片数据并行解决显存。
2. 单层仍放不下时加入 tensor parallel。
3. 层数或集群规模继续增长时加入 pipeline parallel。
4. 长上下文加入 context/sequence parallel。
5. MoE 依据专家大小与流量加入 expert parallel。

每次组合后都重新测量 step time、通信暴露、显存峰值、收敛一致性和故障恢复。
