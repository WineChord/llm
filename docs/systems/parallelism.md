# 并行训练

并行策略决定模型状态和计算沿哪些维度切分。实际系统通常组合多种并行，目标是在显存可容纳的前提下，让高带宽通信留在节点内、低频通信跨节点。

本页保留为选择总览。collective、DDP、ZeRO 与 FSDP 的数据布局见[集合通信与状态分片](collectives-sharding.md)；column/row TP、pipeline schedule、Ring Attention 与 expert parallel 的计算路径见[模型并行](model-parallelism.md)。

若先追踪历史问题怎样移动，[Megatron-LM 与 ZeRO 深读](../landscape/works/megatron-zero.md)把层内算子切分和训练状态分片放在同一张内存—通信账本中比较。

组合并行可以抽象为进程网格

$$
R=D\times T\times P\times C\times E,
$$

分别表示 data、tensor、pipeline、context 与 expert parallel degree。这个乘积只说明 rank 数量，不说明通信是否合理：每个维度还必须绑定 process group、物理链路和 tensor ownership。

## Data Parallel

每个 rank 持有模型副本并处理不同数据，随后同步梯度。Distributed Data Parallel 通常 all-reduce 梯度；ZeRO/FSDP 进一步分片 optimizer state、梯度与参数。

[ZeRO](https://arxiv.org/abs/1910.02054) 将冗余状态分阶段消除。分片降低单卡显存，却增加 all-gather、reduce-scatter 和 checkpoint 复杂度。

设每个 step 需要同步的 gradient 字节为 $S$，ring all-reduce 每个 rank 约传输

$$
2\frac{D-1}{D}S.
$$

这只是带宽项；小 bucket 还受 collective latency 支配。DDP 用 bucket overlap backward，而 ZeRO/FSDP 的参数 all-gather 又受下一层计算 deadline 约束，所以“通信总字节更少”不一定意味着关键路径更短。

## Tensor Parallel

张量并行把单层矩阵沿维度分片。列并行与行并行线性层通过 all-gather、all-reduce 或 reduce-scatter 连接。它能切分大层，但每层都可能通信，因此通常优先放在高速节点内互联。

[Megatron-LM](https://arxiv.org/abs/1909.08053) 给出 Transformer 张量并行的经典实现。现代实现还组合 sequence parallel、context parallel 与专家并行。

以 MLP 为例，第一层按输出列切分后，各 rank 可独立计算局部 activation；第二层按输入行切分，局部结果必须求和。合理的相邻切分让一次 block 只在必要位置 collective，而不是每个线性层都重建完整张量。TP 省下的是参数、激活和算子尺寸，付出的是层内高频通信，因此通常限制在 NVLink/NVSwitch 等低延迟域内。

## Pipeline Parallel

流水线并行把层分为 stage，通过 microbatch 填充流水线。若 stage 数为 $p$，microbatch 数为 $m$，简单 GPipe 调度的空泡比例约为：

$$
\frac{p-1}{m+p-1}
$$

增加 $m$ 可减小空泡，但会改变激活存储与调度。[GPipe](https://arxiv.org/abs/1811.06965) 是代表性方案；1F1B 等调度进一步控制峰值激活。

简单 bubble 公式假设 stage 等时、通信可忽略且没有重算。真实系统还会遇到不均匀层、embedding/输出头、MoE 负载和 pipeline 边界传输。切 stage 时应优化最大 stage time，而不是让每段层数相等；interleaving 可以减小气泡，却增加调度、通信和 activation 生命周期复杂度。

## 三种容易混淆的 token 维切分

- **Sequence parallel** 分片逐 token 激活，常与 TP 配合处理 norm、dropout 或残差，attention 的逻辑可见范围不因此改变。
- **Context parallel** 沿长上下文分片 attention，需要交换 K/V block 或在线 softmax 统计，必须保留全局因果边界。
- **Expert parallel** 不是序列切分；它按 router 结果把 token 发往专家，流量由数据动态决定。

三者都“移动 token”，collective 语义却分别接近 reduce-scatter/all-gather、环式或 all-to-all 通信。把它们合并成一个 sequence dimension，容易在 mask、顺序恢复和负载统计上出错。

## 拓扑映射

并行维度不能只看乘积等于 GPU 数。应把通信最频繁、消息最小且时延敏感的维度映射到最快互联，并避免多个 collective 在同一链路同时拥塞。

一个可解释的映射通常先问：

1. 哪些 process group 完全位于节点内；
2. 哪些 collective 每层发生，哪些每 step 发生；
3. context/expert 流量是否会与 TP 争用同一链路；
4. rank failure 会使哪些 shard 无法恢复；
5. checkpoint 能否在不同并行网格间重分片。

## 从单卡 reference 逐层扩展

1. 先用激活重计算和分片数据并行解决显存。
2. 单层仍放不下时加入 tensor parallel。
3. 层数或集群规模继续增长时加入 pipeline parallel。
4. 长上下文加入 context/sequence parallel。
5. MoE 依据专家大小与流量加入 expert parallel。

每次组合后都重新测量 step time、通信暴露、显存峰值、收敛一致性和故障恢复。

每引入一个维度，都用固定 batch 与单卡高精度 reference 比较 loss、gradient 和更新；否则多维并行同时打开后，很难定位是 shard layout、collective、mask 还是随机数漂移。模型并行的算子级路径见[模型并行](model-parallelism.md)，collective 与 ZeRO 状态见[集合通信与状态分片](collectives-sharding.md)，pipeline checkpoint 与恢复见 [Checkpoint、韧性与可观测性](checkpointing.md)。

## Reference {#reference}

- [ZeRO](https://arxiv.org/abs/1910.02054)
- [Megatron-LM](https://arxiv.org/abs/1909.08053)
- [GPipe](https://arxiv.org/abs/1811.06965)
- [Megatron-LM official implementation](https://github.com/NVIDIA/Megatron-LM)
- [PyTorch Fully Sharded Data Parallel documentation](https://pytorch.org/docs/stable/fsdp.html)
