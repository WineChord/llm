# 稀疏与替代架构

Dense Transformer 同时沿两个方向增长：更多层和通道提高参数容量，更长序列提高注意力计算与缓存成本。MoE、状态空间模型、线性注意力、记忆和混合架构分别尝试解耦其中一部分成本。

本页保留为稳定比较入口：

- [Mixture of Experts](moe.md)：让每个 token 只激活部分参数；
- [状态空间与线性注意力](state-space-linear-attention.md)：用有限状态递推降低长序列成本；
- [记忆架构](memory-architectures.md)：在窗口、段、外部存储与可塑参数之间保存历史；
- [长上下文](long-context.md)：位置扩展、稀疏可见性、分布式 attention 与有效长度；
- [推理时计算](../reasoning/test-time-compute.md)：保持模型结构不变，在回答阶段分配更多采样与搜索。

## 先区分优化对象

| 路线 | 主要解耦对象 | 不会自动解决 |
| --- | --- | --- |
| 稀疏 MoE | 总参数容量与每 token 激活计算 | 序列长度、通信、KV Cache |
| MQA/GQA/MLA | KV Cache 与 query head 数 | attention 的全局二次 score |
| Window/sparse attention | 可见边数量与序列长度 | 窗口外精确内容寻址 |
| SSM/线性注意力 | 历史长度与递推状态大小 | 有限状态的信息容量 |
| 外部检索 | 参数记忆与可更新知识 | 检索错误、额外延迟、证据利用 |
| 测试时搜索 | 单次前向能力与回答计算预算 | 基础模型、verifier 与样本相关性 |

因此“更高效”必须附带明确分母：训练 FLOPs、decode FLOPs、峰值显存、KV bytes、通信量、端到端吞吐、延迟或质量约束。

## 混合架构

不同 token mixer 可以按层或分支组合。常见分工是：

- local/full attention 保留精确内容寻址；
- SSM 或线性注意力压缩大部分历史；
- MoE 增加通道容量；
- 外部检索提供可更新、可引用的非参数信息。

混合比例不是越复杂越好。实现需要回答：

1. 哪些层拥有精确 KV；
2. 哪些层只有固定大小 recurrent state；
3. 状态能否 chunkwise 训练并逐 token 推理；
4. 不同 mixer 的 norm、残差和位置接口是否一致；
5. runtime 是否真正支持相应 kernel 与 cache；
6. 质量收益来自架构还是参数量、数据和训练预算差异。

## 比较协议

公平比较至少固定：

- tokenizer、训练数据与 token 数；
- 总参数、激活参数和每 token FLOPs；
- hidden size、状态大小、层数与优化器；
- 训练硬件、kernel、batch 和 sequence length；
- prefill、decode、短上下文与长上下文四类速度；
- perplexity、关联回忆、复制、长程检索与真实任务；
- checkpoint、量化、并行和服务栈成熟度。

只比较渐近复杂度或单一 benchmark 容易把架构收益、工程成熟度和规模差异混在一起。

## 证据边界

稳定正文优先描述可复现的计算图、复杂度和已公开实现。新模型在自有配方上的结果适合作为案例，不应直接推出“已取代 attention”或“能无限记忆”。涉及前沿方案时，应同时写出公开 checkpoint、kernel、独立复现、测试规模和未验证外推。

系统层的 MoE 通信见[MoE 系统](../systems/moe-systems.md)，序列模型的最小递推与等价性实验见[序列模型手撕实现](../practice/sequence-models.md)。
