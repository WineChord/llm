# 从固定向量到内容寻址

神经机器翻译把语言模型从“预测同一序列的下一个词”推进为“在输入条件下生成另一条序列”。真正改变后续架构的矛盾，是输入究竟应被压进一个固定状态，还是保留为可按内容读取的记忆。

## Seq2Seq 建立条件生成接口

[Sutskever、Vinyals 与 Le 2014](https://proceedings.neurips.cc/paper_files/paper/2014/hash/5a18e133cbf9f257297f410bb7eca942-Abstract.html)用 LSTM encoder 读取源序列，以最终状态初始化 LSTM decoder：

$$
c=h_T^{\mathrm{enc}},\qquad
p(y\mid x)=\prod_{t=1}^{T_y}p(y_t\mid y_{<t},c).
$$

它把变长输入和输出纳入统一的端到端最大似然训练。论文中的源序列反转缩短了部分依赖路径，是当时有效的优化技巧，却没有从结构上消除固定向量瓶颈：所有源端细节仍必须进入同一个 $c$。

[Seq2Seq 与神经对齐深读](../works/seq2seq-and-neural-alignment.md)并排还原两项时间重叠的工作，避免把 Bahdanau attention 写成事后才出现的单线回应。

固定向量的困难不只是“维度太小”。从源位置 $j$ 到目标位置 $t$，信息必须先穿过 encoder 的剩余时间步，再从单个 $c$ 穿过 decoder 的前 $t$ 步。即使 state 维度很大，优化路径仍随位置增长。反转源句能缩短一部分常见对齐的路径，却会拉长另一部分；它改善的是训练几何，不是记忆接口。

## 可微对齐把一个状态变成一张表

[Bahdanau、Cho 与 Bengio](https://arxiv.org/abs/1409.0473)明确把 fixed-length vector 视为瓶颈。该工作 2014 年 9 月首次提交，后发表于 ICLR 2015；它与 2014 年的 Seq2Seq 工作时间重叠，不宜写成在后者正式发表后才出现的单线回应。

encoder 保留每个位置的状态 $h_j$，decoder 在第 $t$ 步用当前需求计算软对齐：

$$
e_{tj}=v^\top\tanh(W_ss_{t-1}+W_hh_j),
$$

$$
\alpha_{tj}
=\frac{\exp(e_{tj})}{\sum_k\exp(e_{tk})},
\qquad
c_t=\sum_j\alpha_{tj}h_j.
$$

这里发生的是表示接口的改变：历史不再只能压缩成一个向量，而成为由 query 按内容读取的外部状态表。对齐是训练目标内部可微的隐变量，不需要逐词对齐标注。不过 encoder 和 decoder 仍然 recurrent，训练与生成依旧沿时间步串行。

Bahdanau 模型还使用双向 encoder annotation，使 $h_j$ 同时包含源位置左右文；decoder query 则来自目标端已生成历史。因此 cross-attention 的两端并不对称：

- memory 在生成前一次性编码；
- query 随每个目标 token 更新；
- 每一步都重新归一化整个源序列上的 $\alpha_{tj}$；
- 生成仍要等待上一 decoder state。

这个 read interface 后来迁移到图文桥接、检索增强和 encoder–decoder 预训练。真正被继承的是“保留一组可寻址状态”，不是某个具体 additive score。

## Transformer 把寻址移进每一层

[Attention Is All You Need](../works/attention-is-all-you-need.md)把 attention 从 encoder–decoder 之间的接口推广为 encoder self-attention、decoder masked self-attention 与 cross-attention：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}
\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

$Q$ 表示当前读取需求，$K$ 表示地址，$V$ 表示被读取的内容。self-attention 让一层中的所有 token 同时构造 query、key 和 value，任意两位置间的网络路径从随序列增长缩短为常数层级；训练不再需要等待前一 recurrent state。

这种变化不是“计算免费”。长度为 $T$、hidden size 为 $d$ 时，标准 self-attention 的 score 与主要注意力计算通常为 $O(T^2)$ 或 $O(T^2d)$；RNN 每层常写作 $O(Td^2)$，但包含 $O(T)$ 个顺序步骤。实际快慢还取决于 $T$、$d$、kernel、内存带宽和硬件并行度。

路径长度也必须分清网络层与执行时间。self-attention 让两个位置在一层内直接通信，最大网络路径近似为 $O(1)$；自回归输出仍有 $T_y$ 个串行决策。Transformer 消除的是训练时的 recurrent dependency，不是生成时的因果顺序。

## 一次接口迁移留下了什么

Transformer 继承了 additive attention 的核心思想：保留多个状态并做内容寻址。它进一步分离地址与内容、并行多个 head，并要求显式注入顺序。由此产生的新问题包括：

- 没有 recurrence 后，顺序必须由位置表示和 mask 定义；
- 全局 score 矩阵带来平方级中间状态与 IO；
- 自回归生成仍按 token 串行，并需要维护 KV Cache；
- attention 权重是计算系数，不自动等于可解释的因果对齐。

还出现了一条看似回头、实则重新组合的支线：状态空间和线性 attention 恢复固定大小 state 以降低长序列成本，再用选择性更新、卷积/scan duality 或周期性 full attention 缓解有限状态瓶颈。它们继续回答 2014 年留下的问题，而不是简单回到旧 RNN，见[从显式寻址到有限状态](linear-time-sequence-models.md)。

cross-attention 随后迁移到图文桥接、条件生成与检索增强；self-attention 则成为通用 token mixer。机制细节见 [Transformer](../../architecture/transformer.md)、[注意力与位置](../../architecture/attention-position.md)和[位置编码](../../architecture/position-encoding.md)，训练/生成分界见 [KV Cache](../../inference/kv-cache.md)，长序列的成本与有效性见[长上下文](../../architecture/long-context.md)。

## Reference {#reference}

- [Sequence to Sequence Learning with Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2014/hash/5a18e133cbf9f257297f410bb7eca942-Abstract.html)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
