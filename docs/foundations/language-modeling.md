# 语言建模

语言模型为序列 $x_{1:T}$ 建模。自回归分解把联合概率写成：

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t})
$$

训练通常最小化负对数似然：

$$
\mathcal{L}_{\text{NLL}}
=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})
$$

它等价于在每个位置对真实 token 做交叉熵分类。教师强制训练时，位置 $t$ 的条件来自真实前缀；自由生成时，条件包含模型自己之前采样的 token，两者存在分布差异。

若要理解这个目标怎样从计数模型走到神经网络，以及因果建模为何后来与掩码、span corruption 分化，可依次读[从计数到可学习状态](../landscape/lineages/counts-to-learned-state.md)和[预训练目标的分化](../landscape/lineages/pretraining-objectives.md)。

## 因果掩码

decoder-only Transformer 使用 causal mask，使位置 $t$ 不能读取未来位置。掩码约束信息流，不负责阻止模型记住训练数据，也不等于产品层面的安全边界。

## 从 logits 到文本

模型输出 logits $z$，温度为 $\tau$ 时：

$$
p_i=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}
$$

- $\tau<1$ 使分布更尖锐；$\tau>1$ 增加随机性。
- top-$k$ 只保留概率最大的 $k$ 个 token。
- top-$p$ 保留累计概率达到阈值的最小集合。
- greedy decoding 每步取最大概率，不保证全序列概率最大，也不保证事实正确。

## 常用度量

平均 token 损失为 $\bar{\mathcal{L}}$ 时，困惑度 $PPL=\exp(\bar{\mathcal{L}})$。它依赖 tokenizer 和数据分布，不能直接跨不同词表、预处理或测试集比较。

## 失效边界

- next-token prediction 优化的是数据分布上的预测，不是“说真话”目标。
- 长序列的平均损失可能掩盖少数关键位置的严重错误。
- 低损失不代表能够遵循新指令、调用工具或满足外部约束。
- 解码策略能改变多样性，但不能创造权重和上下文中不存在的可靠证据。

原始基础见 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)；缩放行为见[缩放与计算](scaling.md)。
