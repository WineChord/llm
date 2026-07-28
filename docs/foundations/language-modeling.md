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

## 概率单位先由 tokenizer 决定

$x_t$ 不是天然的“一个词”，而是 tokenizer 产生的离散 ID。换一套词表，同一文本会得到不同的序列长度和条件事件，因此 token-level loss 也换了测量单位。字符丰富但被切得更碎的文本，可能仅因分词方式承受更多预测项；这正是不同 tokenizer 的 perplexity 不能直接横比的根本原因。

序列还必须明确起止与拼接语义。若文档 $a$ 和 $b$ 被直接拼成 `a || b`，模型会学习跨文档条件 $p(b_1\mid a)$；若中间有 EOS 或 attention boundary，条件分布又不同。数据打包不是存储细节，而是在定义哪些历史允许成为预测证据。分词与边界见[分词与表示](tokenization.md)，批内拼接与 loss mask 见[序列构造与打包](../data/sequence-construction.md)。

## 因果掩码

decoder-only Transformer 使用 causal mask，使位置 $t$ 不能读取未来位置。掩码约束信息流，不负责阻止模型记住训练数据，也不等于产品层面的安全边界。

训练张量通常把输入和标签错开一位：`input_ids[:, :-1]` 预测 `input_ids[:, 1:]`。padding、跨样本拼接和不参与训练的控制 token 还需要独立的 loss mask。一个紧凑 reference 是：

```python
import torch
import torch.nn.functional as F
def causal_lm_loss(logits, tokens, trainable):
    """logits:[B,T,V], tokens/trainable:[B,T]; trainable marks target tokens."""
    if logits.shape[:2] != tokens.shape or tokens.shape != trainable.shape:
        raise ValueError("shape mismatch")
    target = tokens[:, 1:]
    valid = trainable[:, 1:]
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        target.reshape(-1),
        reduction="none",
    ).view_as(target)
    if not valid.any():
        raise ValueError("batch has no trainable target")
    return (loss * valid).sum() / valid.sum()
B, T, V = 2, 4, 11
z = torch.randn(B, T, V, requires_grad=True)
x = torch.randint(V, (B, T))
m = torch.tensor([[0, 1, 1, 1], [0, 1, 0, 0]], dtype=torch.bool)
value = causal_lm_loss(z, x, m)
value.backward()
assert torch.isfinite(value) and torch.isfinite(z.grad).all()
assert torch.count_nonzero(z.grad[:, -1]) == 0
```

最后一个位置的 logits 没有下一个 target，所以梯度必须为零。生产训练还要验证 packed attention boundary 与 `trainable` 一致；只 mask loss 而不 mask attention，会让一个样本读取另一个样本。

## 从 logits 到文本

模型输出 logits $z$，温度为 $\tau$ 时：

$$
p_i=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}
$$

- $\tau<1$ 使分布更尖锐；$\tau>1$ 增加随机性。
- top-$k$ 只保留概率最大的 $k$ 个 token。
- top-$p$ 保留累计概率达到阈值的最小集合。
- greedy decoding 每步取最大概率，不保证全序列概率最大，也不保证事实正确。

这里的 temperature、top-$k$ 和 top-$p$ 都作用于<strong>条件分布的读取方式</strong>，不会改变训练得到的 $\theta$。生成还引入停止条件、重复惩罚、grammar 和工具协议；这些规则会改变实际输出分布，因此比较模型时必须固定。解码算法及其数值边界见[解码](../inference/decoding.md)。

## 常用度量

平均 token 损失为 $\bar{\mathcal{L}}$ 时，困惑度 $PPL=\exp(\bar{\mathcal{L}})$。平均分母必须是有效 target 数，而不是 padding 后的矩形大小。它依赖 tokenizer 和数据分布，不能直接跨不同词表、预处理或测试集比较。

对固定 tokenizer 和固定测试集，可以同时报告总 NLL、有效 token 数、perplexity 与按领域切片的均值。若需要跨 tokenizer 比较，应回到同一原始字节或字符口径，例如 bits per byte：

$$
\operatorname{BPB}
=\frac{-\sum_t\log_2p_\theta(x_t\mid x_{<t})}
{\text{原始字节数}}.
$$

BPB 也不是通用能力分数：低文本压缩损失不能替代指令、事实性、工具或安全评测。完整评测协议见[语言模型评测](../evaluation/language-model-evaluation.md)。

## 失效边界

- next-token prediction 优化的是数据分布上的预测，不是“说真话”目标。
- 长序列的平均损失可能掩盖少数关键位置的严重错误。
- 低损失不代表能够遵循新指令、调用工具或满足外部约束。
- 解码策略能改变多样性，但不能创造权重和上下文中不存在的可靠证据。
- teacher forcing 只训练真实前缀上的条件；模型自由生成后进入的错误前缀可能位于训练分布之外。
- 训练集合上的极低 loss 可能来自重复或记忆，必须与独立、去污染的 holdout 一起解释。

原始基础见 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)；缩放行为见[缩放与计算](scaling.md)。

## Reference {#reference}

- [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
