# Seq2Seq 与 Bahdanau Attention：一个向量为什么不够

2014 年，神经机器翻译出现了两项彼此时间重叠、共同改变条件生成的工作。[Sequence to Sequence Learning with Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2014/hash/5a18e133cbf9f257297f410bb7eca942-Abstract.html)在其 WMT 设置中展示了 LSTM encoder–decoder 可以把变长源序列映射为变长目标序列；[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)则针对 fixed-vector encoder–decoder 家族追问：为什么整个输入必须被压进一个固定向量？

它们不应被写成严格的“先有 Seq2Seq，后来才有人想到 attention”。Bahdanau 论文首次提交于 2014 年 9 月，与当年的 Seq2Seq 研究并行推进，后发表于 ICLR 2015。

## Seq2Seq 建立了条件语言模型

encoder LSTM 读取 $x_{1:T_x}$，把最终 recurrent state 抽象为 context $c$；对 LSTM 而言它包含 hidden 与 cell state，decoder 由此开始生成：

$$
c=\operatorname{Enc}(x_{1:T_x}),
\qquad
p(y\mid x)=\prod_{t=1}^{T_y}
p(y_t\mid y_{<t},c).
$$

这是重要的接口统一：翻译不再依赖短语表和独立对齐模块，encoder 与 decoder 通过最大似然端到端训练。原工作还把源句反转，缩短某些对应词之间的最小时间距离，改善优化。

反转是有效技巧，却没有改变信息瓶颈。源句无论多长，都只能通过固定维度 $c$ 进入 decoder；细节之间还会竞争同一状态容量。

## Alignment 变成可微读取

Bahdanau attention 保留所有 encoder states $h_1,\ldots,h_{T_x}$。decoder 第 $t$ 步根据上一状态 $s_{t-1}$ 计算 additive score：

$$
e_{tj}=v^\top\tanh(W_s s_{t-1}+W_h h_j),
$$

$$
\alpha_{tj}
=\frac{\exp e_{tj}}{\sum_k\exp e_{tk}},
\qquad
c_t=\sum_j\alpha_{tj}h_j.
$$

context 从固定 $c$ 变为每一步不同的 $c_t$。模型可在生成目标词时回到最相关的源位置，而不必让最终 encoder state 无损保存一切。soft alignment 没有逐词标签，它由翻译 loss 共同学习。

## 带 padding mask 的最小 additive attention

```python
import torch
from torch import nn
class AdditiveAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.score = nn.Linear(d, 1, bias=False)
    def forward(self, state, memory, valid):
        energy = self.score(torch.tanh(self.query(state)[:, None] + self.key(memory))).squeeze(-1)
        energy = energy.masked_fill(~valid, torch.finfo(energy.dtype).min)
        weight = energy.softmax(-1)
        return torch.einsum("bt,btd->bd", weight, memory), weight
torch.manual_seed(0)
attn = AdditiveAttention(8)
state = torch.randn(2, 8, requires_grad=True)
memory = torch.randn(2, 5, 8, requires_grad=True)
valid = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
context, weight = attn(state, memory, valid)
assert context.shape == (2, 8)
assert torch.allclose(weight.sum(-1), torch.ones(2), atol=1e-6)
assert torch.count_nonzero(weight[~valid]) == 0
context.square().mean().backward()
assert memory.grad is not None and torch.count_nonzero(memory.grad[~valid]) == 0
```

mask 后的 padding 不获得权重或梯度，固定了可变长度 batch 的关键语义。完整 NMT 还需要双向 encoder、teacher forcing、decoder recurrence、词表投影和 beam search；这些工程不应遮住“按 decoder 状态读取源端记忆”的核心。

## Attention 不是无损解释

$\alpha_{tj}$ 可以画成 alignment heatmap，也经常呈现合理的词序对应。但它首先是生成计算中的权重，不是自动可靠的语言学标注或因果解释。多个位置可能共同提供信息，后续层也会重新混合表示；仅凭一张 attention 图不能证明模型为什么作出最终决定。

## Transformer 继承并改写了什么

Bahdanau attention 已经包含 query、memory 与 weighted read 的核心。Transformer 做了三次推广：

1. 用点积投影替代 additive scoring 的主要形式；
2. 让序列内部也通过 self-attention 互相读取；
3. 移除 recurrent encoder/decoder，使训练沿 token 维并行。

代价从“一个固定 context”转移为位置表示、平方关系和自回归 KV 状态。[Attention Is All You Need 深读](attention-is-all-you-need.md)解释这次转折，[从固定向量到内容寻址](../lineages/transduction-to-attention.md)则把前后脉络放在同一条线上。

## 这组工作的证据边界

Seq2Seq 在当时的机器翻译设置中展示了大规模 LSTM encoder–decoder 的端到端可行性；Bahdanau attention 在其设置中展示了动态软对齐对 fixed-length bottleneck 与长句翻译的改善。它们没有证明：

- 最终状态对所有短序列都无用；
- attention 权重等同人类对齐；
- recurrence 从此没有价值；
- 后来的 scaled dot-product attention 与 additive attention 在所有条件下等价。

有限状态的前序见[LSTM](lstm.md)，现代 cross-attention 在多模态和生成中的迁移见[冻结模型之间怎样架桥](visual-language-bridges.md)与[RAG](rag.md)，实现层的 mask 细节见[张量原语](../../practice/tensor-primitives.md)。

## Reference {#reference}

- [Sequence to Sequence Learning with Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2014/hash/5a18e133cbf9f257297f410bb7eca942-Abstract.html)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
