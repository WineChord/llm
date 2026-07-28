# Attention Is All You Need

[Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)发表于 NeurIPS 2017。它接住的不是“序列模型从未用过 attention”，而是更具体的矛盾：当时强大的 encoder–decoder 已用 attention 读取源端状态，但 encoder 与 decoder 内部仍依赖 recurrence 或 convolution，长依赖路径和时间串行限制了训练。

作者团队公开的历史实现位于 [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)。该仓库现已归档并弃用，适合核对当时的模型与训练接口，不应被当成现代 PyTorch 实现或某一论文 checkpoint 的无条件完全复现。

还要把 2017 配方与今天常见的 decoder block 分开：原论文采用 post-norm、ReLU FFN 与 sinusoidal position encoding；pre-norm、RMSNorm、SwiGLU 和 RoPE 是后来逐步形成的常见组合。下文先固定原始机制，再把现代变体交给 canonical 架构页。

## 它改变了哪条路径

RNN 必须先得到 $h_{t-1}$ 才能计算 $h_t$；self-attention 则让一层中的所有位置同时从整张状态表读取信息：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V,
$$

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}
\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

缩放项控制初始化时点积的方差，$M$ 定义 padding、causal 或其他可见边界。multi-head attention 在不同投影子空间并行寻址，随后拼接；它提供多组通信通道，但不能预先保证每个 head 都对应稳定、可命名的语义。

encoder 使用双向 self-attention；decoder 先用 masked self-attention 读取已生成前缀，再用 cross-attention 读取 encoder 状态。逐位置 FFN 提供 channel mixing，residual 与 LayerNorm 维持深层优化路径，位置编码补回架构本身缺少的顺序信息。

原论文的 Base 模型是六层 encoder 与六层 decoder，$d_{\text{model}}=512$、八个 head、FFN inner dimension 2048；Big 模型把表示扩至 1024、head 增至 16、FFN 扩至 4096。数字本身不是现代默认值，但它们固定了原始结论的实验对象：一个完整 encoder–decoder，而不是今天最常见的 decoder-only LM。

原始 residual 顺序为

$$
\operatorname{LayerNorm}(x+\operatorname{Sublayer}(x)),
$$

位置则用不同频率的正弦与余弦显式注入。若用 pre-norm、RoPE 或 gated FFN 复现，验证的是后来形成的 Transformer 家族，不是对 2017 配方的逐项复刻。

## 最小机制实现

下面只实现单头 scaled dot-product attention。它不是原论文训练系统的复刻，但保留了最关键的缩放、mask、归一化与梯度语义：

```python
import math
import torch
def scaled_attention(q, k, v, allowed=None):
    """q/k:[B,T,D], v:[B,T,Dv], allowed:[T,T] with True for visible pairs."""
    score = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if allowed is not None:
        if allowed.dtype != torch.bool:
            raise TypeError("allowed must be boolean")
        score = score.masked_fill(~allowed, torch.finfo(score.dtype).min)
    prob = score.softmax(dim=-1)
    return prob @ v, prob
torch.manual_seed(7)
B, T, D = 2, 5, 8
x = torch.randn(B, T, D, requires_grad=True)
wq, wk, wv = (torch.randn(D, D) for _ in range(3))
causal = torch.ones(T, T, dtype=torch.bool).tril()
y, prob = scaled_attention(x @ wq, x @ wk, x @ wv, causal)
torch.testing.assert_close(prob.sum(-1), torch.ones(B, T))
assert torch.count_nonzero(prob.masked_select(~causal)) == 0
x_changed = x.detach().clone()
x_changed[:, -1] += 100
y_changed, _ = scaled_attention(
    x_changed @ wq, x_changed @ wk, x_changed @ wv, causal
)
torch.testing.assert_close(y[:, :-1].detach(), y_changed[:, :-1])
y.square().mean().backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
```

修改最后一个 token 不会改变此前位置，直接验证了 causal no-leak。生产实现还要处理 head reshape、padding 与 packed boundary、全 masked row、低精度 softmax、dropout 和增量 KV Cache。

## 训练配方也是论证的一部分

论文不仅替换了层结构，还配套使用 Adam、warmup 后按步数平方根衰减的学习率、label smoothing、dropout 和共享 embedding/softmax 权重。模型在 WMT 2014 英德与英法翻译上评测，并将 encoder 用于 constituency parsing。因而“attention-only 可训练”是架构与优化配方共同支持的结论，不能把其中一个超参数单独解释成决定性原因。

学习率日程可写为

$$
\eta_t
=d_{\text{model}}^{-1/2}
\min(t^{-1/2},t\,w^{-3/2}),
$$

其中 $w$ 是 warmup steps。这个尺度与原始 post-norm 配方相连；现代 pre-norm 大模型通常重新选择初始化、峰值学习率和日程，不能机械复用。

## 并行性换来了三笔账

论文比较的是每层复杂度、顺序操作数和最大路径长度。长度为 $T$、表示维度为 $d$ 时，dense self-attention 通常产生 $T\times T$ score，主要计算为 $O(T^2d)$；RNN 常写作 $O(Td^2)$，但需要 $O(T)$ 个顺序步骤。两者谁更快取决于 shape、kernel 与硬件，不能只凭一个渐近式判断。

self-attention 缩短了信息路径，却留下三笔新债：

- 顺序必须由位置表示显式注入；
- 长上下文的 score、IO 与 KV 状态持续增长；
- 训练可以并行，自回归 decode 仍必须逐 token 推进。

还要区分中间存储和算法关系。FlashAttention 可以通过 tiling 与 online softmax 避免把完整 score 写回高层显存，却仍精确计算 dense attention；KV Cache 可以复用旧 K/V，却仍需为新 query 读取历史。它们分别处理 IO 与重复计算，不等于从模型语义中删除全局寻址。

## 论文证明到哪里，家族又走到哪里

论文直接支持的是其机器翻译与 parsing 设置下，纯 attention encoder–decoder 能达到有竞争力的质量并显著提高训练并行性。它没有证明所有序列长度上都更快、attention 权重可直接解释、或 recurrence/卷积从此没有价值。后续 decoder-only、encoder-only、视觉 Transformer 与稀疏变体扩大了证据范围，也同时改变了训练数据、目标和系统。

Transformer 的持久影响不是一套不可更改的 2017 block。后来的 decoder-only 预训练、双向 encoder、视觉 patch、跨模态桥接、稀疏 attention 和高效 kernel 都保留了“query 寻址 key、读取 value”的接口，同时更改 norm、位置、FFN、head 共享与执行方式。

前序矛盾见[从固定向量到内容寻址](../lineages/transduction-to-attention.md)。现代 block 的 canonical 推导见 [Transformer](../../architecture/transformer.md)、[Decoder Block](../../architecture/decoder-block.md)与[注意力和位置](../../architecture/attention-position.md)；精确 attention 的 IO 路线见 [FlashAttention](flashattention.md)，完整可执行模型和 KV 对照见[手撕：Decoder-only Transformer](../../practice/transformer-from-scratch.md)。

## Reference {#reference}

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
- [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
