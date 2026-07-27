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

## 并行性换来了什么成本

论文比较的是每层复杂度、顺序操作数和最大路径长度。长度为 $T$、表示维度为 $d$ 时，dense self-attention 通常产生 $T\times T$ score，主要计算为 $O(T^2d)$；RNN 常写作 $O(Td^2)$，但需要 $O(T)$ 个顺序步骤。两者谁更快取决于 shape、kernel 与硬件，不能只凭一个渐近式判断。

self-attention 缩短了信息路径，却留下三笔新债：

- 顺序必须由位置表示显式注入；
- 长上下文的 score、IO 与 KV 状态持续增长；
- 训练可以并行，自回归 decode 仍必须逐 token 推进。

## 哪些思想真正迁移了

Transformer 的持久影响不是一套不可更改的 2017 block。后来的 decoder-only 预训练、双向 encoder、视觉 patch、跨模态桥接、稀疏 attention 和高效 kernel 都保留了“query 寻址 key、读取 value”的接口，同时更改 norm、位置、FFN、head 共享与执行方式。

前序矛盾见[从固定向量到内容寻址](../lineages/transduction-to-attention.md)。现代 block 的 canonical 推导见 [Transformer](../../architecture/transformer.md)、[Decoder Block](../../architecture/decoder-block.md)与[注意力和位置](../../architecture/attention-position.md)；完整可执行模型和 KV 对照见[手撕：Decoder-only Transformer](../../practice/transformer-from-scratch.md)。
