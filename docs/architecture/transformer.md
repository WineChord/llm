# Transformer

Transformer 的关键不是“没有循环”这一表述，而是把序列依赖转化为可并行的全局内容寻址，并用残差堆叠交替完成 token mixing 与 channel mixing。

这一步并非从空白开始。[Seq2Seq 与神经对齐](../landscape/works/seq2seq-and-neural-alignment.md)展示 fixed-vector bottleneck 怎样先变成按 decoder 状态读取的 memory，[从递归到注意力](../landscape/lineages/transduction-to-attention.md)再把问题推向 self-attention；[Attention Is All You Need](../landscape/works/attention-is-all-you-need.md)逐项拆解原论文、Tensor2Tensor 实现边界与最小可执行 attention。

## 缩放点积注意力

给定输入 $X\in\mathbb{R}^{T\times d}$：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V
$$

$M$ 是因果或结构掩码。除以 $\sqrt{d_h}$ 用于控制点积方差，避免 softmax 在初始化时过早饱和。

多头注意力把表示分为多个 head，在不同投影子空间中寻址，再拼接输出。head 并不天然对应可解释的固定语义；功能可能分散且随层变化。

若 batch 为 $B$、head 数为 $H$、单头维度为 $d_h$，实现通常把投影重排为 $[B,H,T,d_h]$。score 的最后两维是 query position 与 key position；mask 必须能广播到 $[B,H,T_q,T_k]$。这条 shape 约定比公式里的二维矩阵更重要，因为 padding、packed sample 和增量 decode 都在这里决定可见性。

下面保留多头 self-attention 的不可约语义，不包含 fused kernel 或 KV Cache：

```python
import math
import torch
def multihead_attention(x, wq, wk, wv, wo, heads, allowed):
    """x:[B,T,D], allowed:[B,T,T] with True for visible edges."""
    b, t, d = x.shape
    if d % heads or allowed.shape != (b, t, t):
        raise ValueError("invalid head or mask shape")
    def split(y):
        return y.view(b, t, heads, d // heads).transpose(1, 2)
    q, k, v = (split(x @ w) for w in (wq, wk, wv))
    score = q @ k.transpose(-2, -1) / math.sqrt(d // heads)
    score = score.masked_fill(~allowed[:, None], torch.finfo(score.dtype).min)
    prob = score.softmax(-1)
    y = (prob @ v).transpose(1, 2).contiguous().view(b, t, d)
    return y @ wo, prob
B, T, D, H = 2, 5, 12, 3
x = torch.randn(B, T, D)
weights = [torch.randn(D, D) for _ in range(4)]
mask = torch.ones(B, T, T, dtype=torch.bool).tril()
y, p = multihead_attention(x, *weights, H, mask)
assert y.shape == x.shape and p.shape == (B, H, T, T)
assert torch.count_nonzero(p.masked_select(~mask[:, None])) == 0
```

若某个 query 的整行都不可见，把有限最小值送进 softmax 会得到一行近似均匀概率，而不是“全零”。生产实现要么保证每行至少可见一个合法 key，要么显式处理 all-masked row；这也是 padding 与 packed boundary 最常见的 silent bug 之一。

## Feed-Forward Network

FFN 对每个 token 独立地做通道变换。常见 gated 形式为：

$$
\operatorname{SwiGLU}(x)
=\left(\operatorname{SiLU}(xW_1)\odot xW_3\right)W_2
$$

注意力负责跨位置通信，FFN 提供大量参数容量。MoE 通常把 dense FFN 替换为按 token 路由的专家集合。

## 一个 block 的信息流

pre-norm block 可写为：

$$
y=x+\operatorname{Attention}(\operatorname{Norm}(x))
$$

$$
z=y+\operatorname{FFN}(\operatorname{Norm}(y))
$$

它为梯度提供更直接的残差路径，深层训练通常更稳定。LayerNorm 同时中心化并缩放，RMSNorm 只按均方根缩放；二者的差别需要结合完整初始化与优化配方评估。

原论文实际使用 post-norm，即 $\operatorname{Norm}(x+\operatorname{Sublayer}(x))$。pre-norm 改变了梯度路径，也改变最终 norm、初始化和残差尺度的配方，不能只移动一行代码后继续加载同一 checkpoint。更完整的 block、GQA 与增量一致性测试见 [Decoder Block](decoder-block.md)。

## 复杂度

标准注意力的 score 矩阵为 $T\times T$，时间和中间存储通常随 $T^2$ 增长；FFN 成本通常随 $T$ 线性增长但常数很大。短上下文、大 hidden size 时 FFN/GEMM 可能主导；长上下文时 attention 与 KV 内存更突出。

训练与生成还处在不同 shape 区间：训练常以较大 $B\times T$ 计算整段矩阵，自回归 decode 每步只有一个新 query，却要读取不断增长的 K/V。FlashAttention 可以避免物化完整 score 中间量，但没有把精确 dense attention 的计算关系从 $T^2$ 变为线性；KV Cache 避免重复计算旧 K/V，也没有消除读取历史状态的带宽成本。

## 从训练张量到可复现 checkpoint

- Q/K/V 的 head 数与 head dimension 是否匹配。
- causal mask、padding mask 与 packed sequence 边界是否正确。
- norm 位置、残差精度和 dropout 是否与 checkpoint 配置一致。
- 训练中的 attention kernel 与推理 kernel 是否使用相同语义。
- chat template、position IDs 与 KV Cache 增量位置是否一致。

还应保存 tokenizer、模型配置、参数命名与 tied-weight 规则；只拿到权重张量而缺少这些契约，无法唯一重建前向。原始架构见 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。MHA/GQA/MLA 见[注意力家族](attention-variants.md)，位置外推见[长上下文](long-context.md)，训练与增量前向的等价性见[手撕 Transformer](../practice/transformer-from-scratch.md)，高效执行见[Attention Kernel](../systems/attention-kernels.md)与 [KV Cache](../inference/kv-cache.md)。

## Reference {#reference}

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
