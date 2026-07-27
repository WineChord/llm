# Transformer

Transformer 的关键不是“没有循环”这一表述，而是把序列依赖转化为可并行的全局内容寻址，并用残差堆叠交替完成 token mixing 与 channel mixing。

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

## Feed-Forward Network

FFN 对每个 token 独立地做通道变换。常见 gated 形式为：

$$
\operatorname{SwiGLU}(x)
=\left(\operatorname{SiLU}(xW_1)\odot xW_3\right)W_2
$$

注意力负责跨位置通信，FFN 提供大量参数容量。MoE 通常把 dense FFN 替换为按 token 路由的专家集合。

## Residual 与 Normalization

pre-norm block 可写为：

$$
y=x+\operatorname{Attention}(\operatorname{Norm}(x))
$$

$$
z=y+\operatorname{FFN}(\operatorname{Norm}(y))
$$

它为梯度提供更直接的残差路径，深层训练通常更稳定。LayerNorm 同时中心化并缩放，RMSNorm 只按均方根缩放；二者的差别需要结合完整初始化与优化配方评估。

## 复杂度

标准注意力的 score 矩阵为 $T\times T$，时间和中间存储通常随 $T^2$ 增长；FFN 成本通常随 $T$ 线性增长但常数很大。短上下文、大 hidden size 时 FFN/GEMM 可能主导；长上下文时 attention 与 KV 内存更突出。

## 实现检查

- Q/K/V 的 head 数与 head dimension 是否匹配。
- causal mask、padding mask 与 packed sequence 边界是否正确。
- norm 位置、残差精度和 dropout 是否与 checkpoint 配置一致。
- 训练中的 attention kernel 与推理 kernel 是否使用相同语义。
- chat template、position IDs 与 KV Cache 增量位置是否一致。

原始架构见 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)。子层与归一化见 [Decoder Block](decoder-block.md)，MHA/GQA/MLA 见[注意力家族](attention-variants.md)，位置外推见[长上下文](long-context.md)，高效实现见[Kernel 与性能](../systems/kernels-performance.md)。
