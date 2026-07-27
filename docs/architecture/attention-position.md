# 注意力与位置

注意力机制回答“当前位置从哪些内容读取什么”，位置机制回答“这些内容在序列或空间中的相对关系是什么”。二者共同作用，但属于不同设计轴。

本页保留为稳定入口，避免旧链接失效：

- [注意力家族](attention-variants.md)：MHA、MQA、GQA、MLA、mask 与 KV Cache；
- [位置编码](position-encoding.md)：绝对位置、RoPE、ALiBi 与多维位置；
- [长上下文](long-context.md)：位置扩展、稀疏模式、分布式计算与有效长度；
- [Attention Kernel](../systems/attention-kernels.md)：online softmax、tiling 与硬件执行；
- [KV Cache](../inference/kv-cache.md)：缓存布局、容量与增量解码。

## 两个正交问题

标准注意力写作

$$
Y
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}+M+B_{\text{pos}}
\right)V.
$$

其中：

- $Q,K,V$ 以及 head 的共享方式属于内容路由；
- $M$ 定义因果、窗口或块稀疏等可见性；
- $B_{\text{pos}}$ 或施加在 $Q,K$ 上的位置变换定义顺序结构。

改变 KV head 数主要影响 cache；改变位置编码主要影响模型怎样区分距离；改变 mask 主要影响哪些位置可以互相读取。三者不可互相替代。

## 稳定比较轴

| 设计轴 | 代表选择 | 首要代价 |
| --- | --- | --- |
| KV 共享 | MHA、GQA、MQA、MLA | 表达容量、缓存带宽、实现复杂度 |
| 位置表示 | learned absolute、RoPE、ALiBi | 外推、分辨率、增量位置 |
| 可见模式 | full、window、block sparse、global token | 信息可达性与 kernel 稀疏度 |
| 精确实现 | 朴素 attention、FlashAttention | HBM 访问、并行划分、支持范围 |
| 状态替代 | SSM、线性注意力、混合层 | 有限状态容量与内容寻址能力 |

复杂度表达式必须附带 shape 与实现条件。理论 FLOPs 相同不代表 wall-clock 相同；理论线性复杂度也不保证短序列更快。

## 阅读与诊断顺序

遇到注意力或长上下文问题时，依次确认：

1. 输入序列、位置 ID 和 causal mask 是否正确；
2. query head 到 KV head 的映射是否正确；
3. prefill 与 decode 是否使用同一位置定义；
4. cache 中保存的是旋转前还是旋转后的 K；
5. kernel 是否支持实际 head dimension、dtype 与 mask；
6. 问题来自位置外推、信息不可见、缓存淘汰还是训练分布。

这个顺序能避免把实现错误误判为架构能力不足。

## 前沿术语边界：跨深度 Attention

[Attention Residuals](https://arxiv.org/abs/2603.15031)及其[官方实现](https://github.com/MoonshotAI/Attention-Residuals)把 attention 用在网络深度方向，对先前层表示进行内容相关加权。它改变的是 residual aggregation，不是 token 序列上的 MHA/GQA，也不属于位置编码。

截至公开论文所披露的证据，该方法在作者给定规模、数据和实现中得到验证，并给出 block-level 近似以降低跨层状态与通信成本。其跨模型家族、训练栈和更大规模的通用收益仍需独立证据，因此只作为前沿观察，不纳入稳定注意力分类。

## Reference {#reference}

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Attention Residuals](https://arxiv.org/abs/2603.15031)
- [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)
