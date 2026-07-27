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

## Attention Residuals：沿深度寻址

标准 residual stream 递推地把所有历史层压进一个 $h_l$。它保留了恒等梯度路径，却要求后续层从这份
累积混合物中恢复早期表示。[Attention Residuals](https://arxiv.org/abs/2603.15031)把相同的
“按内容读取”思想转到网络深度轴：第 $l$ 层不再均匀接收单个 residual state，而是用该层独有的
learnable pseudo-query $q_l$ 在 embedding 与所有先前模块输出间选择。

令 $v_0=k_0=h_{\mathrm{emb}}$，$v_i=k_i=f_i(h_i)$ 表示第 $i$ 个模块的输出。对每个 token 独立计算

$$
\phi(q_l,k_i)
=
\exp\left(q_l^\top\operatorname{RMSNorm}(k_i)\right),
$$

$$
\alpha_{i\to l}
=
\frac{\phi(q_l,k_i)}
{\sum_{j=0}^{l-1}\phi(q_l,k_j)},
\qquad
h_l
=
\sum_{i=0}^{l-1}\alpha_{i\to l}v_i.
$$

RMSNorm 防止某一层仅凭表示范数变大而垄断权重；softmax 使新 residual input 是历史表示的凸组合。
pseudo-query 依层而不依 token，但 key 由 token 的层表示产生，所以同一层仍可对不同 token 选择不同
深度来源。这里的 attention 轴是 **layer depth**，不是 sequence position：它不会替代 token 间的
MHA、GQA、KDA，也不负责位置编码。

### 从 Full 到 Block AttnRes

若完整保留 $L$ 层输出，算术为 $O(L^2d)$，持有历史表示和 pipeline stage 间传输为 $O(Ld)$。当
$L<100$ 时，算术未必是主矛盾；activation residency 与跨 stage 通信往往更先成为瓶颈。

Block AttnRes 把连续层分成 $N$ 个 block，并把每个已完成 block 的模块输出求和为一个
representation。当前 block 内只保留一个逐层增长的 partial sum；跨 block 则在 embedding、已完成
block 和当前 partial sum 上做同样的 depth attention。于是持有与通信从 $O(Ld)$ 降到 $O(Nd)$，
推理时可用 online softmax 合并 inter-block 与 intra-block 两部分。

[Kimi K3 技术报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)公开的实例把
主干分成 8 个、每个 12 层的 block，末块允许不满；再计入 embedding，共有 9 个跨 block 来源。
这是一个具体规模选择，不是方法要求的常数。K3 如何把它与 KDA、Gated MLA 和 Stable LatentMoE
组合，见[Kimi K3](../landscape/works/kimi-k3.md)。
PreNorm 的深度稀释、Full/Block 推导、精确 online merge 与 pipeline cache 的完整脉络，见
[Attention Residuals：让 residual stream 沿深度寻址](../landscape/works/attention-residuals.md)。

### 跨深度 attention reference {#attention-residual}

`attention_residual` 把 source 轴放在第 0 维，后续可以是 token 或 batch 轴。零 pseudo-query 时，
所有历史层权重相同，输出退化为均值；这同时锁定归一化轴与凸组合语义。

```python
import torch

def attention_residual(query, sources, eps=1e-6):
    assert sources.ndim >= 2 and query.shape == (sources.size(-1),)
    normalized = sources.float()
    normalized = normalized * normalized.square().mean(-1, keepdim=True).add(eps).rsqrt()
    logits = torch.einsum("d,l...d->l...", query.float(), normalized)
    weight = torch.softmax(logits, dim=0)
    output = (weight[..., None] * sources.float()).sum(0).to(sources.dtype)
    return output, weight

torch.manual_seed(0)
sources = torch.randn(5, 2, 3, 7)
output, weight = attention_residual(torch.zeros(7), sources)
torch.testing.assert_close(weight.sum(0), torch.ones(2, 3))
torch.testing.assert_close(output, sources.mean(0))
assert output.shape == sources.shape[1:] and torch.isfinite(output).all()
```

这是真值级 full form，不包含 block partial sum、checkpointing、pipeline 通信或 online-softmax kernel。
工程实现必须额外固定 embedding 是否单列、block 边界、训练重算与 decode state；论文与
[官方实现](https://github.com/MoonshotAI/Attention-Residuals)提供了完整定义。当前公开效果主要来自
作者给定模型与训练栈，跨家族收益仍需独立复现，因此它应作为有清晰语义但证据仍在积累的架构分支。

## Reference {#reference}

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Attention Residuals](https://arxiv.org/abs/2603.15031)
- [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)
- [Kimi K3 Technical Report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
