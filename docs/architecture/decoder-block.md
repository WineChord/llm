# Decoder Block

现代 decoder-only 模型反复堆叠两类变换：attention 在 token 之间交换信息，MLP 在每个 token 的通道内变换表示。残差与归一化决定这些变换能否在很深的网络中稳定组合。

## 基本计算图

pre-norm block 可写成

$$
u_l=h_l+\operatorname{Attn}(\operatorname{Norm}(h_l)),
$$

$$
h_{l+1}=u_l+\operatorname{MLP}(\operatorname{Norm}(u_l)).
$$

post-norm 则先做残差相加，再归一化：

$$
u_l=\operatorname{Norm}\left(h_l+\operatorname{Attn}(h_l)\right).
$$

pre-norm 给恒等残差路径提供更直接的梯度通道，通常更容易训练深层网络；post-norm 的表示尺度与优化行为不同，不能只替换一行代码而沿用全部超参数。

## LayerNorm 与 RMSNorm

对 $x\in\mathbb{R}^{d}$，LayerNorm 为

$$
\operatorname{LN}(x)
=\gamma\odot
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,
$$

其中

$$
\mu=\frac{1}{d}\sum_i x_i,
\qquad
\sigma^2=\frac{1}{d}\sum_i(x_i-\mu)^2.
$$

[RMSNorm](https://arxiv.org/abs/1910.07467) 不做中心化：

$$
\operatorname{RMSNorm}(x)
=\gamma\odot
\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}}.
$$

RMSNorm 计算更简单，但“没有减均值”不等于可以忽略数值路径。归约精度、$\epsilon$、权重 dtype、残差累加与 fused kernel 都可能影响 checkpoint 一致性。

## MLP 与 gated activation

普通两层 MLP 为

$$
\operatorname{MLP}(x)=\phi(xW_{\text{up}})W_{\text{down}}.
$$

gated MLP 增加一条门控分支。SwiGLU 常写为

$$
\operatorname{SwiGLU}(x)
=\left(\operatorname{SiLU}(xW_g)\odot xW_u\right)W_d.
$$

若为了保持参数量接近而调整中间宽度，必须说明比较的是相同 hidden width、相同参数量还是相同 FLOPs。激活函数名称不能代替矩阵形状。

### 最小语义实现 {#pre-norm-decoder-block}

下面把 pre-norm block 的三项核心语义放在同一计算图里：RMSNorm 用 FP32 归约后转回输入 dtype，SwiGLU 保留 gate/up 两条投影，两个子层都写回同一 residual stream。输入和输出均为 `[batch, time, dim]`；`attn` 是保持该 shape 的可替换模块。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        scale = x.float().square().mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * scale * self.weight.float()).to(x.dtype)

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate, self.up = nn.Linear(dim, hidden), nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class DecoderBlock(nn.Module):
    def __init__(self, dim, hidden, attn):
        super().__init__()
        self.n1, self.attn, self.n2 = RMSNorm(dim), attn, RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden)
    def forward(self, x, **attn_kwargs):
        x = x + self.attn(self.n1(x), **attn_kwargs)
        return x + self.mlp(self.n2(x))

x = torch.randn(2, 5, 16)
y = DecoderBlock(16, 32, nn.Identity())(x)
assert y.shape == x.shape and torch.isfinite(y).all()
x16 = torch.randn(2, 5, 16, dtype=torch.float16)
assert RMSNorm(16)(x16).dtype == x16.dtype
```

这是结构参考而非 checkpoint 兼容层：attention 的 head/position/cache 契约、bias、初始化、dropout、tensor parallel 与 fused residual-norm 都必须由具体模型补齐。逐算子版本见[张量原语：LayerNorm 与 RMSNorm](../practice/tensor-primitives.md#layernorm-rmsnorm)，带 causal attention 和增量缓存的组合见[Decoder-only Transformer：RMSNorm、SwiGLU 与 Block](../practice/transformer-from-scratch.md#rmsnormswiglu-block)。

## 残差尺度

每层都把新分支写回 residual stream。深度增加后，初始化、norm 位置、分支 scale 与残差 dtype 共同决定方差传播。常见控制包括：

- 按深度缩放部分输出投影初始化；
- 为残差分支增加固定或可学习系数；
- 保留更高精度的 residual accumulation；
- 对不同参数类型采用不同学习率或更新尺度；
- 增加跨层连接或多流残差，但必须重新定义状态和缓存。

这些技巧不是可随意叠加的“稳定性插件”。更换残差拓扑会改变函数类、优化条件和并行实现，应在等计算预算下做消融。

## Dropout 与训练—推理差异

attention probability、MLP、residual 或 embedding 都可能使用 dropout。现代大规模预训练有时将其设为零，但这取决于数据规模和配方。推理必须关闭 dropout；activation checkpointing 的重算则必须恢复相同随机状态，否则反向对应的是另一条计算图。

## 实现契约

一个 block 的 checkpoint 兼容性至少取决于：

```text
norm type, epsilon and placement
Q/K/V/O projection shapes and biases
head layout and positional transform
MLP activation, gate order and intermediate size
residual scaling and accumulation dtype
dropout locations
parameter names, tying and tensor layout
```

两个实现输出 shape 相同，不代表权重可直接互换。

## 调试顺序

1. 用 FP32 小张量核对 norm、MLP 和残差的独立输出。
2. 固定权重，逐子层比较 hidden state。
3. 检查训练与增量推理的 position、mask 和 cache。
4. 再启用 fused norm、fused MLP、低精度和 tensor parallel。
5. 记录第一个发生数值分叉的层，而不是只比较最终 logits。

attention 细节见[注意力家族](attention-variants.md)，完整主干见[Transformer](transformer.md)，优化稳定性见[优化与稳定性](../training/optimization.md)。

## Reference {#reference}

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
