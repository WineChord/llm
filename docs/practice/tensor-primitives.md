# 手撕：张量原语

本页使用 shape 注释固定 decoder-only 模型最常见的数值原语。代码追求语义直接；生产 kernel 会融合访存、分块和向量化，但不应改变归一化轴、mask 或 head 映射。

## Stable log-softmax 与交叉熵

直接计算 $\exp(x)$ 可能溢出。减去最大值不改变 softmax：

$$
\log\operatorname{softmax}(x)_i
=x_i-m-\log\sum_j e^{x_j-m},
\qquad m=\max_j x_j.
$$

```python
import math
import torch

def stable_log_softmax(x, dim=-1):
    """x: [..., V] -> log probabilities with the same shape."""
    m = x.max(dim=dim, keepdim=True).values
    z = x - m
    return z - z.exp().sum(dim=dim, keepdim=True).log()

def cross_entropy(logits, labels, ignore_index=-100):
    """logits: [..., V], labels: [...] -> scalar token mean."""
    keep = labels != ignore_index
    safe = labels.masked_fill(~keep, 0)
    loss = -stable_log_softmax(logits).gather(-1, safe[..., None]).squeeze(-1)
    if not keep.any():
        raise ValueError("cross entropy has no valid token")
    return loss[keep].mean()
```

```python
x = torch.tensor([[1000.0, 1001.0, 999.0]], dtype=torch.float64)
y = torch.tensor([1])
torch.testing.assert_close(
    cross_entropy(x, y),
    torch.nn.functional.cross_entropy(x, y),
)
```

若一整行都被 attention mask，不能把全 $-\infty$ 直接送入 softmax；应在调用前定义该行是全零、保留哨兵，还是输入非法。

## LayerNorm 与 RMSNorm

LayerNorm 同时中心化和缩放：

$$
\operatorname{LN}(x)=
\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta.
$$

RMSNorm 只按均方根缩放：

$$
\operatorname{RMSNorm}(x)=
\gamma\odot
\frac{x}{\sqrt{H^{-1}\sum_{j=1}^{H}x_j^2+\epsilon}}.
$$

```python
def layer_norm(x, weight, bias, eps=1e-5):
    """x: [..., H], weight/bias: [H] -> [..., H]."""
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = (xf - mean).square().mean(dim=-1, keepdim=True)
    y = (xf - mean) * (var + eps).rsqrt()
    return y.to(x.dtype) * weight + bias

def rms_norm(x, weight, eps=1e-6):
    """x: [..., H], weight: [H] -> [..., H]."""
    xf = x.float()
    y = xf * (xf.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    return y.to(x.dtype) * weight
```

归约用 FP32 是实现选择的一部分。验证时覆盖常数输入、极大值、小方差与非连续 tensor，并与所用框架的 `eps` 位置保持一致。

## SiLU 与 SwiGLU

$$
\operatorname{SiLU}(x)=x\sigma(x),\qquad
\operatorname{SwiGLU}(x)=
\operatorname{SiLU}(xW_g^\top)\odot(xW_u^\top).
$$

```python
def swiglu(x, w_gate, w_up):
    """x: [..., H], weights: [F, H] -> [..., F]."""
    gate = x @ w_gate.T
    up = x @ w_up.T
    return gate * gate.sigmoid() * up
```

三个投影的命名在实现间不同。加载 checkpoint 时应按 tensor shape 和计算图确认 gate/up/down，不要只依赖键名。

## Rotary Position Embedding

对每个二维通道对，RoPE 施加位置相关旋转：

$$
\begin{bmatrix}x'_{2i}\\x'_{2i+1}\end{bmatrix}
=
\begin{bmatrix}
\cos\theta_i & -\sin\theta_i\\
\sin\theta_i & \cos\theta_i
\end{bmatrix}
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}.
$$

```python
def rope(x, positions, base=10000.0):
    """x: [B, H, T, D], positions: [T], D must be even."""
    d = x.size(-1)
    if d % 2:
        raise ValueError("RoPE head dimension must be even")
    inv = base ** (-torch.arange(0, d, 2, device=x.device).float() / d)
    position = positions.to(device=x.device, dtype=torch.float32)
    angle = position[:, None] * inv[None, :]
    cos = angle.cos().to(x.dtype)[None, None]
    sin = angle.sin().to(x.dtype)[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
```

```python
v = torch.randn(2, 3, 5, 8, dtype=torch.float64)
r = rope(v, torch.arange(5))
torch.testing.assert_close(v.square().sum(-1), r.square().sum(-1))
assert rope(v.bfloat16(), torch.arange(5)).dtype == torch.bfloat16
```

增量解码必须传绝对 token 位置或与训练一致的变换后位置。cache 中保存旋转前还是旋转后 K，也属于 checkpoint 与运行时契约。

## Grouped-Query Attention

设 query head 数为 $A$，KV head 数为 $A_{\mathrm{kv}}$，要求 $A$ 可被 $A_{\mathrm{kv}}$ 整除。reference 显式展开 K/V，便于核对映射：

```python
def gqa(q, k, v, causal=True):
    """q:[B,A,Tq,D], k/v:[B,Akv,Tk,D] -> [B,A,Tq,D]."""
    b, hq, tq, d = q.shape
    _, hkv, tk, _ = k.shape
    if hq % hkv:
        raise ValueError("query heads must be divisible by KV heads")
    group = hq // hkv
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    score = q @ k.transpose(-2, -1) / math.sqrt(d)
    if causal:
        qi = torch.arange(tq, device=q.device) + tk - tq
        kj = torch.arange(tk, device=q.device)
        score = score.masked_fill(kj[None, :] > qi[:, None], float("-inf"))
    return score.softmax(dim=-1) @ v
```

生产 kernel 不应真的复制 K/V，而是在加载时把 query head 映射为 $\lfloor h_q/(A/A_{\mathrm{kv}})\rfloor$。reference 还需覆盖 $T_q=1$、带历史 cache 的 $T_k>T_q$ 和非整除 head 的错误路径。

## Online softmax

对一个 score block，保存最大值 $m$、指数和 $\ell$ 与未归一化输出 $u$：

$$
m=\max_i s_i,\qquad
\ell=\sum_i e^{s_i-m},\qquad
u=\sum_i e^{s_i-m}v_i.
$$

两个 block 可结合，而不保存完整 score matrix：

```python
def merge_softmax(m1, l1, u1, m2, l2, u2):
    """m/l: [...,1], u:[...,D] -> merged online-softmax state."""
    m = torch.maximum(m1, m2)
    a, b = torch.exp(m1 - m), torch.exp(m2 - m)
    return m, a * l1 + b * l2, a * u1 + b * u2

def finish_softmax(state):
    """Return normalized output; every row must have positive mass."""
    _, l, u = state
    if (l <= 0).any():
        raise ValueError("softmax row has no valid element")
    return u / l
```

合并满足结合律的浮点近似版本，因此可以沿 key block 流式计算；不同合并顺序仍可能产生舍入差异。[FlashAttention](https://arxiv.org/abs/2205.14135)把这一思想与 IO-aware tiling 结合，系统实现见[Attention Kernel](../systems/attention-kernels.md)。

## 验证清单

- identity：与框架或完整矩阵实现对照；
- mask：有效 token 不随 padding 内容变化；
- degenerate：全 mask、零方差、单 KV head、单 token；
- dtype：FP64 reference、FP32 累加和目标低精度；
- gradient：有限差分或 `gradcheck`；
- layout：连续与非连续 tensor；
- partition：切 block 后合并与一次计算一致。

这些原语怎样组成完整 block 见[手撕 Decoder-only Transformer](transformer-from-scratch.md)，数学推导见[Decoder Block](../architecture/decoder-block.md)。

## Reference {#reference}

- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
