# 位置编码

没有位置机制时，自注意力对输入 token 的排列具有置换等变性。位置编码向模型注入顺序、距离或多维坐标，使相同内容在不同位置产生不同交互。

位置机制需要回答四个问题：

1. 位置进入 hidden state，还是只进入 attention score；
2. 使用绝对坐标还是相对距离；
3. prefill、增量 decode 和 cache 怎样保持同一坐标系；
4. 超出训练范围时，频率或偏置如何变化。

## 绝对位置

最直接的做法是把 token embedding 与位置向量相加：

$$
h_t^{(0)}=E[x_t]+p_t.
$$

Learned absolute embedding 令 $p_t$ 为参数表，接口简单但天然受表长限制。原始 [Transformer](https://arxiv.org/abs/1706.03762) 使用正弦位置：

$$
p_{t,2i}
=
\sin\left(t/10000^{2i/d}\right),
\qquad
p_{t,2i+1}
=
\cos\left(t/10000^{2i/d}\right).
$$

正弦函数可计算任意位置，但“能计算”不等于模型在训练范围外仍会正确使用。

## RoPE

[Rotary Position Embedding](https://arxiv.org/abs/2104.09864) 把每两个通道视为二维平面，并对位置 $m$ 应用旋转：

$$
R_m(\theta_i)
=
\begin{bmatrix}
\cos(m\theta_i)&-\sin(m\theta_i)\\
\sin(m\theta_i)&\cos(m\theta_i)
\end{bmatrix}.
$$

对 query 和 key 使用

$$
\tilde q_m=R_mq_m,
\qquad
\tilde k_n=R_nk_n.
$$

由于

$$
R_m^\top R_n=R_{n-m},
$$

点积自然依赖相对位移：

$$
\tilde q_m^\top\tilde k_n
=
q_m^\top R_{n-m}k_n.
$$

### 最小语义实现 {#rotary-position-embedding}

`apply_rope` 接收 `[T,D]` 张量和每个 token 的逻辑 position ID，按相邻偶/奇通道配对后返回同 shape 张量。两个断言分别验证旋转保持向量范数，以及 query/key 同时平移时点积只依赖相对位移。

```python
import torch

def apply_rope(x, position, base=10_000.):
    assert x.ndim == 2 and x.shape[-1] % 2 == 0
    half = x.shape[-1] // 2
    index = torch.arange(half, device=x.device)
    inv_freq = base ** (-index / half)
    angle = torch.as_tensor(position, device=x.device)[:, None] * inv_freq[None, :]
    even, odd = x[:, 0::2], x[:, 1::2]
    cos, sin = angle.cos(), angle.sin()
    return torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1
    ).flatten(-2)

torch.manual_seed(0)
q, k = torch.randn(3, 8), torch.randn(3, 8)
position = torch.arange(3)
torch.testing.assert_close(apply_rope(q, position).norm(dim=-1), q.norm(dim=-1))
lhs = (apply_rope(q[:1], [7]) * apply_rope(k[:1], [11])).sum()
rhs = (apply_rope(q[:1], [0]) * apply_rope(k[:1], [4])).sum()
torch.testing.assert_close(lhs, rhs)
```

真实模型还要固定 split-half/interleaved 约定、partial rotary dimension、batch broadcast、角度计算精度与 cache 中 K 的存储形态；仅凭公式无法保证 checkpoint 对齐。支持高维张量的实现见[张量原语：Rotary Position Embedding](../practice/tensor-primitives.md#rotary-position-embedding)，prefill/decode 的 position 与 mask 应同 [Decoder-only Transformer：Attention](../practice/transformer-from-scratch.md#attention) 联合测试。

不同通道使用不同角频率。常见定义为

$$
\theta_i=b^{-2i/d_r},
$$

其中 $d_r$ 是参与旋转的维度，$b$ 是 frequency base。实现必须保存 $d_r$、base、缩放规则和 position ID；仅保存“最大长度”不足以复现。

### 增量解码

若 cache 已含位置 $0,\ldots,T-1$ 的 key，新 token 必须使用位置 $T$。批处理中的 padding、prefix cache、speculative branch 和截断窗口都会使“数组下标”不再等于“逻辑位置”。

需要明确 cache 存储：

- 未旋转 K：读取时按历史位置旋转，灵活但增加计算；
- 已旋转 K：decode 快，但 cache 复用要求位置语义完全一致。

## ALiBi

[ALiBi](https://arxiv.org/abs/2108.12409) 不修改 hidden state，而在每个 head 的 score 上加入距离偏置：

$$
s_{ij}
=
\frac{q_i^\top k_j}{\sqrt{d_h}}
-m_h(i-j),
\qquad j\le i.
$$

不同 head 使用不同斜率 $m_h$，形成多个距离尺度。它不需要位置表，也可计算训练范围外的距离；实际外推质量仍取决于训练分布、任务和注意力模式。

## 相对位置偏置

另一类方法把相对距离映射为 bucket：

$$
b_{ij}=g(\operatorname{bucket}(i-j)),
\qquad
s_{ij}\leftarrow s_{ij}+b_{ij}.
$$

短距离可用细粒度 bucket，长距离用对数 bucket。其优点是直接控制相对距离，代价是 bucket 定义、方向性和跨长度行为都成为模型接口。

## 多维位置

图像和视频 token 不只有一维序号。一个视频 patch 可以具有

$$
(t,h,w)
$$

三个坐标。[Qwen2-VL](https://arxiv.org/abs/2409.12191) 的 M-RoPE 将旋转维度分配给时间、高度和宽度；文本片段则让这些坐标共同前进。

实现要明确：

- 每个轴占多少 rotary dimension；
- dynamic tiling 后 patch 的原始二维坐标；
- 多图之间是否重置坐标；
- padding、缩放和 frame sampling 是否同步更新位置；
- 文本、图像和视频交错时逻辑时间如何推进。

把二维或三维网格直接 flatten 成一维索引虽然简单，却让相邻关系依赖具体扫描顺序。

## 长度扩展

设训练长度为 $L_0$、目标长度为 $L_1$。[Position Interpolation](https://arxiv.org/abs/2306.15595) 把目标位置压回训练范围：

$$
m'=m\frac{L_0}{L_1}.
$$

[YaRN](https://arxiv.org/abs/2309.00071) 对不同频率采取更细粒度的插值与修正；[LongRoPE](https://arxiv.org/abs/2402.13753) 搜索非均匀缩放并处理短上下文恢复。它们改变的是位置频谱，不会自动解决：

- full attention 的二次计算；
- KV Cache 容量；
- 长距离训练数据不足；
- 模型在中部长证据上的利用能力。

这些系统与评测问题见[长上下文](long-context.md)。

## Shape 与实现契约

设

$$
Q,K\in\mathbb R^{B\times H\times T\times d_h}.
$$

若 rotary dimension 为 $d_r\le d_h$，则只旋转 `[..., :d_r]`，其余通道原样保留。实现应固定：

1. interleaved 还是 split-half 配对；
2. 正余弦缓存的 shape 与 broadcast 轴；
3. position IDs 是 $[T]$、$[B,T]$ 还是多维坐标；
4. Q 与 K 是否使用相同 rotary dimension；
5. dtype：角度计算与最终张量是否同精度；
6. cache 保存旋转前还是旋转后 K；
7. 长度扩展配置是否进入 checkpoint metadata。

## 失效模式

- **Off-by-one**：prefill 最后位置与首个 decode 位置重复。
- **Padding 漂移**：用物理下标而不是每个样本的有效位置。
- **配对约定不一致**：训练与推理分别使用 interleaved、split-half。
- **Cache 复用错误**：已旋转 K 被放到不同绝对位置复用。
- **低精度角度误差**：极长位置下相位计算失真。
- **只改配置长度**：模型能分配 cache，却未学会范围外位置。
- **短上下文回退**：扩展频率破坏原训练区间性能。
- **多维坐标丢失**：动态切片后仍按简单一维顺序编码。

## 验证

| 测试 | 判据 |
| --- | --- |
| 相对性 | 同时平移 Q/K 位置后点积保持 |
| Prefill–decode | 全序列计算与逐 token cache 结果对齐 |
| Batch padding | 左右 padding 不改变有效 token 结果 |
| 配对约定 | 与 checkpoint 官方实现逐元素对齐 |
| 长度边界 | $L_0-1,L_0,L_0+1,L_1-1$ 均无突变 |
| 位置扫描 | 相同证据在头、中、尾的性能曲线 |
| 短上下文 | 扩展前后原训练长度内无不可接受回退 |
| 多模态 | crop、tile、frame 顺序变化后坐标仍一致 |

RoPE 与 GQA 的组合实现见[注意力家族](attention-variants.md)，可执行的最小序列实验见[序列模型手撕实现](../practice/sequence-models.md)。

## Reference {#reference}

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Train Short, Test Long: Attention with Linear Biases](https://arxiv.org/abs/2108.12409)
- [Qwen2-VL](https://arxiv.org/abs/2409.12191)
- [Position Interpolation](https://arxiv.org/abs/2306.15595)
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
- [LongRoPE](https://arxiv.org/abs/2402.13753)
