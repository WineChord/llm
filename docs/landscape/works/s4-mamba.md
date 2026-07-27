# 从 S4 到 Mamba：有限状态怎样重新获得选择性

循环模型一直拥有一个推理优势：处理完当前 token 后，只需保留有限状态。它的困难也同样古老——所有历史都被压进同一个 state，长程信息容易衰减或互相覆盖。S4 与 Mamba 的路线没有简单回到 RNN，而是先用结构化动态系统解决长序列训练，再让状态更新依赖输入，以接近 attention 的内容选择能力。

## HiPPO：状态可以是在线投影

[HiPPO](https://proceedings.neurips.cc/paper/2020/hash/102f0bb6efb3a6128a3c750dd16729be-Abstract.html)把“记住历史”写成在线函数逼近：状态系数追踪过去信号在一组基函数上的投影。它提供了一种原则化设计长时记忆矩阵的方法，而不只是随机挑选 RNN transition。

连续状态空间模型写作

$$
\dot h(t)=Ah(t)+Bx(t),\qquad y(t)=Ch(t)+Dx(t).
$$

以步长 $\Delta$ 离散化：

$$
\bar A=e^{\Delta A},\qquad
\bar B=(\Delta A)^{-1}(e^{\Delta A}-I)\Delta B,
$$

得到 $h_t=\bar Ah_{t-1}+\bar Bx_t$。实际实现会用数值稳定形式处理 $A$ 接近零和不同参数化。

## S4：递推与卷积的双重接口

线性时不变 SSM 可以展开为 convolution：

$$
y_t=\sum_{j=0}^{t}K_jx_{t-j},\qquad
K_j=C\bar A^j\bar B.
$$

[S4](https://arxiv.org/abs/2111.00396)利用结构化 $A$ 和高效 kernel，使很长序列上的并行训练成为现实。下面用标量对角情形验证 recurrence 与 convolution 等价：

```python
import torch
def lti_scan(x, a, b, c):
    state = torch.zeros((), dtype=x.dtype)
    out = []
    for value in x:
        state = a * state + b * value
        out.append(c * state)
    return torch.stack(out)
def lti_convolution(x, a, b, c):
    kernel = c * b * a ** torch.arange(len(x), dtype=x.dtype)
    return torch.stack([(x[:t + 1].flip(0) * kernel[:t + 1]).sum() for t in range(len(x))])
x = torch.tensor([1., -2., 3., .5])
scan = lti_scan(x, torch.tensor(.8), torch.tensor(.4), torch.tensor(1.2))
conv = lti_convolution(x, torch.tensor(.8), torch.tensor(.4), torch.tensor(1.2))
assert torch.allclose(scan, conv, atol=1e-6)
```

训练可批量计算 convolution，decode 可递推维护 state；这份 duality 是 SSM 系统价值的核心。

## 固定 dynamics 的局限

若 $\bar A,\bar B,C$ 对所有输入固定，模型对不同 token 使用同一种遗忘与写入规则。离散内容任务往往需要选择：分隔符可能要求 reset，关键词需要长期保留，填充和普通 token 应被忽略。attention 通过 query–key 匹配天然提供这种内容选择，固定 LTI SSM 较难表达。

这不是说 S4 没有非线性。完整 block 会在通道混合、门控和层堆叠中加入非线性；问题在于单个 state transition 对当前内容的适应性有限。

## Mamba：参数由输入生成

[Mamba](https://arxiv.org/abs/2312.00752)让 $\Delta,B,C$ 依赖输入：

$$
h_t=\bar A(\Delta_t)h_{t-1}+\bar B(\Delta_t,B_t)x_t,
\qquad y_t=C_t h_t+D x_t.
$$

$\Delta_t$ 可以控制当前位置更新多快，$B_t$ 决定怎样写入，$C_t$ 决定怎样读出。一个最小选择性 scan 如下：

```python
def selective_scan(x, a, b, c, state=None, reset=None):
    state = torch.zeros(x.size(-1), dtype=x.dtype) if state is None else state
    out = []
    reset = torch.zeros(len(x), dtype=torch.bool) if reset is None else reset
    for t in range(len(x)):
        state = torch.where(reset[t], torch.zeros_like(state), state)
        state = a[t] * state + b[t] * x[t]
        out.append(c[t] * state)
    return torch.stack(out), state
torch.manual_seed(0)
x = torch.randn(6, 3)
a = torch.sigmoid(torch.randn(6, 3))
b, c = torch.randn(6, 3), torch.randn(6, 3)
full, final = selective_scan(x, a, b, c)
left, state = selective_scan(x[:3], a[:3], b[:3], c[:3])
right, state = selective_scan(x[3:], a[3:], b[3:], c[3:], state)
assert torch.allclose(torch.cat([left, right]), full)
assert torch.allclose(state, final)
reset_out, _ = selective_scan(x, a, b, c, reset=torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.bool))
fresh, _ = selective_scan(x[3:], a[3:], b[3:], c[3:])
assert torch.allclose(reset_out[3:], fresh)
```

chunk equivalence 和 reset 是服务系统必须保持的状态语义。若 chunk 边界悄悄清零，长序列训练与推理就不再是同一个模型。

## Selective scan 为什么是必要的系统工作

输入依赖后，固定 convolution kernel 不再成立。逐 token Python 循环又无法充分利用 GPU。Mamba 的 selective scan 需要在片上状态、并行 prefix scan、重计算和融合 kernel 之间权衡，避免把整条状态轨迹写回 HBM。

所以 Mamba 的贡献不能只写成一组方程：selection mechanism 与 hardware-aware scan 一起决定实际速度。渐进 $O(T)$ 不保证任意长度、batch 和设备上都快于高度优化的 FlashAttention。

## Mamba-2 与 SSD

[Mamba-2](https://arxiv.org/abs/2405.21060)通过 structured state space duality，把一类 SSM 映射到半可分序列矩阵，并设计更适合大矩阵乘的 block 算法。它说明 attention 与 SSM 在代数结构上有交汇区，不意味着两类模型在状态容量、softmax 归一化和任意参数下等价。

## 证据与后续边界

- S4 首次预印本在 2021 年，发表于 ICLR 2022；
- Mamba 首次预印本在 2023 年，页面不把它倒写成当年的已确认会议论文；
- Mamba-2 / SSD 发表于 ICML 2024；
- [S4 官方实现](https://github.com/state-spaces/s4)与 [Mamba 官方实现](https://github.com/state-spaces/mamba)分别对应不同代码路径和 kernel 依赖。

比较 SSM、attention 与 hybrid 时，应固定参数、训练 token、上下文、精度、batch、prefill/decode 形态和真实硬件。完整谱系见[从显式寻址到有限状态](../lineages/linear-time-sequence-models.md)，机制与更多 delta-rule 实现见[状态空间与线性注意力](../../architecture/state-space-linear-attention.md)和[递推与记忆](../../practice/sequence-models.md)。
