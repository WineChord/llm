# LSTM：加性记忆如何穿过时间

循环网络把可变长度历史压进状态，却让梯度反复乘以同一个 transition Jacobian。小于一的方向指数衰减，大于一的方向指数爆炸。1997 年的 [Long Short-Term Memory](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory)试图建立一条近似恒定的误差通道：memory cell 通过加性状态更新跨越时间，门控决定何时写入和读出。

## 它不是在 2003 年神经语言模型之后才出现

RNN dynamic state 与连续分布式词表示是较早并行发展的两条路线。[Elman 1990](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1)研究 recurrent hidden state，[LSTM 1997](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory)处理长时梯度；[Bengio NPLM 2003](https://www.jmlr.org/papers/v3/bengio03a.html)则从固定窗口和连续 embedding 缓解离散 n-gram 稀疏。后来神经语言模型把这些思想汇合。

把历史写成“n-gram → NPLM → RNN → LSTM”会颠倒真实时间，也掩盖两条研究动机。

## 现代 LSTM 方程

今天常见的 cell 写作

$$
\begin{aligned}
i_t&=\sigma(W_i x_t+U_i h_{t-1}+b_i),\\
f_t&=\sigma(W_f x_t+U_f h_{t-1}+b_f),\\
g_t&=\tanh(W_g x_t+U_g h_{t-1}+b_g),\\
o_t&=\sigma(W_o x_t+U_o h_{t-1}+b_o),\\
c_t&=f_t\odot c_{t-1}+i_t\odot g_t,\\
h_t&=o_t\odot\tanh(c_t).
\end{aligned}
$$

必须保留一个历史边界：1997 原始结构并不包含今天惯用的 forget gate；[Learning to Forget](https://direct.mit.edu/neco/article/12/10/2451/6415/Learning-to-Forget-Continual-Prediction-with-LSTM)在 2000 年引入相应机制。现代方程适合教学和实现，但不应倒写为原论文逐项结构。

## 加性路径怎样影响梯度

忽略门值对旧状态的间接依赖，有

$$
\frac{\partial c_T}{\partial c_0}
\approx\prod_{t=1}^{T}f_t.
$$

若 $f_t$ 接近 1，梯度可比普通 tanh recurrence 保留更久；若任务需要忘记，门又能主动缩小旧状态。

```python
import torch
def lstm_state(x, c0, forget, inp, candidate):
    state, out = c0, []
    for t in range(len(x)):
        state = forget[t] * state + inp[t] * torch.tanh(candidate[t] * x[t])
        out.append(state)
    return torch.stack(out)
x = torch.tensor([1., -1., .5, 2.])
c0 = torch.tensor(0.7, requires_grad=True)
forget = torch.tensor([.9, .8, .95, .85])
inp = torch.tensor([.2, .3, .1, .4])
candidate = torch.tensor([1.1, .7, .4, .9])
states = lstm_state(x, c0, forget, inp, candidate)
gradient, = torch.autograd.grad(states[-1], c0)
assert torch.allclose(gradient, forget.prod(), atol=1e-7)
reset = lstm_state(x[2:], torch.tensor(0.), forget[2:], inp[2:], candidate[2:])
assert not torch.allclose(states[2:], reset)
```

这段 reference 只隔离 cell state 的加性记忆，不实现依赖 $x_t,h_{t-1}$ 的门网络。第二个断言强调 state reset 属于数据契约：跨样本错误复用状态会造成隐式泄漏。

## LSTM 解决了什么，又留下什么

LSTM 显著缓解长程梯度问题，却没有消除：

- 时间步串行，训练难以像 self-attention 一样沿序列并行；
- 固定维度 state 仍是信息瓶颈；
- 长程精确寻址需要模型把内容编码进状态，而不是直接访问某个历史位置；
- 门饱和、初始化和 truncation 仍影响有效记忆长度。

Seq2seq 把 encoder 最终状态用作整句摘要后，这个瓶颈变得尤为明显；[Bahdanau attention](seq2seq-and-neural-alignment.md)通过保留所有 encoder states 让 decoder 动态寻址。Transformer 随后进一步移除 recurrence，但以 $T^2$ 关系和 KV cache 换取全局访问。

## 它为何在 SSM 时代重新重要

现代状态空间模型同样研究有限状态怎样吸收历史，只是使用结构化线性 dynamics、卷积 duality 和硬件 scan。[Mamba](s4-mamba.md)又引入输入依赖的选择性更新。它们并不是 LSTM 的直接放大版，却继承了同一个根问题：有限状态应该何时写、何时忘、怎样在训练和推理之间保持一致。

前序历史见[从计数、分布式表示到可学习状态](../lineages/counts-to-learned-state.md)，后续寻址转折见[从固定向量到全局内容寻址](../lineages/transduction-to-attention.md)，现代递推实现见[递推与记忆](../../practice/sequence-models.md)。

## Reference {#reference}

- [Hochreiter 与 Schmidhuber 1997](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory)
- [Elman 1990](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1)
- [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html)
- [Gers、Schmidhuber 与 Cummins 2000](https://direct.mit.edu/neco/article/12/10/2451/6415/Learning-to-Forget-Continual-Prediction-with-LSTM)
