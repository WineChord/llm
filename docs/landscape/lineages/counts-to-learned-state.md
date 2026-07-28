# 从计数到可学习状态

语言模型的早期演化不是一条按年份首尾相接的直线。统计估计、连续分布式表示和循环状态曾长期并行发展：Elman 的循环网络与 LSTM 分别发表于 1990 和 1997 年，都早于 2003 年的前馈神经概率语言模型。更有解释力的读法，是观察模型怎样在“可靠计数、跨上下文共享和长程记忆”之间移动瓶颈。

## 有限历史让概率可以估计

链式法则本身不限制上下文：

$$
p(w_{1:T})=\prod_{t=1}^{T}p(w_t\mid w_{<t}).
$$

$n$-gram 用有限阶 Markov 假设换取可估计性：

$$
p(w_t\mid w_{<t})
\approx p(w_t\mid w_{t-n+1:t-1}).
$$

最大似然计数对见过的上下文有效，却会给未见组合零概率。[Katz backoff](https://ieeexplore.ieee.org/document/1165125) 对可靠计数折扣，把剩余概率质量交给更短历史：

$$
p_{\mathrm{BO}}(w\mid h)=
\begin{cases}
\dfrac{d(c(h,w))}{c(h)},&c(h,w)>0,\\[6pt]
\alpha(h)p_{\mathrm{BO}}(w\mid \operatorname{suffix}(h)),&c(h,w)=0.
\end{cases}
$$

$\alpha(h)$ 负责重新归一化，而不是任意的回退权重。[Brown 等人的 class-based $n$-gram](https://aclanthology.org/J92-4003/) 进一步让词类共享统计。平滑和聚类缓解了稀疏性，但没有移除固定窗口，也不能让语义相近的词自然共享连续参数。

## 连续表示让相似历史共享参数

[Bengio 等人的神经概率语言模型](https://www.jmlr.org/papers/v3/bengio03a.html)把离散词映射到连续向量，再用前馈网络预测下一个词：

$$
e_j=Cw_{t-j},\qquad
h=\tanh\!\left(b+W[e_1;\ldots;e_{n-1}]\right),
$$

$$
p(w_t\mid w_{t-n+1:t-1})=\operatorname{softmax}(Uh+b_o).
$$

相近词和相近历史可以通过 embedding 与隐藏层共享统计强度，组合不再彼此完全独立。代价也随之变化：上下文仍被窗口截断，词表 softmax 昂贵，网络只会处理训练时定义好的固定长度输入。

## 循环状态把窗口变成动态记忆

[Elman 1990](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1) 展示了循环隐藏状态怎样保存序列中的动态结构；[Mikolov 等人 2010](https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.html) 则把简单 RNN 与经典 backoff 语言模型作了有影响力的语言建模比较：

$$
h_t=\phi(W_xe_t+W_hh_{t-1}+b),\qquad
p(w_{t+1}\mid w_{\le t})=\operatorname{softmax}(Uh_t).
$$

模型不再显式截断到 $n-1$ 个词，但“可读取整个历史”和“能可靠保存整个历史”并不是一回事。反向传播包含许多状态 Jacobian 的连乘：

$$
\frac{\partial h_T}{\partial h_t}
=\prod_{k=t+1}^{T}\frac{\partial h_k}{\partial h_{k-1}},
$$

其范数可能迅速消失或爆炸。历史还必须被压进固定维度的 $h_t$，并沿时间步串行更新。

## LSTM 改写梯度路径

[Hochreiter 与 Schmidhuber 1997](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory) 以加性的 cell path 改善长时间间隔上的误差信号。现代常用记号可写为

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t,\qquad
h_t=o_t\odot\tanh(c_t).
$$

这里的 $f_t$ 是后来成为标准组件的 forget gate；它应追溯到 [Gers、Schmidhuber 与 Cummins 2000](https://direct.mit.edu/neco/article/12/10/2451/6415/Learning-to-Forget-Continual-Prediction-with-LSTM)，而不能倒写进 1997 年原始结构。门控让模型学习保留、写入和读出，但没有消除固定状态容量与时间串行。

原始 cell、后来补入的 forget gate 与现代常用方程之间的区别见 [LSTM 深读](../works/lstm.md)。

## 瓶颈怎样继续迁移

这些路线留下了三种持久思想：

- embedding 把离散符号映射为可共享的连续几何；
- recurrent state 把历史压成可增量更新的有限状态；
- gated update 让模型决定哪些信息应保留或覆盖。

encoder–decoder 随后把 recurrent state 用作条件生成接口，却把整个输入压成一个向量；[注意力机制](transduction-to-attention.md)转而保留一张可寻址的状态表。更晚的[状态空间与线性注意力](../../architecture/state-space-linear-attention.md)又回到有限状态，但用结构化动力学、并行 scan 和输入相关选择重新设计它。

概率分解、困惑度和 tokenizer 口径见[语言建模](../../foundations/language-modeling.md)、[概率、损失与梯度](../../foundations/probability-objectives.md)与[分词](../../foundations/tokenization.md)；可执行递推 reference 见[手撕：递推与记忆](../../practice/sequence-models.md)。

## Reference {#reference}

- [Katz backoff](https://ieeexplore.ieee.org/document/1165125)
- [Brown 等人的 class-based $n$-gram](https://aclanthology.org/J92-4003/)
- [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html)
- [Elman 1990](https://onlinelibrary.wiley.com/doi/10.1207/s15516709cog1402_1)
- [Mikolov 等人 2010](https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.html)
- [Hochreiter 与 Schmidhuber 1997](https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory)
- [Gers、Schmidhuber 与 Cummins 2000](https://direct.mit.edu/neco/article/12/10/2451/6415/Learning-to-Forget-Continual-Prediction-with-LSTM)
