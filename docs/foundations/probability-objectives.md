# 概率、损失与梯度

语言模型最终输出的是词表上的 logits。理解 softmax、交叉熵和梯度，不只是为了会写公式：数值稳定、mask、归一化分母和 label shift 的任何错误，都会直接改变训练目标。

## 从 logits 到概率

设某个位置的 logits 为 $z\in\mathbb{R}^{V}$，softmax 定义为

$$
p_i=\frac{\exp(z_i)}{\sum_{j=1}^{V}\exp(z_j)}.
$$

softmax 对整体平移不敏感：

$$
\operatorname{softmax}(z)=\operatorname{softmax}(z-c).
$$

因此实现时取 $c=\max_j z_j$，让最大的指数项为 $1$，避免直接计算大正数的指数。对应的 log-sum-exp 为

$$
\operatorname{LSE}(z)
=m+\log\sum_j\exp(z_j-m),
\qquad m=\max_j z_j.
$$

log-softmax 可稳定写成

$$
\log p_i=z_i-\operatorname{LSE}(z).
$$

先做 softmax、再取对数会丢失小概率项的有效精度；训练实现应直接使用融合的 log-softmax 或 cross-entropy。

## 交叉熵

真实类别为 $y$ 时，单位置负对数似然为

$$
\ell(z,y)=-\log p_y
=-z_y+\operatorname{LSE}(z).
$$

对 one-hot 标签，logit 梯度具有简洁形式：

$$
\frac{\partial\ell}{\partial z_i}=p_i-\mathbf 1[i=y].
$$

这个式子解释了两件事：正确类别概率不足时梯度为负，更新会抬高对应 logit；其他类别按当前概率受到正梯度，概率越高，压低得越多。

带 label smoothing 的目标分布可写为

$$
q_i=(1-\varepsilon)\mathbf 1[i=y]+\frac{\varepsilon}{V},
\qquad
\ell=-\sum_i q_i\log p_i.
$$

它改变的不只是“防过拟合”，还改变校准、梯度幅度和稀有 token 的学习信号。语言模型是否使用、在哪些阶段使用，应以具体配方为准。

## 序列损失

decoder-only 模型用位置 $t$ 的隐藏状态预测 $x_{t+1}$：

$$
\mathcal L
=-\frac{1}{\sum_t m_t}
\sum_t m_t\log p_\theta(x_{t+1}\mid x_{\le t}),
$$

其中 $m_t\in\{0,1\}$ 是 loss mask。分母必须是有效目标 token 数，而不是包含 padding 的张量长度。SFT 中还可能只保留 assistant response；packed sequence 则要同时处理文档边界和 label 边界。

常见的错位包括：

- 输入和标签没有平移，模型被训练成复制当前 token；
- BOS、EOS 或 role token 是否计入损失与模板定义不一致；
- padding 已从 attention 中屏蔽，却仍进入 loss；
- 多条样本打包后，后一条样本读取了前一条的 token；
- 梯度累积时先对每个 microbatch 求均值，再直接相加，导致不同有效长度的样本权重失真。

序列构造的完整契约见[序列构造与打包](../data/sequence-construction.md)。

## 困惑度

若有效 token 的平均负对数似然为 $\bar{\mathcal L}$，困惑度为

$$
\operatorname{PPL}=\exp(\bar{\mathcal L}).
$$

它可解释为模型在当前 tokenization 与测试分布上的平均不确定性尺度，但不能跨 tokenizer 直接比较。滑动窗口评测还要说明上下文重叠和每个 token 被计分几次，否则同一个 checkpoint 也可能得到不同结果。

## 自动微分与梯度检查

自动微分沿计算图应用链式法则，不会替代目标审计。最小检查顺序是：

1. 用很小的 logits 手算 softmax 与交叉熵；
2. 比较解析梯度 $p-y$；
3. 用中心差分

$$
\frac{\partial f}{\partial x_i}
\approx
\frac{f(x+\epsilon e_i)-f(x-\epsilon e_i)}{2\epsilon}
$$

检查自定义算子；
4. 再加入 mixed precision、mask、分布式归约和融合 kernel。

有限差分的 $\epsilon$ 过小会被浮点舍入淹没，过大又偏离局部线性；梯度检查通常用 FP64 小张量，并关闭随机算子。

## 数值审计清单

- softmax 是否减去行最大值，归约维度是否正确；
- masked row 是否可能全部为 $-\infty$；
- loss 的分子与有效 token 分母是否在分布式 rank 间一致归约；
- logits、log-probability 和 probability 是否被混用；
- reference、behavior 与 current policy 的 log-prob 是否使用同一 tokenizer、模板和 action mask；
- 出现 NaN 时是否找到第一个非有限张量，而不是只降低学习率。

概率目标连接[语言建模](language-modeling.md)、[监督微调](../training/supervised-finetuning.md)和[策略优化](../agentic-rl/math-algorithms.md)。
