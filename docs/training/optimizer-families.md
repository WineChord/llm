# 优化器家族

优化器把梯度转换为参数更新。大模型配方不能只写一个名字：参数分组、更新尺度、weight decay、精度、分片与通信共同决定实际算法。

## SGD 与动量

最基本的更新为

$$
\theta_{t+1}=\theta_t-\eta g_t.
$$

动量维护平滑方向：

$$
m_t=\beta m_{t-1}+(1-\beta)g_t,
\qquad
\theta_{t+1}=\theta_t-\eta m_t.
$$

它的状态较小，但不同参数方向共享相同的标量学习率，对尺度差异很大的 Transformer 参数较难直接调优。

## AdamW

Adam 维护一阶与二阶矩：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

经 bias correction 后，[AdamW](https://arxiv.org/abs/1711.05101)把 weight decay 与自适应更新解耦：

$$
\theta_{t+1}
=(1-\eta\lambda)\theta_t
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

常见实现不对 bias、norm scale 或 embedding 的某些参数做 decay。若参数分组规则不同，即使超参数同名，更新也不相同。

## Muon

Muon 一类方法对二维矩阵参数的动量更新做近似正交化，使不同奇异方向的更新尺度更均衡。抽象地说，先得到矩阵更新 $M$，再近似

$$
O\approx UV^\top,
\qquad
M=U\Sigma V^\top.
$$

实际实现通常不显式做完整 SVD，而使用 Newton–Schulz 迭代等低成本近似。[Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)强调 weight decay 与按参数调整更新尺度对扩展到更大训练的重要性。

Muon 通常只应用于隐藏层的二维矩阵；embedding、norm、bias 和标量参数仍使用 AdamW 或其他优化器。因此“使用 Muon”实际上是混合优化器与参数路由规则。

## 更新尺度

观察相对更新量比只看梯度范数更直接：

$$
\rho_l=
\frac{\lVert\Delta\theta_l\rVert_F}
{\lVert\theta_l\rVert_F+\epsilon}.
$$

不同层的 $\rho_l$ 若相差数个数量级，可能意味着参数化、学习率、正交化 scale 或分组异常。还应按 attention、MLP、embedding、norm 和 router 分组观察。

## Gradient clipping

全局范数裁剪为

$$
g\leftarrow
g\cdot\min\left(
1,\frac{c}{\lVert g\rVert_2+\epsilon}
\right).
$$

分布式实现必须先得到正确的全局范数。按 rank 局部裁剪会改变算法；ZeRO/FSDP 下参数已分片，更需要由框架统一归约。裁剪可以限制偶发尖峰，不能修复错误数据、loss mask 或 overflow。

## 学习率日程

常见配方由 warmup、峰值和 decay 组成。比较两个优化器时，要区分：

- 相同峰值学习率；
- 各自调优后的学习率；
- 相同训练 token 或相同 wall-clock；
- 相同 batch 与梯度噪声尺度；
- 相同 weight decay 和参数分组。

只给新优化器更多调参预算，结论会混入搜索偏差。

## 状态、显存与分片

AdamW 通常为每个参数保存两个 FP32 moment，某些实现还保留 FP32 master weight。Muon 对矩阵保存动量，并增加正交化临时 buffer。优化器状态可由 ZeRO stage 1/FSDP 分片；更新计算和分片通信仍应计入 step time。

低精度 optimizer state 能节省内存，但量化误差可能集中在小梯度或稀有参数上。应同时评估训练 loss、下游质量、恢复一致性与异常尖峰。

## 可复现清单

```text
optimizer implementation and version
parameter-group selection
learning rate, betas or momentum
epsilon, weight decay and exclusions
update scaling / orthogonalization iterations
warmup and decay in tokens
gradient clipping and accumulation
state/master-weight precision
distributed sharding and communication
resume conversion rules
```

混合精度与故障诊断见[优化与稳定性](optimization.md)，状态分片见[集合通信与分片](../systems/collectives-sharding.md)。

## Reference {#reference}

- [On the Importance of Initialization and Momentum in Deep Learning](https://proceedings.mlr.press/v28/sutskever13.html)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
