# 从显式寻址到有限状态

Self-attention 的魅力在于每个 token 都能按内容直接寻找历史位置；代价是长度为 $T$ 时需要形成或隐式处理 $T^2$ 个 query–key 关系。线性注意力和状态空间模型都试图把历史压进固定或缓慢增长的状态，却从不同方向出发：前者重排 attention 的计算，后者从动态系统定义递推。二者后来在 fast weights 与半可分矩阵处相遇，但不能被简化成同一个算法。

## 线性注意力：先改变运算顺序

若相似度可分解为

$$
\operatorname{sim}(q,k)=\phi(q)^\top\phi(k),
$$

则

$$
y_t=
\frac{\phi(q_t)^\top\sum_{j\le t}\phi(k_j)v_j^\top}
{\phi(q_t)^\top\sum_{j\le t}\phi(k_j)+\varepsilon}.
$$

定义状态

$$
S_t=S_{t-1}+\phi(k_t)v_t^\top,\qquad
z_t=z_{t-1}+\phi(k_t),
$$

即可逐 token 递推。[Linear Transformer](https://proceedings.mlr.press/v119/katharopoulos20a.html) 展示了这种结合律带来的 recurrent form；[Performer](https://arxiv.org/abs/2009.14794) 则用随机特征近似 softmax kernel。

两者的边界不同：前者选择一个可精确重排的 feature kernel，通常不再等同 softmax attention；后者尝试近似 softmax，但误差受特征数和随机性影响。“线性 attention”并不自动意味着“与原 attention 完全等价”。

## Fast weights：状态其实是一块可写记忆

将 $S_t$ 看成 key 到 value 的线性映射，线性注意力就像在序列中持续写入 fast weight matrix。[Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html) 沿这条视角解释递推 attention。简单累加写入会产生冲突：相似 key 的 value 互相污染。

Delta rule 用当前预测误差修正写入：

$$
S_t=S_{t-1}
+\beta_t k_t\left(v_t-S_{t-1}^{\top}k_t\right)^\top.
$$

它不增加状态大小，却让写入考虑旧记忆已经预测了什么。有限状态仍然存在容量上限；长程精确 recall 不是仅靠更新规则就能无限保持。

## SSM：从连续动态系统获得递推

状态空间模型从

$$
\dot h(t)=Ah(t)+Bx(t),\qquad y(t)=Ch(t)+Dx(t)
$$

出发，离散化后得到

$$
h_t=\bar Ah_{t-1}+\bar Bx_t,\qquad
y_t=Ch_t+Dx_t.
$$

若参数不随输入变化，这是线性时不变系统，可以在 recurrence 与 convolution 两种形式之间切换。训练时卷积并行、推理时递推常数状态，构成 SSM 的重要吸引力。

[HiPPO](https://proceedings.neurips.cc/paper/2020/hash/102f0bb6efb3a6128a3c750dd16729be-Abstract.html) 把在线历史压缩表述为投影问题；[S4](https://arxiv.org/abs/2111.00396) 通过结构化状态矩阵与高效 kernel，使 SSM 能在长序列上高效并行训练并取得有竞争力的结果。细节见 [S4 到 Mamba 深读](../works/s4-mamba.md)。

## Mamba：让状态更新依赖输入

固定 SSM 对所有 token 使用同一动态，难以像 attention 一样按内容选择“记住什么、忽略什么”。[Mamba](https://arxiv.org/abs/2312.00752) 让离散步长 $\Delta$、输入映射 $B$ 和读出 $C$ 依赖当前输入，使 selection 进入递推：

$$
h_t=\bar A(x_t)h_{t-1}+\bar B(x_t)x_t,\qquad
y_t=C(x_t)h_t+D x_t.
$$

输入依赖破坏了固定卷积 kernel，需要 selective scan 在硬件上并行组织递推。这是算法与 kernel 共同完成的转折：模型表达力提升的同时，简单 FFT convolution 不再直接适用。

[Mamba-2 / SSD](https://arxiv.org/abs/2405.21060) 从 structured state space duality 出发，展示某类 SSM 计算与半可分矩阵、attention 式 block 算法之间的联系。它不是“所有 Transformer 都等于 SSM”，而是为一组受结构约束的序列变换建立共同代数。

## 线性复杂度为何仍可能不快

$O(T)$ 只描述长度渐进量级。真实速度还取决于：

- 状态维度与 expand ratio；
- scan 是否融合，是否频繁读写 HBM；
- batch、序列长度与硬件并行度；
- chunk state 的保存、reset 与跨请求复用；
- 混合 attention 层仍产生的 KV cache。

短序列或高并行训练中，成熟 attention kernel 可能更快；需要精确任意位置 recall 时，有限状态也可能需要更大维度或混合 attention。成本比较见[性能模型](../../systems/performance-model.md)与 [Attention Kernel](../../systems/attention-kernels.md)。

## 两条路线在哪里汇合

线性注意力强调 query–key 形式怎样折叠成状态，SSM 强调状态转移怎样产生序列算子。它们在以下对象上相遇：

```text
finite recurrent state
associative / block scan
state update as memory write
semiseparable sequence matrix
train-parallel and decode-recurrent duality
```

这提供了混合设计空间，也要求更严格的比较：state size、recall、训练稳定性、prefill 速度、decode state、chunk equivalence 与 wall-clock 缺一不可。最小实现见[递推与记忆](../../practice/sequence-models.md)，完整机制边界见[状态空间与线性注意力](../../architecture/state-space-linear-attention.md)。

## Reference {#reference}

- [Linear Transformer](https://proceedings.mlr.press/v119/katharopoulos20a.html)
- [Performer](https://arxiv.org/abs/2009.14794)
- [Fast Weight Programmers](https://proceedings.mlr.press/v139/schlag21a.html)
- [HiPPO](https://proceedings.neurips.cc/paper/2020/hash/102f0bb6efb3a6128a3c750dd16729be-Abstract.html)
- [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Transformers are SSMs](https://arxiv.org/abs/2405.21060)
