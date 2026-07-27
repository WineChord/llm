# 状态空间与线性注意力

标准全注意力保留每个历史 token 的 K/V，并让新 query 直接寻址全部历史。状态空间模型（State Space Model, SSM）与线性注意力改用固定或受控大小的状态压缩历史，目标是在序列变长时保持近似线性的计算与常数级 decode state。

代价是明确的：历史被压入有限状态后，不再拥有无损、任意位置的精确内容寻址。理解这一信息瓶颈，比只记住复杂度更重要。

[线性时间序列模型](../landscape/lineages/linear-time-sequence-models.md)从 RNN、长卷积和结构化状态空间梳理到选择性递推；[LSTM](../landscape/works/lstm.md)给出门控有限状态的早期转折，[S4、Mamba 与 Mamba-2](../landscape/works/s4-mamba.md)进一步给出连续系统离散化、scan 等价和最小 selective recurrence。

## 统一递推

最一般的离散递推可写为

$$
h_t=A_th_{t-1}+B_tx_t,
\qquad
y_t=C_th_t+D_tx_t.
$$

若 $h_t\in\mathbb R^n$，逐 token 推理只需保留当前状态。固定 $A,B,C$ 对输入内容使用同一动力学；让它们依赖 $x_t$ 可实现选择性写入、遗忘和读取。

## 从连续 SSM 到离散递推

连续线性系统为

$$
\frac{dh(t)}{dt}=Ah(t)+Bx(t),
\qquad
y(t)=Ch(t)+Dx(t).
$$

以步长 $\Delta$ 离散化后：

$$
\bar A=e^{\Delta A},
\qquad
\bar B
=
(\Delta A)^{-1}(e^{\Delta A}-I)\Delta B,
$$

得到

$$
h_t=\bar Ah_{t-1}+\bar Bx_t.
$$

[S4](https://arxiv.org/abs/2111.00396)利用结构化状态矩阵和卷积/递推双重视角，使长序列训练与流式推理能够共享同一模型。实际实现通常不会显式计算一般矩阵指数，而会利用对角或特殊结构。

## 卷积与递推的等价

在参数不随输入变化时，展开递推：

$$
y_t
=
\sum_{i=1}^{t}
C\bar A^{t-i}\bar Bx_i
+Dx_t.
$$

令

$$
K_j=C\bar A^j\bar B,
$$

即可写成因果卷积。训练可以并行计算卷积，推理使用 recurrent state；两条路径应在相同精度下数值对齐。

## Selective SSM

[Mamba](https://arxiv.org/abs/2312.00752)让离散化步长和输入、输出映射依赖当前 token：

$$
\Delta_t=\operatorname{softplus}(W_\Delta x_t),
\qquad
B_t=W_Bx_t,
\qquad
C_t=W_Cx_t.
$$

输入相关离散参数下，状态更新为

$$
h_t=\bar A_th_{t-1}+\bar B_tx_t,
\qquad
y_t=C_th_t.
$$

输入相关参数使模型能按内容控制“保留还是遗忘”。同时，它破坏了固定卷积核，因此训练依赖硬件感知的 parallel scan 或 chunkwise 算法。[Mamba 官方实现](https://github.com/state-spaces/mamba)是核对算子语义、状态 shape 和增量接口的首选参考。

## State Space Duality

[Mamba-2](https://arxiv.org/abs/2405.21060)把一类结构化 SSM 与半可分矩阵联系起来。展开后，位置 $i$ 对位置 $j$ 的线性映射可写成

$$
M_{j,i}
=
C_j^\top
\left(
A_jA_{j-1}\cdots A_{i+1}
\right)
B_i,
\qquad j\ge i.
$$

这使序列递推可以按 block 分解，并复用矩阵乘法硬件。它提供 attention 与 SSM 之间的代数桥梁，但不意味着两者具有相同的信息容量或 softmax 归一化。

## 线性注意力

若核函数满足

$$
\operatorname{sim}(q,k)=\phi(q)^\top\phi(k),
$$

则

$$
\sum_{i\le t}
\operatorname{sim}(q_t,k_i)v_i
=
\phi(q_t)^\top
\left(
\sum_{i\le t}\phi(k_i)v_i^\top
\right).
$$

定义状态

$$
S_t=S_{t-1}+\phi(k_t)v_t^\top,
\qquad
z_t=z_{t-1}+\phi(k_t),
$$

归一化输出为

$$
y_t
=
\frac{\phi(q_t)^\top S_t}
{\phi(q_t)^\top z_t+\varepsilon}.
$$

计算顺序从显式 $T\times T$ score 矩阵改成固定状态累积。若 feature dimension 很大，状态 $S_t$ 的二次通道成本仍可能显著。

## Retention、衰减与 Delta Rule

[RetNet](https://arxiv.org/abs/2307.08621)将带衰减的 retention 写成 parallel、recurrent 与 chunkwise 等价形式。一般状态可写为

$$
S_t=\gamma_t\odot S_{t-1}+k_tv_t^\top.
$$

简单累加会让新值与旧值叠加而不纠错。Delta rule 用预测误差更新 fast weight：

$$
S_t
=
S_{t-1}
+
\beta_t
k_t
\left(v_t-S_{t-1}^\top k_t\right)^\top.
$$

这里 $S_{t-1}\in\mathbb R^{d_k\times d_v}$，所以 $S_{t-1}^\top k_t\in\mathbb R^{d_v}$ 是旧状态对当前 key 的预测，$k_t(v_t-S_{t-1}^\top k_t)^\top$ 是 $d_k\times d_v$ 的残差写入。它改善有限状态的关联记忆，但仍受到状态秩、key 冲突和数值稳定性的约束。

## 其他稳定比较对象

- [RWKV](https://arxiv.org/abs/2305.13048)：以时间混合和通道混合实现 Transformer 风格并行训练、RNN 风格推理。
- [Hyena](https://arxiv.org/abs/2302.10866)：以长卷积和数据控制门替代标准 attention。
- [Griffin](https://arxiv.org/abs/2402.19427)：将 gated linear recurrence 与 local attention 混合。
- [xLSTM](https://arxiv.org/abs/2405.04517)：重新设计 LSTM 的门控、记忆和并行形式。

这些方案应按状态大小、精确寻址、并行训练、decode 成本和 kernel 成熟度比较，而不是排列成单向“继任者”时间线。

## Shape 与实现契约

以多 head 状态为例：

$$
x\in\mathbb R^{B\times T\times d},
\quad
S\in\mathbb R^{B\times H\times d_k\times d_v}.
$$

实现需要固定：

1. recurrent state 的 batch、head 与 feature 维；
2. sequence reset 和 padding 位置是否更新状态；
3. prefill 使用 scan、chunk 还是逐 token 路径；
4. decode state 的 dtype；
5. chunk 边界是否携带相同的终态；
6. normalization 的 $\varepsilon$ 与计算精度；
7. bidirectional 训练是否可用于 causal decode；
8. checkpoint 是否保存卷积状态与 recurrent state 的转换约定。

## 复杂度不能脱离状态大小

设状态维为 $n$。对角 SSM 的逐 token 更新可接近 $O(n)$；一般矩阵状态可能达到 $O(n^2)$。线性注意力若 $S\in\mathbb R^{d_k\times d_v}$，每 token 更新为 $O(d_kd_v)$。

因此报告“关于 $T$ 线性”时，还必须报告：

- 状态维与 head 数；
- training scan 的额外存储；
- kernel 是否融合；
- 短序列、长 prefill 与单 token decode 的 wall-clock；
- 与 full attention 相同质量下需要的宽度和层数。

## 失效模式

- **有限状态瓶颈**：多个相似 key 覆盖彼此，精确复制失败。
- **Recall gap**：语言建模 loss 接近，但 associative recall 明显落后。
- **路径不一致**：parallel、chunkwise、recurrent 输出随长度漂移。
- **状态污染**：padding、请求边界或 batch 重排后未 reset。
- **低精度累积**：长序列递推在 FP16/BF16 中漂移或溢出。
- **局部速度错觉**：渐近更优，但短上下文 kernel 启动与状态操作更慢。
- **双向泄漏**：训练使用未来信息，部署却按 causal state 推理。
- **生态断层**：量化、并行、checkpoint 与服务 runtime 缺少对应实现。

[Zoology](https://arxiv.org/abs/2312.04927)用受控任务展示了不同序列 mixer 在回忆能力上的结构性差异，适合作为“语言建模平均分数之外”的诊断依据。

## 验证矩阵

| 验证 | 方法 |
| --- | --- |
| 代数等价 | 显式下三角矩阵、parallel scan、逐 token 递推对齐 |
| Chunk 边界 | 随机切分同一序列，终态与输出不变 |
| Reset | 拼 batch 与逐样本运行结果相同 |
| 精度 | FP32 参考对照 BF16/FP16，扫描长度逐级扩大 |
| Recall | copy、selective copy、MQAR、冲突 key |
| 长度 | 训练内、训练外、极长流式分别测试 |
| 系统 | prefill、decode、峰值内存、state bytes、吞吐 |
| 质量 | 相同数据、token、激活参数与计算预算比较 |

## 前沿观察：Kimi Linear

[Kimi Linear](https://arxiv.org/abs/2510.26692)提出 Kimi Delta Attention，并公开了[官方实现](https://github.com/MoonshotAI/Kimi-Linear)。论文报告在其控制配方中以 KDA 与 MLA 混合，获得长上下文缓存和吞吐收益。

适合确认的事实是：论文、kernel、模型与实现接口已经公开，可直接研究 delta rule、chunkwise 算法和混合层。尚不能由此推出：

- 任意数据与规模下线性注意力都优于 full attention；
- 论文中的最长上下文质量可外推到其他 checkpoint；
- 理论或单套硬件结果等于所有 runtime 的端到端收益。

因此该路线应保留在观察层，等独立复现、不同规模和多种服务栈证据积累后再调整定位。

最小 selective scan、delta rule 与路径等价实验见[序列模型手撕实现](../practice/sequence-models.md)，与精确注意力的缓存比较见[KV Cache](../inference/kv-cache.md)。
