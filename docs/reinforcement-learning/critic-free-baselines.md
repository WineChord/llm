# 无 learned critic 的策略梯度：REINFORCE、ReMax、RLOO 与 GRPO

“无 critic”不等于“无 baseline”。语言模型在线 RL 中，response reward、同 prompt 多采样和可验证结果提供了几种不训练 value model 的方差控制方式。它们省去 critic 参数与 value pretraining，却把成本转移到额外 rollout、组同步和 reward 分布。

## 从 REINFORCE 开始

对 prompt $x$、response $y\sim\pi_\theta$ 与 sequence reward $R(x,y)$：

$$
\nabla_\theta J
=\mathbb E
\left[
(R-b(x))
\nabla_\theta\log\pi_\theta(y\mid x)
\right].
$$

若 baseline $b(x)$ 不依赖当前采样 action，它不会改变期望，只改变方差。语言模型中

$$
\log\pi_\theta(y\mid x)
=\sum_t\log\pi_\theta(y_t\mid x,y_{<t}),
$$

因此同一个 sequence advantage 可以乘到所有 action token。token sum 与 token mean 对长度的权重不同，必须明确。

## Self-including mean 不是 action-independent baseline

不同 prompt 难度差异很大。用全 batch reward mean 会让容易题与难题共享 baseline：

$$
A_i=R_i-\frac1N\sum_jR_j.
$$

这里的均值包含 $R_i$，因而依赖当前样本的 action，不能直接套用“baseline 不改变期望”的结论。把自身拆出来可得

$$
R_i-\frac1N\sum_jR_j
=\frac{N-1}{N}
\left(
R_i-\frac1{N-1}\sum_{j\ne i}R_j
\right).
$$

在同一 prompt、候选独立同分布且使用共同 reduction 的特殊情形下，它是 leave-one-out estimator 的固定缩放；混合不同 prompt 时，还会额外混入 prompt difficulty。更合适的 baseline 通常是同 prompt 的 leave-one-out 候选、与当前动作独立的 greedy response，或 learned value。

## ReMax：用 greedy response 作基线

[ReMax](https://arxiv.org/abs/2310.10505)对同一 prompt 同时生成 sampled response 与 greedy response：

$$
A=
R(y_{\mathrm{sample}})
-R(y_{\mathrm{greedy}}).
$$

优点：

- 不训练 critic；
- baseline 与 prompt 自适应；
- 只额外需要一个确定性候选。

边界：

- greedy rollout 仍消耗推理；
- stochastic environment 中两次 rollout 的初始状态需可比；
- greedy reward 很差时 baseline 降方差效果有限；
- 多轮环境的 greedy 分支可能访问完全不同状态。

## RLOO：leave-one-out baseline {#rloo}

同一 prompt 采样 $K\ge2$ 个 response，reward 为 $R_1,\ldots,R_K$。对第 $i$ 个样本：

$$
A_i
=R_i-\frac{1}{K-1}\sum_{j\ne i}R_j.
$$

leave-one-out baseline 不包含当前 reward。若改用组均值：

$$
R_i-\frac1K\sum_jR_j
=\frac{K-1}{K}A_i^{\mathrm{RLOO}},
$$

每个样本的梯度都按 $(K-1)/K$ 缩小。固定 $K$ 时两者方向相同，但 self-including mean 不是 action-independent baseline，得到的是 RLOO estimator 的固定缩放，而不是原始 estimator 本身；跨实验比较还需统一 learning rate、group size 和 reduction。

[Back to Basics](https://arxiv.org/abs/2402.14740)在其 RLHF 设置中系统比较了 REINFORCE-style 方法与 PPO；结论依赖模型、reward、数据和预算，不能推广成“critic 永远无用”。

## GRPO：组内中心化与标准化

[DeepSeekMath](https://arxiv.org/abs/2402.03300)中的常见形式为

$$
\widehat A_i
=
\frac{R_i-\bar R}
{\operatorname{std}(R)+\varepsilon}.
$$

分子中的 $\bar R$ 包含当前 reward，但它与 RLOO 有上面的固定缩放关系；分母却是依赖整组 reward 的随机量，也依赖当前 action。于是 GRPO 标准化不能仅凭 baseline theorem 宣称对原始 expected-reward policy gradient 无偏：它会按组内 reward dispersion 重新加权 prompt，且这种权重随采样组变化。更准确的说法是 **group-relative normalized estimator**，而不是 action-independent baseline。

本页保留 GRPO 与其他 baseline 的家族关系；原始 token objective、population/sample std、process supervision、response-length weighting、Dr. GRPO 与 dynamic sampling 的完整推导见[GRPO：组相对优势、PPO 更新与长度权重](grpo.md)。

组标准差使不同 prompt 的 advantage 尺度更接近，也产生新的退化：

- 全组 reward 相同，分子全为零；
- $K=1$ 无法形成相对信号；
- 极小 std 使 $\varepsilon$ 主导尺度；
- 离散 reward 下 all-correct/all-wrong 组没有梯度；
- 同组要等最慢 rollout，长尾任务形成 barrier。

动态重采样“有信号组”会改变训练任务分布。报告优化 token 时还要报告为筛选付出的全部 rollout token。

## 分母决定谁主导梯度

设第 $i$ 个 response 长度为 $L_i$。至少有三种 reduction：

### Sequence sum

$$
\mathcal L
=-\frac1K\sum_i
A_i\sum_{t=1}^{L_i}\log\pi(y_{i,t}).
$$

长 response 有更多 token 梯度项。

### Per-response token mean

$$
\mathcal L
=-\frac1K\sum_i
A_i\frac1{L_i}\sum_t\log\pi(y_{i,t}).
$$

每条 response 等权，但改变长度偏好。

### Global token mean

$$
\mathcal L
=-\frac{
\sum_i\sum_t A_i\log\pi(y_{i,t})
}{
\sum_iL_i
}.
$$

每个 token 等权。三者不是实现细节，应与论文公式、分布式 reduction 和指标保持一致。

## 变体应按修正项理解

下列方法均为 2025 年公开的较新工作，证据主要来自论文给定的模型、任务与预算；它们提供的是可检验的目标修正，不是跨场景定论。

| 工作 | 主要修正 | 不应误写成 |
| --- | --- | --- |
| [Dr. GRPO](grpo.md#group-std) | 移除 group std 与按 response length 的归一化 | 一套完整通用 trainer |
| [DAPO](../landscape/works/dapo.md) | asymmetric clip、动态采样、token loss、长度处理等 recipe | 只等于 dynamic sampling |
| [GSPO](ratio-clipping-gating.md#gspo) | length-normalized sequence ratio 与 sequence-level clipping | 任意多轮 episode 都自然是一个 sequence |
| [SAPO](ratio-clipping-gating.md#sapo) | sequence-coherent、token-adaptive 的平滑 ratio gate | 自动消除 off-policy 偏差 |

这些工作常同时改变 sampling、filter、clipping 和 denominator。比较时必须逐项消融，并固定生成预算。

## Critic-free 的适用边界

更适合：

- response/episode reward 可靠；
- 同 prompt 可以并行产生多个候选；
- episode 长度相近；
- group barrier 可接受；
- value model 成本或冷启动不可接受。

更需要 critic：

- 每个 prompt 只有一条 rollout；
- 多轮状态需要细粒度 advantage；
- 轨迹很长且 reward 稀疏；
- 需要 bootstrap 或在线 value；
- 组内 reward 经常无方差。

这不是二选一。可以用 critic 提供 token/turn advantage，同时用 prompt group 做额外中心化；但 estimator 的偏差和 denominator 必须重新定义。

## 诊断

1. 验证 RLOO baseline 不包含自身。
2. 全同 reward 时 GRPO advantage 精确为零。
3. 报告 all-correct、all-wrong、mixed group 比例。
4. 同时报告 sampled 与 accepted rollout token。
5. 比较 sequence sum、response mean 与 token mean。
6. 按 group size、长度和 reward std 分层。
7. 环境/基础设施错误先排除，不把它们当普通零 reward。
8. 与 SFT、rejection sampling 和 PPO 在相同预算下比较。

可执行实现见[RLOO 与 GRPO 的组内信号](../practice/llm-policy-optimization.md#rloo-grpo)，与 learned critic 的比较见[Actor–Critic](actor-critic.md)。

## Reference {#reference}

- Williams, [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696)
- Li et al., [ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models](https://arxiv.org/abs/2310.10505)
- Ahmadian et al., [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback](https://arxiv.org/abs/2402.14740)
- Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- Liu et al., [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
- Yu et al., [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- Zheng et al., [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
- Gao et al., [Soft Adaptive Policy Optimization](https://arxiv.org/abs/2511.20347)
