# MDP、POMDP 与回报

强化学习首先是一个序贯决策问题：动作不仅带来即时 reward，还会改变下一时刻的状态与未来能够获得的信息。若这层对象没有定义清楚，后面的 value、advantage、policy gradient 都只是在优化一个含义不明的数字。

## 从一次预测到一条轨迹

监督学习通常在固定数据分布上预测标签；强化学习中的策略会改变数据分布。令时刻 $t$ 的一次转移写成

$$
(s_t,a_t,r_t,s_{t+1}),
$$

其中 reward $r_t$ 属于从 $s_t$ 执行动作 $a_t$ 后到达 $s_{t+1}$ 的这次转移。策略与环境交替生成轨迹：

$$
s_0\sim\rho_0,\qquad
a_t\sim\pi(\cdot\mid s_t),\qquad
(r_t,s_{t+1})\sim P(\cdot,\cdot\mid s_t,a_t).
$$

策略要最大化的不是每一步 reward，而是回报

$$
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k},
\qquad 0\le\gamma\le1.
$$

$\gamma$ 同时决定时间偏好和 Bellman 算子的数学性质。它不应被随意解释成“未来 reward 的可信度”：在 continuing task 中，discount 也可对应随机终止或长期平均目标的一种近似；在有限 episode 中，即使 $\gamma=1$，回报也可能有界。

## MDP：把决策对象写完整

一个 discounted Markov decision process 可写为

$$
\mathcal M=
(\mathcal S,\mathcal A,P,r,\rho_0,\gamma),
$$

其中

$$
P(s'\mid s,a)
=\Pr(s_{t+1}=s'\mid s_t=s,a_t=a),
$$

$$
r(s,a,s')
=\mathbb E[r_t\mid s_t=s,a_t=a,s_{t+1}=s'].
$$

Markov 性要求当前状态足以预测下一步：

$$
\Pr(s_{t+1},r_t\mid s_{0:t},a_{0:t})
=\Pr(s_{t+1},r_t\mid s_t,a_t).
$$

这不是说“环境没有历史”，而是说历史对未来有用的信息已经压入 $s_t$。物理位置、库存、权限、剩余预算和已完成子目标都可能是状态的一部分。若删掉其中任何一项会改变同一动作的未来分布，删减后的表示通常就不再 Markov。

历史上，Markov 过程先研究无控制的随机演化；Bellman 的动态规划把动作和最优性递推加入其中，形成现代 MDP 控制的基础。随后，[价值函数与 Bellman 递推](values-bellman.md)把“长期表现”写成可解的固定点，[Monte Carlo、TD 与控制](prediction-control.md)再把已知模型下的递推变成只依赖样本的学习。

## Policy、trajectory 与 occupancy

随机策略给出动作的条件分布 $\pi(a\mid s)$。在有限 horizon 下，一条轨迹的概率分解为

$$
p_\pi(\tau)
=\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi(a_t\mid s_t)
P(s_{t+1},r_t\mid s_t,a_t).
$$

策略不仅改变动作频率，还通过转移改变以后访问哪些状态。对 $0\le\gamma<1$，归一化的 discounted state occupancy 可写为

$$
d^\pi_\gamma(s)
=(1-\gamma)\sum_{t=0}^{\infty}
\gamma^t\Pr_\pi(s_t=s).
$$

因此，拿旧策略收集的数据更新新策略时，差异不只存在于单步 action probability，也存在于整条 state visitation distribution。这是 [off-policy 校正](off-policy-correction.md)无法仅凭局部 ratio 完全消除分布外缺口的根源。

## Episode、终止与截断

数据边界不等于环境终态。至少要区分：

| 状态 | 含义 | 下一状态是否 bootstrap |
| --- | --- | --- |
| `terminated=True` | MDP 内的成功、失败或吸收终态 | 否 |
| `truncated=True` | 时间、token、交互预算等外部边界 | 通常是 |
| 基础设施错误 | 工具、网络或 worker 没有给出有效环境结果 | 不应自动记成零回报 |

一步 target 因此应写为

$$
y_t=r_t+\gamma(1-d_t)V(s_{t+1}),
\qquad
d_t=\mathbf 1[\text{terminated at }t].
$$

`truncated` 告诉采样器停止记录当前片段，但通常不把 continuation value 置零。若任务定义本身就把时间上限纳入状态，并规定超时为真正终态，那么它才属于 `terminated`；关键是环境语义，而不是 API 字段的名字。

## POMDP：看到的不是状态

在 partially observable MDP 中，环境还有不可直接观察的真实状态。一个常见定义是

$$
\mathcal P=
(\mathcal S,\mathcal A,P,r,\mathcal O,O,\rho_0,\gamma),
$$

其中 observation kernel 为

$$
O(o_{t+1}\mid s_{t+1},a_t).
$$

agent 根据 action-observation history

$$
h_t=(o_0,a_0,r_0,o_1,\ldots,o_t)
$$

选择动作。若模型与初始分布已知，history 可压缩为 belief state

$$
b_t(s)=\Pr(s_t=s\mid h_t).
$$

离散 belief 的 Bayes 更新为

$$
b_{t+1}(s')
=\eta\,
O(o_{t+1}\mid s',a_t)
\sum_sP(s'\mid s,a_t)b_t(s),
$$

其中 $\eta$ 是归一化常数。belief 本身构成一个连续状态 MDP，但精确维护通常代价很高；循环网络、上下文、外部 memory 与结构化 ledger 都是在逼近可用于控制的 sufficient statistic，并不自动证明 Markov 性。

[Åström 1965 年的部分可观察 Markov 控制工作](https://lup.lub.lu.se/record/8867084)已把不完整状态信息与 Bayesian state estimation 放进控制问题；后来的有限 POMDP 理论进一步刻画了 belief-space value 的结构。今天用神经网络编码 history，改变的是近似工具，不是部分可观察性的基本定义。

## 一个可手算的 belief update

下面只保留状态转移、观察更新和终止语义。`P[s, a, s_next]` 与 `O[a, s_next, observation]` 都是显式条件分布。

```python
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Transition:
    s: int
    a: int
    r: float
    s_next: int
    terminated: bool
    truncated: bool

def belief_step(b, a, o, P, O):
    pred = b @ P[:, a, :]
    post = pred * O[a, :, o]
    z = post.sum()
    assert z > 0, "observation 在当前模型下必须有正概率"
    return post / z

def td_target(r, v_next, gamma, terminated):
    return r if terminated else r + gamma * v_next

P = np.array([
    [[0.8, 0.2], [0.1, 0.9]],
    [[0.3, 0.7], [0.6, 0.4]],
])
O = np.array([
    [[0.9, 0.1], [0.2, 0.8]],
    [[0.8, 0.2], [0.1, 0.9]],
])
b1 = belief_step(np.array([0.6, 0.4]), a=0, o=1, P=P, O=O)
assert np.all(b1 >= 0) and np.isclose(b1.sum(), 1.0)
cut = Transition(0, 1, 1.0, 1, terminated=False, truncated=True)
assert np.isclose(td_target(cut.r, 5.0, 0.9, cut.terminated), 5.5)
assert np.isclose(td_target(1.0, 5.0, 0.9, terminated=True), 1.0)
```

最后两个断言是容易遗漏的契约：真正终止不 bootstrap，外部截断仍保留 $V(s_{t+1})$。这段实现只适用于小型、模型已知的离散 POMDP；大状态空间需要近似 filtering 或直接学习 history representation。

有限 MDP 的完整 value-iteration 对照与边界断言见[手撕：强化学习](../practice/reinforcement-learning.md)。

## 映射到语言模型

语言任务可落在不同决策抽象上：

| 场景 | 状态或观察 | 动作 | 更自然的抽象 |
| --- | --- | --- | --- |
| 固定 prompt、一次完整回答 | prompt | 整个 response | 一步 MDP / contextual bandit |
| 自回归生成 | prompt 与 prefix | 下一个 token | 确定性前缀转移的 episodic MDP |
| 多轮工具任务 | 对话、工具返回、外部状态投影 | assistant turn / tool call | 通常是 POMDP |

在 token 级抽象中，拼接 token 的转移近乎确定，但 reward 和真实任务状态未必只由可见 prefix 决定。在 agent 场景中，同样的对话文本可能对应不同文件、权限、网页或数据库状态，因此 history 只是 observation history。

[语言模型作为策略](language-model-policy.md)继续区分 token、response、turn 与 episode 粒度；[语言模型中的信用分配](credit-assignment.md)说明终局 reward 怎样沿这些时间尺度传播。

## 常见误区

1. **“把最近若干帧或全部文本拼起来，就得到 Markov state。”** 拼接能增加信息，却不能恢复从未观察到的外部状态，也不能保证有限窗口充分。
2. **“POMDP 只是 observation 有噪声的 MDP。”** 状态别名、隐藏权限、未观测目标和对手意图都能产生部分可观察性，不要求显式加性噪声。
3. **“$\gamma<1$ 只是为了偏爱眼前收益。”** 它还影响回报是否有限、Bellman contraction 和有效时间尺度。
4. **“episode 结束就应把 bootstrap 置零。”** 只有真正 `terminated` 才如此；`truncated` 常需要 continuation value。
5. **“response 是离散动作，所以可以枚举最优动作。”** token 字典可枚举，不代表指数多的完整序列可枚举。
6. **“reward 是状态的属性。”** 一般 reward 属于转移，可能依赖 $s_t,a_t,s_{t+1}$，也可能是随机变量。

## Reference {#reference}

- Sutton and Barto, [Reinforcement Learning: An Introduction, Second Edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
- Puterman, [Markov Decision Processes: Discrete Stochastic Dynamic Programming](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887)
- Åström, [Optimal Control of Markov Processes with Incomplete State Information](https://lup.lub.lu.se/record/8867084)
- Kaelbling, Littman, and Cassandra, [Planning and Acting in Partially Observable Stochastic Domains](https://doi.org/10.1016/S0004-3702%2898%2900023-X)
- Gymnasium, [Handling Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
- Gymnasium, [Env API: Terminated and Truncated](https://gymnasium.farama.org/api/env/)
