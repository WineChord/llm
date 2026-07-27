# 函数逼近与致命三元组

表格方法为每个状态保存一个独立数值；函数逼近则让许多状态共享参数。共享使强化学习可以处理连续状态、图像和长文本，也意味着一次错误 bootstrap 会同时改变大片状态空间。规模来自泛化，失稳也来自泛化。

本文聚焦 value approximation。策略参数化见[Policy Gradient](policy-gradient.md)，Bellman target 的来源见[价值函数与 Bellman 递推](values-bellman.md)。

## 从查表到投影

线性 value function 写作

$$
\widehat V_w(s)=x(s)^\top w,
$$

神经网络则以非线性映射 $\widehat V_w(s)=f_w(s)$ 取代固定特征。给定一步 transition，TD residual 为

$$
\delta_t
=r_t+\gamma(1-d_t)\widehat V_w(s_{t+1})
-\widehat V_w(s_t).
$$

semi-gradient TD(0) 更新

$$
w_{t+1}
=w_t+\alpha\delta_t\nabla_w\widehat V_w(s_t)
$$

只对当前预测求梯度，把 bootstrap target 当作常数。若对 target 里的同一个网络也反向传播，得到的是另一个算法；它不再是经典 semi-gradient TD。

在状态分布 $d$ 诱导的加权空间里，线性 TD 试图求解 projected Bellman fixed point：

$$
\widehat V_w=\Pi_d\mathcal T^\pi\widehat V_w.
$$

因此即使收敛，它一般也不是对真实 $V^\pi$ 做普通 least squares。投影分布由数据访问频率决定；behavior policy 改变，拟合重点和固定点都会改变。

## On-policy 线性结论有多窄

在遍历性、合适步长、特征矩阵满秩等条件下，on-policy 线性 TD(0) 可收敛到 projected Bellman solution。以下变化都不能自动继承该保证：

- behavior policy 与 target policy 不同；
- 非线性网络改变表示和 target；
- control 中 policy 随 $Q$ 同时变化；
- replay buffer 混合多个历史策略；
- 常数步长只在固定点附近形成稳态噪声，而非精确收敛。

这也是为什么[Monte Carlo、TD 与控制](prediction-control.md)中的表格 Q-learning 收敛结论，不能仅把表换成神经网络后继续引用。

## 致命三元组

经典 deadly triad 指三个因素同时出现：

| 因素 | 含义 | 它带来的困难 |
| --- | --- | --- |
| Function approximation | 多个状态共享参数 | 一个样本改变未访问状态 |
| Bootstrapping | target 含当前估计 | 错误可被再次当作监督 |
| Off-policy learning | 数据分布不同于目标策略 | 投影分布与 Bellman 动力学错位 |

三项都很常见，却不是说任何包含其中一项的方法必然发散。危险在于它们的组合让 expected update 不再是收缩映射，甚至在线性、零 reward、正确 importance ratio 的小例子里也会发散。

## 手撕：terminal 与 truncation 下的 semi-gradient

`truncated` 只切断 episode 数据，不应清零真实后继状态的 bootstrap：

```python
import numpy as np

def linear_td0(w, x, reward, x_next, terminated, truncated,
               alpha=0.1, gamma=0.99):
    w, x = np.asarray(w, float), np.asarray(x, float)
    if not terminated and x_next is None:
        raise ValueError("non-terminal transition needs its true successor")
    next_value = 0.0 if terminated else np.asarray(x_next, float) @ w
    delta = reward + gamma * next_value - x @ w
    return w + alpha * delta * x, delta, bool(terminated or truncated)

w = np.array([2.0])
ordinary, d0, reset0 = linear_td0(
    w, [1.0], 1.0, [4.0], False, False, gamma=0.5)
limited, d1, reset1 = linear_td0(
    w, [1.0], 1.0, [4.0], False, True, gamma=0.5)
terminal, d2, reset2 = linear_td0(
    w, [1.0], 1.0, None, True, False, gamma=0.5)
np.testing.assert_allclose(ordinary, limited)
assert d0 == d1 == 3.0 and reset0 is False and reset1 is True
assert d2 == -1.0 and reset2 is True
```

代码返回的 `reset` 用于清空 recurrent state 或 eligibility trace；它和 TD target 的 bootstrap mask 不是同一个量。

## 手撕：Baird 反例的 expected update

Baird 反例有 7 个状态、8 维线性特征、零 reward。behavior policy 的稳态状态分布均匀，target policy 总是转到下方状态。对动作 importance ratio 求完期望后，off-policy semi-gradient TD 的一步期望更新仍可写成

$$
w_{k+1}=(I+\alpha A)w_k,\qquad
A=\mathbb E_{s\sim d_\mu}
\left[x(s)\left(\gamma x(s')-x(s)\right)^\top\right].
$$

若更新矩阵的 spectral radius 大于 $1$，就存在会指数增长的初值：

```python
import numpy as np

phi = np.zeros((7, 8))
phi[:6, :6] = 2.0 * np.eye(6)
phi[:6, 7] = 1.0
phi[6, 6], phi[6, 7] = 1.0, 2.0
phi /= np.sqrt(5.0)

def baird_expected_matrix(features, gamma=0.99):
    lower = np.repeat(features[-1][None, :], len(features), axis=0)
    return features.T @ (gamma * lower - features) / len(features)

alpha = 0.01
update = np.eye(phi.shape[1]) + alpha * baird_expected_matrix(phi)
radius = np.abs(np.linalg.eigvals(update)).max()
assert radius > 1.0
assert np.allclose(phi @ np.zeros(8), 0.0)  # true value under zero reward
```

这里没有高维观察、深网络或 noisy reward；发散来自 bootstrap、共享参数与 behavior-state distribution 的组合。importance sampling 能校正已观察动作的概率，却不能凭空把 behavior 访问分布变成 target policy 的稳态分布。

## 深度强化学习怎样缓解失稳

常见机制各自只处理问题的一部分：

| 机制 | 主要作用 | 不能保证什么 |
| --- | --- | --- |
| Target network | 让 bootstrap target 暂时慢变 | 不保证整体收敛 |
| Replay buffer | 打散相关性、提高样本复用 | 同时让数据更 off-policy |
| Double estimator | 减少 $\max$ 与噪声耦合造成的高估 | 不消除表示漂移 |
| Gradient clipping / normalization | 限制数值爆炸 | 不修复错误固定点 |
| GTD / emphatic TD | 在线性 off-policy 预测下恢复特定收敛性质 | 不直接解决任意深度 control |

[DQN 的原始工作](https://www.nature.com/articles/nature14236)把 replay 与 target network 组合起来取得经验稳定性；这是一套有效工程配方，不是 deadly triad 已被普遍证明解决。

## 目标、梯度与诊断

只看训练 TD loss 可能产生误判。target 与 prediction 一起漂移时，loss 下降不代表 value 更接近真实回报。至少同时检查：

1. 固定 holdout trajectory 上的 Monte Carlo calibration；
2. value mean、scale、分位数和跨 checkpoint drift；
3. Bellman residual 与真正 return error 是否同向；
4. target network lag、replay policy age 和 importance ratio；
5. terminal、truncated、invalid、infrastructure error 的占比；
6. reward rescale 后 value 与 optimizer 超参数是否同步变化。

对 off-policy ratio、support 和 policy lag 的系统处理见[Off-policy 校正](off-policy-correction.md)。

## 语言模型桥梁

语言模型 actor–critic 通常在 transformer 上增加 value head，输入是 prompt、prefix、tool observation 或整段历史。共享 trunk 同时带来强泛化与强耦合：

- value loss 会改变 policy representation，反之亦然；
- 用旧 policy rollout、当前 critic bootstrap 时已经存在 off-policy drift；
- observation 和 prompt 可进入 value 条件，但不能因此进入 policy action mask；
- reward model 估计“这段输出得到多少外部评分”，critic 估计“当前 policy 从该状态继续的期望 return”，二者不是同一个函数；
- time-limit 或 token-budget truncation 通常 bootstrap，环境成功或失败 terminal 不 bootstrap。

长轨迹中 value 的角色见[语言模型信用分配](credit-assignment.md)，状态与动作粒度见[语言模型作为策略](language-model-policy.md)。当 rollout 与 learner 解耦时，应同时记录 behavior policy version、old log-prob、critic version 和 final observation。

## 历史脉络

线性 approximation 早于深度强化学习。1990 年代的分析已经表明：on-policy 线性 TD 可以有清楚的收敛边界，而 Baird 反例揭示 off-policy bootstrap 的发散。随后 projected Bellman error、gradient TD 与 emphatic TD 发展出更严格的 off-policy prediction 方法。2015 年 DQN 则以 replay、target network 和卷积表示展示了经验稳定的深度 control 路线。

这段历史的重要教训不是“旧反例已过时”，而是必须区分三层证据：特定线性算法的证明、特定 benchmark 的经验配方，以及任意非线性系统的实际诊断。

## 常见误区

1. **用了神经网络才有 deadly triad。** Baird 反例本身就是线性的。
2. **target network 让 target 固定，所以已经变成监督学习。** target 仍会周期更新，数据分布也随策略改变。
3. **replay 总能稳定训练。** 它降低时间相关性，却增加 policy age 与 off-policy 程度。
4. **TD loss 越小，value 越准。** 自举 target 可与错误 prediction 一起移动。
5. **terminal 和 truncation 都应令 target 等于 reward。** truncation 通常要从 final observation bootstrap，只在 episode bookkeeping 上重置。

## Reference {#reference}

- Sutton and Barto, [Reinforcement Learning: An Introduction, Second Edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
- Baird, [Residual Algorithms: Reinforcement Learning with Function Approximation](https://leemon.com/papers/1995b.pdf)
- Tsitsiklis and Van Roy, [An Analysis of Temporal-Difference Learning with Function Approximation](https://web.stanford.edu/~bvr/pubs/td.pdf)
- Sutton et al., [Fast Gradient-Descent Methods for Temporal-Difference Learning with Linear Function Approximation](https://proceedings.mlr.press/v26/sutton12a.html)
- Sutton, Mahmood, and White, [An Emphatic Approach to the Problem of Off-policy Temporal-Difference Learning](https://jmlr.org/papers/v17/14-488.html)
- Mnih et al., [Human-level Control through Deep Reinforcement Learning](https://www.nature.com/articles/nature14236)
