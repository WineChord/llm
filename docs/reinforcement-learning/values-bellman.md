# 价值函数与 Bellman 递推

价值函数把一条尚未发生的未来压成当前状态或动作的一个期望。Bellman 递推进一步指出：长期回报可以拆成“一步 reward + 下一状态的剩余价值”。这条自洽关系连接了动态规划、TD、Q-learning、critic 与语言模型中的 value head。

## 从回报到价值

在策略 $\pi$ 下，state value 与 action value 定义为

$$
V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s],
\qquad Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].
$$

advantage 比较一个动作与当前策略在同一状态下的平均表现：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

因为

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a),
$$

所以对 $a\sim\pi(\cdot\mid s)$ 有

$$
\mathbb E_\pi[A^\pi(s,a)\mid s]=0.
$$

value 不是即时 reward，也不是单条轨迹的 realized return。它是由策略、环境动力学、终止规则和 discount 共同决定的条件期望。MDP 对象与转移约定见 [MDP、POMDP 与回报](decision-processes.md)。

## Bellman expectation equation

把回报展开一步：

$$
G_t=r_t+\gamma G_{t+1}.
$$

对下一状态和动作取期望，得到

$$
V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)
\left[r(s,a,s')+\gamma(1-d(s,a,s'))V^\pi(s')\right],
$$

其中 $d(s,a,s')$ 表示这次转移是否真正终止。对应的 action-value 递推为

$$
Q^\pi(s,a)=\sum_{s'}P(s'\mid s,a)
\left[r(s,a,s')+\gamma(1-d)\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')\right].
$$

定义 Bellman expectation operator

$$
(\mathcal T^\pi V)(s)=
\mathbb E_\pi[r_t+\gamma(1-d_t)V(s_{t+1})\mid s_t=s].
$$

则真实价值是固定点：

$$
V^\pi=\mathcal T^\pi V^\pi.
$$

当 $0\le\gamma<1$ 且 reward 有界时，

$$
\|\mathcal T^\pi V-\mathcal T^\pi U\|_\infty
\le\gamma\|V-U\|_\infty.
$$

因此 $\mathcal T^\pi$ 是 sup norm 下的 contraction，反复应用会收敛到唯一固定点。这是迭代 policy evaluation 的数学基础；它不是说有限样本、非线性函数逼近和 off-policy 更新也自动收敛。

## Bellman optimality equation

最优价值定义为

$$
V^*(s)=\max_\pi V^\pi(s),\qquad Q^*(s,a)=\max_\pi Q^\pi(s,a).
$$

Bellman optimality operator 为

$$
(\mathcal T^*V)(s)=\max_a\sum_{s'}P(s'\mid s,a)
\left[r(s,a,s')+\gamma(1-d)V(s')\right],
$$

并满足

$$
V^*=\mathcal T^*V^*.
$$

一旦得到 $Q^*$，任意只选择最大值动作的策略都是最优策略：

$$
\pi^*(s)\in\arg\max_a Q^*(s,a).
$$

这里的 $\max$ 是控制假设的一部分：agent 能在每个状态自由选择候选动作。带 entropy 或 KL regularization 的控制问题会把 hard max 换成 soft value；完整 response 无法枚举时，也不能把序列空间上的 $\max$ 当成可直接计算的操作。

## Dynamic programming 的三个循环

Bellman 在 1950 年代建立 dynamic programming 和最优性原理：一个最优决策的剩余部分，在到达下一状态后仍必须对子问题最优。已知有限 MDP 中，由此得到三个基本过程。

### Policy evaluation

固定 $\pi$，迭代 $V_{k+1}=\mathcal T^\pi V_k$，直到 Bellman
residual $\|\mathcal T^\pi V_k-V_k\|_\infty$ 足够小。也可把有限状态方程写成线性系统直接求解。

### Policy improvement

由当前 $V^\pi$ 构造 greedy policy：

$$
\pi'(s)\in
\arg\max_a
\mathbb E[r_t+\gamma(1-d_t)V^\pi(s_{t+1})\mid s_t=s,a_t=a].
$$

policy improvement theorem 保证 $V^{\pi'}(s)\ge V^\pi(s)$。反复“精确评估 → greedy 改进”就是 policy iteration。

### Value iteration

将评估和改进压成一次 optimality backup：

$$
V_{k+1}=\mathcal T^*V_k.
$$

它在每轮只做有限评估就立即 greedy 改进。policy iteration 与 value iteration 不是两套互不相关的技巧，而是 generalized policy iteration 的两个极端。

## 手撕 value iteration

下面的 `P[s, a, s_next]`、`R[s, a, s_next]` 与 `terminated[s, a, s_next]` 直接对应转移语义。外部截断不进入 `terminated`。

```python
import numpy as np

def q_from_v(P, R, terminated, v, gamma):
    continuation = (1.0 - terminated) * v[None, None, :]
    return np.sum(P * (R + gamma * continuation), axis=-1)

def value_iteration(P, R, terminated, gamma=0.99, tol=1e-10):
    assert P.ndim == 3 and P.shape == R.shape == terminated.shape
    assert np.allclose(P.sum(axis=-1), 1.0)
    v = np.zeros(P.shape[0])
    for _ in range(100_000):
        q = q_from_v(P, R, terminated, v, gamma)
        v_new = q.max(axis=1)
        if np.max(np.abs(v_new - v)) < tol:
            v = v_new
            break
        v = v_new
    q = q_from_v(P, R, terminated, v, gamma)
    residual = np.max(np.abs(q.max(axis=1) - v))
    assert residual < 10 * tol
    return v, q.argmax(axis=1)

P = np.zeros((2, 2, 2))
P[0, 0, 0] = P[0, 1, 1] = 1.0
P[1, :, 1] = 1.0
R = np.zeros_like(P)
R[0, 0, 0], R[0, 1, 1] = 1.0, 5.0
terminated = np.zeros_like(P)
terminated[0, 1, 1] = terminated[1, :, 1] = 1.0
v, policy = value_iteration(P, R, terminated, gamma=0.9)
assert np.isclose(v[0], 10.0, atol=1e-8)
assert policy[0] == 0
```

状态 0 的动作 0 每步得到 1 并继续，价值是

$$
1+0.9+0.9^2+\cdots=10;
$$

动作 1 立即获得 5 后终止。因此最优动作是继续。该实现是有限、模型已知 MDP 的语义参考，不包含大状态空间的采样、稀疏矩阵或异步更新优化。

## 从模型 backup 到样本 backup

动态规划计算完整期望

$$
\sum_{s'}P(s'\mid s,a)\left[r+\gamma V(s')\right].
$$

未知模型时，可以用一次实际转移

$$
r_t+\gamma(1-d_t)V(s_{t+1})
$$

作为随机估计。这一步把 Bellman equation
变成 TD update；若等待完整 episode，则可用 Monte Carlo return 作为 value
target。[Monte Carlo、TD 与控制](prediction-control.md)比较二者的偏差、方差与在线性。

函数逼近时，目标通常含当前或旧参数产生的 bootstrap value。它不是静态真值标签；共享参数、off-policy 数据和 bootstrap 相遇时还会形成[致命三元组](function-approximation.md)。

## 映射到语言模型

对自回归 policy，history $h_t=(x,y_{<t})$ 可作为可见状态，token $y_t$ 是动作：

$$
V^\pi(h_t)=\mathbb E_\pi[R\mid h_t],
\qquad Q^\pi(h_t,y_t)=\mathbb E_\pi[R\mid h_t,y_t].
$$

value head 学的是当前 policy 继续生成时的期望未来回报，而 reward model 评估的是某种反馈目标；两者训练分布、条件变量和用途都不同。critic 的 TD residual 可近似 advantage，但 critic 不是 verifier，也不负责定义什么是“好回答”。

若只有 response 末尾 reward，Bellman 结构仍存在，只是中间 token 的 $r_t$ 多为零。真正 EOS 或环境成功可令 bootstrap 为零；token budget、context limit 或 rollout cutoff 通常是截断，应该保留边界 value。对应的 action mask 与多时间尺度契约见[语言模型作为策略](language-model-policy.md)和[语言模型中的信用分配](credit-assignment.md)。

## 常见误区

1. **“Bellman equation 就是 value iteration。”** 前者是价值的固定点关系；后者是在已知模型上寻找固定点的一种算法。
2. **“value 是 reward 的平滑版本。”** value 还依赖未来策略、转移、终止和 discount；同一 reward 函数在不同策略下产生不同 value。
3. **“bootstrap target 是 ground truth。”** target 本身依赖估计值，参数漂移会改变监督信号。
4. **“Bellman contraction 保证神经 TD 收敛。”** contraction 适用于精确函数空间中的算子；投影、非线性参数化和 off-policy 采样会改变动力学。
5. **“最大 Q 就是无偏的最优价值估计。”** 对多个带噪估计取最大值通常产生 overestimation bias。
6. **“截断和终止都令 $V(s_{t+1})=0$。”** 只有 MDP 真终态如此；外部截断通常仍有 continuation value。
7. **“语言动作是 token，所以可以计算精确 $V^*$。”** token 分支与长 horizon 造成指数序列空间，环境 reward 也常不可查询。

## Reference {#reference}

- Bellman, [Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming)
- Sutton and Barto, [Reinforcement Learning: An Introduction, Second Edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
- Puterman, [Markov Decision Processes: Discrete Stochastic Dynamic Programming](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887)
- Silver, [Reinforcement Learning Course](https://davidstarsilver.wordpress.com/teaching/)
