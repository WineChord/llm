# Monte Carlo、TD 与控制

在模型未知时，Bellman 期望无法直接枚举，但环境会给出样本转移 $(s_t,a_t,r_t,s_{t+1})$。Monte Carlo 用完整采样回报学习，temporal-difference learning 用一步 reward 加当前 value 进行 bootstrap；控制算法再把“评估一个策略”变成“边评估边改进策略”。

## 先分 prediction 与 control

- **Prediction / policy evaluation**：策略 $\pi$ 固定，估计 $V^\pi$ 或 $Q^\pi$。
- **Control**：同时寻找更好的策略，目标是 $V^*$ 或 $Q^*$。

这一区分比算法名称更重要。TD(0) 可以做 prediction；SARSA 和 Q-learning 则用 TD target 做 control。[价值函数与 Bellman 递推](values-bellman.md)给出它们共同逼近的固定点，[MDP、POMDP 与回报](decision-processes.md)定义样本所属的决策过程。

## Monte Carlo：等到回报落地

若真正终止的 episode 含 transition $0,\ldots,T-1$，时刻 $t$ 的采样回报是

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k.
$$

增量更新可写为

$$
V(s_t)\leftarrow V(s_t)+\alpha[G_t-V(s_t)].
$$

若 $\alpha=1/N(s_t)$，就是 sample mean。first-visit MC 每个 episode 只更新某状态第一次出现的位置；every-visit MC 更新所有出现位置。两者在适当条件下都收敛到 $V^\pi$，有限数据下方差和权重不同。

MC target 不依赖当前 value，因而没有 bootstrap bias；代价是必须等待未来 reward，长 horizon 下方差高，也无法自然处理永不终止的 continuing task。

外部截断尤其需要小心。若轨迹只是因为时间或 token 预算停止，截断点之后的 return 并不等于零：

$$
\widehat G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k}
+\gamma^nV(s_{t+n}).
$$

加入边界 $V$ 后，它已经是 bootstrapped n-step target，而不是纯粹的 complete-episode MC。

## TD(0)：一步以后交给当前估计

TD error 为

$$
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),
$$

更新为

$$
V(s_t)\leftarrow V(s_t)+\alpha\delta_t.
$$

其中 $d_t=\mathbf 1[\text{terminated at }t]$，而不是
`terminated or truncated`。TD 可以在每一步在线更新，并把新信息逐步向前传播；它的 target 方差通常低于 MC，却依赖当前估计，因此会引入 bootstrap bias。

Sutton 在 1988 年系统化了 TD prediction，并展示它如何在 complete return 与动态规划之间搭桥。后来的 n-step return、TD($\lambda$) 与 GAE 都沿用这个“采样多少步、其余部分 bootstrap”的连续谱，见[多步回报、资格迹与 GAE](multistep-traces.md)。

## MC 与 TD 不是简单优劣关系

| 维度 | Monte Carlo | TD(0) |
| --- | --- | --- |
| target | 完整采样回报 | 一步 reward + bootstrap |
| episode 未结束能否更新 | 通常不能 | 能 |
| 对 transition model 的要求 | 不需要 | 不需要 |
| target 方差 | 通常较高 | 通常较低 |
| bootstrap bias | 无 | 有 |
| continuing task | 不自然 | 自然 |
| 对初始 value 的依赖 | target 不依赖 | target 依赖 |

MC 的“无偏”也有条件：episode 必须来自目标策略，return 必须完整，且没有基于结果的选择性过滤。TD 的低方差也不保证稳定；结合函数逼近和 off-policy 数据时，需要考虑[致命三元组](function-approximation.md)。

## 从 prediction 到 control

control 需要一边估计 action value，一边让 behavior policy 保持探索并逐步变得 greedy。常见的 $\epsilon$-greedy policy 为

$$
\pi(a\mid s)=
\begin{cases}
1-\epsilon+\epsilon/|\mathcal A|,&a\in\arg\max_{a'}Q(s,a'),\\
\epsilon/|\mathcal A|,&\text{otherwise},
\end{cases}
$$

有多个 greedy action 时，应把 $1-\epsilon$ 均分给这些并列动作。

### SARSA：更新当前 behavior policy

SARSA 使用实际选择的下一动作：

$$
y_t^{\text{SARSA}}=r_t+\gamma(1-d_t)Q(s_{t+1},a_{t+1}).
$$

若 $a_{t+1}$ 来自当前 $\epsilon$-greedy behavior，target 就在评估这条含探索的策略，因此 SARSA 是 on-policy control。

### Expected SARSA：对下一动作取期望

$$
y_t^{\text{ExpSARSA}}=r_t+\gamma(1-d_t)
\sum_{a'}\pi(a'\mid s_{t+1})Q(s_{t+1},a').
$$

它把 SARSA 下一动作的采样噪声积分掉，但需要知道目标策略的动作概率。

### Q-learning：直接逼近 optimality target

$$
y_t^Q=r_t+\gamma(1-d_t)\max_{a'}Q(s_{t+1},a').
$$

behavior 可以是 $\epsilon$-greedy，target 却是 greedy policy，因此 Q-learning 是 off-policy control。Watkins 与 Dayan 给出的表格收敛结果要求有限 MDP、充分访问状态动作、合适学习率等条件；它不能直接推出 neural Q-learning 稳定。

三个算法的差别集中在同一个位置：SARSA 采样 behavior 的下一动作，Expected
SARSA 对 target policy 取期望，Q-learning 则使用 greedy target。

## 紧凑的 target 与更新

下面的实现只保留最关键的 target 语义。`truncated=True` 会结束当前采样片段，但不抹掉下一状态价值。

```python
import numpy as np

def discounted_returns(rewards, gamma, bootstrap=0.0):
    out = np.empty(len(rewards), dtype=float)
    g = float(bootstrap)
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out

def epsilon_probs(q, eps):
    greedy = np.isclose(q, q.max())
    p = np.full(q.size, eps / q.size)
    p[greedy] += (1.0 - eps) / greedy.sum()
    assert np.isclose(p.sum(), 1.0)
    return p

def control_target(Q, r, s_next, terminated, truncated, gamma,
                   method, eps=0.1, a_next=None):
    assert not (terminated and truncated)
    if terminated:
        bootstrap = 0.0
    elif method == "sarsa":
        assert a_next is not None
        bootstrap = Q[s_next, a_next]
    elif method == "expected_sarsa":
        bootstrap = epsilon_probs(Q[s_next], eps) @ Q[s_next]
    elif method == "q_learning":
        bootstrap = Q[s_next].max()
    else:
        raise ValueError(method)
    return r + gamma * bootstrap

def update_q(Q, s, a, target, alpha):
    Q[s, a] += alpha * (target - Q[s, a])

assert np.allclose(
    discounted_returns([1.0, 2.0], gamma=0.5, bootstrap=4.0),
    [3.0, 4.0],
)
Q = np.array([[0.0, 0.0], [4.0, 2.0]])
q_cut = control_target(Q, 1.0, 1, False, True, 0.9, "q_learning")
q_end = control_target(Q, 1.0, 1, True, False, 0.9, "q_learning")
exp = control_target(Q, 1.0, 1, False, False, 0.9,
                     "expected_sarsa", eps=0.2)
assert np.isclose(q_cut, 4.6) and np.isclose(q_end, 1.0)
assert np.isclose(exp, 4.42)
update_q(Q, s=0, a=0, target=q_cut, alpha=0.5)
assert np.isclose(Q[0, 0], 2.3)
```

`discounted_returns` 的 `bootstrap` 只有在真终态才应设为零；截断时可传入边界 value。`control_target` 同时展示了 SARSA、Expected SARSA 与 Q-learning 的唯一核心分叉。完整训练循环还需要环境 reset、探索 schedule 和 visitation 条件，但这些不改变这里的估计量。

## On-policy、off-policy 与探索

需要分开三个角色：

- behavior policy $\mu$：实际产生转移；
- target policy $\pi$：想要评估或改进的策略；
- learned table/function：当前的 $V$ 或 $Q$ 估计。

SARSA 常令 $\mu=\pi$。Q-learning 允许 $\mu$ 探索而 target greedy，但这不意味着任意历史数据都安全：behavior 必须覆盖目标动作，函数逼近会引入分布与投影问题，旧数据还可能来自不同环境版本。

表格控制的经典充分探索条件常写成 GLIE：所有状态动作被无限访问，同时策略在极限上变得 greedy。实际深度 RL 很少严格满足它，只能通过 replay coverage、entropy、随机化和独立评测观察是否发生探索坍缩。[Off-policy 校正](off-policy-correction.md)继续处理目标与行为分布不同带来的 ratio 和方差。

## 映射到语言模型

对固定 prompt 的整段 response，终局 reward 可以直接形成 Monte Carlo policy-gradient 信号；若训练 value head，则可用同一 response return 做回归。对 token-level MDP，中间 reward 常为零：

$$
r_0=\cdots=r_{T-1}=0,\qquad r_T=R(x,y).
$$

这会让 TD 信息一次只向前传播一步，n-step return 或 GAE 更常见。value head 在 prefix 上预测当前 policy 的未来 response reward，不能把 reward model 的输出直接当作每个 prefix 的真实 $V^\pi$。

Q-learning 通常不是开放式语言生成的默认选择。虽然单个 token 动作有限，但：

- horizon 很长，reward 极稀疏；
- vocabulary 上的逐 token greedy 不等于全序列最优；
- policy 本身能直接给采样动作的 log-probability；
- 多轮工具环境常部分可观察且状态分布随策略显著变化。

因此语言后训练多沿 policy-gradient 或 actor–critic 路线展开。response reward 的 token、turn 与 episode 归因见[语言模型中的信用分配](credit-assignment.md)，动作尺度与 mask 见[语言模型作为策略](language-model-policy.md)。

## 常见误区

1. **“MC 不 bootstrap，所以总是更准确。”** 单条 complete return 方差可能极高；是否更好取决于 horizon、reward 噪声与 value bias。
2. **“TD target 是环境标签。”** 它含当前 value estimate；学习目标会随参数变化。
3. **“SARSA 比 Q-learning 更保守，所以一定更差。”** SARSA 优化的是含探索的实际行为，在危险路径或探索代价存在时可能学到不同且更合适的策略。
4. **“Q-learning 使用 off-policy target，所以任何离线数据都能训练。”** 仍需要支持集覆盖；函数逼近还可能在分布外动作上产生任意高估。
5. **“Expected SARSA 就是 Q-learning。”** 只有 target policy 退化为 deterministic greedy 时，两者 target 才相同。
6. **“所有 episode 边界都把 bootstrap 清零。”** 外部截断通常保留 continuation value，真正终止才清零。
7. **“token 是离散动作，因此 DQN 自然适合语言模型。”** 可枚举一步动作不解决长序列组合优化、稀疏反馈和部分可观察性。

## Reference {#reference}

- Sutton, [Learning to Predict by the Methods of Temporal Differences](https://link.springer.com/article/10.1007/BF00115009)
- Watkins and Dayan, [Q-learning](https://link.springer.com/article/10.1007/BF00992698)
- Sutton and Barto, [Reinforcement Learning: An Introduction, Second Edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
- van Seijen et al., [A Theoretical and Empirical Analysis of Expected Sarsa](https://ojs.aaai.org/index.php/AAAI/article/view/9597)
- Gymnasium, [Handling Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
