# 强化学习基础

Agentic RL 建立在经典强化学习对象上，但状态通常是长历史，动作可能是一段 token 或工具调用，环境又昂贵且部分可观测。先把 value、policy、采样分布和终止语义分清，才能理解 PPO、GRPO 与异步 rollout。

## MDP 与 POMDP

Markov Decision Process 可写为

$$
\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma).
$$

策略 $\pi(a\mid s)$ 产生动作，环境按 $P(s'\mid s,a)$ 转移。回报为

$$
G_t=\sum_{k=0}^{T-t}\gamma^kr_{t+k}.
$$

语言 agent 通常只能看到 observation $o_t$，真实环境状态 $s_t$ 不完全可见，因此策略依赖历史

$$
h_t=(o_0,a_0,\ldots,o_t).
$$

上下文窗口、结构化任务状态和外部 memory 都是在近似 belief state；摘要丢失信息会破坏近似 Markov 性。

## Value 与 Q

状态价值：

$$
V^\pi(s)
=\mathbb E_\pi[G_t\mid s_t=s].
$$

动作价值：

$$
Q^\pi(s,a)
=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].
$$

优势函数：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

优势描述“这个动作相对当前状态的平均行为好多少”，比绝对回报更适合降低 policy-gradient 方差。

## Bellman 方程

对固定策略，

$$
V^\pi(s)
=\mathbb E_{a\sim\pi,s'\sim P}
\left[r(s,a)+\gamma V^\pi(s')\right].
$$

最优动作价值满足

$$
Q^*(s,a)
=\mathbb E_{s'}
\left[
r(s,a)+\gamma\max_{a'}Q^*(s',a')
\right].
$$

动态规划需要已知环境模型或能枚举转移；大多数语言环境只能通过 rollout 采样。

## Monte Carlo 与 TD

Monte Carlo 用 episode 完整回报 $G_t$ 监督 value，无 bootstrap 偏差但方差高、必须等终局。TD(0) 使用

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),
$$

在轨迹未结束时即可更新，但 target 依赖当前 value 估计。TD($\lambda$)与 GAE 用指数权重混合多步 target，在偏差和方差间折中。

长时 agent 的最终奖励延迟很长，纯 MC 方差大；错误的 step reward 或 critic 又会把偏差传播到所有早期动作。过程 verifier 的质量因此是算法的一部分。

## Value-based 与 policy-based

Q-learning 用

$$
y_t=r_t+\gamma\max_{a'}Q_{\bar\theta}(s_{t+1},a')
$$

构造 bootstrap target，适合离散、可枚举动作。语言模型词表虽离散，但完整 action span 的组合空间巨大，工具参数也有结构约束；直接对所有完整动作估计 Q 通常不现实。

Policy gradient 直接优化参数化策略：

$$
\nabla_\theta J
=\mathbb E
\left[
\nabla_\theta\log\pi_\theta(a_t\mid h_t)
\hat A_t
\right].
$$

它自然支持巨大动作空间，却依赖 on-policy 样本且方差高。

## Actor–Critic

actor 是 $\pi_\theta$，critic 是 $V_\phi$ 或 $Q_\phi$。critic 提供 advantage，actor 更新策略。二者互相影响：

- critic 欠拟合，advantage 方差大；
- critic 过拟合或分布外，产生系统偏差；
- actor 更新太快，critic target 持续漂移；
- rollout 过旧，二者都在 off-policy 数据上学习。

PPO 用 clipped probability ratio 限制 actor 单次更新；GAE 常用于构造 advantage。完整推导见[数学与算法](math-algorithms.md)。

## On-policy、off-policy 与 replay

数据由 behavior policy $\mu$ 产生，目标是更新 $\pi$。若 $\mu\ne\pi$，直接使用 on-policy estimator 会有偏差。单步 importance ratio 为

$$
\rho_t=\frac{\pi(a_t\mid h_t)}{\mu(a_t\mid h_t)}.
$$

长序列比率乘积方差会迅速增大，因此工程上常用 clipping、截断校正、限制 policy lag 或只消费新鲜轨迹。经典 replay buffer 可提高样本利用率，但对快速变化的语言策略并非免费收益。

## 探索

离散控制可用 $\epsilon$-greedy；语言策略通常用温度、top-$p$、多样化 prompt、不同 seed 或层级任务采样。探索要同时考虑：

- 生成是否真正多样；
- verifier 能否区分新策略；
- 环境动作是否安全；
- 失败成本与预算；
- 采样分布是否仍可计算 log-prob。

过强探索产生大量无效轨迹，过弱探索则让组内奖励相同、没有相对学习信号。

## 终止

对 value target，必须区分：

- `terminated`：到达环境终态，bootstrap 值通常为零；
- `truncated`：因时间或预算截断，环境本可继续，可能仍需 bootstrap；
- `infrastructure_error`：不是策略结果，应单独处理。

把所有 timeout 当作零回报会训练策略回避耗时任务，也会低估截断状态价值。

## 映射到语言 agent

| RL 对象 | 语言/工具系统中的实例 |
| --- | --- |
| state/history | prompt、消息、任务 ledger、环境状态 |
| action | token、整条消息、tool call、代码或终止 |
| transition | 工具执行、文件变化、网页响应 |
| reward | verifier、偏好模型、成本与权限惩罚 |
| episode | 从任务 reset 到成功、失败或截断 |
| behavior policy | 生成轨迹的精确 checkpoint 与解码策略 |

动作粒度决定 log-prob、credit assignment 和 replay 语义。下一步读[轨迹与策略契约](trajectory-contract.md)，再进入[数学与算法](math-algorithms.md)。

## Reference {#reference}

- [Reinforcement Learning: An Introduction, Second Edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
