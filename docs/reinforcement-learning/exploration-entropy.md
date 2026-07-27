# 探索与最大熵

探索不是“让 policy 更随机”这么简单。探索的目标是取得能改善未来决策的信息；随机性只是产生不同经历的一种手段；entropy regularization 则直接改变被优化的目标。三者相关，却不能互换。

本页先从 bandit 的可辨认权衡出发，再进入序贯控制、maximum-entropy RL 与 SAC。状态、回报和终止语义可先参考[MDP、POMDP 与回报](decision-processes.md)。

## 探索究竟在优化什么

对未知环境，greedy action 利用当前估计，exploratory action 可能牺牲即时 reward 来降低不确定性。常见目标至少有三种：

| 目标 | 典型量 | 关心的时间尺度 |
| --- | --- | --- |
| Regret minimization | 累积 reward 与最优动作之差 | 在线交互全过程 |
| Best-policy identification | 最终策略出错概率 | 探索阶段结束后 |
| Coverage / representation | 状态、动作或特征访问 | 后续训练与迁移 |

因此同一种随机策略不可能在所有问题上都“探索充分”。只有不确定性、访问分布和评测目标都明确时，探索策略才可比较。

## Bandit：最小探索模型

$\epsilon$-greedy 以概率 $1-\epsilon$ 选当前最优臂，其余时间随机；UCB 则把估计均值和不确定性 bonus 相加：

$$
a_t=\arg\max_a
\left[
\widehat Q_t(a)
+c\sqrt{\frac{\log t}{N_t(a)}}
\right].
$$

未尝试动作的 bonus 视为无穷大。下面的实现只保留选择逻辑：

```python
import numpy as np

def ucb_action(value, count, step, c=np.sqrt(2.0)):
    value, count = np.asarray(value, float), np.asarray(count, int)
    unseen = np.flatnonzero(count == 0)
    if unseen.size:
        return int(unseen[0])
    score = value + c * np.sqrt(np.log(step) / count)
    return int(np.argmax(score))

q = np.array([0.2, 0.6, 0.5])
n = np.array([1, 0, 3])
assert ucb_action(q, n, step=4) == 1
n[1] = 20
assert 0 <= ucb_action(q, n, step=24) < 3
```

UCB 的置信界依赖 reward 范围、独立性和所选版本的假设。把同一公式直接贴到非平稳深度 RL 的 neural Q 上，不再自动拥有 bandit regret guarantee。

探索如何进入 MC、TD 与策略更新，可结合[手撕：强化学习](../practice/reinforcement-learning.md)中的最小估计量逐项对照。

## 序贯探索为何更难

MDP 中动作会改变未来能看到的状态。一个即时 reward 很低的动作，可能打开新的状态区域；一个高 entropy policy，也可能只在熟悉状态里随机打转。常见路线包括：

- optimism：给不确定状态动作更高 value；
- posterior sampling：按环境假设的 posterior 采样一套决策模型；
- count 或 pseudo-count bonus：奖励低访问区域；
- information gain：奖励能显著更新世界模型或表示的 transition；
- demonstrations、curriculum 与 resets：从数据和环境设计改善覆盖。

这些方法优化的对象不同。Intrinsic reward 还是 reward，会改变 return 与 credit assignment；若只在训练时使用，必须说明部署目标和 shaping 是否保持原最优策略。

## Maximum-entropy RL

最大熵控制把 policy entropy 放进每一步目标：

$$
J(\pi)
=\mathbb E_{\tau\sim\pi}
\left[
\sum_{t=0}^{\infty}\gamma^t
\left(r_t+\alpha\mathcal H(\pi(\cdot\mid s_t))\right)
\right].
$$

等价地，对采样动作使用 shaped reward

$$
r_t-\alpha\log\pi(a_t\mid s_t).
$$

固定 policy 的 soft value 满足

$$
V^\pi(s)
=\mathbb E_{a\sim\pi}
\left[Q^\pi(s,a)-\alpha\log\pi(a\mid s)\right],
$$

$$
Q^\pi(s,a)
=\mathbb E_{s'}
\left[r(s,a,s')+\gamma V^\pi(s')\right].
$$

对离散动作的最优 soft value，

$$
V^\star(s)
=\alpha\log\sum_a
\exp\left(\frac{Q^\star(s,a)}{\alpha}\right).
$$

$\alpha\to0$ 时 log-sum-exp 逼近 $\max$；$\alpha>0$ 时最优策略本身发生变化。熵奖励可能改善覆盖和鲁棒性，但它不是“免费探索 bonus”。

## Entropy 与 reference KL

若 reference policy 是均匀分布，负 KL 与 entropy 只差常数。一般情况下，

$$
-\beta D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})
=\beta\mathcal H(\pi)
+\beta\mathbb E_{a\sim\pi}
\left[\log\pi_{\mathrm{ref}}(a\mid s)\right].
$$

第二项把策略拉向非均匀先验。因此语言模型中的[KL 正则化控制](kl-regularized-control.md)既包含熵，也包含 reference 的语法、知识和风格偏好，不能简写成“增加随机性”。

## SAC：现代常用配方

Soft Actor-Critic 是面向连续控制的 off-policy maximum-entropy actor–critic。现代常用版本学习 stochastic actor 和两个 Q-function：

$$
y_t=r_t+\gamma(1-d_t)
\left[
\min_{i=1,2}Q_{\bar\phi_i}(s_{t+1},a_{t+1})
-\alpha\log\pi_\theta(a_{t+1}\mid s_{t+1})
\right],
$$

$$
L_\pi
=\mathbb E
\left[
\alpha\log\pi_\theta(a\mid s)
-\min_iQ_{\phi_i}(s,a)
\right].
$$

$a_{t+1}$ 通过 reparameterization 从当前 actor 采样，Q target 必须 stop-gradient。双 Q 取较小值用于缓解过高估计，不代表它是无偏 estimator。

### 两代 SAC 不应混写

- 2018 年 ICML 原始 SAC 的算法描述含显式 soft value network 及其 target，温度通常作为固定超参数；
- 随后的 *Soft Actor-Critic Algorithms and Applications* 配方去掉显式 value network，直接用 twin target Q 构造 soft target，并给出约束形式的自动 temperature tuning；
- 现代库里写作“SAC”的通常是后者，但固定 $\alpha$ 与自动 $\alpha$ 仍是两种不同实验合同。

引用结果或复现代码时，应同时记录 value network 是否存在、Q 数量、target 构造、temperature 是否学习、action scaling 和 entropy reduction。

## 手撕：tanh Gaussian 与 soft target

连续动作常用 Gaussian 经 `tanh` 映射到 $(-1,1)$。change-of-variables 不能省略：

$$
\log\pi(a\mid s)
=\log\mathcal N(u;\mu,\sigma)
-\sum_i\log(1-\tanh^2u_i),\qquad a=\tanh u.
$$

下面用稳定恒等式计算 Jacobian，并明确区分 terminal 与 truncation：

```python
import math
import torch
import torch.nn.functional as F
from torch.distributions import Normal
def squashed_normal(mu, log_std, noise):
    log_std = log_std.clamp(-20.0, 2.0)
    std = log_std.exp()
    u = mu + std * noise
    action = torch.tanh(u)
    logp = Normal(mu, std).log_prob(u)
    log_det = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
    return action, (logp - log_det).sum(-1)
@torch.no_grad()
def sac_target(reward, next_q1, next_q2, next_logp, terminated, truncated,
               gamma=.99, alpha=.2):
    tensors = (next_q1, next_q2, next_logp, terminated, truncated)
    if any(x.shape != reward.shape for x in tensors):
        raise ValueError("all transition tensors must have shape [B]")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("termination masks must be boolean")
    bootstrap = torch.minimum(next_q1, next_q2) - alpha * next_logp
    target = reward + gamma * torch.where(terminated, 0., bootstrap)
    return target, terminated | truncated
mu = torch.zeros(2, 1)
log_std = torch.zeros_like(mu)
action, logp = squashed_normal(mu, log_std, torch.tensor([[0.0], [20.0]]))
assert torch.isfinite(logp).all() and (action.abs() <= 1.0).all()
r = torch.ones(2)
q1 = torch.tensor([float("nan"), 4.], requires_grad=True)
q2 = torch.tensor([float("nan"), 5.])
target, reset = sac_target(
    r, q1, q2, torch.tensor([float("nan"), -1.]),
    torch.tensor([True, False]), torch.tensor([False, True]), gamma=.5, alpha=.2)
torch.testing.assert_close(target, torch.tensor([1.0, 3.1]))
assert reset.tolist() == [True, True] and not target.requires_grad
```

第二个样本在时间限制处 bootstrap 到 final observation 的 soft value，但 replay 序列必须重置；工具故障或缺失 observation 不应伪造成 reward 为零的 truncation。

## Temperature 的语义

固定 $\alpha$ 直接决定 reward 与 entropy 的相对单位。自动 temperature 常将约束写为期望 entropy 不低于目标值 $\bar{\mathcal H}$，并优化

$$
L(\alpha)
=\mathbb E_{a\sim\pi}
\left[-\alpha\left(\log\pi(a\mid s)+\bar{\mathcal H}\right)\right].
$$

实现常参数化 `log_alpha` 保证正值，并对括号内 stop-gradient。target entropy 是任务选择，不是由 SAC 自动发现的自然常数；action 维数、单位和 squashing 都会影响其尺度。

## 语言模型桥梁

在[语言模型作为策略](language-model-policy.md)中，temperature、top-$k$、top-$p$ 和 grammar mask 决定实际 behavior distribution。提高采样 temperature 会增加多样性，但不代表定向获取信息：

- decoder 截断后的概率必须重新归一化，importance ratio 的分母要对应实际采样分布；
- 多采几条 response 扩大了局部候选覆盖，却仍受 prompt、verifier 和模型支持集限制；
- 逐 token entropy 累加会产生长度依赖，不能未经说明地当作 sequence-level regularizer；
- SAC 的连续动作、replay 和 soft Q 配方不能因 token vocabulary 是离散的就直接移植到完整 response 空间；
- RLHF 的 reference KL 与 PPO old-policy ratio 各有不同作用，均不等同于 SAC entropy。

若旧 rollout 被反复训练，还需结合[Off-policy 校正](off-policy-correction.md)检查 support、policy lag 与 ratio，而不是把更多随机采样当作分布校正。

## 历史脉络

Bandit 理论先把 exploration–exploitation 写成可分析的 regret 问题，UCB 用 optimism under uncertainty 给出有限时间界。序贯 RL 随后发展出 optimism、posterior sampling、count bonus 和 information gain 等不同路线。Maximum-entropy control 则从 control-as-inference 视角把随机策略写进目标。2018 年 SAC 将这套目标与 off-policy stochastic actor–critic 结合，并在同年末扩展出自动 temperature 与现代常用配方。

## 常见误区

1. **高 entropy 就是有效探索。** policy 可能在同一狭小状态区域内随机。
2. **entropy bonus 不改变任务。** $\alpha>0$ 优化的是 regularized objective，其最优策略一般不同。
3. **reference KL 就是 entropy penalty。** 非均匀 reference 还贡献显式先验项。
4. **所有 SAC 公式都可混用。** 显式 value network、现代 twin-Q target 和自动 temperature 属于不同版本合同。
5. **`done` 可直接屏蔽 SAC bootstrap。** 真正 terminal 才屏蔽；time-limit truncation 通常仍 bootstrap。
6. **语言模型动作离散，所以离散 SAC 可直接处理整段回答。** 完整序列动作空间呈指数增长，粒度和 soft value 定义必须重新建立。

## Reference {#reference}

- Auer, Cesa-Bianchi, and Fischer, [Finite-time Analysis of the Multiarmed Bandit Problem](https://link.springer.com/article/10.1023/A%3A1013689704352)
- Sutton and Barto, [Reinforcement Learning: An Introduction, Second Edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
- Haarnoja et al., [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://proceedings.mlr.press/v80/haarnoja18b.html)
- Haarnoja et al., [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
- Haarnoja et al., [Official Soft Actor-Critic Implementation](https://github.com/haarnoja/sac)
- Levine, [Reinforcement Learning and Control as Probabilistic Inference](https://arxiv.org/abs/1805.00909)
- OpenAI, [Spinning Up: Soft Actor-Critic](https://spinningup.openai.com/en/latest/algorithms/sac.html)
