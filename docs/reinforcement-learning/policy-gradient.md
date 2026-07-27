# Policy Gradient：直接沿着行为概率学习

价值方法先问“哪个动作更值”，再由价值诱导策略；policy gradient 则把策略本身写成可微分分布，直接提高高回报轨迹的概率。这个转向对语言模型尤其重要：词表虽然有限，但完整回答、工具调用和多轮轨迹的组合空间几乎无法枚举，显式求解每个完整动作的 $Q$ 值通常并不现实。

本页从 likelihood-ratio trick 推出 REINFORCE，再解释 reward-to-go、baseline、序列 mask 与可变长度归一化。状态、动作和终止怎样映射到语言模型，先见[语言模型作为策略](language-model-policy.md)；value function 怎样进一步降低方差，继续读 [Actor–Critic](actor-critic.md)。

## 从期望回报到可采样梯度

设有限时域轨迹为

$$
\tau=(s_0,a_0,r_0,\ldots,s_T),
$$

策略与环境共同给出轨迹概率

$$
p_\theta(\tau)
=\rho_0(s_0)
\prod_{t=0}^{T-1}
\pi_\theta(a_t\mid h_t)
P(s_{t+1}\mid s_t,a_t),
$$

其中 $h_t$ 可以是 Markov state，也可以是 POMDP 中的完整可见历史。目标是

$$
J(\theta)
=\mathbb E_{\tau\sim p_\theta}
\left[R(\tau)\right].
$$

对期望直接求导：

$$
\begin{aligned}
\nabla_\theta J
&=\int \nabla_\theta p_\theta(\tau)R(\tau)\,d\tau\\
&=\int p_\theta(\tau)
\nabla_\theta\log p_\theta(\tau)
R(\tau)\,d\tau\\
&=\mathbb E_\tau
\left[
\nabla_\theta\log p_\theta(\tau)R(\tau)
\right].
\end{aligned}
$$

若环境转移不依赖 $\theta$，只有策略项留下：

$$
\nabla_\theta\log p_\theta(\tau)
=\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t\mid h_t).
$$

于是得到最朴素的 REINFORCE estimator：

$$
\widehat g
=\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid h_t)
R(\tau).
$$

它只需要从当前策略采样、计算 log-probability 并观察回报，不要求对环境求导。代价是方差很高：一条轨迹中所有动作都乘上同一个随机回报，早期动作还会被与它无关的先前奖励干扰。

## 因果性带来 reward-to-go

动作 $a_t$ 不会改变它发生之前的奖励，因此可把整轨迹回报替换为从 $t$ 开始的回报：

$$
G_t
=\sum_{k=t}^{T-1}
\gamma^{k-t}r_k.
$$

梯度估计变成

$$
\widehat g
=\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid h_t)G_t.
$$

这不是启发式删项。对 $k<t$，在给定 $h_t$ 时有

$$
\mathbb E_{a_t\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(a_t\mid h_t)
\right]
=\nabla_\theta\sum_a\pi_\theta(a\mid h_t)
=0,
$$

所以过去奖励对应项的期望为零，只增加采样噪声。

折扣 $\gamma$ 同时定义任务偏好与估计尺度。把 $\gamma<1$ 当作纯数值技巧会改变最优策略；在固定长度问答中常取 $\gamma=1$，而长时环境中的时间成本、风险与截断语义需要显式建模。

## Baseline 为什么不引入偏差

可以减去任何不依赖当前动作的 baseline $b(h_t)$：

$$
\widehat g
=\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid h_t)
\left(G_t-b(h_t)\right).
$$

因为

$$
\mathbb E_{a_t\sim\pi_\theta}
\left[
b(h_t)\nabla_\theta\log\pi_\theta(a_t\mid h_t)
\right]=0.
$$

baseline 不改变期望，却能显著降低方差。常见选择包括：

- 整个 batch 的平均回报；
- 同一 prompt 其他采样的 leave-one-out 均值；
- greedy response 的回报；
- 学习得到的状态价值 $V_\phi(h_t)$。

前三者仍是 Monte Carlo baseline；最后一种把方法带到 actor–critic。组均值、组标准差和长度归一化都会改变估计量的尺度与样本权重，不能统称为“只是 normalization”。无 critic 的语言模型基线见[无 critic 的基线](critic-free-baselines.md)。

## Policy-gradient theorem 的视角

对持续任务，可以用折扣状态访问分布

$$
d^\pi_\gamma(s)
=(1-\gamma)\sum_{t\ge0}\gamma^t
P(s_t=s\mid\pi)
$$

写成

$$
\nabla_\theta J
\propto
\mathbb E_{
s\sim d^\pi_\gamma,\,
a\sim\pi_\theta
}
\left[
\nabla_\theta\log\pi_\theta(a\mid s)
Q^\pi(s,a)
\right].
$$

定理最重要的结论不是一个新公式，而是：虽然状态分布也随策略变化，最终梯度仍可由访问到的状态、采样动作和对应 $Q^\pi$ 表达。这为 actor–critic、natural policy gradient 和 trust-region 方法提供了共同起点。

## 语言模型中的“一个动作”

给定 prompt $x$ 和输出 token $y_{1:L}$，

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{L}
\log\pi_\theta(y_t\mid x,y_{<t}).
$$

若整条回答只有一个结果奖励 $R$，最简单的序列级 REINFORCE 为

$$
\mathcal L_{\text{PG}}
=-
\frac{
\sum_{i,t}m_{i,t}
\log\pi_\theta(y_{i,t}\mid h_{i,t})
A_i
}{
\sum_{i,t}m_{i,t}
},
$$

其中 $m_{i,t}$ 只覆盖 policy 实际采样的 token。system、user、工具 observation、padding 与仅用于条件化的前缀都不是动作。

这里有三个不同但经常被混在一起的约定：

1. **token mean**：每个有效 token 等权，长序列贡献更多 token；
2. **sequence mean**：先对每条序列求和或求均值，再在序列间平均；
3. **action-span mean**：工具调用或一轮消息作为一个动作单位。

三者对应不同优化目标。尤其当同一个 sequence reward 被复制到所有 token 时，先求和会让长回答获得更大梯度；先做 token mean 又会改变原始序列 log-probability 的尺度。归一化方式必须与实验口径一起保存。

## 一个最小、可检查的实现

下面的实现接收已经对齐的 action-token log-probability、advantage 和 mask。它不负责采样，也不尝试从文本猜测哪些 token 属于 policy。

```python
import torch

def policy_gradient_loss(logp, adv, action_mask):
    """logp, adv, action_mask: [batch, time]."""
    if logp.shape != adv.shape or logp.shape != action_mask.shape:
        raise ValueError("shape mismatch")
    mask = action_mask.to(logp.dtype)
    denom = mask.sum().clamp_min(1)
    return -((logp * adv.detach()) * mask).sum() / denom

logp = torch.tensor([[-0.2, -0.7, -0.4]], requires_grad=True)
adv = torch.tensor([[1.5, 1.5, 1.5]])
mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
loss = policy_gradient_loss(logp, adv, mask)
loss.backward()
assert logp.grad[0, 0] == 0
assert torch.all(logp.grad[0, 1:] < 0)
```

`adv.detach()` 很关键：policy loss 不应通过 advantage target 反向修改 reward 或 critic。若需要端到端可微环境，那已经是另一类 estimator，不能悄悄沿用这里的推导。

## 实现契约

一次可重放的 policy-gradient update 至少固定：

```text
policy revision and tokenizer/template revision
sampled action IDs and exact action mask
sampling distribution and decoding processors
terminal / truncated / invalid / infrastructure status
reward components and reward/verifier revision
return convention: gamma, bootstrap and normalization
loss reduction: token / sequence / action span
```

训练使用的 `logp` 必须对应实际采样分布。若 rollout 经 temperature、top-$k$、top-$p$ 或 grammar mask 后采样，却用未截断基础分布冒充 behavior probability，后续 importance ratio 就没有声明中的语义。详细版本边界见[轨迹与策略契约](../agentic-rl/trajectory-contract.md)。

## 失败边界

- **把 loss 符号写反**：正 advantage 应提高动作 log-probability。
- **advantage 未停止梯度**：actor loss 意外训练 critic 或 reward model。
- **prompt 进入 action mask**：模型被奖励去“生成”它从未选择的条件 token。
- **把所有 timeout 当 terminal**：本可继续的状态失去 bootstrap，回报系统性偏低。
- **跨 prompt 直接比较 raw reward**：难度差异主导梯度，需有合适 baseline 或分层。
- **标准化后声称 estimator 未变**：组标准差、长度均值和样本过滤都会改变权重。
- **使用过时轨迹却不校正**：上述推导是 on-policy；异步数据见[Off-policy 校正](off-policy-correction.md)。
- **只提高采样 reward**：策略可能利用 reward 漏洞；独立评测与诊断见[强化学习评测与调试](evaluation-debugging.md)。

## 从 REINFORCE 向后走

Williams 的 REINFORCE 给出了简单、通用的 likelihood-ratio estimator；policy-gradient theorem 把它整理为状态访问分布下的局部期望。接下来的发展主要沿两条线降低它的实际代价：

- 用学习得到的 value baseline 降方差，形成 [Actor–Critic](actor-critic.md)；
- 限制一次更新对策略分布的破坏，形成 [Trust Region 与 PPO](trust-region-ppo.md)。

它们没有改变“提高有正优势动作的概率”这一核心，只是重新设计 advantage 和更新步长。

## Reference {#reference}

- [Williams, Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696)
- [Sutton et al., Policy Gradient Theorems for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
- [Schulman et al., Gradient Estimation Using Stochastic Computation Graphs](https://arxiv.org/abs/1506.05254)
- [Sutton and Barto, Reinforcement Learning: An Introduction, Second Edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
