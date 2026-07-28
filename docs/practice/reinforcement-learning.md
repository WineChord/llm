# 手撕：强化学习

强化学习实现最容易混淆的是时间语义：奖励属于哪次转移，终止状态是否 bootstrap，数据由哪一个 policy 产生，以及 loss 对哪些动作取平均。本页用小状态空间和小张量固定这些契约；代码是可执行的语义 reference，不是并行采样器或生产 trainer。

每段断言都在验证一个不变量：Bellman fixed point、terminal mask、baseline 梯度、PPO 裁剪方向、behavior support、action mask 或 reference KL。改写为向量化和分布式实现后，这些断言仍应成立。

## 有限 MDP 与 Value Iteration

有限 MDP 的 Bellman optimality backup 为

$$
Q(s,a)=\sum_{s'}P(s'|s,a)\left[R(s,a,s')+\gamma V(s')\right],
\qquad
V(s)=\max_a Q(s,a).
$$

下面的三状态 MDP 中，状态 $2$ 是吸收终态。状态 $0$ 直接结束可得 $1$，先去状态 $1$ 再结束可得折扣后的 $1.8$，因此最优动作是后者。

```python
import torch
def value_iteration(transition, reward, gamma=0.9, tol=1e-12):
    """transition/reward:[S,A,S] -> value:[S], greedy policy:[S]."""
    if transition.shape != reward.shape or transition.ndim != 3:
        raise ValueError("expected matching [S,A,S] tensors")
    torch.testing.assert_close(
        transition.sum(-1), torch.ones_like(transition.sum(-1))
    )
    value = torch.zeros(transition.size(0), dtype=transition.dtype)
    while True:
        q = (transition * (reward + gamma * value[None, None, :])).sum(-1)
        updated = q.max(-1).values
        if (updated - value).abs().max() < tol:
            return updated, q.argmax(-1)
        value = updated
p = torch.zeros(3, 2, 3, dtype=torch.float64)
r = torch.zeros_like(p)
p[0, 0, 1], p[0, 1, 2] = 1, 1
p[1, 0, 2], p[1, 1, 2] = 1, 1
p[2, :, 2] = 1
r[0, 1, 2], r[1, 0, 2] = 1, 2
v, greedy = value_iteration(p, r)
torch.testing.assert_close(v, torch.tensor([1.8, 2.0, 0.0], dtype=v.dtype))
assert greedy.tolist() == [0, 0, 0]
```

这个 reference 已知完整转移矩阵；model-free 方法则只能从采样轨迹估计同一个固定点。

## Monte Carlo 与 TD(0)

Monte Carlo 等 episode 结束后使用完整 return：

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}R_k.
$$

TD(0) 在每次转移后 bootstrap：

$$
V(S_t)\leftarrow V(S_t)+
\alpha\left(R_t+\gamma(1-d_t)V(S_{t+1})-V(S_t)\right).
$$

```python
import torch
def returns(reward, gamma):
    out, carry = torch.empty_like(reward), reward.new_zeros(())
    for t in range(reward.numel() - 1, -1, -1):
        carry = reward[t] + gamma * carry
        out[t] = carry
    return out
def first_visit_mc(states, reward, gamma):
    value, count, seen = torch.zeros(2), torch.zeros(2), set()
    ret = returns(reward, gamma)
    for t, state in enumerate(states.tolist()):
        if state not in seen:
            seen.add(state)
            count[state] += 1
            value[state] += (ret[t] - value[state]) / count[state]
    return value
def td0(states, next_states, reward, done, gamma, alpha, episodes):
    value = torch.zeros(2)
    for _ in range(episodes):
        for state, nxt, rew, terminal in zip(states, next_states, reward, done):
            target = rew if terminal else rew + gamma * value[nxt]
            value[state] += alpha * (target - value[state])
    return value
states = torch.tensor([0, 1])
next_states = torch.tensor([1, 0])
reward = torch.tensor([0.0, 1.0])
done = torch.tensor([False, True])
mc = first_visit_mc(states, reward, gamma=0.9)
td = td0(states, next_states, reward, done, gamma=0.9, alpha=0.2, episodes=100)
torch.testing.assert_close(mc, torch.tensor([0.9, 1.0]))
torch.testing.assert_close(td, mc, atol=1e-5, rtol=0)
```

真正终止时乘数为零；time-limit truncation 通常仍应从最终 observation bootstrap，但 trace 不应跨到下一条 episode。

## n-step Return 与 Eligibility Trace {#n-step-return}

$n$-step return 在 Monte Carlo 与一步 TD 之间插值：

$$
G_t^{(n)}
=\sum_{k=0}^{n-1}\gamma^kR_{t+k}
+\gamma^nV(S_{t+n}).
$$

Backward-view TD($\lambda$) 用 eligibility trace 将后来的 TD error 传播给较早访问的状态：

```python
import torch
def n_step_return(reward, value, start, steps, gamma):
    end = min(start + steps, reward.numel())
    result = reward.new_zeros(())
    for t in range(start, end):
        result += gamma ** (t - start) * reward[t]
    if start + steps < reward.numel():
        result += gamma ** steps * value[start + steps]
    return result
def td_lambda_episode(gamma, lam, alpha):
    value, trace = torch.zeros(2), torch.zeros(2)
    transitions = [(0, 0.0, 1, False), (1, 1.0, 0, True)]
    for state, reward, nxt, done in transitions:
        target = reward if done else reward + gamma * value[nxt]
        delta = target - value[state]
        trace.mul_(gamma * lam)
        trace[state] += 1
        value.add_(trace, alpha=alpha * delta)
    return value
reward = torch.tensor([0.0, 0.0, 1.0])
value = torch.tensor([0.5, 0.6, 0.7])
torch.testing.assert_close(
    n_step_return(reward, value, 0, 2, 0.9), torch.tensor(0.9 ** 2 * 0.7)
)
torch.testing.assert_close(
    n_step_return(reward, value, 0, 3, 0.9), torch.tensor(0.9 ** 2)
)
td_zero = td_lambda_episode(gamma=0.9, lam=0.0, alpha=0.1)
td_one = td_lambda_episode(gamma=0.9, lam=1.0, alpha=0.1)
torch.testing.assert_close(td_zero, torch.tensor([0.0, 0.1]))
torch.testing.assert_close(td_one, torch.tensor([0.09, 0.1]))
```

这里使用 accumulating trace。Replacing trace、Dutch trace、off-policy trace 与 function approximation 各有额外语义，不能只改一个 $\lambda$ 参数便视作同一算法。

## REINFORCE

[REINFORCE](https://link.springer.com/article/10.1007/BF00992696) 直接用采样 return 加权 score function：

$$
\nabla_\theta J
=\mathbb E\left[
(G_t-b(S_t))\nabla_\theta\log\pi_\theta(A_t|S_t)
\right].
$$

只要 baseline 不依赖当前采样动作，它可以降方差而不改变期望。下面的两臂 bandit 梯度会提高高回报动作的 logit。

```python
import torch
from torch.distributions import Categorical
logits = torch.zeros(2, requires_grad=True)
actions = torch.tensor([0, 1])
episode_return = torch.tensor([1.0, -1.0])
baseline = episode_return.mean()
policy = Categorical(logits=logits.expand(actions.numel(), -1))
advantage = episode_return - baseline
loss = -(policy.log_prob(actions) * advantage.detach()).mean()
loss.backward()
assert logits.grad[0] < 0
assert logits.grad[1] > 0
with torch.no_grad():
    updated = logits - 0.1 * logits.grad
assert updated[0] > updated[1]
```

梯度下降沿负梯度更新，因此第一个 logit 的负梯度意味着其概率上升。语言模型中的 episode return 常包含任务 reward 与 reference-policy penalty，二者的时间分配必须显式定义。

## One-step Actor–Critic

Actor–critic 用 learned value 构造状态相关 baseline。最小的一步目标为

$$
y_t=R_t+\gamma(1-d_t)V_{\bar\phi}(S_{t+1}),
\qquad
\widehat A_t=y_t-V_\phi(S_t).
$$

target 与 actor advantage 通常停止梯度，避免 policy loss 反向修改 critic 或 bootstrap target。

```python
import torch
import torch.nn.functional as F
def actor_critic_loss(logits, value, state, action, reward, nxt, done, gamma):
    bootstrap = torch.where(done, torch.zeros_like(reward), value[nxt].detach())
    target = reward + gamma * bootstrap
    advantage = target - value[state]
    logp = F.log_softmax(logits[state], -1).gather(1, action[:, None]).squeeze(1)
    actor = -(logp * advantage.detach()).mean()
    critic = 0.5 * advantage.square().mean()
    return actor + critic, target
logits = torch.zeros(2, 2, requires_grad=True)
value = torch.tensor([0., 0., float("nan")], requires_grad=True)
state = torch.tensor([0, 1])
action = torch.tensor([0, 1])
reward = torch.tensor([0.0, 1.0])
nxt = torch.tensor([1, 2])
done = torch.tensor([False, True])
loss, target = actor_critic_loss(
    logits, value, state, action, reward, nxt, done, gamma=0.9
)
assert not target.requires_grad
loss.backward()
torch.testing.assert_close(logits.grad[0], torch.zeros(2))
assert logits.grad[1, 1] < 0
assert value.grad[1] < 0
assert value.grad[2] == 0 and torch.isfinite(target).all()
```

共享 actor/critic backbone 时，“detach target”并不会自动隔离共享参数上的两种 loss；需要用 loss 权重、更新频率和梯度统计判断相互干扰。

## PPO Clipping

[PPO](https://arxiv.org/abs/1707.06347) 使用 old policy ratio：

$$
\rho_t=\exp(\log\pi_\theta-\log\pi_{\mathrm{old}}),
\qquad
L^{\mathrm{clip}}
=\min\left(\rho_t\widehat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t\right).
$$

裁剪方向依赖 advantage 符号，并不是把所有越界 ratio 都置零。

```python
import torch
def ppo_loss(new_logp, old_logp, advantage, eps):
    ratio = (new_logp - old_logp).exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - eps, 1 + eps) * advantage
    objective = torch.minimum(unclipped, clipped)
    return -objective.mean(), objective
ratio = torch.tensor([1.3, 0.7, 1.3, 0.7])
new_logp = ratio.log().requires_grad_()
old_logp = torch.zeros_like(new_logp)
advantage = torch.tensor([1.0, 1.0, -1.0, -1.0])
loss, objective = ppo_loss(new_logp, old_logp, advantage, eps=0.2)
torch.testing.assert_close(
    objective, torch.tensor([1.2, 0.7, -1.3, -0.8])
)
loss.backward()
torch.testing.assert_close(new_logp.grad[[0, 3]], torch.zeros(2))
assert new_logp.grad[1] < 0
assert new_logp.grad[2] > 0
```

PPO clipping 只是 surrogate 的单侧饱和，不保证实际 KL 落在固定区间；训练仍应观察 ratio 分布、KL、clip fraction、entropy 与有效 token denominator。

## Importance Sampling 与 Off-policy 数据

若样本来自 behavior policy $\mu$，目标 policy 为 $\pi$：

$$
w(a)=\frac{\pi(a)}{\mu(a)},
\qquad
\mathbb E_{a\sim\mu}[w(a)R(a)]
=\mathbb E_{a\sim\pi}[R(a)].
$$

前提是 $\mu(a)>0$ 覆盖目标 policy 的支持集。权重裁剪降低方差，却通常引入偏差。

```python
import torch
def importance_estimate(action, reward, target, behavior, clip=None):
    if torch.any(behavior[action] <= 0):
        raise ValueError("behavior policy lacks target support")
    weight = target[action] / behavior[action]
    if clip is not None:
        weight = weight.clamp_max(clip)
    ordinary = (weight * reward).mean()
    normalized = (weight * reward).sum() / weight.sum()
    ess = weight.sum().square() / weight.square().sum()
    return ordinary, normalized, ess
action = torch.tensor([0, 1])
reward = torch.tensor([1.0, 0.0])
target = torch.tensor([0.8, 0.2])
behavior = torch.tensor([0.5, 0.5])
ordinary, normalized, ess = importance_estimate(
    action, reward, target, behavior
)
torch.testing.assert_close(ordinary, torch.tensor(0.8))
torch.testing.assert_close(normalized, torch.tensor(0.8))
torch.testing.assert_close(ess, torch.tensor(4.0 / 2.72))
clipped, _, _ = importance_estimate(
    action, reward, target, behavior, clip=1.0
)
torch.testing.assert_close(clipped, torch.tensor(0.5))
try:
    importance_estimate(torch.tensor([0]), torch.ones(1), target, torch.tensor([0.0, 1.0]))
except ValueError:
    pass
else:
    raise AssertionError("unsupported action was accepted")
```

长序列的 trajectory weight 是 token ratio 的乘积，方差会随 horizon 急剧放大。[V-trace](https://arxiv.org/abs/1802.01561)、截断 IS 与 token rejection 是不同的偏差—方差选择。

## LLM Action Mask {#llm-action-mask}

Decoder 在位置 $t$ 的 logits 预测 token $t+1$。prompt、padding 与工具 observation 可以进入状态，却不应自动进入 policy loss：

```python
import torch
import torch.nn.functional as F
def masked_action_logprob(logits, tokens, action_mask):
    """logits:[B,T-1,V], tokens/action_mask:[B,T] -> sequence sums, counts."""
    if logits.shape[:2] != (tokens.size(0), tokens.size(1) - 1):
        raise ValueError("logits must predict tokens[:,1:]")
    if tokens.shape != action_mask.shape:
        raise ValueError("tokens and action_mask must align")
    target, mask = tokens[:, 1:], action_mask[:, 1:].bool()
    count = mask.sum(-1)
    if torch.any(count == 0):
        raise ValueError("each sequence needs at least one action token")
    selected_logits, selected_tokens = logits[mask], target[mask]
    if not torch.isfinite(selected_logits).all():
        raise ValueError("selected action logits must be finite")
    if torch.any((selected_tokens < 0) | (selected_tokens >= logits.size(-1))):
        raise ValueError("selected action token is outside the vocabulary")
    chosen = F.log_softmax(selected_logits, -1).gather(1, selected_tokens[:, None]).squeeze(1)
    row = torch.arange(logits.size(0), device=logits.device)[:, None].expand_as(mask)[mask]
    return logits.new_zeros(logits.size(0)).scatter_add(0, row, chosen), count
tokens = torch.tensor([[0, -100, 2, 1]])
mask = torch.tensor([[False, False, True, True]])
logits = torch.zeros(1, 3, 3)
logits[:, 0] = float("nan")
base, count = masked_action_logprob(logits, tokens, mask)
prompt_changed = logits.clone()
prompt_changed[:, 0] = torch.tensor([9.0, -9.0, -9.0])
same, _ = masked_action_logprob(prompt_changed, tokens, mask)
torch.testing.assert_close(base, same)
action_changed = logits.clone()
action_changed[:, 1] = torch.tensor([9.0, -9.0, -9.0])
different, _ = masked_action_logprob(action_changed, tokens, mask)
assert not torch.allclose(base, different)
assert count.item() == 2
```

mask 与 exact token IDs 必须来自 rollout，而不是训练时重新套模板或重新 tokenize。全局 token mean 应先汇总所有 rank 的 loss numerator 和有效 token denominator，再相除。

## Reference-policy KL {#reference-policy-kl}

LLM post-training 常用固定 reference policy 约束漂移：

$$
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
=\sum_a\pi_\theta(a|s)
\left[\log\pi_\theta(a|s)-\log\pi_{\mathrm{ref}}(a|s)\right].
$$

exact KL 非负，但单个采样动作的 log-ratio 可以为负。后者只有在动作确实来自 $\pi_\theta$ 时，样本均值才估计 forward KL。

<details class="code-disclosure">
<summary id="reference-policy-kl-code">Exact KL 与 sampled log-ratio <span class="code-disclosure__meta">Python · 43 行</span></summary>
<div class="code-disclosure__body" markdown="1">

```python
import torch
import torch.nn.functional as F
def select_distributions(policy, reference, action_mask):
    if policy.shape != reference.shape or policy.shape[:-1] != action_mask.shape:
        raise ValueError("policy, reference and mask shapes must align")
    valid = action_mask.bool()
    if torch.any(valid.sum(-1) == 0):
        raise ValueError("every sequence needs a KL action")
    policy, reference = policy[valid], reference.detach()[valid]
    invalid = lambda x: torch.isnan(x) | torch.isposinf(x)
    if invalid(policy).any() or invalid(reference).any():
        raise ValueError("selected logits cannot contain NaN or positive infinity")
    policy_support, reference_support = ~torch.isneginf(policy), ~torch.isneginf(reference)
    if torch.any(~policy_support.any(-1)) or torch.any(policy_support & ~reference_support):
        raise ValueError("policy support must be non-empty and covered by reference")
    return policy, reference, policy_support, valid
def masked_exact_kl(policy, reference, action_mask):
    policy, reference, support, _ = select_distributions(policy, reference, action_mask)
    logp, logq = F.log_softmax(policy, -1), F.log_softmax(reference, -1)
    log_ratio = torch.where(support, logp, 0.) - torch.where(support, logq, 0.)
    return (torch.where(support, logp, -torch.inf).exp() * log_ratio).sum(-1).mean()
def sampled_log_ratio(policy, reference, action, action_mask):
    if action.shape != action_mask.shape:
        raise ValueError("one sampled action is required per prefix")
    policy, reference, support, valid = select_distributions(policy, reference, action_mask)
    action = action[valid]
    if torch.any((action < 0) | (action >= policy.size(-1))):
        raise ValueError("sampled action is outside the vocabulary")
    if not support.gather(1, action[:, None]).all():
        raise ValueError("sampled action must lie in policy support")
    logp, logq = F.log_softmax(policy, -1), F.log_softmax(reference, -1)
    return (logp - logq).gather(1, action[:, None]).mean()
policy = torch.tensor([[[2., 0., -torch.inf], [float("nan")] * 3]], requires_grad=True)
reference = torch.tensor([[[0., 0., -torch.inf], [float("nan")] * 3]])
mask, action = torch.tensor([[True, False]]), torch.tensor([[1, -100]])
exact = masked_exact_kl(policy, reference, mask)
sampled = sampled_log_ratio(policy, reference, action, mask)
assert exact > 0 and sampled < 0
subset = masked_exact_kl(torch.tensor([[[0., -torch.inf]]]),
                         torch.tensor([[[0., 0.]]]), torch.tensor([[True]]))
torch.testing.assert_close(subset, torch.log(torch.tensor(2.)))
exact.backward()
torch.testing.assert_close(policy.grad[:, 1], torch.zeros(1, 3))
```

</div>
</details>

reference policy、产生样本的 behavior policy 与 PPO old policy 是三个不同角色，即使某个同步实现偶尔让它们指向同一 checkpoint。相关目标与 mask 细节见[手撕训练目标](training-objectives.md)和[在线强化学习](../training/online-rl.md)。

## Reference {#reference}

- Sutton and Barto, [Reinforcement Learning: An Introduction, Second Edition](https://incompleteideas.net/book/the-book-2nd.html)。
- Sutton, [Learning to Predict by the Methods of Temporal Differences](https://link.springer.com/article/10.1007/BF00115009)。
- Williams, [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696)。
- Konda and Tsitsiklis, [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper_files/paper/1999/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)。
- Schulman et al., [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)。
- Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)。
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)。
- Espeholt et al., [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)。
- Ziegler et al., [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)。
