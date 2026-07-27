# Actor–Critic：用价值估计缩短信号路径

REINFORCE 等到整条轨迹结束，再用 Monte Carlo 回报评价每个动作；它无须学习环境模型，却把大量未来随机性都带进梯度。Actor–Critic 增加一个 critic，用当前状态的价值预测构造低方差 advantage，让 actor 不必只依赖终局回报。

这是一种统计分工，而不是两个模型名称的组合：actor 表示要优化的策略，critic 估计该策略下的回报。policy-gradient 的无偏起点见 [Policy Gradient](policy-gradient.md)；critic 提供的 advantage 如何进入受控更新，见 [Trust Region 与 PPO](trust-region-ppo.md)。

## 从 Monte Carlo 到 bootstrap

对固定策略 $\pi$，状态价值为

$$
V^\pi(s_t)
=\mathbb E_\pi[G_t\mid s_t].
$$

若完整 episode 已结束，可用 Monte Carlo target

$$
\widehat V_t^{\text{MC}}=G_t.
$$

它不依赖当前 value 预测，但同一状态后的所有环境与动作随机性都会进入 target。一步 TD 改用

$$
\widehat V_t^{\text{TD}}
=r_t+\gamma V_\phi(s_{t+1}),
$$

对应 TD residual

$$
\delta_t
=r_t+\gamma(1-d_t)V_\phi(s_{t+1})
-V_\phi(s_t),
$$

其中 $d_t=1$ 只表示真正终态。TD 用 bootstrap 降低方差，也引入来自 $V_\phi$ 的近似偏差。

Actor 以

$$
\widehat A_t\approx Q^\pi(s_t,a_t)-V^\pi(s_t)
$$

更新：

$$
\mathcal L_{\text{actor}}
=-\mathbb E_t
\left[
\log\pi_\theta(a_t\mid h_t)
\operatorname{stopgrad}(\widehat A_t)
\right].
$$

critic 则最小化 value regression：

$$
\mathcal L_{\text{critic}}
=\frac12
\mathbb E_t
\left[
V_\phi(h_t)-\widehat V_t
\right]^2.
$$

两个目标可以共享网络 trunk，但 target 的梯度边界必须清楚：bootstrap target 通常停止梯度，否则 critic 会通过目标分支追逐自身。完整双边界推导见[Advantage 估计与 GAE](advantage-estimation-gae.md#boundaries)。

## $n$-step return 与 GAE

一步 TD 信号短但偏差更依赖 critic；Monte Carlo 偏差小但方差高。$n$-step return 位于两者之间：

$$
\widehat V_t^{(n)}
=\sum_{l=0}^{n-1}\gamma^l r_{t+l}
+\gamma^nV_\phi(s_{t+n}).
$$

Generalized Advantage Estimation 用 TD residual 的指数加权和：

$$
\widehat A_t^{\gamma,\lambda}
=\sum_{l=0}^{T-t-1}
(\gamma\lambda)^l\delta_{t+l}.
$$

反向递推形式更适合实现：

$$
\widehat A_t
=\delta_t
+\gamma\lambda(1-b_t)\widehat A_{t+1},
$$

其中 $d_t$ 只控制当前 transition 是否 bootstrap，$b_t$ 则表示 trace 是否必须在这里停止。普通 transition 有 $d_t=b_t=0$，真正 terminal 有 $d_t=b_t=1$；具备真实 final observation 的 truncation 则是 $d_t=0,b_t=1$。因此一个 `done` 不能同时承担两个角色。

- $\lambda=0$ 接近一步 TD，方差低、critic 偏差影响大；
- $\lambda=1$ 在有限轨迹中接近 reward-to-go 减 value baseline；
- 中间值在 bias–variance 之间折中，但没有脱离 critic 质量。

常用 value target 为

$$
\widehat V_t^{\text{target}}
=\widehat A_t+V_\phi(s_t).
$$

这里右侧的旧 value 和 advantage 都应视作冻结 target。

## Terminal、truncation 与边界价值

结束标记至少分三类：

| 状态 | 是否环境终态 | 通常是否 bootstrap |
| --- | --- | --- |
| `terminated` | 是 | 否，边界值为零 |
| `truncated` | 否，只是预算或时间耗尽 | 是，若能估计边界状态 |
| `infrastructure_error` | 未形成有效策略结果 | 不应直接作为普通终止训练 |

若把所有 timeout 都设成 $d_t=1$，critic 会把“尚未完成但仍有希望”的尾部状态估成零；长任务越常被预算截断，偏差越强。反过来，在真正失败终态上继续 bootstrap 也会虚构未来价值。

## 语言模型中的 critic 放在哪里

语言策略可以在多种粒度上定义 value：

1. **response-level**：只在 prompt 末端预测整条回答的回报；
2. **token-level**：每个生成前缀都有一个 value；
3. **turn-level**：每条 assistant message 或工具 action 结束时预测；
4. **environment-state level**：工具执行后，根据新 observation 预测。

token-level critic 易与 causal LM hidden state 对齐，却不意味着每个 token 都是独立环境决策。若 reward 只在工具执行后出现，把同一 transition 任意摊给工具名、JSON 参数和解释文本会改变信用分配。更完整的粒度选择见[信用分配](credit-assignment.md)。

对包含 observation 的轨迹，critic 可以读取 observation，因为它估计条件状态价值；actor loss 仍只能覆盖 policy action。两套 mask 因此可能不同：

```text
attention mask: 可作为上下文读取的 token
action mask: policy 实际选择的 token
value mask: 需要训练或评估 value 的位置
bootstrap mask: 终态边界是否保留下一状态价值
```

把四者压成一个 `attention_mask` 是常见但危险的捷径。

## 一个最小 GAE 实现

```python
import torch
@torch.no_grad()
def generalized_advantage(reward, value, bootstrap_mask, trace_mask, gamma=.99, lam=.95):
    if (reward.ndim != 2 or value.shape != (*reward.shape[:-1], reward.shape[-1] + 1)
            or bootstrap_mask.shape != reward.shape or trace_mask.shape != reward.shape):
        raise ValueError("reward, masks and bootstrap value must align")
    if bootstrap_mask.dtype != torch.bool or trace_mask.dtype != torch.bool:
        raise ValueError("bootstrap and trace masks must be boolean")
    if torch.any(trace_mask & ~bootstrap_mask):
        raise ValueError("trace continuation requires a valid bootstrap transition")
    adv = torch.zeros_like(reward)
    carry = torch.zeros_like(reward[:, 0])
    for t in range(reward.shape[1] - 1, -1, -1):
        boot, trace = bootstrap_mask[:, t], trace_mask[:, t]
        next_value = torch.where(boot, value[:, t + 1], 0.)
        delta = reward[:, t] + gamma * next_value - value[:, t]
        carry = delta + gamma * lam * torch.where(trace, carry, 0.)
        adv[:, t] = carry
    target = adv + value[:, :-1]
    return adv, target
r = torch.tensor([[1.0, 2.0]])
v = torch.tensor([[0.3, 0.5, float("nan")]], requires_grad=True)
bootstrap = torch.tensor([[True, False]])
trace = torch.tensor([[True, False]])
a, target = generalized_advantage(
    r, v, bootstrap, trace, gamma=1.0, lam=1.0
)
assert torch.allclose(a, torch.tensor([[2.7, 1.5]]))
assert torch.allclose(target, torch.tensor([[3.0, 2.0]]))
assert torch.isfinite(a).all() and not a.requires_grad and not target.requires_grad
try: generalized_advantage(r, v, bootstrap, torch.tensor([[True, True]]))
except ValueError: pass
else: raise AssertionError("terminal transitions cannot continue the trace")
```

正常 transition 的两个 mask 都为真，真实 terminal 都为假；有真实 final observation 的 truncation 则 `bootstrap=True, trace=False`。这个实现假设 batch 中每条轨迹已经按时间对齐。padding 位置必须在外层用有效步 mask 排除，不能仅靠把 reward 置零；否则 padding 上的 value、loss 分母和递推仍可能污染结果。完整推导与边界测试见[Advantage 估计与 GAE](advantage-estimation-gae.md)。

packed trajectory、bootstrap 与 valid-step mask 的组合实现见[手撕：LLM 策略优化 · GAE](../practice/llm-policy-optimization.md#gae)。

## Critic 的训练节奏

critic 面对的目标分布随 actor 变化。常见稳定手段包括：

- 在 actor 更新前用同批或更广轨迹预热 critic；
- 让 critic 每轮比 actor 多做若干受控更新；
- 限制 actor KL，使 value target 不至于瞬间漂移；
- 按任务难度、轨迹长度和终止类型检查 explained variance；
- 对旧 policy 的数据记录 staleness，而非混进同一回归集；
- 在共享 trunk 时分别监控 actor 与 critic 梯度范数。

训练得更久不一定更好。critic 对一批 Monte Carlo return 过拟合，会在同批轨迹上给出漂亮 loss，却对更新后的 actor 和新状态产生系统偏差。长时异步场景中的 critic、policy lag 和 trajectory freshness 见[训练系统](../agentic-rl/training-systems.md)。

## 实现契约

可审计的 actor–critic batch 至少包含：

```text
ordered state/action transitions
reward components before weighting
terminated and truncated separately
behavior policy revision and action log-probability
value revision used to build targets
gamma, lambda, horizon and bootstrap convention
attention, action, value and valid-step masks
advantage normalization scope
actor/critic update counts and optimizer revisions
```

最小测试应覆盖：

1. 一步终态的 target 等于即时奖励；
2. truncation 保留给定的边界 value；
3. padding 不进入递推、value loss 或分母；
4. 给所有 advantage 加零均值 baseline 不改变符号预期；
5. critic target 分支没有梯度；
6. 保存再恢复后，value revision 与 rollout revision 仍可辨认。

## 失败边界

- **critic 被当成真值**：它只是当前策略分布上的近似，分布外状态可能任意错误。
- **终态和截断合并**：bootstrap 语义错误。
- **actor loss 回传进 advantage**：策略可通过修改 critic 来“改善”自身目标。
- **value target 未停止梯度**：网络同时移动预测与靶点。
- **只看 value MSE**：回报尺度大时 MSE 难解释；还需 explained variance、分层校准和策略结果。
- **advantage 在全局随意 whitening**：跨任务难度与长度的相对权重被改变。
- **共享 trunk 互相拖拽**：critic 梯度可能破坏语言表示，actor 梯度也可能让 value 退化。
- **旧轨迹无限复用**：critic 和 actor 都离开声明的 on-policy 分布；校正见 [Off-policy 校正](off-policy-correction.md)。

## 历史位置

早期 actor–critic 把 policy-gradient actor 与 TD critic 组合起来；A3C 展示了并行 actor 产生多样经验、异步更新共享模型的深度 RL 实现；GAE 则把多步 TD residual 组织成可调的 advantage estimator。现代语言模型 RL 的模型更大、动作更长、rollout 更昂贵，但核心矛盾没有改变：critic 用偏差换方差，actor 的更新速度又决定 critic 的目标漂移速度。

## Reference {#reference}

- [Konda and Tsitsiklis, Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1999/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- [Mnih et al., Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [Schulman et al., High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [Sutton et al., Policy Gradient Theorems for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
