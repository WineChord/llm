# 手撕：LLM 策略优化

语言模型强化学习的公式通常不难，难的是让公式、张量和 rollout 语义完全对齐。一次训练更新至少同时涉及四种坐标：样本属于哪个 prompt group，哪些位置真是 policy action，数据由哪个 policy 产生，以及最终 loss 按 token、response 还是 prompt 归约。

本页不实现分布式 trainer，而用小张量冻结十个最容易漂移的接口。完整概念分别见 [GAE](../reinforcement-learning/advantage-estimation-gae.md)、[PPO](../reinforcement-learning/trust-region-ppo.md)、[GRPO](../reinforcement-learning/grpo.md)和 [ratio、clipping 与 gate](../reinforcement-learning/ratio-clipping-gating.md)。

## Packed trajectory 上的 GAE {#gae}

`terminated` 决定 TD target 是否 bootstrap，`boundary` 决定 trace 是否继续。time-limit truncation 通常令 `terminated=False, boundary=True`：它使用真实 final observation 的 value，却不能把下一条物理相邻样本接进来。

```python
import torch
def packed_gae(reward, value, next_value, terminated, boundary, gamma=.99, lam=.95):
    delta = reward + gamma * (~terminated) * next_value - value
    advantage = torch.empty_like(delta)
    carry = delta.new_zeros(())
    for t in range(delta.numel() - 1, -1, -1):
        carry = delta[t] + gamma * lam * (~boundary[t]) * carry
        advantage[t] = carry
    return advantage
reward = torch.tensor([1., 2., 10.])
value = torch.tensor([.5, 1., 3.])
next_value = torch.tensor([1., 4., 99.])
terminated = torch.tensor([False, False, True])
boundary = torch.tensor([False, True, True])
advantage = packed_gae(reward, value, next_value, terminated, boundary, .5, .8)
delta = reward + .5 * (~terminated) * next_value - value
torch.testing.assert_close(advantage, torch.tensor([delta[0] + .4 * delta[1], delta[1], delta[2]]))
assert delta[1] == 3
assert delta[2] == 7
```

真实 LLM batch 还需要 `trajectory_id`、action mask 和 final-observation value。只用一个 `done` 往往无法同时表达 bootstrap 与 trace reset。

## PPO 的符号相关裁剪

PPO 的 `min` 不是装饰。正 advantage 只截住过度增大的 ratio，负 advantage 只截住过度减小的 ratio；另一侧仍保留梯度。

```python
import torch
def ppo_objective(new_logp, old_logp, advantage, eps=.2):
    ratio = (new_logp - old_logp).exp()
    raw = ratio * advantage
    clipped = ratio.clamp(1 - eps, 1 + eps) * advantage
    return torch.minimum(raw, clipped), ratio
ratio = torch.tensor([1.3, .7, 1.3, .7])
new_logp = ratio.log().requires_grad_()
objective, observed = ppo_objective(new_logp, torch.zeros(4), torch.tensor([1., 1., -1., -1.]))
torch.testing.assert_close(objective, torch.tensor([1.2, .7, -1.3, -.8]))
(-objective.mean()).backward()
torch.testing.assert_close(new_logp.grad[[0, 3]], torch.zeros(2))
assert new_logp.grad[1] < 0
assert new_logp.grad[2] > 0
```

因此“先把 ratio 全部 clamp，再乘 advantage”不是同一个 surrogate。PPO 也不把任意陈旧 rollout 变回 on-policy 数据；它只让当前样本上的局部改进更保守。

## RLOO 与 GRPO 的组内信号 {#rloo-grpo}

RLOO 不把当前样本放进 baseline；GRPO 常用组均值与组标准差。二者都要求同一 prompt 下有多个独立 rollout，但其缩放语义不同。

```python
import torch
def rloo_advantage(reward):
    if reward.size(1) < 2:
        raise ValueError("RLOO needs at least two rollouts")
    return reward - (reward.sum(1, keepdim=True) - reward) / (reward.size(1) - 1)
def grpo_advantage(reward, eps=1e-6):
    centered = reward - reward.mean(1, keepdim=True)
    scale = reward.std(1, keepdim=True, unbiased=False)
    return torch.where(scale > eps, centered / scale, torch.zeros_like(centered))
reward = torch.tensor([[0., 1., 1.], [1., 1., 1.]])
rloo = rloo_advantage(reward)
grpo = grpo_advantage(reward)
torch.testing.assert_close(rloo[0], 1.5 * (reward[0] - reward[0].mean()))
torch.testing.assert_close(grpo.mean(1), torch.zeros(2))
assert torch.count_nonzero(grpo[1]) == 0
```

全对或全错组没有相对排序信息。给分母加极小常数只能避免除零，不能创造学习信号。若训练过滤这类组，就同时改变了 prompt 的有效采样分布。

## Reduction 决定隐式权重

同一个 per-token objective，可以产生三种不同训练目标：每条 response 等权、每个 action token 等权，或用固定生成预算作分母。把它们都称作“取平均”会掩盖长度偏置。

```python
import torch
def reduce_losses(loss, mask, mode, budget=None):
    mask = mask.to(loss.dtype)
    per_response = (loss * mask).sum(1)
    length = mask.sum(1).clamp_min(1)
    if mode == "response":
        return (per_response / length).mean()
    if mode == "token":
        return per_response.sum() / mask.sum().clamp_min(1)
    if mode == "fixed":
        return (per_response / budget).mean()
    raise ValueError(mode)
loss = torch.tensor([[2., 0., 0., 0.], [1., 1., 1., 1.]])
mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1]])
assert reduce_losses(loss, mask, "response") == 1.5
assert reduce_losses(loss, mask, "token") == 1.2
assert reduce_losses(loss, mask, "fixed", budget=4) == .75
```

DAPO 使用 global token mean，是为了让有效 token 等权；Dr. GRPO 用固定生成预算讨论对原始 policy-gradient objective 的无偏实现。二者改的是不同分母，不能互换名称。

## DAPO 的 mixed-group 动态采样

binary verifier 下，只有同时含正确与错误回答的组会产生 group-relative 排序信号。动态采样可以保证 learner batch 中的有效组数，却要额外记录被拒绝组和总 rollout 成本。

```python
import torch
def keep_mixed_groups(reward):
    if reward.ndim != 2:
        raise ValueError("expected [prompt, rollout]")
    return (reward.max(1).values > reward.min(1).values)
reward = torch.tensor([[0., 0., 0.], [0., 1., 0.], [1., 1., 1.], [1., 0., 1.]])
keep = keep_mixed_groups(reward)
assert keep.tolist() == [False, True, False, True]
kept = reward[keep]
assert torch.all(kept.std(1, unbiased=False) > 0)
assert kept.numel() < reward.numel()
```

它提高的是 learner batch 的有效信号密度，不等于免费提升采样效率。公平实验必须同时报告生成过但被拒绝的 token 数、不同难度 prompt 的保留率和 wall-clock。

## VAPO 的 length-adaptive GAE {#vapo}

VAPO 让 policy 使用的 $\lambda$ 随轨迹长度变化：

$$
\lambda_{\text{policy}}(l)=1-\frac{1}{\alpha l}.
$$

实现时必须约束合法区间，并说明 $l$ 是 action token、环境 step 还是另一种时间单位。

```python
import torch
def length_adaptive_lambda(length, alpha=.05):
    length = torch.as_tensor(length, dtype=torch.float32)
    if torch.any(length <= 0) or alpha <= 0:
        raise ValueError("positive length and alpha required")
    return (1 - 1 / (alpha * length)).clamp(0, 1)
length = torch.tensor([16, 64, 256, 1024])
lam = length_adaptive_lambda(length)
assert torch.all(lam[1:] > lam[:-1])
assert lam[0] == 0
torch.testing.assert_close(lam[-1], torch.tensor(1 - 1 / (0.05 * 1024)))
```

这段公式只改变 advantage residual 的衰减尺度。critic 预训练、critic target 的 $\lambda$、正样本 NLL 与 Clip-Higher 是 VAPO recipe 中其他独立组件。

## Sequence ratio 与 GSPO

对 response $y=(y_1,\ldots,y_T)$，完整 sequence likelihood ratio 是 token ratio 的乘积。直接相乘容易上下溢，因此先累加 log-ratio；GSPO 使用长度归一化后的几何平均 ratio，使不同长度的数值尺度更可比。

```python
import torch
def geometric_sequence_ratio(new_logp, old_logp, mask):
    mask = mask.to(new_logp.dtype)
    length = mask.sum(1).clamp_min(1)
    return (((new_logp - old_logp) * mask).sum(1) / length).exp()
old_logp = torch.tensor([[-1., -2., 0.], [-1., -2., -3.]])
new_logp = old_logp + torch.tensor([[.1, .3, 9.], [.1, .3, .2]])
mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
ratio = geometric_sequence_ratio(new_logp, old_logp, mask)
torch.testing.assert_close(ratio, torch.tensor([.2, .2]).exp())
assert ratio[0] == ratio[1]
```

sequence-coherent gate 会让整条回答共同保留或共同饱和。它更贴近 sequence reward 的粒度，却也可能因少数异常 token 放弃整条轨迹的信号。

## SAPO 的平滑 gate

硬 clip 在阈值处突然改变梯度。SAPO 的 loss 本体使用 sigmoid surrogate；`sech²` 是它求导后出现的 gate，不应直接写回 loss。

```python
import torch
def sapo_term(new_logp, old_logp, advantage, tau_pos=1., tau_neg=2.):
    ratio = (new_logp - old_logp).exp()
    tau = torch.where(advantage > 0, tau_pos, tau_neg)
    return 4 / tau * torch.sigmoid(tau * (ratio - 1)) * advantage, ratio, tau
new_logp = torch.tensor([-.7, 0., .7], requires_grad=True)
advantage = torch.tensor([1., -1., 1.])
term, ratio, tau = sapo_term(new_logp, torch.zeros(3), advantage)
term.sum().backward()
gate = torch.cosh(tau * (ratio - 1) / 2).reciprocal().square()
torch.testing.assert_close(new_logp.grad, ratio * gate * advantage)
assert gate[1] == 1
assert gate[2] < gate[0]
```

对 $\log\pi$ 的梯度系数是 $\rho\,\operatorname{sech}^2(\tau(\rho-1)/2)A$。平滑不意味着无偏，也不自动消除 train–inference mismatch；正负温度与归约层级共同决定远策略信号怎样衰减。

## CISPO、TIS 与 IcePop 的 coefficient

三者都可写成 detached coefficient 乘 policy log-prob，但 ratio 的含义不同：CISPO 约束 current–old update ratio；TIS 与 IcePop 校正同一 checkpoint 在训练引擎与 rollout 引擎间的 engine ratio。前者饱和两侧权重，后两者分别截断上尾和拒绝双侧尾部。

```python
import torch
def detached_ratio_controls(update_ratio, engine_ratio, is_low=.2, is_high=.2, tis_cap=2., ice_low=.8, ice_high=1.2):
    cispo = update_ratio.clamp(1 - is_low, 1 + is_high).detach()
    tis = engine_ratio.clamp(max=tis_cap).detach()
    accept = (engine_ratio >= ice_low) & (engine_ratio <= ice_high)
    icepop = torch.where(accept, engine_ratio, torch.zeros_like(engine_ratio)).detach()
    return cispo, tis, icepop, accept
update = torch.tensor([.5, 1., 2.], requires_grad=True)
engine = torch.tensor([.5, 1., 3.], requires_grad=True)
cispo, tis, icepop, accept = detached_ratio_controls(update, engine)
torch.testing.assert_close(cispo, torch.tensor([.8, 1., 1.2]))
torch.testing.assert_close(tis, torch.tensor([.5, 1., 2.]))
torch.testing.assert_close(icepop, torch.tensor([0., 1., 0.]))
assert accept.tolist() == [False, True, False]
assert not cispo.requires_grad and not tis.requires_grad and not icepop.requires_grad
```

`detach` 只冻结校正权重；真实 loss 仍要让底层 `logp` 提供梯度。实现时还应分别记录 update ratio 和 engine ratio 的分位数，不能把两种偏移压成同一个直方图。公式与适用边界见 [CISPO、TIS 与 IcePop](../reinforcement-learning/ratio-clipping-gating.md#cispo)及[训推分布与策略滞后](../reinforcement-learning/training-inference-discrepancy.md#tis)。

## DIS 的双侧硬门

DIS（Direct Double-Sided Importance Sampling）保留落在非对称区间内的 token-level importance-weighted 更新，并让越界 token 直接失去 policy-gradient 信号。它与 PPO 的“悲观二选一”不同：PPO 的某些越界方向仍保留 raw 分支，DIS 则显式做 keep/drop。

```python
import torch
def detached_dis_coefficient(new_logp, behavior_logp, low=.2, high=.28):
    ratio = (new_logp - behavior_logp).exp()
    keep = (ratio > 1 - low) & (ratio < 1 + high)
    return torch.where(keep, ratio, torch.zeros_like(ratio)).detach(), keep
ratio = torch.tensor([.7, .81, 1., 1.27, 1.4])
new_logp = ratio.log().requires_grad_()
coefficient, keep = detached_dis_coefficient(new_logp, torch.zeros(5))
loss = -(coefficient * new_logp).sum()
loss.backward()
assert keep.tolist() == [False, True, True, True, False]
torch.testing.assert_close(new_logp.grad, -coefficient)
```

这段 reference 明确选择了 detached importance coefficient。SAO 论文公式没有显式写 stop-gradient，且暂无完整官方训练代码裁决；不 detach 会得到额外导数项，因此真实实现必须把选择写进契约。异步队列若偏向短轨迹，仅靠 gate 也不能修复选择偏差。

## 一组必须保留的测试

把这些 reference 接进真实 trainer 时，至少保留以下不变量：

1. current 与 behavior log-prob 完全相同时，所有有效 ratio 为 $1$；
2. prompt、observation 与 padding token 永不进入 policy-loss 分母；
3. terminal 不 bootstrap，truncation 使用正确 final observation 后终止 trace；
4. 全同 reward 组不会通过数值常数伪造优势；
5. response、token 和 fixed-budget reduction 的选择写入配置与实验记录；
6. dynamic sampling 报告全部生成成本，而非只报告 retained batch；
7. sequence gate 与 token gate 分别统计有效样本率；
8. behavior、old、current、reference 的 checkpoint 和 log-prob convention 可追溯；
9. 学习信号、采样吞吐和最终 held-out 能力分开作图。

方法选择与证据解释见[推理 RL 配方地图](../reinforcement-learning/reasoning-rl-recipes.md)，异步场景的概率契约见[训推分布与策略滞后](../reinforcement-learning/training-inference-discrepancy.md)。

## Reference {#reference}

- Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- Yu et al., [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- Yue et al., [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)
- MiniMax et al., [MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention](https://arxiv.org/abs/2506.13585)
- Zheng et al., [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
- Yao et al., [On the Rollout-Training Mismatch in Modern RL Systems](https://www.opt-ml.org/papers/2025/paper116.pdf)
- Ling Team et al., [Every Step Evolves: Scaling Reinforcement Learning for Trillion-Scale Thinking Model](https://arxiv.org/abs/2510.18855)
- Gao et al., [Soft Adaptive Policy Optimization](https://arxiv.org/abs/2511.20347)
- Hou et al., [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508)
