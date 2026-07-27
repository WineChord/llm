# PPO：从 Clipped Surrogate 到训练契约

Policy gradient 给出上升方向，却没有保证一步走多远。神经策略的一次大更新可能让新策略几乎不再访问旧数据中的状态，使原先估计的 advantage 立刻失效。PPO 用 clipped surrogate 让 sampled action 在“有利方向”越界后不再获得额外收益，并换取一阶优化器、minibatch 与多 epoch 训练的便利。

PPO 继承 trust-region 路线的动机，却不等价于硬 KL 约束。[Trust Region 与 TRPO](trust-region.md)独立推导 performance-difference、Fisher、conjugate gradient 与 line search；本页集中解释 PPO 的精确分段、policy 身份、LLM reduction、batch lifecycle 与诊断。梯度起点见 [Policy Gradient](policy-gradient.md)，advantage 的构造见 [Advantage 与 GAE](advantage-estimation-gae.md)。

## 从 trust region 到 clipped surrogate {#trpo}

旧轨迹只在策略仍靠近采样分布时提供可信的局部改进信号。[Trust Region 与 TRPO](trust-region.md)从 performance-difference identity 出发，把“靠近”写成显式 KL 约束，再用 Fisher、conjugate gradient 与 line search 求解。PPO 保留同一个局部更新动机，却不再求解二阶约束问题。

对当前训练批次，先在训练引擎上冻结旧策略 $\pi_{\mathrm{old}}^{\mathrm{train}}$，定义 token ratio

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid h_t)}
{\pi_{\mathrm{old}}^{\mathrm{train}}(a_t\mid h_t)}
=
\exp\left(
\log\pi_\theta(a_t\mid h_t)
-\log\pi_{\mathrm{old}}^{\mathrm{train}}(a_t\mid h_t)
\right).
$$

在标准同步 on-policy 实现里，真实 behavior $\mu^{\mathrm{rollout}}$ 与这个冻结旧策略一致；异步队列、推理引擎数值差异或 sampling processor 会破坏该等式。PPO ratio 描述 current–old update，不能自动完成 current–behavior 的 off-policy correction。四种 policy 身份与三种 ratio 见[策略身份、训推分布与策略滞后](training-inference-discrepancy.md)。

## PPO-Clip：用悲观 surrogate 限制收益

PPO-Clip 定义

$$
\mathcal L_{\text{clip}}(\theta)
=-
\mathbb E_t
\left[
\min\left(
r_t\widehat A_t,\,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)
\widehat A_t
\right)
\right].
$$

理解 `min` 要按 advantage 符号分开：

- $\widehat A_t>0$：提高动作概率是好事，但 $r_t>1+\epsilon$ 后不再从 surrogate 获得额外收益；
- $\widehat A_t<0$：降低动作概率是好事，但 $r_t<1-\epsilon$ 后不再从 surrogate 获得额外收益。

因此 clip 不是把所有 ratio 截到区间后再训练，而是选择更保守的一侧。一个常见错误实现是

```python
import torch
ratio, eps = torch.tensor([1.5]), 0.2
advantage = torch.tensor([1.0])
wrong_surrogate = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
torch.testing.assert_close(wrong_surrogate, torch.tensor([1.2]))
```

它只保留 clipped 分支，丢失未裁剪目标，和 PPO surrogate 不等价。

精确分段更直接：

$$
\widehat A_t\ge0:
\quad
\min(r_t\widehat A_t,\operatorname{clip}(r_t)\widehat A_t)
=\widehat A_t\min(r_t,1+\epsilon),
$$

$$
\widehat A_t<0:
\quad
\min(r_t\widehat A_t,\operatorname{clip}(r_t)\widehat A_t)
=\widehat A_t\max(r_t,1-\epsilon).
$$

因此正 advantage 只在 $r_t>1+\epsilon$ 后饱和，负 advantage 只在 $r_t<1-\epsilon$ 后饱和。正 advantage 且 $r_t<1-\epsilon$ 仍要提高概率；负 advantage 且 $r_t>1+\epsilon$ 仍要降低概率。PPO、Clip-Higher、CISPO、GSPO 与 SAPO 的真实梯度差异见[Ratio、Clipping 与 Gate](ratio-clipping-gating.md)。

PPO 也不保证实际 KL 小于某个硬阈值：未被当前样本覆盖的动作仍可变化，多个 minibatch epoch 还会持续推动策略。因此实践中常同时监控：

- approximate KL；
- clip fraction；
- ratio 的分位数与极值；
- entropy；
- advantage、value target 和 explained variance；
- 每个 epoch 后的新旧策略差异。

必要时用 KL early stopping 或降低学习率，而不是把 $\epsilon$ 当作完整 trust region。

## Current、old、behavior 与 reference 不是一个对象

LLM 后训练至少要区分四种策略：

| 策略 | 作用 | 更新节奏 |
| --- | --- | --- |
| $\pi_\theta$ | 当前待优化 actor | 每个 minibatch 改变 |
| $\pi_{\text{old}}^{\mathrm{train}}$ | PPO surrogate 的冻结更新基准 | 对该批数据冻结 |
| $\mu^{\mathrm{rollout}}$ | 真正生成 sampled token 的 behavior distribution | 由 checkpoint、引擎与采样器共同定义 |
| $\pi_{\text{ref}}$ | 定义行为先验或 KL 成本 | 通常跨多批冻结 |

PPO ratio 使用 $\pi_{\text{old}}$：

$$
r_t
=\frac{\pi_\theta(a_t\mid h_t)}
{\pi_{\text{old}}(a_t\mid h_t)}.
$$

对齐目标中的 KL 则相对 $\pi_{\text{ref}}$。同步、同精度、无解码变换时，old training policy 可以近似真实 behavior；使用不同推理引擎、量化、top-$p$、grammar 或异步队列后，$\mu^{\mathrm{rollout}}\ne\pi_{\text{old}}^{\mathrm{train}}$。此时还要处理

$$
\frac{\pi_{\text{old}}^{\mathrm{train}}}{\mu^{\mathrm{rollout}}},
$$

而不能把 reference log-prob 填进分母。完整四策略与三种 ratio 见[策略身份、训推分布与策略滞后](training-inference-discrepancy.md)。

## LLM 中的 token 与序列归约

对 response token，常见实现先得到

$$
\ell_{i,t}
=-
\min
\left(
r_{i,t}A_{i,t},
\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_{i,t}
\right).
$$

随后如何归约决定了训练权重：

$$
\mathcal L_{\text{token}}
=
\frac{\sum_{i,t}m_{i,t}\ell_{i,t}}
{\sum_{i,t}m_{i,t}},
$$

或

$$
\mathcal L_{\text{sequence}}
=
\frac1B\sum_i
\frac{\sum_tm_{i,t}\ell_{i,t}}
{\sum_tm_{i,t}}.
$$

第一种让每个 action token 等权，长回答贡献更多 token；第二种让每条序列近似等权。二者都可能合理，但不能把不同 reduction 的实验只写成“PPO”。工具轨迹还可能需要按 action span 或 turn 归约。

如果 reward 是 sequence-level，而 ratio 是 token-level，还要解释 $A_{i,t}$ 是整条序列共享、由 critic 逐 token 给出，还是在工具 transition 处对齐。这个选择属于信用分配，不是 tensor reshape。

## 最小 PPO surrogate

```python
import torch

def ppo_policy_loss(new_logp, old_logp, adv, action_mask, eps=0.2):
    if not (new_logp.shape == old_logp.shape == adv.shape == action_mask.shape):
        raise ValueError("shape mismatch")
    mask = action_mask.bool()
    if not mask.any() or not 0 < eps < 1:
        raise ValueError("PPO needs actions and a clipping radius in (0, 1)")
    if not all(torch.isfinite(x[mask]).all() for x in (new_logp, old_logp, adv)):
        raise ValueError("selected PPO terms must be finite")
    log_ratio = new_logp[mask] - old_logp[mask]
    advantage = adv.detach()[mask]
    ratio = log_ratio.exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - eps, 1 + eps) * advantage
    token_loss = -torch.minimum(unclipped, clipped)
    loss = token_loss.mean()
    approx_kl = ((ratio - 1) - log_ratio).mean()
    clipfrac = ((ratio - 1).abs() > eps).float().mean()
    return loss, approx_kl, clipfrac

old = torch.tensor([[0., 0., float("nan")]])
new = torch.tensor([[1.5, .5, float("nan")]]).log()
adv = torch.tensor([[1., -1., float("nan")]])
loss, approx_kl, clipfrac = ppo_policy_loss(
    new, old, adv, torch.tensor([[True, True, False]]), eps=0.2
)
torch.testing.assert_close(loss, torch.tensor(-0.2))
assert approx_kl > 0
torch.testing.assert_close(clipfrac, torch.ones(()))
try:
    ppo_policy_loss(new, old, adv, torch.zeros_like(adv, dtype=torch.bool))
except ValueError:
    pass
else:
    raise AssertionError("an empty PPO action set must be rejected")
```

这里的 `approx_kl` 是采样估计和诊断量，不是精确全词表 KL，也不是硬约束。若训练需要精确 token-distribution KL，需要保留 policy/reference logits 或计算相应分布，成本与语义都不同。

符号相关裁剪与 token / sequence reduction 的组合测试见[手撕：LLM 策略优化 · Reduction](../practice/llm-policy-optimization.md#loss-reduction)。

## Update epoch 与数据新鲜度

PPO 常在同一批 rollout 上做多个 minibatch epoch，提高昂贵样本的利用率。但每走一步，数据相对当前参数就更旧：

```text
rollout under pi_old
  -> compute fixed returns / advantages
  -> epoch 1 minibatches
  -> epoch 2 minibatches
  -> ...
  -> stop before ratios and KL drift too far
```

应冻结 old log-prob、return 和 advantage target，记录实际 update epoch，并在每轮检查 KL、clip fraction 与有效样本比例。更多 epoch 不是免费数据增强；它同时增加 overfitting 与 off-policy 程度。

大规模异步 rollout 中，轨迹甚至在进入第一轮更新前就已落后，此时单靠 PPO clip 无法恢复 on-policy 等价性，见 [Off-policy 校正](off-policy-correction.md)。

## 实现契约

```text
current, old-training, behavior and reference policy revisions
exact sampled token IDs and rollout behavior log-probabilities
recomputed old-training log-probabilities for the PPO ratio
action, valid-token and terminal/bootstrap masks
advantage/value revision and normalization scope
clip epsilon, KL estimator and early-stop threshold
token/sequence/action-span reduction
number and order of minibatch epochs
sampling processors and log-probability convention
```

至少验证：

1. `new_logp == old_logp` 时所有有效 ratio 为 $1$；
2. 正 advantage 的梯度提高动作概率，负 advantage 相反；
3. prompt 与 padding 的 log-prob 改变不影响 loss；
4. 正负 advantage 两侧都覆盖超出 clip 区间的测试；
5. minibatch 重排不改变单 epoch 的全量 reduction；
6. KL、clip fraction、长度和 reward 按任务 slice 联合报告。

## 失败边界

- **clip 等同于 KL 约束**：PPO 没有自动满足 TRPO 的硬约束。
- **old/ref 混用**：importance ratio 和行为正则失去语义。
- **只计算 clipped 分支**：实现不再是 PPO surrogate。
- **把重算 old log-prob 当作 behavior**：tokenizer、模板、量化或 sampler 变化后，training-side 重算值不能冒充真实 rollout probability。
- **ratio 先平均再 exponentiate**：sequence ratio 与 token ratio 是不同目标。
- **sequence reward 复制到 token 后求和**：长度成为隐式权重。
- **无限复用 rollout**：clip 不能把任意旧数据变回 on-policy。
- **只看 mean KL**：少量极端 token、长序列或特定任务可能主导失败。
- **高 reward 代替独立评测**：策略可能只学会利用 reward 或 verifier；见[强化学习评测与调试](evaluation-debugging.md)。

## 历史位置

TRPO 把 natural policy gradient、局部 surrogate 和 KL 邻域组合成可操作算法；PPO 则牺牲显式二阶约束，换取一阶优化器和 minibatch 训练的便利。它因此成为深度 RL 与早期 LLM RLHF 的常用基线。便利不等于定义模糊：ratio、advantage、mask、reduction、KL 与数据新鲜度仍共同决定“PPO”实际优化的对象。

## Reference {#reference}

- [Schulman et al., Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- [Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Kakade, A Natural Policy Gradient](https://proceedings.neurips.cc/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html)
- [Ouyang et al., Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- [Engstrom et al., Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729)
