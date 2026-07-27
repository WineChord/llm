# Trust Region 与 PPO：让策略更新保持可控

Policy gradient 给出上升方向，却没有保证一步走多远。神经策略的一次大更新可能让新策略几乎不再访问旧数据中的状态，使原先估计的 advantage 立刻失效。Trust Region Policy Optimization（TRPO）把“不要离旧策略太远”写成 KL 约束；Proximal Policy Optimization（PPO）用更容易实现的 surrogate 近似这一目标。

本页关注二者共同的统计结构，而不是把 `clip` 当成一个孤立公式。梯度起点见 [Policy Gradient](policy-gradient.md)，advantage 的构造见 [Actor–Critic](actor-critic.md)，语言模型中 reference policy 的另一种 KL 角色见 [KL 正则化控制](kl-regularized-control.md)。

## 为什么旧策略数据会很快失效

策略性能差可以用 performance-difference identity 表示：

$$
J(\pi)-J(\pi_{\text{old}})
=\frac{1}{1-\gamma}
\mathbb E_{
s\sim d^\pi,\,
a\sim\pi
}
\left[
A^{\pi_{\text{old}}}(s,a)
\right].
$$

困难在于右侧状态分布是新策略的 $d^\pi$，而手中轨迹来自 $d^{\pi_{\text{old}}}$。若更新很小，可先固定旧状态分布，构造局部 surrogate：

$$
L_{\pi_{\text{old}}}(\pi)
=
\mathbb E_{
s\sim d^{\pi_{\text{old}}},
a\sim\pi_{\text{old}}
}
\left[
\frac{\pi(a\mid s)}
{\pi_{\text{old}}(a\mid s)}
A^{\pi_{\text{old}}}(s,a)
\right].
$$

概率比

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid h_t)}
{\pi_{\text{old}}(a_t\mid h_t)}
=
\exp\left(
\log\pi_\theta(a_t\mid h_t)
-\log\pi_{\text{old}}(a_t\mid h_t)
\right)
$$

把旧策略采样的动作重新加权到新策略。更新越大，固定旧状态分布的近似越不可信，importance ratio 的方差也越高。

## TRPO：把邻域写成约束

TRPO 求解近似约束问题

$$
\begin{aligned}
\max_\theta\quad
&\widehat{\mathbb E}_t
\left[
r_t(\theta)\widehat A_t
\right],\\
\text{s.t.}\quad
&\widehat{\mathbb E}_t
\left[
D_{\mathrm{KL}}
\left(
\pi_{\text{old}}(\cdot\mid h_t)
\;\|\;
\pi_\theta(\cdot\mid h_t)
\right)
\right]
\le\delta.
\end{aligned}
$$

在旧参数附近，KL 的二阶展开给出 Fisher 信息矩阵；natural-gradient 方向近似为

$$
\Delta\theta
\propto F^{-1}g,
$$

再用 conjugate gradient 求近似方向，并通过 line search 检查实际 surrogate 与 KL。它比普通一阶更新更接近“在策略分布空间走固定距离”，但实现和分布式训练成本较高。

TRPO 的理论单调改进依赖精确或受控近似、足够准确的 advantage 和真实 KL 约束。神经网络、有限 batch、函数逼近与近似求解都使它成为条件性保证，而不是部署承诺。

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
torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
```

它只保留 clipped 分支，丢失未裁剪目标，和 PPO surrogate 不等价。

PPO 也不保证实际 KL 小于某个硬阈值：未被当前样本覆盖的动作仍可变化，多个 minibatch epoch 还会持续推动策略。因此实践中常同时监控：

- approximate KL；
- clip fraction；
- ratio 的分位数与极值；
- entropy；
- advantage、value target 和 explained variance；
- 每个 epoch 后的新旧策略差异。

必要时用 KL early stopping 或降低学习率，而不是把 $\epsilon$ 当作完整 trust region。

## Old policy 与 reference policy 不是一个对象

LLM 后训练常同时出现三种策略：

| 策略 | 作用 | 更新节奏 |
| --- | --- | --- |
| $\pi_\theta$ | 当前待优化 actor | 每个 minibatch 改变 |
| $\pi_{\text{old}}$ | 产生 rollout 的 behavior policy | 对该批数据冻结 |
| $\pi_{\text{ref}}$ | 定义行为先验或 KL 成本 | 通常跨多批冻结 |

PPO ratio 使用 $\pi_{\text{old}}$：

$$
r_t
=\frac{\pi_\theta(a_t\mid h_t)}
{\pi_{\text{old}}(a_t\mid h_t)}.
$$

对齐目标中的 KL 则相对 $\pi_{\text{ref}}$。即使 old 与 ref 在某个时刻权重相同，它们的语义也不同：前者回答“这条样本由谁产生”，后者回答“行为允许偏离哪个锚点”。详细策略角色与保存字段见[语言模型作为策略](language-model-policy.md)和[轨迹与策略契约](../agentic-rl/trajectory-contract.md)。

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
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()
    unclipped = ratio * adv.detach()
    clipped = ratio.clamp(1 - eps, 1 + eps) * adv.detach()
    token_loss = -torch.minimum(unclipped, clipped)
    mask = action_mask.to(token_loss.dtype)
    loss = (token_loss * mask).sum() / mask.sum().clamp_min(1)
    approx_kl = ((ratio - 1) - log_ratio) * mask
    approx_kl = approx_kl.sum() / mask.sum().clamp_min(1)
    clipfrac = (((ratio - 1).abs() > eps) * action_mask).float().sum()
    clipfrac = clipfrac / action_mask.sum().clamp_min(1)
    return loss, approx_kl, clipfrac
```

这里的 `approx_kl` 是采样估计和诊断量，不是精确全词表 KL，也不是硬约束。若训练需要精确 token-distribution KL，需要保留 policy/reference logits 或计算相应分布，成本与语义都不同。

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
current, behavior and reference policy revisions
exact sampled token IDs and behavior log-probabilities
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
- **重算 old log-prob**：tokenizer、模板、量化或 sampler 变化会伪造 behavior probability。
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
