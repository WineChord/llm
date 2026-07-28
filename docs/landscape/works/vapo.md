# VAPO：长推理中的 critic、GAE 与稀疏正样本

[VAPO](https://arxiv.org/abs/2504.05118) 回答了一个与 DAPO 不同的问题：如果终局 reward 很稀疏、response 长度差异巨大，而 learned value 又容易在训练早期产生系统偏差，怎样让 value-based PPO 仍然工作？

它不是“length-adaptive GAE”的另一个名字，而是七项组件组成的 recipe：

1. value model pretraining；
2. actor 与 critic 使用不同的 GAE $\lambda$；
3. policy 使用 length-adaptive GAE；
4. Clip-Higher；
5. global token-level policy loss；
6. positive-example LM loss；
7. group sampling。

前三项直接围绕 critic 与长期信用，后四项处理探索、长度权重与稀疏正样本。整套方法的历史位置见[推理策略优化谱系](../lineages/reasoning-policy-optimization.md)，GAE 的通用推导见 [Advantage 估计](../../reinforcement-learning/advantage-estimation-gae.md)。

## 为什么 value model 会在长 CoT 中失灵

value function 需要估计当前 policy 从 prefix $h_t$ 出发的未来回报：

$$
V^\pi(h_t)
=
\mathbb E_\pi
\left[
\sum_{k\ge t}\gamma^{k-t}r_k
\mid h_t
\right].
$$

语言 RLHF 常用 reward model 初始化 value model。二者的训练语义却不同：

- reward model 通常只在完整 response 末端比较或评分；
- value model 必须在每个 prefix 上估计未来 return；
- 不完整 prefix 不是“低质量完整回答”，却可能被 reward-model initialization 赋较低分。

设估计误差为 $\widehat V(h_t)=V^\pi(h_t)+e_t$。估计 TD residual 相对真实 residual 的条件偏差为

$$
\operatorname{bias}(\widehat\delta_t)
=
\gamma\,\mathbb E[e_{t+1}\mid h_t,a_t]-e_t.
$$

因此“prefix value 整体偏低”本身还不够推出 residual 正偏；只有较早位置低估得更重，或误差沿轨迹以相应速度收敛时，下面的

$$
\delta_t
=
r_t+\gamma V(h_{t+1})-V(h_t)
$$

才会产生正偏并沿递推传播。与此同时，终局 reward 通过固定 $\lambda<1$ 向前传播时，一般按

$$
(\gamma\lambda)^{T-t}
$$

衰减。VAPO 的 RLHF 实验固定 $\gamma=1$，论文后续才简写成 $\lambda^{T-t}$。对数百或数千 token 的 response，真实 reward 到早期 token 的直接贡献可以接近零，advantage 几乎完全由尚未学准的 bootstrap value 决定。

VAPO 因而把问题拆成三件事：先让 critic 进入可用区域，再分开 critic target 与 actor advantage 的偏差—方差需求，最后让不同长度 response 使用不同传播尺度。

## Value pretraining：先修 baseline，再动 actor {#value-pretraining}

VAPO 在 actor RL 开始前先训练 value model。其动机是避免：

```text
biased initial value
  -> biased GAE
  -> actor moves toward a distorted target
  -> data distribution changes
  -> critic chases a moving policy
```

value warmup 不是把 reward model score 复制到所有 token。它仍要构造 prefix 上的 return target，让 critic 学习“从这里继续生成，期望会得到多少”。

这个阶段的可审计量包括：

- terminal return 与各 prefix value 的 calibration；
- value bias 随 token position 和 response length 的变化；
- positive/negative outcome 上的 explained variance；
- warmup data 与正式 rollout 的 policy/data distribution；
- critic 初始化来自 actor、reward model 还是独立 checkpoint。

VAPO 论文的 ablation 显示移除 value pretraining 会在其设置中显著损害结果；这支持“critic cold start 是该实验的重要瓶颈”，不证明固定 warmup steps 可以跨模型直接复用。

## Decoupled GAE：actor 与 critic 不必用同一个 λ {#decoupled-gae}

GAE 递推为

$$
\widehat A_t^{(\lambda)}
=
\delta_t
+\gamma\lambda m_t^{\mathrm{trace}}
\widehat A_{t+1}^{(\lambda)}.
$$

actor 希望得到低方差、可快速优化的 advantage；critic 希望 value target 尽量包含真实长期 return。两个目标并不要求相同的 $\lambda$。

VAPO 使用：

$$
\lambda_{\mathrm{critic}}=1,
$$

让 critic target 更接近完整 Monte Carlo return；

$$
\lambda_{\mathrm{policy}}<1,
$$

让 actor advantage 保留一定 bootstrap，降低采样方差。若写成

$$
\widehat V_t^{\mathrm{target}}
=
V(h_t)+\widehat A_t^{(\lambda_{\mathrm{critic}})},
$$

$$
\widehat A_t^{\mathrm{actor}}
=
\widehat A_t^{(\lambda_{\mathrm{policy}})},
$$

就能看出两个张量虽来自同一批 TD residual，却服务于不同 loss。实现中必须分别 stop-gradient、归一化和记录版本，不能先算一份 advantage 再同时喂给 actor/critic。

## Length-adaptive GAE：让传播尺度随长度变化 {#length-adaptive-gae}

一般 GAE 的有效 residual 窗口近似为

$$
\sum_{k=0}^{\infty}(\gamma\lambda)^k
=
\frac1{1-\gamma\lambda}.
$$

VAPO 的实验取 $\gamma=1$，因此其推导化为

$$
\sum_{k=0}^{\infty}\lambda^k
\approx
\frac1{1-\lambda}.
$$

VAPO 希望它与 response length $l$ 成正比：

$$
\frac1{1-\lambda_{\mathrm{policy}}}
=
\alpha l.
$$

解得

$$
\lambda_{\mathrm{policy}}(l)
=
1-\frac1{\alpha l}.
$$

短 response 使用较小 $\lambda$，更多依赖 critic、降低有限样本方差；长 response 使用更接近 $1$ 的 $\lambda$，让终局 reward 和较远 TD residual 不至于过早衰减。

这个公式有四个实现边界：

1. 当 $\alpha l<1$ 时原式会给出负值，需要明确 clamp 或最小长度；
2. $l$ 是 action token 数、所有 response token 数还是 environment step 数；
3. truncation 与 terminal 使用不同 bootstrap/trace mask；
4. packed batch 中每条 response 的 $\lambda$ 不同，不能用单个全局 scalar。

论文使用 $\alpha=0.05$；后续其他工作出现的不同 $\alpha$ 属于另一实验设置，不能回填为 VAPO 原参数。若任务采用 $\gamma\ne1$，直接照搬 $\lambda=1-1/(\alpha l)$ 不再保持相同的有效窗口，需要从 $1/(1-\gamma\lambda)$ 重新推导。

## Token-level loss 与 Clip-Higher 从 DAPO 接过来

VAPO 的 PPO surrogate 使用非对称区间：

$$
\min\left(
\rho_{i,t}\widehat A_{i,t},\,
\operatorname{clip}
(\rho_{i,t},1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})
\widehat A_{i,t}
\right),
$$

并以全 batch 有效 token 数归约：

$$
\mathcal L_{\mathrm{policy}}
=
-
\frac{
\sum_{i,t}m_{i,t}j_{i,t}
}{
\sum_{i,t}m_{i,t}
}.
$$

它们分别处理两类问题：

- Clip-Higher 扩大正 advantage、低概率 token 的上侧活动区；
- global token mean 防止长 response 内每个 token 被 $1/|y_i|$ 过度缩小。

这两项来自 DAPO 路线，但 VAPO 的 advantage 由 critic/GAE 提供，而不是 group std-normalized outcome reward。相同 gate 不代表相同算法，见 [Ratio、Clipping 与 Gate](../../reinforcement-learning/ratio-clipping-gating.md) 和 [DAPO 深读](dapo.md)。

## Positive-example LM loss：把稀有成功当示范

当 verifier reward 极稀疏时，一条正确 response 可能经过大量失败探索才得到。纯 policy gradient 会按 advantage 增强它，但 clipped ratio、minibatch 噪声和有限更新步数可能不足以充分利用这条轨迹。

VAPO 对正确 response 集合 $\mathcal T^+$ 追加 NLL：

$$
\mathcal L_{\mathrm{NLL}}
=
-
\frac1{\sum_{i\in\mathcal T^+}|y_i|}
\sum_{i\in\mathcal T^+}\sum_t
\log\pi_\theta(y_{i,t}\mid h_{i,t}),
$$

$$
\mathcal L
=
\mathcal L_{\mathrm{PPO}}
+\mu\mathcal L_{\mathrm{NLL}}.
$$

它相当于对 on-policy 成功轨迹做一小步 imitation learning。优点是稀有正样本的所有 token 都得到稳定监督；代价是：

- 不再是纯粹的 policy-gradient objective；
- 成功 response 中的冗余、捷径和偶然格式也会被模仿；
- verifier 漏洞会被正样本 NLL 进一步放大；
- $\mu$ 改变 RL 与 behavior cloning 的相对尺度。

当当前 minibatch 的 $\mathcal T^+$ 为空时，NLL 项应记为零或显式跳过，并记录 positive-response 与 positive-token count；不能让分母除零，也不能通过未计入成本的重采样伪造稳定正样本率。

应同时评估 verifier-correct、独立 correctness、风格、长度与多样性，而不是只看训练 reward。

## Group sampling 与 GRPO 不是同一件事

VAPO 仍使用 value-based PPO，但在固定生成预算下减少 distinct prompts、增加每个 prompt 的重复采样。这样可获得更多同题正负对照，也提高遇到稀有正确 response 的概率。

它与 GRPO 的区别是：

- GRPO 必须用同组 reward 构造 baseline；
- VAPO 的 actor advantage 来自 critic/GAE；
- VAPO group sampling 是计算预算分配与正样本获取策略；
- 即使一组 reward 全同，critic advantage 也不必严格为零。

因此 `group_size > 1` 不足以判断 trainer 在使用 GRPO。要看 advantage 的来源，而不是 batch 的外形。

## 最小可执行语义

下面实现两套 $\lambda$ 的反向递推，并验证长 response 获得更大的 policy $\lambda$。`bootstrap` 与 `trace` 是两个 mask；代码没有包含 value warmup、模型采样或 positive NLL。

```python
import torch
def adaptive_lambda(length, alpha=.05):
    return (1 - 1 / (alpha * length.float())).clamp(0, 1)
def gae_with_lambda(reward, value, next_value, bootstrap, trace, lam, gamma=1.):
    delta = reward + gamma * bootstrap * next_value - value
    advantage = torch.empty_like(delta)
    carry = torch.zeros_like(delta[:, 0])
    for t in range(delta.size(1) - 1, -1, -1):
        carry = delta[:, t] + gamma * lam * trace[:, t] * carry
        advantage[:, t] = carry
    return advantage
reward = torch.zeros(2, 64)
reward[0, 31] = 1.
reward[1, 63] = 1.
value = torch.zeros_like(reward)
next_value = torch.zeros_like(reward)
bootstrap = torch.zeros_like(reward)
trace = torch.zeros_like(reward)
trace[0, :31] = 1.
trace[1, :63] = 1.
length = torch.tensor([32., 64.])
policy_lam = adaptive_lambda(length)
actor_adv = gae_with_lambda(reward, value, next_value, bootstrap, trace, policy_lam)
critic_adv = gae_with_lambda(reward, value, next_value, bootstrap, trace, torch.ones(2))
assert policy_lam[1] > policy_lam[0]
assert actor_adv[1, 0] > actor_adv[0, 0]
torch.testing.assert_close(critic_adv[0, :32], torch.ones(32))
torch.testing.assert_close(critic_adv[0, 32:], torch.zeros(32))
```

真实实现还要确保 `length` 与递推时间轴相同；若 $\lambda$ 按 token 数计算、GAE 却沿 turn 递推，公式就失去原语义。更多边界断言见[手撕 LLM 策略优化](../../practice/llm-policy-optimization.md#vapo)。

## 七组件怎样形成一条因果链

可以把 VAPO 读成三层：

```text
critic 可用性
  value pretraining
  decoupled GAE
        |
        v
长短轨迹的信用尺度
  length-adaptive policy lambda
  global token reduction
        |
        v
稀疏成功的探索与利用
  Clip-Higher
  positive-example NLL
  group sampling
```

每层都交换了一种成本：

- 更可靠 critic 增加预训练与 value 计算；
- 更长信用传播可能增加 bias/variance 敏感性；
- token weighting 会改变长度分布；
- positive NLL 会增强成功轨迹中的所有模式；
- group sampling 减少 prompt breadth。

这比把 VAPO 记作一个单独 loss 更接近其真实设计。

## 实验到底支持什么

论文在 Qwen2.5-32B、数学 reasoning 与 AIME 2024 avg@32 设置中，将七项修改加入 vanilla PPO，并报告 VAPO 达到约 60.4；论文还逐项移除组件，报告 value pretraining、decoupled GAE 与 length-adaptive GAE 在该设置中有显著影响。

这些结果支持：

- critic cold start、固定 $\lambda$ 与长短混合是该训练设置中的真实瓶颈；
- value-based PPO 经协同设计后可以在长 CoT RL 中保持竞争力；
- DAPO 路线的一些探索与归约技术可以与 critic/GAE 组合；
- 七组件整体在论文设置中表现稳定，并得到重复实验支持。

它们不支持：

- 60.4 可归因于 length-adaptive GAE 一项；
- $\alpha=.05$ 或 warmup 50 steps 是跨模型最优；
- critic-based PPO 普遍优于 critic-free GRPO；
- AIME 上更长 reasoning 自动意味着更正确、更忠实；
- positive NLL 对开放式 reward 或 noisy verifier 同样安全；
- 七项消融差可当作互不干扰的独立因果效应。

组件彼此耦合：去掉 value pretraining 会改变后续 GAE 质量，去掉 token loss 会改变不同长度上的梯度，因此单项 ablation 只描述“从完整 recipe 移除该项”的条件效应。

## 复现边界

论文公开了目标、关键超参数、训练动态和消融，但没有给出作者链接的完整官方训练仓库。公开框架中的第三方实现可用于核对公式，不能自动当作原训练系统。

严谨复现至少还需固定：

```text
value initialization and warmup data
actor/critic update frequencies
two lambda values and length definition
terminal/truncation masks
token denominator and packed-batch semantics
positive-sample parser and NLL coefficient
group allocation and total rollout tokens
sampling processors and behavior log-prob
verifier version and infrastructure-failure handling
```

缺少这些信息时，可以复现机制，不应声称复现论文完整结果。

## 从 VAPO 到长程 Agentic RL

VAPO 仍主要处理单轮长 reasoning：trajectory 长，但状态通常是 prompt 加已生成 prefix。Agentic RL 增加外部 observation、工具调用、异步队列与 context compaction 后，时间轴会再次改变。

[SAO](sao-compactionrl.md#sao) 继承 VAPO 的 value pretraining、length-adaptive GAE 等思想，却为单轨迹异步系统加入 direct behavior ratio、critic 更新节奏和 Skip-Observation GAE；[CompactionRL](sao-compactionrl.md#compactionrl) 则处理 context segment 被摘要切开的信用传播。

这条连接说明 GAE 的 $\lambda$ 不是独立魔法参数。先定义 action step、observation、segment 与 behavior policy，再谈如何让 residual 穿过长轨迹。

## Reference {#reference}

- Yue et al., [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)
- Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Yu et al., [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
