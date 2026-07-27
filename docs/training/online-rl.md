# 在线 RL 与可验证奖励

在线强化学习从当前或近期策略采样，再根据 reward、verifier 或环境终态更新策略。它能探索离线数据中不存在的行为，也带来非平稳数据、昂贵 rollout、策略滞后和奖励投机。

[InstructGPT](../landscape/works/instructgpt.md)适合核对经典 RLHF 中 SFT、reward model 与 PPO 的角色；[DeepSeek-R1](../landscape/works/deepseek-r1.md)则展示可验证奖励、group-relative 更新与蒸馏之间必须分开的证据边界。两者之间的演进见[后训练与对齐](../landscape/lineages/training-alignment.md)。

本页只保留 online rollout 的数据闭环。算法选择先看[推理 RL 配方地图](../reinforcement-learning/reasoning-rl-recipes.md)；PPO、GAE、GRPO 与分布校正的 canonical 推导分别见[PPO](../reinforcement-learning/trust-region-ppo.md)、[Advantage 与 GAE](../reinforcement-learning/advantage-estimation-gae.md)、[GRPO](../reinforcement-learning/grpo.md)和[训推分布与策略滞后](../reinforcement-learning/training-inference-discrepancy.md)。

## 策略与轨迹契约

一次训练更新至少涉及四个不同策略身份：

| 符号 | 角色 | 是否变化 |
| --- | --- | --- |
| $\pi_\theta$ | 正在计算梯度的新 policy | 当前 update 内变化 |
| $\pi_{\text{old}}^{\mathrm{train}}$ | PPO surrogate 的冻结更新基准 | 对该批轨迹冻结 |
| $\mu^{\mathrm{rollout}}$ | 真正产生 token 的 behavior distribution | 由 checkpoint、引擎与采样器共同定义 |
| $\pi_{\text{ref}}$ | KL 或行为先验的 reference | 通常长期冻结 |

$\pi_{\text{old}}$ 决定 PPO update ratio，$\mu$ 决定真实 off-policy correction，$\pi_{\text{ref}}$ 定义策略偏离。同步且无训推差异时 old 可近似 behavior；这不是由 checkpoint ID 自动保证的事实。完整分解见[策略身份、训推分布与策略滞后](../reinforcement-learning/training-inference-discrepancy.md)。

每个 action token 至少记录：

```text
prompt / environment and trajectory ID
token or action ID and action mask
behavior policy version and exact rollout log-prob
old-training policy version and recomputed frozen log-prob
reference version and reference log-prob
reward components and verifier version
terminal / truncated / invalid / timeout / infra error
sampling configuration and RNG
```

工具 observation、system token、prompt 和 padding 不是 policy action，不应进入 policy ratio。

## 算法怎样插入数据闭环

在线系统不应在这里重新定义每个 optimizer，而应固定它们消费和产生的数据：

| 环节 | 闭环中的职责 | 深入入口 |
| --- | --- | --- |
| advantage / return | 把 reward 与未来状态变成局部学习信号 | [GAE](../reinforcement-learning/advantage-estimation-gae.md)、[无 critic baseline](../reinforcement-learning/critic-free-baselines.md)、[GRPO](../reinforcement-learning/grpo.md) |
| policy update | 在冻结 old-training 坐标中更新 current policy | [PPO](../reinforcement-learning/trust-region-ppo.md)、[Ratio、Clipping 与 Gate](../reinforcement-learning/ratio-clipping-gating.md) |
| distribution correction | 处理真实 behavior、训练重算与 current 之间的差异 | [训推分布与策略滞后](../reinforcement-learning/training-inference-discrepancy.md)、[Off-policy 校正](../reinforcement-learning/off-policy-correction.md) |
| recipe | 联合采样、归约、长度处理、critic 与系统节奏 | [推理 RL 配方地图](../reinforcement-learning/reasoning-rl-recipes.md) |

一次可审计迭代应按状态转换理解：

```text
prompt/environment snapshot
  -> rollout under recorded behavior distribution
  -> validate terminal, truncation and infrastructure status
  -> score with versioned reward/verifier
  -> freeze old-training coordinates and build targets
  -> update actor/critic under explicit reductions
  -> evaluate held-out capability and failure slices
  -> promote, retain or reject the new policy revision
```

同步系统可以让 rollout、target 构造与 learner step 严格成批；异步系统则以更高设备利用率换取 policy lag、队列选择偏差和更复杂的版本治理。无论使用哪种算法，都应同时核算生成 token、保留样本、训练 token、wall-clock 与最终能力，而不只比较 learner steps。

## Reward 与 verifier

可验证任务通常把最终答案、测试或环境状态转成 reward。必须区分：

```text
correct
wrong
invalid action / format
timeout
environment failure
infrastructure error
```

infra error 不应自动记为零 reward；否则策略会学习规避某些机器或工具状态。reward 还应分解为任务、格式、长度、成本和安全项，并保留未加权原值。

reward model 的校准与不可辨识性见[奖励建模](reward-modeling.md)，推理任务中的 verifier 设计见[推理后训练](reasoning-posttraining.md)。

## 正确性与失效

- **old/ref 混淆**：PPO ratio 与 KL penalty 使用错误概率。
- **训练模板重算 old log-prob**：重算结果不再是 behavior probability。
- **prompt 与 observation 进入 action loss**：优化非策略生成内容。
- **terminal/truncated 合并**：value target 错误。
- **组内 std 为零**：NaN 或噪声被放大。
- **把缺失 reward 当失败**：基础设施故障污染策略。
- **sequence reward 复制到每个 token 后再求和**：长回答被重复加权。
- **只监控 mean reward**：reward hacking、长度漂移和多样性坍缩被掩盖。
- **异步数据无限复用**：importance ratio 极端，更新由过时策略主导。
- **新 recipe 缺少同预算基线**：收益混入更多 rollout、过滤或调参。

## 何时不应在线 RL

高质量示范或离线 pair 已覆盖目标、reward 无法可靠判断核心正确性、环境昂贵且不可重放、或高风险动作缺少外部权限控制时，不应先上在线 RL。Rejection sampling、Best-of-$N$、SFT 或离线偏好往往是更容易验证的基线。

## 验证

1. 当 $\pi_\theta=\pi_{\text{old}}$ 时，所有有效 action 的 $\rho_t=1$。
2. 保持 action-state 输入不变、只修改被 action mask 排除的 prompt/padding log-prob 张量时，policy loss 不变。
3. 用两三步轨迹手算 return、GAE、terminal 与 truncated。
4. RLOO 检查 $K<2$；GRPO 检查全同 reward、极小 std 和缺失结果。
5. 按 policy lag、ratio、长度、group success rate 和 verifier 状态分层。
6. 固定生成 token、样本数、训练 token 和调参预算比较 PPO、RLOO、GRPO 与 rejection sampling。
7. 对 reward 高但人工或隐藏 verifier 失败的轨迹优先审计。
8. save/resume 后 behavior revision/log-prob、old-training revision/recomputed log-prob、data cursor 和 reference 必须连续。

目标函数的最小实现见[手撕 LLM 策略优化](../practice/llm-policy-optimization.md)，多步动作与异步轨迹见[轨迹与策略契约](../agentic-rl/trajectory-contract.md)。

## Reference {#reference}

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
