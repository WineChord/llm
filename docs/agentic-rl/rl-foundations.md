# 从经典强化学习到语言 Agent

语言 Agent 没有发明一套脱离经典强化学习的新数学。它改变的是问题尺度：state 由长历史近似，action 可以持续数千 token，environment 包含工具与外部系统，reward 又常在任务结束后才出现。本页只负责连接两端；MDP、Bellman、TD、policy gradient 与 PPO 的完整推导放在[强化学习](../reinforcement-learning/index.md)。

## 三种建模层次

### Contextual bandit

若每个 prompt 独立，模型生成完整 response 后立即得到 reward，且回答不影响未来任务，可近似为：

$$
x\sim\mathcal D,\qquad
y\sim\pi_\theta(\cdot\mid x),\qquad
R=R(x,y).
$$

这里没有跨 turn transition。许多单轮 RLHF、DPO 与 response-level RLVR 在这个抽象下最清楚。

### Token-level episodic MDP

把生成前缀视为状态、下一个 token 视为 action：

$$
h_t=(x,y_{<t}),
\qquad
a_t=y_t.
$$

transition 是确定性的字符串追加，直到 EOS 或长度上限。该建模解释了 sequence log-prob 怎样拆成 token log-prob，却不意味着每个 token 都有独立 reward。完整动作尺度见[语言模型作为策略](../reinforcement-learning/language-model-policy.md)。

### 多轮 POMDP / SMDP

工具调用和外部操作会改变未来观察：

$$
a_t\sim\pi_\theta(\cdot\mid h_t),
\qquad
s_{t+1}\sim P(\cdot\mid s_t,a_t),
\qquad
o_{t+1}\sim\Omega(\cdot\mid s_{t+1}).
$$

模型只看到 $h_t$，真实文件、权限、网页或用户状态可能不可见，因此更接近 [POMDP](../reinforcement-learning/decision-processes.md)。一次 tool call 持续时间不同，又可用 [SMDP 与 option](../reinforcement-learning/models-planning-hierarchy.md)描述。

## 六个对象怎样变化

| 对象 | 经典抽象 | 语言 Agent 中的难点 |
| --- | --- | --- |
| State | $s_t$ | context 只是部分观察；外部状态会漂移 |
| Action | $a_t$ | token、span、turn、tool call 或完整 episode |
| Transition | $P(s'\mid s,a)$ | 工具、网络、文件和用户模拟器 |
| Reward | $R_t$ | human/AI preference、verifier、成本与权限 |
| Time | 固定 step | token、turn、wall-clock 与调用成本并存 |
| Policy | $\pi(a\mid s)$ | tokenizer、模板、decoder 与权重共同定义 |

任何算法选择之前，先把这六项写成可重放契约。

## History 不是 state 的同义词

完整 history 只有在包含所有影响未来的变量时才可能是 Markov state。现实系统经常缺少：

- 工具内部状态；
- 文件或数据库 revision；
- 访问权限与预算；
- 被截断的早期约束；
- 并发 actor 的修改；
- 用户后续行为。

摘要、retrieval 与 memory 是 belief-state engineering。它们提高可用信息，不证明信息充分。[长时任务](long-horizon.md)讨论上下文压缩与恢复，[CompactionRL](../landscape/works/sao-compactionrl.md#compactionrl)则把 summary 本身作为可训练 action。

## Observation 不参与 policy gradient

轨迹可写成

$$
\tau=(a_0,o_0,a_1,o_1,\ldots).
$$

只有 policy 实际采样的 $a_i$ 进入 action loss；$o_i$ 作为下一动作的条件。tool result 很长时，不能因为它占据很多 token，就让这些 token 产生 policy ratio 或梯度。

精确字段、mask、old log-prob 与 version 见[轨迹与策略契约](trajectory-contract.md)。

## Reward 来源不决定 optimizer

[反馈制度](../reinforcement-learning/feedback-regimes.md)把几个常被混淆的轴分开：

- RLHF、RLAIF、RLVR 描述反馈来源；
- online/offline 描述数据是否持续刷新；
- on/off-policy 描述数据与目标策略关系；
- PPO、RLOO、GRPO 描述更新或 baseline。

一个 coding Agent 可以使用程序 verifier reward、learned critic、PPO update 与异步 rollout；“RLVR”只覆盖其中 reward 的来源。

## 长时信用

终局成功要传回早期决策。可选接口包括：

- episode Monte Carlo return；
- turn/span reward；
- critic + TD/GAE；
- process verifier；
- 层级 option；
- 搜索产生的 state value；
- 跨压缩 segment 修正。

更细的 reward 不一定更准确。一个有偏 PRM 会把错误监督传播到更多 token。完整比较见[语言模型信用分配](../reinforcement-learning/credit-assignment.md)。

## 探索与安全

语言模型通过 temperature、多采样、任务 curriculum、工具选择和环境状态探索。探索强度不能只看 entropy：

- 高 temperature 可能只改变措辞；
- 同组 reward 全同意味着没有有效行为差异；
- 高风险 action 不能靠负 reward 事后纠正；
- verifier coverage 决定“新行为”是否可评价；
- sandbox 与权限 guard 必须先于执行。

[探索与最大熵](../reinforcement-learning/exploration-entropy.md)给出经典基础，[Agent 安全](../applications/agent-security.md)给出运行时边界。

## 终止、截断与故障

| 状态 | 是否 bootstrap | 是否作为策略结果 |
| --- | --- | --- |
| 成功 / 明确失败 | 通常否 | 是 |
| 时间、token 或调用预算截断 | 依任务而定 | 未完成但可归因 |
| 非法动作 | 按环境契约 | 是 |
| Environment / verifier / infrastructure error | 不应自动记零 reward | 通常单列 |

将 timeout 全部当 terminal 会低估长任务 value，也会鼓励过早结束。公式与例子见[序贯决策](../reinforcement-learning/decision-processes.md)和[多步回报](../reinforcement-learning/multistep-traces.md)。

## 阅读路线

| 想解决的问题 | 先读 | 再读 |
| --- | --- | --- |
| MDP、value、TD 不熟 | [强化学习总览](../reinforcement-learning/index.md) | [Bellman](../reinforcement-learning/values-bellman.md)、[MC/TD](../reinforcement-learning/prediction-control.md) |
| PPO 与 critic | [Policy Gradient](../reinforcement-learning/policy-gradient.md) | [Actor–Critic](../reinforcement-learning/actor-critic.md)、[TRPO/PPO](../reinforcement-learning/trust-region-ppo.md) |
| RLOO / GRPO | [无 critic baseline](../reinforcement-learning/critic-free-baselines.md) | [在线 RL](../training/online-rl.md) |
| 异步 rollout | [Off-policy 校正](../reinforcement-learning/off-policy-correction.md) | [训练系统](training-systems.md) |
| 多轮工具任务 | 本页 | [轨迹契约](trajectory-contract.md)、[数据与环境](data-environments.md) |
| 长上下文与终局 reward | [信用分配](../reinforcement-learning/credit-assignment.md) | [长时任务](long-horizon.md) |

## Reference {#reference}

- Sutton and Barto, [Reinforcement Learning: An Introduction, Second Edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
- Kaelbling, Littman, and Cassandra, [Planning and Acting in Partially Observable Stochastic Domains](https://doi.org/10.1016/S0004-3702(98)00023-X)
- Sutton, Precup, and Singh, [Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction](https://doi.org/10.1016/S0004-3702(99)00052-1)
- Ouyang et al., [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
