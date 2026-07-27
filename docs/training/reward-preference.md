# 奖励、偏好与策略学习

“让模型更好”不是一个可直接优化的标量。人类选择、原则评审、单样本好坏标签、单元测试和环境终态提供的是不同观察；reward model、偏好损失和强化学习再把这些观察变成更新信号。先固定反馈接口，才有理由选择 DPO、PPO 或其他算法。

本页保留为稳定入口。成对评分的统计细节见[奖励建模](reward-modeling.md)，固定数据上的策略学习见[离线偏好优化](offline-preference.md)，会持续改变采样分布的问题见[强化学习](../reinforcement-learning/index.md)。

## 五个彼此独立的轴

| 轴 | 典型选择 | 它决定什么 |
| --- | --- | --- |
| 反馈来源 | 人类、AI、程序、环境 | 目标偏差与可扩展性 |
| 监督形态 | 示范、pair、list、标量、过程标签、终态 | 可使用的损失 |
| 数据制度 | 固定数据 / 持续采样 | 是否能探索新行为 |
| 策略关系 | on-policy / off-policy | 是否需要分布校正 |
| 正则锚点 | reference、constraint、无显式锚点 | 允许偏离多远 |

[反馈制度](../reinforcement-learning/feedback-regimes.md)进一步说明：RLHF、RLAIF 与 RLVR 说的是反馈从哪里来；offline/online 说的是数据会不会刷新；on/off-policy 说的是数据由哪个策略产生。它们不能用一个缩写互相替代。

## 从观察到目标

### 示范回答

若数据直接给出希望模型模仿的 $y^\star$，最自然的起点是 response-masked SFT：

$$
\mathcal L_{\mathrm{SFT}}
=-\sum_t m_t\log\pi_\theta(y_t^\star\mid x,y_{<t}^\star).
$$

它把示范中的每个目标 token 都当作正例，却不知道未展示的其他回答是否也正确。高质量覆盖比复杂 optimizer 更重要。

### 成对偏好

若只知道 $y_w\succ y_l$，可以先学习标量 reward：

$$
P(y_w\succ y_l\mid x)
=\sigma\!\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right),
$$

也可以像 DPO 一样直接学习相对 reference 的 policy log-ratio。前者得到可用于在线采样的评分器，但要承担分布外校准问题；后者省去独立 reward model，却仍受固定 pair 的 support、标签噪声和 reference 约束。两条路线的完整推导分别见[奖励建模](reward-modeling.md)与[离线偏好优化](offline-preference.md)。

### 可执行结果

数学等价检查、编译测试、游戏终局或可查询环境状态能给出可重复反馈。这类信号通常比开放式 judge 更精确，却只覆盖 verifier 实际检查的性质。策略可能通过格式漏洞、测试盲区或环境副作用得分；[RLVR](../reinforcement-learning/rlvr.md)和 [Verifier 与奖励塑形](../reinforcement-learning/verifiers-reward-shaping.md)专门讨论这个边界。

### 过程反馈

step、span 或 turn 级标签缩短了信用路径，但标签粒度更细不等于更真实。过程评分器可能偏爱冗长格式，局部正确也不保证最终可达。[语言模型信用分配](../reinforcement-learning/credit-assignment.md)说明 outcome、process、critic 与 search value 怎样落到 token 或 action。

## 一个选择顺序

```text
目标能否由确定性规则直接验证？
  是 -> 先建立 verifier、隐藏测试与失败分类
  否 -> 人类或模型反馈的语义能否稳定标注？
          否 -> 先改任务定义、工具或评测
          是 -> 固定数据是否覆盖希望学到的行为？
                  是 -> SFT / 离线偏好作为基线
                  否 -> 考虑在线采样，并承担分布漂移与探索成本
```

在线 RL 最有说服力的场景，是新采样能够发现离线 support 外的有效行为，而且 reward 在当前策略分布上仍可信。它也可用于持续吸收新反馈、适应不断变化的采样分布，或让 reward 直接重加权当前策略产生的轨迹；这些收益都要与 rollout、分布校正和验证成本一起比较。[RLHF 数据闭环](../reinforcement-learning/rlhf-pipeline.md)把标注、reward model、rollout、策略更新与独立评测连接为可审计的数据循环。

## Reference、old 与 behavior

在线更新常同时出现四种策略身份：

- $\pi_\theta^{\mathrm{train}}$：正在更新的 learner；
- $\pi_{\mathrm{old}}^{\mathrm{train}}$：一轮更新开始时冻结的 training-side policy；
- $\mu^{\mathrm{rollout}}$：由 checkpoint、推理引擎与 sampling processor 共同定义的真实 behavior；
- $\pi_{\mathrm{ref}}$：定义偏离成本的冻结 reference。

$\pi_{\mathrm{old}}^{\mathrm{train}}$ 用于 current–old update ratio，$\mu^{\mathrm{rollout}}$ 决定数据分布，$\pi_{\mathrm{ref}}$ 用于 KL anchor。三者权重一度相同不代表语义相同。模板、tokenizer、sampling processor 或 action mask 不一致时，旧 log-prob、reference log-prob 与真实采样概率也不再可比较。完整约束见[语言模型作为策略](../reinforcement-learning/language-model-policy.md)、[KL 正则化控制](../reinforcement-learning/kl-regularized-control.md)、[训推分布](../reinforcement-learning/training-inference-discrepancy.md)和[轨迹契约](../agentic-rl/trajectory-contract.md)。

## 常见的错误顺序

- 先决定使用 DPO 或 PPO，再寻找能塞进目标的数据；
- 用 reward model 的绝对分数解释偏好强度，而忽略可加常数与分布漂移；
- 把“AI 产生反馈”误写成一种 optimizer；
- 用 online rollout，却没有记录 behavior policy version 与采样概率；
- 用程序 verifier，却不区分错误答案、超时、无效动作和基础设施故障；
- 在同一 judge 上训练、筛选和报告最终效果；
- 只看平均 reward，不检查长度、风格、覆盖、多样性与真实任务成功。

## 最小验证矩阵

| 接口 | 必测退化 | 独立结果 |
| --- | --- | --- |
| Pair preference | 对调顺序、长度匹配、同分 pair | 人工一致性与分层胜率 |
| Reward model | 常数平移、OOD 生成器、格式扰动 | 校准、margin 与真实任务 |
| Offline preference | reference 更换、support 外 prompt | 新鲜人工或 verifier 评测 |
| Online RL | policy lag、极端 ratio、reward 缺失 | 冻结任务集与成本 |
| Verifier | 隐藏测试、对抗格式、环境故障 | 人工审计与第二评分器 |
| Process feedback | step 边界变化、局部对全局反例 | 最终成功与错误恢复 |

对应的最小目标实现见[训练目标](../practice/training-objectives.md)和[手撕强化学习](../practice/reinforcement-learning.md)，实验中的 reward 上升是否代表真实能力见[实验诊断](../reinforcement-learning/evaluation-debugging.md)。

## Reference {#reference}

- Christiano et al., [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)
- Ouyang et al., [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- Bai et al., [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- Lightman et al., [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
