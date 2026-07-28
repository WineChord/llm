# 迭代式 RLHF：反馈、数据与策略共同演化

本页把 **RLHF pipeline** 限定为一条迭代闭环：当前策略产生候选，人类比较训练显式 reward model，policy-gradient optimization 改变策略，新策略再产生下一轮候选。SFT 常用于初始化；PPO 也可以替换成其他策略梯度 recipe。

这是本站为保持页面边界采用的术语约定，不是对领域缩写的统一定义。固定 demonstration、一次性离线 preference optimization 和人类参与的 model selection 都属于更宽的 alignment from human feedback，但不属于本页讨论的显式 RM + 迭代 policy-gradient pipeline。反馈制度的完整坐标见[反馈制度](feedback-regimes.md)。

## 目标从哪里来

传统 RL 假设环境直接返回 reward。开放式语言任务中，“有帮助、诚实、合适”难以写成程序，人类比较提供一种相对监督：

$$
y_w\succ y_l\mid x.
$$

Bradley–Terry reward model 假设

$$
P(y_w\succ y_l\mid x)
=\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right).
$$

它只能从差值学习排序，reward 零点不可辨识；尺度还会影响后续 RL 强度。完整标签语义、ties 与多维 rubric 见[奖励建模](../training/reward-modeling.md)。

## 闭环的六个阶段

```text
任务与目标分布
  -> 示范 / SFT
  -> 当前策略生成候选
  -> 人类比较或评分
  -> 显式 reward model
  -> reward / return / advantage
  -> policy-gradient optimization 与独立评测
  -> 新策略重新生成候选
```

### 1. 定义目标分布

先固定谁的任务、哪类语言和风险边界。训练集里的 prompt 分布、标注者分布与部署分布不是同一个对象。若目标混入多个相互冲突群体，单个 scalar reward 会把聚合规则隐藏进模型。

### 2. SFT 建立行为先验

SFT 提供可读、可控的初始策略，降低在线探索成本。它也是 KL reference 的常见起点。SFT 太弱，RL 会花大量 rollout 学基本格式；SFT 过窄，则可能限制探索与输出多样性。

### 3. 从当前策略采样候选

偏好数据不是静态标签集。候选质量、相似度和错误类型取决于采样 policy、temperature 与 prompt 选择。两个几乎相同的候选难以标注；一个显然错误、一个显然正确又提供很少边界信息。active query selection 的目标是提高信息量，而不是只收集容易一致的 pair。

### 4. 收集反馈

反馈形式决定可学习对象：

| 反馈 | 数据单位 | 能表达什么 | 主要限制 |
| --- | --- | --- | --- |
| Demonstration | 单个 response | 目标行为实例 | 成本高、只有一条路径 |
| Pairwise preference | chosen/rejected | 相对排序 | 不给绝对尺度 |
| Listwise ranking | 多候选 | 更完整局部排序 | 标注负担更高 |
| Scalar rating | response + score | 绝对等级近似 | 标注者量表漂移 |
| Critique / rubric | 文本与维度 | 错误类型和理由 | 解析与一致性复杂 |

标注协议、展示顺序、参考答案、时间限制和是否允许跳过都属于数据生成机制。

### 5. 从 preference 学 reward，再构造 advantage

原始 pairwise preference 只是比较标签，不是可直接乘到 $\nabla\log\pi$ 上的 advantage。显式 reward model 先拟合

$$
P(y_w\succ y_l\mid x)
=\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right).
$$

对新 rollout 得到的 $r_\phi(x,y)$ 还要与 reference KL、规则约束或成本合成为 reward，再通过 Monte Carlo return、critic/GAE、RLOO 等机制形成 advantage：

```text
raw preference -> reward model
new rollout -> scalar reward -> return / advantage -> policy gradient
```

[DPO 类离线偏好](../training/offline-preference.md)走的是另一条路径：pair 直接定义 chosen/rejected log-probability loss，使用 ordinary supervised gradient，而不是把 preference 当作 policy-gradient advantage。周期性重采样可以让 DPO 形成迭代数据闭环，但它仍不是本页限定的显式 RM + policy-gradient pipeline。

### 6. 优化、评测并重新采样

在线或周期式优化可用 PPO、RLOO、GRPO 或其他 policy-gradient recipe。训练 reward 上升只说明策略更会取得这个反馈；最终仍要用 held-out 人类、隐藏 verifier、能力与安全评测判断是否真的改善。只有把新策略再次送回候选采样，才闭合本页所说的迭代 RLHF pipeline。

## 三种分布漂移

### Policy drift

策略优化后，候选离开 reward model 的训练分布。RM 可能对新型答案过度自信，产生 Goodhart 效应。

### Preference drift

标注标准、产品目标与人群偏好会变化。把时间上不一致的数据合并成一个静态 reward，会将真实变化解释为噪声。

### Evaluator coupling

teacher、reward model、judge 与训练数据若共享模型或来源，错误可能高度相关。训练和最终评测需要尽可能独立的证据通路。

## 闭环更新频率

| 模式 | 新策略是否生成新候选 | 与本页闭环的关系 | 风险 |
| --- | --- | --- | --- |
| 固定离线 preference | 否 | 闭环外的对照；可用于预训练 RM | 覆盖受旧 policy 限制 |
| 周期式 RLHF | 分轮生成 | 每轮采样、标注、RM 与 policy 都可审计 | 周期长、版本复杂 |
| 持续在线 RLHF | 持续生成 | 闭环最紧，但反馈和 policy 同时变化 | policy lag、反馈漂移、安全风险 |
| 混合 human + verifier | 分轮或持续生成 | 人类定义主观目标，程序反馈补充可验证分量 | 分量尺度与覆盖边界 |

真实系统常混合：离线 SFT 和 preference 作为先验，在线 verifier 提供探索信号，再把高质量轨迹蒸馏回稳定 policy。只要人类 preference 不再刷新，就应明确哪些更新属于 RLHF 闭环，哪些只是固定 RM 上的继续优化。

## 数据版本契约

每条候选或比较至少绑定：

```text
task / prompt family and split
candidate policy and decoding configuration
exact token IDs and template version
labeler / evaluator pool and rubric revision
presentation order and reference visibility
raw feedback, tie / skip and disagreement
reward-model training split
timestamp and provenance class
```

模型、数据和 evaluator 都变化时，只保存 `chosen/rejected` 文本无法解释结果。

## Reward hacking 不是偶发 bug

reward model 是有限数据上的函数逼近，而 policy optimization 会主动寻找高分区域。常见症状包括：

- 长度、格式或自信措辞替代真实质量；
- 引用数量增加但证据不支持 claim；
- verifier parser 被特殊字符串绕过；
- 安全拒答模板在不该拒绝时获得高分；
- reward 上升而 held-out 人类偏好下降。

缓解手段包括扩展对抗数据、分解 reward、限制 KL、隐藏 verifier、保留人工审计和用独立终态测量；没有任何一项能单独证明目标已对齐。

## 与 RLAIF、Constitutional AI 的关系

把反馈者换成模型不会改变闭环结构，只会改变反馈生成机制。RLAIF 可以扩大规模、固定 rubric，也会继承 judge 的偏差、prompt 敏感性和自我偏好。[Constitutional AI](https://arxiv.org/abs/2212.08073) 还加入 critique/revision 与原则驱动的反馈生产；原则、judge 与最终人类目标仍需分别评测。

## 最小评测矩阵

1. pairwise agreement：按任务、长度、语言、风险和标注者分层；
2. reward calibration：只在有明确定义的标签概率上解释；
3. policy win rate：使用 held-out evaluator 与盲化顺序；
4. 能力保留：通用知识、推理、代码和校准；
5. reward hacking：高 reward、低真实质量的反例搜索；
6. 分布外：新 prompt、长回答、多轮与 adversarial slice；
7. 成本：反馈小时、rollout token、训练 token 和失败重跑；
8. 版本隔离：训练 RM 与最终 judge 不共享泄漏路径。

策略目标见 [KL 正则化控制](kl-regularized-control.md)，在线 estimator 见[在线 RL](../training/online-rl.md)，整个历史脉络见[从续写到偏好与在线学习](../landscape/lineages/training-alignment.md)。

## Reference {#reference}

- Christiano et al., [Deep Reinforcement Learning from Human Preferences](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html)
- Ziegler et al., [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)
- Stiennon et al., [Learning to Summarize with Human Feedback](https://proceedings.neurips.cc/paper/2020/hash/1f89885d556929e98d3ef9b86448f951-Abstract.html)
- Ouyang et al., [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- Bai et al., [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
