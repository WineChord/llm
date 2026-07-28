# RLHF、RLAIF、RLVR：先分反馈来源，再谈算法

RLHF、RLAIF 与 RLVR 首先描述反馈怎样产生或被组织，不是互斥的优化器。PPO、RLOO、GRPO 或 REINFORCE 都可以消费由人类偏好、AI 偏好或 verifier 构造的 reward；DPO 一类方法则把 preference pair 直接变成监督式 loss，不使用 policy-gradient estimator。反馈来源、训练 loss 和梯度估计器必须分层命名。

## 四条彼此独立的轴

### 数据何时产生

| 制度 | policy 是否产生新数据 | 主要限制 |
| --- | --- | --- |
| 固定离线 | 训练期间不再采样 | 支持集受旧 policy 限制 |
| 迭代式 | 每轮更新后重新采样 | 版本与轮次耦合 |
| 持续在线 | rollout 持续进入 learner | policy lag、漂移与安全 |

### 反馈由谁给出

| 来源 | 例子 | 优点 | 风险 |
| --- | --- | --- | --- |
| 人类 | pairwise preference、rubric | 接近真实主观目标 | 成本、分歧、疲劳与群体偏差 |
| AI | critique、ranking、constitutional feedback | 可扩展、协议一致 | judge 偏差、自我偏好、串谋 |
| 程序 verifier | 单元测试、证明检查、exact answer | 可重复、可规模化 | 覆盖有限、parser 与测试漏洞 |
| 环境 | 游戏分数、任务终态、真实成本 | 直接反映交互结果 | 昂贵、随机、可能有副作用 |

### 反馈怎样变成训练信号

raw feedback 本身不是 reward，更不是 advantage。一个 pair

$$
y_w\succ y_l\mid x
$$

至少要沿以下一条路径进入优化：

```text
preference -> learned reward -> sampled return -> advantage -> policy gradient
preference -> direct preference loss -> ordinary supervised gradient
verifier status -> explicit reward mapping -> return / advantage -> policy gradient
```

第一条路径通常先用 Bradley–Terry 等模型拟合 reward difference；第二条路径如 DPO 直接优化 chosen/rejected 的 log-probability margin；第三条路径仍要决定 invalid、timeout、部分通过和成本怎样映射为数值 reward。任何路径都不能把原始 preference 标签直接称为 advantage。

### 使用策略梯度时怎样估计 advantage

- learned critic / TD / GAE；
- leave-one-out 或 greedy baseline；
- group mean / standard deviation；
- Monte Carlo return 与状态无关 baseline；
- off-policy importance correction。

这四条轴正交。把 `RLVR = GRPO`、`RLAIF = DPO`，或把 DPO 列作 policy-gradient estimator，都会把反馈、目标函数与 estimator 混在一起。

## RLHF 的术语约定

RLHF 在论文和工程语境中没有唯一、统一的外延。本站为避免同一缩写在相邻页面漂移，采用以下局部约定，而不宣称这是整个领域的标准定义：

- **strict RLHF**：人类比较训练显式 reward model，再由在线或周期式 policy-gradient optimization 更新 policy；SFT 常作为初始化，但不是定义中必需的梯度阶段；
- **alignment from human feedback**：更宽的上位描述，包括 demonstration/SFT、critique、直接 preference loss、显式 RM + RL，以及人类参与的 policy selection。

[InstructGPT](../landscape/works/instructgpt.md) 式 SFT → reward model → PPO 是 strict RLHF 的代表性实现。DPO/IPO 可以使用同一种人类 preference 数据，但本站把它们记作 direct preference optimization，而不是 policy-gradient RLHF；需要涵盖二者时使用更宽的 alignment from human feedback。

## RLAIF

RLAIF 用模型产生偏好、critique 或 rubric score。若 AI feedback 训练 reward model，再用 PPO 更新 policy，训练路径是显式 RM + 在线 actor–critic；若直接对 AI 生成的 pair 做 DPO，则是离线或迭代式 direct preference optimization。两者共享反馈来源，不共享目标函数。

[Constitutional AI](https://arxiv.org/abs/2212.08073) 包含两条不同数据路径：

1. critique/revision 生成监督数据；
2. 原则驱动的 AI preference 进入 RL 阶段。

它的贡献不只是“把人换成模型”，而是显式组织反馈原则与自我修订。原则覆盖、judge 能力与最终人类评测仍是独立环节。

## RLVR

RLVR 使用可自动核验的 reward，例如：

- 数学答案匹配或符号验证；
- 代码编译、单元测试和隐藏测试；
- 定理证明 kernel；
- 环境终态；
- 结构化约束与安全沙箱结果。

它降低反馈扩展成本，并允许当前 policy 在线探索。但可验证的不一定是完整目标：测试通过不保证代码可维护，最终答案正确不保证推理可靠，格式合法也不保证内容真实。完整机制见[可验证奖励](rlvr.md)。

## 一个训练可以混合多种反馈

总 reward 可写为

$$
R=
w_hR_{\mathrm{human}}
+w_aR_{\mathrm{AI}}
+w_vR_{\mathrm{verifier}}
-w_cC.
$$

这个表达看似简单，实际有三层问题：

1. **尺度**：不同分量的数值范围与噪声不同；
2. **聚合**：线性权重把不可补偿约束变成可交易分数；
3. **可识别性**：总 reward 上升时，无法直接知道哪一分量驱动。

安全或权限约束通常更适合硬门禁或 constrained objective，而不是允许高任务分抵消违规惩罚。

## Offline / online 与 on-policy / off-policy

两组术语也不能合并：

- **online/offline** 描述训练期间是否从当前策略或环境获取新数据；
- **on-policy/off-policy** 描述梯度 estimator 的目标策略与数据行为策略是否一致。

周期式 PPO 使用刚采样 rollout 时接近 on-policy；对同一数据做多个 epoch 后已经产生偏移。持续在线系统不断采样，却可能因为队列和 learner 更新而消费 off-policy 数据。Online DPO 会重新生成 preference pair，但每轮优化仍可在固定 pair 上进行。

## 选择反馈制度

| 任务条件 | 更自然的反馈起点 |
| --- | --- |
| 专家行为可直接示范 | SFT / imitation |
| 质量主观、相对比较容易 | 人类或 AI preference |
| 终态可精确执行判断 | RLVR / rejection sampling |
| 中间状态可可靠核验 | process verifier + outcome |
| 环境昂贵且不可重放 | 离线数据、保守更新 |
| 风险动作不可试错 | sandbox、权限门禁、离线评测优先 |

算法选择应在反馈可靠性之后。reward 不可信时，换一种 advantage estimator 只会更稳定地优化错误目标。

## 反馈通道隔离

训练与最终评测最好使用不同证据通路：

```text
training reward / preference
  != model-selection judge
  != hidden final evaluator
  != deployment safety gate
```

如果同一模型同时生成训练 critique、reward、最终 judge 和解释，错误会高度相关。至少保留隐藏测试、人类抽检、对抗样例或独立 evaluator。

## 记录与评测

每次实验应报告：

- feedback source 与版本；
- candidate policy 与采样参数；
- online/offline 数据制度；
- behavior/target/reference policy；
- raw feedback 到 reward 或 direct loss 的映射，以及 reward 到 advantage 的 estimator；
- reward 原始分量与聚合；
- estimator 与 denominator；
- evaluator 是否参与训练；
- 覆盖不到的目标与未知。

[RLHF 数据闭环](rlhf-pipeline.md)展开人类反馈，[奖励建模](../training/reward-modeling.md)展开 learned reward，[Verifier 与 reward shaping](verifiers-reward-shaping.md) 展开程序反馈。

## Reference {#reference}

- Christiano et al., [Deep Reinforcement Learning from Human Preferences](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html)
- Bai et al., [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300)
