# 从续写到偏好与在线学习

语言模型首先学会的是条件预测，而不是“怎样成为一个可靠助手”。后训练的演进也不是 SFT、PPO、DPO、GRPO 的算法接力赛；真正推动下一步工作的，是监督信号、采样分布与信用分配不断暴露新的缺口。

理解这条谱系时，应始终分开三个问题：

1. **反馈从哪里来**：示范、人工偏好、AI 偏好、学习式 reward，还是确定性 verifier；
2. **用什么目标更新**：token NLL、pairwise classification、PPO、DPO、RLOO 或 GRPO；
3. **数据怎样产生**：固定离线数据，还是当前策略的在线 rollout。

同一 verifier 可以配合不同 policy optimizer，同一个 optimizer 也可以消费不同来源的 reward。方法名称只有放回这三个坐标后才有意义。

## 预测目标没有定义助手行为

自回归预训练与监督微调都使用 teacher-forced token likelihood：

$$
\mathcal L_{\mathrm{NLL}}
=-\mathbb E_{(x,y)}
\sum_t m_t\log\pi_\theta(y_t\mid x,y_{<t}),
$$

其中 $m_t$ 决定哪些 token 属于训练目标。预训练在自然文本上学习广泛统计结构；SFT 则把 $x$ 组织成任务或对话，把 $y$ 组织成希望模型模仿的回答。

[FLAN](https://arxiv.org/abs/2109.01652) 展示了跨任务 instruction tuning 可以改善未见任务上的 zero-shot 表现。它解决的是“怎样把自然语言指令变成统一任务接口”，却仍然要求数据给出一个可模仿的 completion。对于写作、对话和安全边界，多个答案可能都合理，单个示范很难表达细微取舍。

因此，SFT 是有用的行为先验，而不是完整效用函数。它还继承示范分布的覆盖限制：数据里从未出现的策略，不能仅靠提高示范似然自动被发现。

## 成对偏好把“唯一答案”改成“相对更好”

相对判断通常比编写唯一标准答案容易。给定同一 prompt 下的 chosen $y_w$ 与 rejected $y_l$，Bradley–Terry 模型写成

$$
P(y_w\succ y_l\mid x)
=\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right),
$$

对应损失为

$$
\mathcal L_{\mathrm{RM}}
=-\mathbb E\log\sigma\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right).
$$

早期的 [Learning from Human Preferences](https://arxiv.org/abs/1706.03741)、语言任务上的 [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593) 与摘要研究 [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) 逐步建立了“比较回答—拟合 reward—优化 policy”的接口。

这一步扩大了反馈表达力，也引入新问题：reward model 必须把有限比较外推到新回答。策略一旦针对它持续优化，就可能进入标注数据之外的区域，放大长度、格式或评分器捷径。[Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) 用受控 proxy/gold reward 实验研究了这种现象；其中经验曲线不能直接当作真实人类偏好上的普适常数。

## InstructGPT 把三阶段流程连起来

[InstructGPT](../works/instructgpt.md) 将流程组织为：

```text
demonstrations -> SFT policy
comparisons    -> reward model
fresh rollout  -> PPO with KL / pretraining constraints
```

带 reference policy 的理想化目标常写成

$$
\max_\pi\;
\mathbb E_{y\sim\pi(\cdot\mid x)}[r_\phi(x,y)]
-\beta D_{\mathrm{KL}}
\left(\pi(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\right).
$$

$r_\phi$ 给出优化方向，$\pi_{\mathrm{ref}}$ 给出偏离坐标，$\beta$ 控制二者权衡。PPO 还需要冻结 training-side update 基准 $\pi_{\mathrm{old}}$：

$$
\rho_t(\theta)
=\exp\left[
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\mathrm{old}}(a_t\mid s_t)
\right].
$$

真实产生 rollout 的分布另记为 $\mu^{\mathrm{rollout}}$。只有同步、同精度且 sampling processor 一致时，它才近似 $\pi_{\mathrm{old}}$。old 用于 update ratio，behavior 决定数据分布，reference 用于 KL 或行为锚定；即使三者某一时刻权重相同，语义也不能合并。完整实现契约见[在线 RL](../../training/online-rl.md) 和[训推分布](../../reinforcement-learning/training-inference-discrepancy.md)。

## Constitutional AI 改变反馈生产者

[Constitutional AI](https://arxiv.org/abs/2212.08073) 先让模型依据一组原则批评并修订回答，再让 AI 对回答做比较，用这些比较训练 preference model 并执行 RLAIF。

它改变的是反馈生产过程：

```text
principle -> critique -> revision -> comparison -> reward / policy update
```

它没有创造一种独立于 SFT、reward modeling 或 RL 的通用 optimizer。人仍然选择原则、任务分布与评测方式；实验也不能证明任意 constitution 都会带来形式化安全保证。

## DPO 消去显式 reward model，不消去偏好假设

对 KL 正则化目标，最优策略满足

$$
\pi^*(y\mid x)
\propto
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right).
$$

同一 prompt 的 chosen/rejected 做差时，归一化常数抵消。[DPO](../works/dpo.md) 因而可以直接优化 policy/reference log-ratio：

$$
\mathcal L_{\mathrm{DPO}}
=-\mathbb E\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
\right]\right).
$$

它显著简化了固定偏好数据上的训练，但没有在线生成新候选，也无法自动探索离线 support 之外的高质量行为。reference、pair 生成策略、模板、长度与标签噪声仍然定义了实际学习问题。更多假设与变体见[离线偏好优化](../../training/offline-preference.md)。

## 在线方法重新获得探索，也重新承担方差

当 reward 能够对当前策略的新回答评分时，在线 rollout 可以发现离线 pair 中没有的行为。代价是数据分布随 policy 变化，训练必须处理 rollout 成本、policy lag、importance ratio 与 reward drift。

无 critic 的方法改变的是 advantage estimator：

$$
A_i^{\mathrm{RLOO}}
=R_i-\frac{1}{K-1}\sum_{j\ne i}R_j,
$$

$$
A_i^{\mathrm{GRPO}}
=\frac{R_i-\bar R}{\operatorname{std}(R)+\varepsilon}.
$$

[Back to Basics](https://arxiv.org/abs/2402.14740) 在特定 sequence-reward RLHF 设置中重新评估了 REINFORCE/RLOO；[DeepSeekMath](https://arxiv.org/abs/2402.03300) 描述了数学训练中的 GRPO 配方。二者都不意味着 PPO 在带环境状态、value bootstrap 或长时信用分配的任务中普遍多余。

组内奖励全相等时，GRPO 没有相对学习信号。强行除以极小标准差不会创造信息，只会制造数值噪声。组大小、采样温度、reward 离散度和 response 长度共同决定方差。

当 rollout 扩展为长程环境交互，组内等待和上下文膨胀会成为新的物理瓶颈。[SAO 与 CompactionRL](../works/sao-compactionrl.md) 分别把 single-rollout 异步更新和策略生成的上下文摘要接回 critic-based PPO；这不是算法接力，而是数据形态改变后对调度、状态表示与信用分配的重新设计。

如果要继续追踪 GRPO 之后为什么出现 Dr. GRPO、DAPO、VAPO、GSPO、SAPO、TIS 与 DIS，应进入[推理策略优化谱系](reasoning-policy-optimization.md)。那里把 group baseline、长度分母、ratio gate 与训推系统拆成独立坐标，避免把所有工作压进“在线方法”一个段落。

## RLVR 改变 reward provenance

数学答案、代码测试、结构化约束和环境终态有时可以由确定性程序检查。此时可以用

$$
R(x,y)=v_{\mathrm{exec}}(x,y)
$$

替代或补充 learned reward model。[Tülu 3](https://arxiv.org/abs/2411.15124) 将这类阶段称为 Reinforcement Learning with Verifiable Rewards；其[官方 Open Instruct 实现](https://github.com/allenai/open-instruct)公开了相应训练配方。

RLVR 描述奖励来源，并不指定 PPO、RLOO 或 GRPO。verifier 也只验证规格覆盖的性质：测试通过不等于需求完整，格式正确不等于推理正确。怎样把搜索、验证与在线训练闭合，见[从外显推理到可验证搜索](reasoning-verification.md)。

## 一张决策表

| 方法 | 反馈 | 数据分布 | 关键估计器 | 能否主动探索 |
| --- | --- | --- | --- | --- |
| SFT | 示范 token | 离线 | token NLL | 否 |
| RM + PPO | 学习式 scalar reward | 在线 | clipped policy gradient + value | 是 |
| DPO | chosen/rejected pair | 离线 | policy/reference pair log-ratio | 否 |
| RLOO | scalar reward | 在线分组 | leave-one-out baseline | 是 |
| GRPO | scalar reward | 在线分组 | group-normalized advantage | 是 |
| RLVR | 可执行 verifier | 通常在线 | 由所选 optimizer 决定 | 取决于 optimizer |
| RLAIF | AI 产生的比较或评分 | 离线或在线 | 由所选目标决定 | 取决于数据流程 |

## 证据边界

- [InstructGPT](../works/instructgpt.md) 支持特定 API prompt 与标注者分布上的偏好改善，不证明对齐后的小模型通常比大模型更有能力。
- [DPO](../works/dpo.md) 支持若干离线偏好任务上的简洁有效训练，不证明在线 RL 已经没有价值。
- GRPO 的数学形式不提供可靠 verifier，也不自动消除长度偏差、奖励投机或污染。
- Constitutional AI 降低部分人工反馈依赖，不等于移除人类规范选择。
- RLVR 在可判定领域提供高精度 reward，不代表开放式写作、事实综合与安全判断都可被确定性验证。

机制细节分别见[后训练总览](../../training/post-training.md)、[奖励建模](../../training/reward-modeling.md)、[离线偏好优化](../../training/offline-preference.md)与[在线 RL](../../training/online-rl.md)。

## Reference {#reference}

- [Finetuned Language Models Are Zero-Shot Learners / FLAN](https://arxiv.org/abs/2109.01652)
- [Learning from Human Preferences](https://arxiv.org/abs/1706.03741)
- [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)
- [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback](https://arxiv.org/abs/2402.14740)
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)
