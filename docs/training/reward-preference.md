# 奖励建模与偏好优化

偏好学习处理的是“相对更好”，而不是天然存在的绝对奖励。数据、偏好模型、参考策略和 divergence 共同定义最终目标；方法名称不能替代这些假设。

## Bradley–Terry 奖励模型

给定 prompt $x$ 与成对回答 $y_w\succ y_l$，标量奖励模型常假设

$$
P(y_w\succ y_l\mid x)
=\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right),
$$

并最小化

$$
\mathcal L_{\text{RM}}
=-\log\sigma\left(r_w-r_l\right).
$$

奖励只在差值上可辨识：给所有回答加同一常数不改变偏好概率。训练与评测应关注 pairwise accuracy、margin、校准和分布外行为，而不是奖励绝对值看起来多大。

## 奖励模型的捷径

模型可能学习：

- 长回答通常被偏好；
- 某种标题、礼貌语或拒答模板；
- 生成器身份与格式；
- 引用数量而非引用支持性；
- 代码块存在而非代码正确；
- 过程更长而非推理更可靠。

需要构造长度匹配、风格扰动和单一错误的 hard negatives，并用可执行 verifier 与人工抽检分解评分器到底学到了什么。

## KL 正则化的 RLHF

经典形式在提高期望奖励的同时限制策略偏离 reference：

$$
\max_\pi
\mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)]
-\beta D_{\mathrm{KL}}
\left(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\right).
$$

reference 不是“旧权重备份”这么简单：它定义了偏离成本的坐标。tokenizer、模板和 support 不一致时，KL 失去清晰语义。[InstructGPT 深读](../landscape/works/instructgpt.md)给出 SFT、reward model 与 PPO 串联的代表性管线、最小 clipped objective 和公开证据边界。

## Direct Preference Optimization

[DPO 深读](../landscape/works/dpo.md)将上述带 reverse-KL 正则的最优策略关系代回 Bradley–Terry 模型，得到分类式目标。定义

$$
\Delta_\theta
=\log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x),
$$

$$
\Delta_{\text{ref}}
=\log\pi_{\text{ref}}(y_w\mid x)-\log\pi_{\text{ref}}(y_l\mid x),
$$

则

$$
\mathcal L_{\text{DPO}}
=-\log\sigma\left(
\beta(\Delta_\theta-\Delta_{\text{ref}})
\right).
$$

DPO 不需要在线 rollout 或独立 reward model，工程更简单；它仍受偏好覆盖、reference、序列长度归一化、标签噪声与离线分布限制。

## IPO、KTO 与目标选择

[IPO](https://arxiv.org/abs/2310.12036)来自更一般的偏好优化框架，强调直接使用成对偏好并分析 DPO/RLHF 的近似；[KTO](https://arxiv.org/abs/2402.01306)则允许只使用 desirable/undesirable 的二元反馈，并引入相对 reference 的效用构造。

它们解决的数据接口不同：

| 数据条件 | 自然起点 | 仍需解决 |
| --- | --- | --- |
| 高质量成对偏好 | DPO / IPO 类 | pair 构造、reference、长度偏置 |
| 只有单样本好坏标签 | KTO 类 | 类别不平衡、基准效用 |
| 可在线采样且奖励可验证 | PPO / group-relative RL | rollout 成本、off-policy 与奖励攻击 |
| 多步轨迹偏好 | trajectory preference 或过程监督 | 信用分配、环境版本与动作边界 |

不存在对所有数据都最优的偏好损失；选择应由监督语义决定。

## 序列 log-probability

回答的 log-probability 是 token log-prob 之和：

$$
\log\pi(y\mid x)=\sum_{t=1}^{T}\log\pi(y_t\mid x,y_{<t}).
$$

因此长回答的数值绝对值更大。目标是否使用总和、均值或其他长度校正会改变偏好。mask 必须只覆盖策略真正选择的 token；prompt、padding 和环境 observation 不应混入 action probability。

## Offline 与 online

离线偏好优化只在已收集回答上学习，稳定且易复现，但无法直接探索新行为。在线 RL 根据当前策略采样，可以发现新解，也会让数据分布随训练变化，放大奖励模型分布外误差。

从离线切换到在线前应验证：

1. reward/verifier 对当前策略样本仍校准；
2. rollout 能记录不可变 policy version 与旧 log-prob；
3. 生成、训练与 reference 模板完全一致；
4. reward 突增会触发轨迹审计；
5. 成本、长度和安全回归不被单一奖励掩盖。

## 评测

至少分开报告：

- 成对偏好胜率与人工一致性；
- 事实性、任务成功和可执行验证；
- 长度、风格、多样性与校准；
- 对 reference 的 KL 或 log-ratio 分布；
- chosen/rejected 来源分层结果；
- 通用能力和安全边界回归；
- judge 更换后的桥接评测。

偏好数据契约见[偏好、过程与轨迹数据](../data/feedback-trajectories.md)，在线策略算法见[Agentic RL 数学与算法](../agentic-rl/math-algorithms.md)。

## Reference {#reference}

- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- [A General Theoretical Paradigm to Understand Learning from Human Preferences / IPO](https://arxiv.org/abs/2310.12036)
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
