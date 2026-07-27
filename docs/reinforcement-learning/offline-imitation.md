# 模仿学习、逆强化学习与 Offline RL

离线数据可以包含专家示范、旧策略轨迹、偏好对或混合质量经验。Behavior Cloning、Inverse RL 与 Offline RL 都使用这些数据，却回答不同问题：模仿动作、推断目标，或在不新增环境交互的条件下改进策略。

## Behavior Cloning

给定示范数据

$$
\mathcal D=\{(s_i,a_i)\}_{i=1}^N,
$$

Behavior Cloning 最小化监督负对数似然：

$$
\mathcal L_{\mathrm{BC}}
=-\mathbb E_{(s,a)\sim\mathcal D}
\log\pi_\theta(a\mid s).
$$

对语言模型，这就是 response/action-only SFT。它简单、稳定，不需要 reward；但训练 state 来自专家，部署 state 来自 learner。一旦早期动作偏离，后续状态可能落到数据未覆盖区域，错误会累积。

## Covariate shift 与 DAgger

[DAgger](https://proceedings.mlr.press/v15/ross11a.html)迭代执行当前 policy，在它实际访问的状态上查询专家，再聚合数据。它把训练分布逐渐拉向 learner occupancy：

```text
train policy
  -> execute current policy
  -> expert labels visited states
  -> aggregate
  -> retrain
```

这需要安全、可查询的专家。语言 Agent 的状态包含外部工具和历史，重新标注可能昂贵或不可逆。

## Inverse Reinforcement Learning

IRL 假设示范近似优化某个未知 reward，目标是推断 reward 而非直接复制动作。reward 通常不可唯一识别：多个 reward 可以诱导相同最优 policy。

Maximum-entropy IRL 在固定环境动力学下，用可行轨迹的 base measure 与指数 reward 共同定义分布：

$$
p_\theta(\tau)
=\frac{1}{Z_\theta}
p(s_0)
\prod_t P(s_{t+1}\mid s_t,a_t)
\exp\left(R_\theta(\tau)\right).
$$

环境初始分布与转移决定哪些轨迹可行，指数项才在这些轨迹之间偏向更高 reward；简写成 $p_\theta(\tau)\propto e^{R_\theta(\tau)}$ 时，这个 base measure 不能被理解为消失。它与[KL 正则化控制](kl-regularized-control.md)的指数倾斜有共同结构，但状态、数据和规范化对象不同。

## GAIL

[GAIL](https://proceedings.neurips.cc/paper/2016/hash/cc7e2b878868cbae992d1fb743995d8f-Abstract.html)用 discriminator 区分 expert 与 policy occupancy，再通过 adversarial training 让两者接近。它避免显式恢复 reward，却仍需要在线采样 policy 轨迹；因此不属于纯 offline 方法。

## Offline RL

Offline RL 在固定 transition 数据集上优化 expected return，不再与环境交互。关键困难是 distributional shift：Bellman target 或 policy improvement 会查询数据很少支持的动作。

设 behavior policy 为 $\mu$，若新 policy 在某状态选择 $\mu$ 几乎从未采取的动作，$Q(s,a)$ 只能外推。function approximation 的小误差会在 $\max$ 或 policy optimization 中被放大。

## Support 约束

最保守的原则是：

$$
\pi(a\mid s)>0
\Longrightarrow
\mu(a\mid s)>0,
$$

但连续或组合动作空间中无法精确满足。实践中使用：

- behavior regularization；
- conservative value penalty；
- expectile / advantage-weighted regression；
- uncertainty 或 pessimism；
- policy constraint 与 OOD action detection。

## CQL 与 IQL 的不同取向

[Conservative Q-Learning](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)惩罚数据外动作的高 Q，试图学习保守价值下界。

[Implicit Q-Learning](https://arxiv.org/abs/2110.06169)避免在训练 value 时直接评估数据外动作，使用 expectile value 与 advantage-weighted behavior cloning 提取 policy。两者都依赖固定数据质量与超参数，不能将“offline”理解为无需环境假设。

## 与离线偏好优化的关系

DPO 使用固定 chosen/rejected response，也常被称为 offline preference optimization。它与一般 Offline RL 的共同点是受 behavior support 限制；不同点是：

- 数据通常是 preference pair，不是完整 transition/reward；
- 目标来自 KL-regularized preference model 推导；
- 不显式做 Bellman backup；
- response 可能被视为 contextual-bandit action。

因此 DPO 不是 CQL/IQL 的语言版本；完整推导见[离线偏好优化](../training/offline-preference.md)。

## Offline-to-online

常见路线是先用离线数据初始化，再有限在线探索：

1. BC/SFT 建立可用 policy；
2. offline value 或 preference 学习；
3. 在 sandbox / verifier 下在线采样；
4. 用新轨迹校正旧数据覆盖；
5. 保留离线回归集防止遗忘。

切换时要记录 behavior policy、数据时间、reward/verifier version 和 online mixture。否则旧数据 replay 会在新目标下产生不可解释梯度。

## 语言 Agent 的特殊困难

- response/action space 组合爆炸；
- observation 与外部状态难以重放；
- 工具版本和权限改变 transition；
- expert demonstration 可能只展示成功路径；
- 失败轨迹缺少“应该怎样恢复”；
- reward 可能只在终态给出；
- 旧文本重新 tokenize 不等于旧行为概率。

当现有数据缺少失败邻域与恢复行为时，高质量失败、恢复和截断轨迹可以补充仅靠成功示范难以覆盖的状态。

## 选择方法

| 条件 | 起点 |
| --- | --- |
| 高质量示范充足、目标行为清楚 | BC / SFT |
| 可持续查询专家 | DAgger 式迭代 |
| 想从行为推断可迁移目标 | IRL / preference learning |
| 固定 transition + reward 数据 | Offline RL |
| 固定成对偏好 | DPO / IPO 等 |
| 有可靠安全环境与 verifier | Offline initialization + online RL |

## 验证

1. 按 behavior policy 和时间切分，不随机泄漏同一轨迹。
2. 报告 dataset coverage、成功/失败、长度与终止分布。
3. 在 behavior actions 上与 OOD actions 上分别测 Q。
4. BC 是必须比较的强基线。
5. policy improvement 后用独立环境或 hidden verifier 验证。
6. 不能交互时，明确 offline evaluation 的模型假设。
7. offline-to-online 时审计旧数据权重和 policy lag。
8. Agent 数据保留 exact action、observation、环境和版本。

轨迹字段见[语言模型作为策略](language-model-policy.md)和[轨迹契约](../agentic-rl/trajectory-contract.md)，函数逼近风险见[致命三元组](function-approximation.md)。

## Reference {#reference}

- Ross, Gordon, and Bagnell, [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html)
- Ziebart et al., [Maximum Entropy Inverse Reinforcement Learning](https://publications.ri.cmu.edu/maximum-entropy-inverse-reinforcement-learning/)
- Ho and Ermon, [Generative Adversarial Imitation Learning](https://proceedings.neurips.cc/paper/2016/hash/cc7e2b878868cbae992d1fb743995d8f-Abstract.html)
- Kumar et al., [Conservative Q-Learning for Offline Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)
- Kostrikov, Nair, and Levine, [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169)
