# 在线 RL 与可验证奖励

在线强化学习从当前或近期策略采样，再根据 reward、verifier 或环境终态更新策略。它能探索离线数据中不存在的行为，也带来非平稳数据、昂贵 rollout、策略滞后和奖励投机。

[InstructGPT](../landscape/works/instructgpt.md)适合核对经典 RLHF 中 SFT、reward model 与 PPO 的角色；[DeepSeek-R1](../landscape/works/deepseek-r1.md)则展示可验证奖励、group-relative 更新与蒸馏之间必须分开的证据边界。两者之间的演进见[后训练与对齐](../landscape/lineages/training-alignment.md)。

## 策略与轨迹契约

一次训练更新可能涉及三个不同策略：

| 符号 | 角色 | 是否变化 |
| --- | --- | --- |
| $\pi_\theta$ | 正在计算梯度的新 policy | 当前 update 内变化 |
| $\pi_{\text{old}}$ | 产生 rollout 的 behavior policy | 对该批轨迹冻结 |
| $\pi_{\text{ref}}$ | KL 或行为先验的 reference | 通常长期冻结 |

$\pi_{\text{old}}$ 决定 importance ratio，$\pi_{\text{ref}}$ 定义策略偏离。二者权重偶尔相同，也不能把 old log-prob 与 reference log-prob 混用。

每个 action token 至少记录：

```text
prompt / environment and trajectory ID
token or action ID and action mask
behavior policy version and old log-prob
reference version and reference log-prob
reward components and verifier version
terminal / truncated / invalid / timeout / infra error
sampling configuration and RNG
```

工具 observation、system token、prompt 和 padding 不是 policy action，不应进入 policy ratio。

## PPO

importance ratio 为

$$
\rho_t(\theta)
=
\exp\left[
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\text{old}}(a_t\mid s_t)
\right].
$$

[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) 的 clipped surrogate 为

$$
\mathcal L_{\text{policy}}
=-\mathbb E_t
\left[
\min\left(
\rho_tA_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right].
$$

clip 约束的是相对 behavior policy 的 update，不是相对 reference 的 KL。后者有两种不同实现：

$$
\begin{aligned}
r_t^{\text{rollout}}
&=r_t^{\text{task}}
-\beta\left(
\log\pi_{\text{old}}(a_t\mid s_t)
-\log\pi_{\text{ref}}(a_t\mid s_t)
\right),\\
\mathcal J_{\text{current}}
&=\mathcal J_{\text{policy}}
-\beta D_{\mathrm{KL}}
\left(\pi_\theta\,\|\,\pi_{\text{ref}}\right).
\end{aligned}
$$

第一种在采样轨迹上把 behavior/reference log-ratio 计入 reward，第二种在更新时直接约束当前策略；二者的梯度、估计偏差和统计不同，不能在同一实现中无说明地混用。系数、位置与版本都应分别记录。

### GAE

对 value $V$，TD residual 为

$$
\delta_t
=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),
$$

$$
A_t
=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}.
$$

$d_t$ 应表示真正 terminal。时间预算导致的 truncated episode 仍可能有 bootstrap value；把 truncated 当 terminal 会系统性低估尾部价值。

[InstructGPT](https://arxiv.org/abs/2203.02155) 给出了语言模型 SFT、reward model 与 PPO 的代表性组合。该流程不是所有任务的默认最优解：critic、reward 和多轮 rollout 都增加系统与统计复杂度。

## 无 critic 的 baseline

### RLOO

同一 prompt 采样 $K\ge2$ 个回答，sequence reward 为 $R_i$。Leave-One-Out baseline 给出

$$
A_i
=R_i-\frac{1}{K-1}\sum_{j\ne i}R_j.
$$

baseline 不包含自身 reward；若使用组均值

$$
R_i-\frac{1}{K}\sum_jR_j,
$$

则幅度缩小且 baseline 与自身样本耦合。[Back to Basics](https://arxiv.org/abs/2402.14740) 重新研究了 REINFORCE、RLOO 与 PPO 在 RLHF 中的比较。

### ReMax

[ReMax](https://arxiv.org/abs/2310.10505) 用 greedy response 的 reward 作为 prompt-level baseline，不训练 critic。它减少 value model 成本，但 baseline 质量依赖 greedy 解码和 reward 稳定性；greedy 生成也要计入 rollout 成本。

## Group-relative 方法

[DeepSeekMath](https://arxiv.org/abs/2402.03300) 描述的 GRPO 配方以同 prompt 一组回答的 reward 统计构造优势：

$$
\hat A_i
=
\frac{R_i-\bar R}
{\operatorname{std}(R)+\varepsilon}.
$$

这引入几个重要边界：

- 全组 reward 相同，则没有相对学习信号；
- std 很小时，$\varepsilon$ 与数值精度决定尺度；
- 每组只含成功或只含失败时，增加采样也未必产生梯度；
- group size、reward 离散度和采样温度共同决定方差；
- sequence reward 怎样分配到 token，会改变长回答权重。

如果全组相同，应明确输出零优势或跳过，而不是除以零、把 infra error 混进组，或人为制造排名。

[DAPO](https://arxiv.org/abs/2503.14476) 与 [Dr. GRPO](https://arxiv.org/abs/2503.20783) 分析并修改了 clipping、动态采样、长度与归一化等具体配方。这些是有公开实验的较新 recipe，不应被写成跨任务普适结论；应逐项与清晰的 PPO/RLOO/GRPO baseline 消融。

## Off-policy 与策略滞后

异步 rollout 会使行为策略落后于训练策略。精确 importance weight 依赖：

- 产生该 action 的 exact policy revision；
- 相同 tokenizer、模板、action boundary 和 sampling processor；
- 保存的 old log-prob 与实际采样概率一致。

若轨迹过旧，可丢弃、限制 ratio，或使用 off-policy correction。[IMPALA](https://arxiv.org/abs/1802.01561) 的 V-trace 使用 clipped importance weights 构造 value target：

$$
\delta_t^V
=\bar\rho_t
\left(r_t+\gamma V(s_{t+1})-V(s_t)\right),
$$

再用 $\bar c_t$ 控制多步修正传播。clip 降低方差也引入偏差，不能把极旧数据“修正”为等价 on-policy。

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
2. response/action mask 改变 prompt 与 padding 时，policy loss 不变。
3. 用两三步轨迹手算 return、GAE、terminal 与 truncated。
4. RLOO 检查 $K<2$；GRPO 检查全同 reward、极小 std 和缺失结果。
5. 按 policy lag、ratio、长度、group success rate 和 verifier 状态分层。
6. 固定生成 token、样本数、训练 token 和调参预算比较 PPO、RLOO、GRPO 与 rejection sampling。
7. 对 reward 高但人工或隐藏 verifier 失败的轨迹优先审计。
8. save/resume 后 policy version、old log-prob、data cursor 和 reference 必须连续。

目标函数的最小实现见[训练目标实现](../practice/training-objectives.md)，多步动作与异步轨迹见[轨迹与策略契约](../agentic-rl/trajectory-contract.md)。
