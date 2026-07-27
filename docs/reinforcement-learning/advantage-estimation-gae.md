# Advantage 估计与 GAE：把长期回报变成局部学习信号

策略梯度需要回答的不是“这条轨迹最后得了多少分”，而是“在当时已经看到的信息下，这个动作比策略通常会做的选择好多少”。前者是 return，后者是 advantage。两者只差一个 baseline，却对应完全不同的统计问题：return 决定任务目标，advantage estimator 决定有限样本怎样把这个目标分配给每一次更新。

本页从 TD residual 推出 Generalized Advantage Estimation（GAE），重点解释它的边界语义、actor 与 critic 的不同 target，以及语言模型中 token、turn、segment 三条时间轴。多步 return 与 eligibility trace 的共同背景见[多步回报、资格迹与 GAE](multistep-traces.md)，策略梯度为什么允许减去 baseline 见 [Policy Gradient](policy-gradient.md)，value function 的训练耦合见 [Actor–Critic](actor-critic.md)。

## Advantage 为什么是策略更新的接口

对策略 $\pi$，状态价值、动作价值与优势函数分别为

$$
V^\pi(s)
=
\mathbb E_\pi[G_t\mid S_t=s],
$$

$$
Q^\pi(s,a)
=
\mathbb E_\pi[G_t\mid S_t=s,A_t=a],
$$

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

$Q^\pi$ 回答“先做这个动作，再按 $\pi$ 行动会得到什么”，$V^\pi$ 回答“按 $\pi$ 在这里通常会得到什么”。二者相减后，正 advantage 表示该动作优于当前策略的条件平均水平，负 advantage 表示更差。Policy-gradient theorem 因而可以写成

$$
\nabla_\theta J(\theta)
=
\mathbb E_{s\sim d^{\pi_\theta},\,a\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(a\mid s)
A^{\pi_\theta}(s,a)
\right].
$$

真实 $A^\pi$ 不可直接查询，只能用 rollout 与 learned value 估计。最简单的 Monte Carlo estimator 是

$$
\widehat A_t=G_t-\widehat V(s_t).
$$

它等待完整回报，较少依赖 bootstrap，却会把后续所有随机性都带回当前时刻。一步 TD estimator 则是

$$
\widehat A_t^{(1)}
=
r_t+\gamma\widehat V(s_{t+1})-\widehat V(s_t),
$$

方差通常更低，但直接依赖 value 的局部准确性。GAE 不是一种新的 reward，也不改变环境目标；它在这两端之间组合多个 TD residual。

## 从 TD residual 推出 GAE

先固定一份 value snapshot $\widehat V$。对非终止 transition 定义

$$
\delta_t
=
r_t+\gamma\widehat V(s_{t+1})-\widehat V(s_t).
$$

把连续两个 residual 相加：

$$
\delta_t+\gamma\delta_{t+1}
=
r_t+\gamma r_{t+1}
+\gamma^2\widehat V(s_{t+2})
-\widehat V(s_t).
$$

中间的 $\widehat V(s_{t+1})$ 正好抵消，得到两步 bootstrap return 减去当前 value。一般地，

$$
\sum_{\ell=0}^{n-1}\gamma^\ell\delta_{t+\ell}
=
G_t^{(n)}-\widehat V(s_t).
$$

GAE 再用 $\lambda$ 对这些不同长度的 advantage estimator 做指数加权。最常用的 residual 形式为

$$
\widehat A_t^{\operatorname{GAE}(\gamma,\lambda)}
=
\sum_{\ell=0}^{\infty}
(\gamma\lambda)^\ell\delta_{t+\ell}.
$$

在没有边界的理想情形下：

- $\lambda=0$ 只保留一步 TD residual；
- $\lambda\to1$ 且展开到真实 episode 终点时，望远镜消去中间 value，接近 Monte Carlo return 减 baseline；
- 中间的 $\lambda$ 让较远 residual 的影响按 $(\gamma\lambda)^\ell$ 衰减。

这里 $\gamma$ 与 $\lambda$ 不能互换。$\gamma$ 首先定义任务怎样权衡远期 reward；改变它通常改变优化目标。$\lambda$ 主要决定 advantage estimator 对 bootstrap bias 与采样 variance 的权衡。把 $\lambda$ 称为“第二个 discount”会掩盖这个区别。

有限轨迹上不应真的构造所有 $n$-step return。反向递推即可得到同一固定 value snapshot 下的结果：

$$
\widehat A_t
=
\delta_t+\gamma\lambda\widehat A_{t+1}.
$$

真正困难的部分不是这行递推，而是“下一项是否存在”“下一状态是否还能 bootstrap”以及“一步究竟对应 token、turn 还是环境 transition”。

## Terminal 与 truncated 是两个不同问题 {#boundaries}

令

$$
d_t=\mathbf 1[\text{transition }t\text{ 后真正终止}],
$$

$$
b_t=\mathbf 1[\text{transition }t\text{ 后轨迹边界}],
$$

其中轨迹边界包含 `terminated` 与 `truncated`。于是 TD residual 应写成

$$
\delta_t
=
r_t+\gamma(1-d_t)\widehat V(s_{t+1})-\widehat V(s_t),
$$

GAE 递推则写成

$$
\widehat A_t
=
\delta_t
+\gamma\lambda(1-b_t)\widehat A_{t+1}.
$$

两个 mask 承担不同职责：

- `terminated` 表示任务动力学已经结束，未来价值为零，因此禁止 bootstrap；
- `truncated` 表示时间预算、最大步数或采样窗口结束，任务本身未必结束，因此通常仍从真实 final observation bootstrap；
- 两者都会结束当前存储片段，所以 GAE carry 不得穿进 batch 中物理相邻的下一条轨迹；
- timeout、工具崩溃、reward 服务缺失等基础设施错误不应被悄悄编码成普通 terminal 或零 reward。

一个常见错误是只保存 `done = terminated or truncated`，并同时用它关闭 bootstrap 与 trace。这样会把 time-limit truncation 的尾部价值系统性压低。另一个错误是在截断处读取 packed tensor 的 `value[t + 1]`；那个位置可能已经属于下一条 episode。正确的数据对象应显式保存 transition 自己的 `next_value`，它由该 transition 的真实后继 observation 计算。

## Actor target 与 critic target 不能混成一个张量

GAE 首先产生 actor 使用的 advantage target：

$$
\widehat A_t
=
\operatorname{stopgrad}
\left[
\operatorname{GAE}
(r,\widehat V_{\text{old}},d,b)
\right].
$$

停止梯度很重要。策略损失不应通过 $\widehat A_t$ 反向修改 value network，尤其不能让 actor 通过改变 baseline 人为降低自身 loss。若 actor 与 critic 共享 backbone，仍应在 target 边界 detach，再由各自损失对共享参数产生明确梯度。

critic 常用的回归 target 为

$$
\widehat R_t^\lambda
=
\widehat A_t+\widehat V_{\text{old}}(s_t),
$$

并优化

$$
\mathcal L_V
=
\frac12
\mathbb E_t
\left[
\left(
V_\phi(s_t)-\operatorname{stopgrad}(\widehat R_t^\lambda)
\right)^2
\right].
$$

这里加回的是生成 target 时冻结的 value snapshot，而不是 optimizer 已更新后的当前 $V_\phi$。若同一批 rollout 做多个 minibatch epoch，return 与 advantage 通常保持冻结；每个 epoch 重新用变化中的 critic 计算 GAE，会让 actor target 在一批数据内部漂移。

actor 与 critic 还可能使用不同的 mask 和 reduction：

- actor 只在 policy 实际选择的 action 上计算 log-probability loss；
- critic 可以在每个有明确定义 value target 的决策状态上回归；
- observation token 可能用于构造状态，却不一定是动作；
- padding、缺失 reward 与无效轨迹必须同时从分子和分母排除。

Advantage whitening 也不是 GAE 定义的一部分。按整个 batch、每个 prompt group、每条序列或每个任务分别中心化，会改变不同样本的相对权重。分布式训练若只在单个 rank 上计算均值与方差，还会让同一 global batch 因切分方式不同而得到不同更新。

## 三条时间轴：token、turn 与 segment

[语言模型作为策略](language-model-policy.md)指出，同一段文本可以同时包含 prompt、模型动作、工具 observation 与环境状态。GAE 必须先选择决策时间轴，再谈递推。

### Token 时间轴

把每个生成 token 视作动作时，

$$
s_t=(x,y_{<t}),\qquad a_t=y_t.
$$

value head 需要解释为“给定当前 prefix，继续按策略生成的期望未来回报”。若只有 response 末尾的 outcome reward，中间 token 的 TD residual 主要来自相邻 value 差；这给出细粒度信用，却要求 token-level value 真正校准。

prompt、system token、tool 返回和复制进上下文的 observation 通常不属于 policy action。实现至少要分开：

```text
action mask: 哪些位置进入 actor loss
trace mask: 哪些 transition 沿同一决策链传播 GAE
bootstrap mask: 哪些后继状态仍有未来价值
valid mask: 哪些位置进入统计分母
```

只用一个 attention mask 代替这四种语义，通常会把“模型能够读取”误写成“模型选择了这个动作”。

### Turn 时间轴

在工具 Agent 中，更自然的一步可能是一次 assistant message、一次 tool call 或一个 action span。环境返回 observation 后才进入下一决策状态：

```text
state h_t
  -> assistant action a_t
  -> environment / tool transition
  -> reward r_t and next state h_{t+1}
```

此时 $\gamma$ 描述的是跨 turn 的延迟，而不是跨 token 的距离。一个很长的 JSON tool call 是否比一个短调用多折扣数百次，应由任务语义决定，不能由 tokenizer 偶然决定。turn-level critic 可以显著缩短 GAE 序列，但会失去 action span 内部的信用分辨率。

### Segment 时间轴

长程 Agent 可能把一次 episode 切成多个训练 segment，或在上下文接近上限时生成摘要再继续。segment boundary 只是存储或表示边界，不应自动成为环境 terminal。

如果后一段仍属于同一 episode，就需要保存：

- segment 的全局顺序与原始 episode ID；
- 前段末状态和后段初状态的对应关系；
- 摘要是策略动作、外部变换还是只读 observation；
- 后续 value 能否在压缩后的信息状态上可靠 bootstrap；
- GAE carry 是否跨 segment，以及跨越时按哪种时间距离衰减。

跨 segment 递推并不自动正确。若压缩丢失了影响未来 reward 的信息，critic 实际在一个近似 belief state 上预测；延长 trace 只能传播 estimator，不能恢复已经丢失的状态信息。长轨迹中的 action、observation 与层级信用边界见[语言模型信用分配](credit-assignment.md)和[轨迹与策略契约](../agentic-rl/trajectory-contract.md)。

## 一个两步手算

设 $\gamma=0.9,\lambda=0.95$。第一步是普通 transition：

$$
r_0=0,\qquad V_0=0.4,\qquad V_1=0.6.
$$

第二步因时间预算而 truncation，但真实 final observation 的 value 为 $0.8$：

$$
r_1=1,\qquad V_1=0.6,\qquad V_{\mathrm{final}}=0.8.
$$

truncation 允许当前 TD target bootstrap，却必须在这里截断 trace，因此

$$
\delta_1=1+0.9\times0.8-0.6=1.12,
\qquad
\widehat A_1=1.12.
$$

第一步的 residual 与 advantage 为

$$
\delta_0=0+0.9\times0.6-0.4=0.14,
$$

$$
\widehat A_0
=0.14+(0.9\times0.95)\times1.12
=1.0976.
$$

如果第二步是真正 terminal，同样的即时 reward 会得到 $\delta_1=1-0.6=0.4$；差异来自是否存在可继续估值的环境状态，而不是数组是否在这里结束。可执行 packed-tensor 版本及退化断言集中在[手撕 LLM 策略优化](../practice/llm-policy-optimization.md)。

## 偏差、方差与失败边界

$\lambda$ 增大通常减少对局部 critic 的依赖，也让更远 reward 噪声进入当前 advantage。这个经验方向不等于固定排序：

- critic 在某类状态上系统性偏高时，较短 trace 会继承局部 bias；
- episode 很长、reward 极稀疏时，较长 trace 的 variance 和数值范围可能迅速增加；
- policy lag 使 residual 对应的 value、reward 与当前策略不一致，调小 $\lambda$ 不能替代[Off-policy 校正](off-policy-correction.md)；
- value pretraining、actor/critic 更新频率和 target normalization 都会改变同一 $\lambda$ 的表现；
- response 长度、turn 数与 segment 数不同，固定 $\gamma\lambda$ 会给不同任务不同的有效信用半衰期。

诊断时不应只看 advantage 的全局均值和方差。至少按 trajectory length、reward 类型、terminal/truncated、任务类别与 policy version 分层，并检查：

1. $\lambda=0$ 时 advantage 是否精确等于 TD residual；
2. 真正 terminal 是否关闭 bootstrap；
3. truncation 是否使用真实 final observation 的 value；
4. carry 是否在每条 episode 或存储边界正确重置；
5. actor mask 改动 prompt、observation 与 padding 后，policy loss 是否保持不变；
6. critic target 是否由冻结 snapshot 构造；
7. advantage whitening 的统计范围是否跨全部参与 rank 一致；
8. token、turn、segment 三种长度变化时，优势尺度与有效样本权重怎样变化。

GAE 最值得保留的不是一条反向循环，而是一组明确的接口：reward 属于哪一步、value 预测哪个策略的未来、什么会终止任务、什么只会截断存储、actor 对哪些动作负责。接口没有固定时，两个都叫“GAE”的实现可能在优化不同对象。

## Reference {#reference}

- Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- Sutton and Barto, [Reinforcement Learning: An Introduction, Second Edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
- Pardo et al., [Time Limits in Reinforcement Learning](https://arxiv.org/abs/1712.00378)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
