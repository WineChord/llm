# Off-policy 校正：当数据不是由当前策略产生

On-policy 推导假设轨迹来自正在优化的策略；真实系统却常把 rollout 与 learner 解耦，复用 replay，或让许多推理 worker 使用稍旧的 checkpoint。此时 behavior policy $\mu$ 负责产生数据，target policy $\pi$ 才是要评估或优化的对象。

Off-policy correction 试图修正这两个分布之间的差异，但不存在把任意旧轨迹“无损变新”的技巧。importance sampling 无偏却可能高方差；截断权重降低方差又引入偏差；分布支持已经丢失时，任何重加权都无法恢复未采到的动作。

先读 [Policy Gradient](policy-gradient.md)理解 on-policy estimator，再读 [Actor–Critic](actor-critic.md)理解 value target。本页最后把经典 importance sampling、V-trace 与异步 LLM/Agentic RL 的版本契约连起来。

## Behavior 与 target policy

设轨迹由

$$
a_t\sim\mu(\cdot\mid h_t)
$$

产生，而目标是估计或优化 $\pi$。单步 importance ratio 为

$$
\rho_t
=\frac{\pi(a_t\mid h_t)}
{\mu(a_t\mid h_t)}.
$$

只要 $\mu(a\mid h)>0$ 覆盖所有 $\pi(a\mid h)>0$ 的动作，就有

$$
\mathbb E_{a\sim\mu}
\left[
\rho(a\mid h)f(a)
\right]
=
\mathbb E_{a\sim\pi}[f(a)].
$$

整条轨迹的权重则是 ratio 乘积：

$$
w_{0:T}
=\prod_{t=0}^{T}\rho_t.
$$

长轨迹中，即使每个 $\rho_t$ 只略偏离 $1$，乘积也可能指数级变大或趋近零。这是 off-policy 长程语言 agent 特别困难的原因：一次 episode 可包含成百上千个 token 和多次环境转移。

## Per-decision importance sampling

估计折扣回报时，无须让早期 reward 乘上未来动作 ratio。Per-decision IS 写成

$$
\widehat V^\pi(h_0)
=
\sum_{t=0}^{T}
\gamma^t
\left(
\prod_{i=0}^{t}\rho_i
\right)r_t.
$$

它利用因果结构减少不必要的乘积，但方差仍随 horizon 增长。Self-normalized IS 可把权重除以 batch 权重和，通常更稳定，却不再严格无偏，并且 batch composition 会影响每个样本的有效权重。

因此工程上还会采用：

- 截断 $\rho_t$ 或轨迹总权重；
- 限制 replay 年龄与 policy lag；
- 只保留新旧策略重叠良好的 token；
- 使用 learned value bootstrap 缩短乘积；
- 直接丢弃无法可靠校正的轨迹。

每一种都在样本效率、方差和偏差之间重新取舍。

## V-trace

IMPALA 面对许多 actor 落后于 learner 的场景，使用截断 importance weight 构造 value target。定义

$$
\rho_t
=\frac{\pi(a_t\mid h_t)}
{\mu(a_t\mid h_t)},\qquad
\bar\rho_t=\min(\bar\rho,\rho_t),\qquad
c_t=\min(\bar c,\rho_t).
$$

带校正的 TD residual 为

$$
\delta_t^V
=
\bar\rho_t
\left[
r_t+\gamma(1-d_t)V(h_{t+1})-V(h_t)
\right].
$$

从位置 $s$ 开始的 V-trace target：

$$
v_s
=V(h_s)
+\sum_{t=s}^{s+n-1}
\gamma^{t-s}
\left(
\prod_{i=s}^{t-1}
c_i
\right)
\delta_t^V.
$$

actor 可使用

$$
\widehat A_s^{\text{V-trace}}
=
\bar\rho_s
\left[
r_s+\gamma(1-d_s)v_{s+1}-V(h_s)
\right].
$$

$\bar\rho$ 主要控制目标 policy 的校正强度，$\bar c$ 控制多步信息向前传播的方差。截断让 estimator 更稳定，也把 fixed point 推向由 $\pi$、$\mu$ 和截断共同定义的策略；它不是“数值上更稳但完全等价”。

## 一个紧凑的 V-trace target

```python
import torch

def vtrace(reward, value, logp, behavior_logp, terminated,
           gamma=0.99, rho_bar=1.0, c_bar=1.0):
    """reward/logp/terminated: [B,T], value: [B,T+1]."""
    if value.shape[-1] != reward.shape[-1] + 1:
        raise ValueError("value must include bootstrap state")
    ratio = (logp - behavior_logp).exp()
    rho = ratio.clamp(max=rho_bar)
    c = ratio.clamp(max=c_bar)
    cont = 1.0 - terminated.to(reward.dtype)
    delta = rho * (reward + gamma * cont * value[:, 1:] - value[:, :-1])
    target = torch.empty_like(reward)
    carry = value[:, -1]
    for t in range(reward.shape[1] - 1, -1, -1):
        carry = value[:, t] + delta[:, t] + (
            gamma * cont[:, t] * c[:, t] * (carry - value[:, t + 1])
        )
        target[:, t] = carry
    next_target = torch.cat([target[:, 1:], value[:, -1:]], dim=1)
    advantage = rho * (
        reward + gamma * cont * next_target - value[:, :-1]
    )
    return target, advantage, ratio
```

这段代码没有处理 padding。调用方必须使每条 batch row 的有效 transition 连续，或在递推中加入单独的 valid-step mask；把 padding reward 设零仍会让 `value` 和 `c` 穿过伪时间步传播。

## Retrace、ACER 与不同校正目标

V-trace 不是唯一选择：

- **Retrace($\lambda$)** 用截断 ratio 与 eligibility trace 构造安全、低方差的 action-value target；
- **ACER** 把经验回放、截断 IS 与 bias correction 结合到 actor–critic；
- **off-policy policy gradient** 直接在 behavior state distribution 下推导目标，但优化口径不同于原始 on-policy performance；
- **PPO-style clipping** 限制 policy update，却不能单独修复任意 policy lag。

选择方法前要先明确校正对象：是 value evaluation、policy gradient、整条 sequence likelihood，还是异步系统中“哪些 token 仍可消费”。把不同论文里的 clip 常数拼在一起，不会自动形成一致 estimator。

## LLM rollout 的真实 behavior distribution

语言模型系统常只保存 checkpoint 名称，然后用该 checkpoint 重新计算 old log-prob。这不足以重建 behavior policy，因为实际采样还依赖：

```text
tokenizer and chat template
temperature
top-k / top-p / typical sampling
repetition and length processors
grammar or tool-schema masks
precision, quantization and inference kernel
adapter set
random-number implementation
```

若 token 从 top-$p$ 截断分布采样，$\mu$ 可以指截断后的归一化分布，也可以指 processor 前的基础模型分布；两种 ratio 对应不同目标。必须保存或明确重建实际采样概率，不能在更新时静默切换口径。

工具 observation 不是 policy action，不应有 importance ratio；assistant 生成的工具名、参数或文本才属于动作。若一次工具调用按 span 视为动作，则

$$
\log\rho_{\text{span}}
=\sum_{t\in\text{span}}
\left(
\log\pi_t-\log\mu_t
\right).
$$

长 span 的 ratio 更容易极端。改成 token 平均 log-ratio 会改善尺度，却改变 estimator，应作为独立算法选择记录。

## Policy lag 不只是版本号之差

异步系统中，behavior checkpoint 为 $v_b$，learner 当前 checkpoint 为 $v_l$。`v_l-v_b` 只能粗略表示 lag；两次小更新可能比一次大更新更接近，也可能相反。更有意义的诊断包括：

- action token 上的 current/behavior KL；
- log-ratio 的 p50、p95、p99 与极值；
- 被截断、丢弃或降权的 token/sequence 比例；
- 按任务、长度、worker 与环境类型分层的 lag；
- 轨迹从生成、验证、排队到消费的 wall-clock age；
- 更新后 reward、success 与独立评测是否随 lag 退化。

系统应给数据设新鲜度预算。高方差校正不如直接重新 rollout 时，丢弃旧数据是正确选择，不是样本浪费。轨迹需要保存的不可变字段见[轨迹与策略契约](../agentic-rl/trajectory-contract.md)，rollout–learner 调度见[训练系统](../agentic-rl/training-systems.md)。

## 与 PPO、SAO 的边界

[Trust Region 与 PPO](trust-region-ppo.md)通常假设一批轨迹由冻结的 $\pi_{\text{old}}$ 产生，再做有限次更新。异步系统打破的是“采完再学”的整齐边界：

```text
worker samples under behavior v_b
  -> environment / verifier latency
  -> queueing
  -> learner already at v_l
  -> correction, filtering or discard
```

AReaL 等系统工作把 staleness 视作一等指标；SAO 则针对长尾 Agentic rollout，保存真实 behavior log-prob，并用双侧 importance 区间选择仍可用于更新的 token。详细方法与实验边界见 [SAO 与 CompactionRL](../landscape/works/sao-compactionrl.md#sao)。这类筛选提高异步吞吐，但被丢弃的 tail 可能产生选择偏差，critic 也需要跟上新策略。

## 实现契约

```text
target and behavior policy immutable revisions
actual sampled action IDs and behavior log-probabilities
sampling processor and probability convention
ordered transition, action and valid-step masks
terminal versus truncation bootstrap semantics
rho/c clipping thresholds and estimator definition
trajectory age, update lag and replay count
discard/downweight reason for every rejected sample
```

最小验证：

1. $\pi=\mu$ 时所有 ratio 为 $1$，V-trace 退化为普通多步 TD target；
2. 一步真正终态不读取边界 value；
3. truncation 可保留明确提供的 bootstrap value；
4. 人工构造极大、极小 ratio，检查 clip 与 bias 方向；
5. 改变 observation token 的 log-prob 不影响 policy correction；
6. 按 lag bucket 重跑固定评测，而不只比较训练 reward。

## 失败边界

- **support 不重叠**：behavior 从未采到的动作不能靠重加权恢复。
- **重算 behavior log-prob**：模板、processor 或 kernel 漂移会产生虚假 ratio。
- **长序列 ratio 连乘**：数值下溢只是表象，统计方差同样爆炸。
- **clip 被写成无偏校正**：截断明确引入偏差。
- **PPO clip 替代 off-policy 分析**：有限更新约束不能修复任意陈旧数据。
- **版本差替代分布差**：checkpoint ID 距离不等于 KL。
- **padding 穿过递推**：伪 transition 污染 value target。
- **基础设施失败记作低 reward**：策略被训练去回避机器故障。
- **只报告吞吐提升**：更旧的数据可能换来更差的有效学习率与最终能力。

## 历史位置

Importance sampling 为不同策略间的期望转换提供了数学起点，但长轨迹暴露了它的方差极限。Retrace、ACER 和 V-trace 逐步把截断校正、bootstrap 与分布式 actor–learner 结合起来。LLM 与 Agentic RL 把同一问题推到更长序列、更昂贵环境和更复杂采样器上：系统吞吐与统计新鲜度必须一起优化。

## Reference {#reference}

- [Degris, White, and Sutton, Off-Policy Actor-Critic](https://arxiv.org/abs/1205.4839)
- [Munos et al., Safe and Efficient Off-Policy Reinforcement Learning / Retrace](https://arxiv.org/abs/1606.02647)
- [Wang et al., Sample Efficient Actor-Critic with Experience Replay / ACER](https://arxiv.org/abs/1611.01224)
- [Espeholt et al., IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
- [Fu et al., AReaL: A Large-Scale Asynchronous Reinforcement Learning System](https://arxiv.org/abs/2505.24298)
