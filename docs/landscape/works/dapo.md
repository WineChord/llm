# DAPO：把 GRPO 从目标函数变成可训练的 RLVR 配方

[DAPO](https://arxiv.org/abs/2503.14476) 的重要性不在于又增加一个 policy-optimization 缩写，而在于公开了一次失败如何被拆开：naive GRPO 在长 reasoning 训练中出现 entropy collapse、无信号组、长度权重与截断噪声；作者没有用单一公式解释全部问题，而把算法、采样、归约、reward 和系统实现一起改动。

它最适合记成四项耦合组件：

```text
Clip-Higher
  + Dynamic Sampling
  + Global Token-Level Policy Gradient Loss
  + Overlong Filtering / Soft Punishment
```

DAPO 仍使用 group-normalized advantage 与 PPO-style surrogate。它不是 dynamic sampling 的别名，也不是一个脱离 verifier、数据和 rollout budget 的普适 optimizer。

## 前一条路线具体卡在哪里

[DeepSeekMath](https://arxiv.org/abs/2402.03300) 中的 GRPO 为每个 prompt 采样一组 response，用组内 reward 均值与标准差构造优势，再做 clipped token update。这个骨架省去 critic，却没有自动规定：

- 好动作的概率最多可以提高多少；
- 全对或全错组是否进入 learner batch；
- 长短 response 的 token 怎样加权；
- 到达 generation limit 的样本是错误、缺失还是有效但未完成；
- reference KL 是否适合当前 reasoning 目标。

DAPO 作者以 Qwen2.5-32B 的长 CoT 数学训练为实验环境，报告 naive GRPO baseline 在 AIME 2024 上明显低于目标结果，并观察 entropy、reward 与 response length 的不稳定。其四项技术分别作用在不同接口，不能把总收益归因给其中任意一项。

完整 baseline 语义与后续 bias 分析见[GRPO](../../reinforcement-learning/grpo.md)，这一系列方法为何出现见[推理策略优化谱系](../lineages/reasoning-policy-optimization.md)。

## 一个公式容纳不了整套 DAPO

对 prompt $q$ 的 $G$ 个 response，reward 标准化为

$$
\widehat A_i
=
\frac{
R_i-\operatorname{mean}(R_1,\ldots,R_G)
}{
\operatorname{std}(R_1,\ldots,R_G)
}.
$$

token update ratio 为

$$
\rho_{i,t}
=
\frac{
\pi_\theta(o_{i,t}\mid q,o_{i,<t})
}{
\pi_{\mathrm{old}}(o_{i,t}\mid q,o_{i,<t})
}.
$$

DAPO 的核心 surrogate 写成

$$
J_{\mathrm{DAPO}}
=
\mathbb E
\left[
\frac1{\sum_i|o_i|}
\sum_i\sum_t
\min\left(
\rho_{i,t}\widehat A_i,\,
\operatorname{clip}
(\rho_{i,t},1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})
\widehat A_i
\right)
\right],
$$

并要求 learner batch 中的 group 满足

$$
0
<
\#\{o_i:\operatorname{correct}(o_i)\}
<
G.
$$

第一行同时体现 asymmetric clip 与 global token denominator，约束体现 dynamic sampling；overlong reward 还不在这个式子里。因此只复制 loss function，并没有复制 DAPO。

## Clip-Higher：上界为何单独放宽 {#clip-higher}

对正 advantage token，PPO 在

$$
\rho>1+\epsilon
$$

后停止给“继续提高概率”提供收益。若 old probability 是 $0.01$，$\epsilon=0.2$ 只允许在 surrogate 活动区升到约 $0.012$；若 old probability 是 $0.9$，同一相对上界对应 $1.08$，实际概率上限会先饱和在 $1$。低概率探索 token 的绝对增长空间因此更受限制。

DAPO 使用

$$
\epsilon_{\mathrm{high}}>\epsilon_{\mathrm{low}}
$$

扩大正 advantage 的上侧活动区，同时保留较紧的 lower side。论文实验配置使用 $0.2/0.28$，这是特定模型、数据和训练预算上的选择，不是通用默认值。

从公式可见，Clip-Higher 改的是 sampled action 的局部梯度几何。它不保证未采动作的概率、全分布 KL 或最终 entropy；这些仍需通过 ratio tail、entropy、采样多样性和 held-out 能力观察。它与 CISPO、GSPO、SAPO 的区别见[Ratio、Clipping 与 Gate](../../reinforcement-learning/ratio-clipping-gating.md)。

## Dynamic Sampling：保持有效组，而非创造信号 {#dynamic-sampling}

binary verifier 下，如果一个 group 全对或全错，则

$$
R_i-\bar R=0
$$

对所有 $i$ 成立。随着 policy 变强，越来越多简单 prompt 变成全对；极难 prompt 则可能长期全错。固定 prompt batch 中真正有梯度的组数会不断变化。

DAPO 持续 oversample，再只保留 mixed group，直到 learner batch 拥有固定数量的有效 prompt。这能稳定每步信号密度，却改变了两种分布：

1. **计算分布**：一次 learner step 需要的 rollout 数成为随机变量；
2. **训练分布**：容易和极难 prompt 被更频繁拒绝，保留集偏向当前成功率中等的题。

同步生成时，wall-clock 可能已由最慢 response 决定，额外短样本被长尾延迟“遮住”；异步系统中这个结论未必成立。公平报告应包括所有生成 token，而不仅是 retained groups。

## Global token loss：分母就是课程 {#token-loss}

原始 GRPO 常先对 response 内 token 求平均：

$$
J_{\mathrm{response}}
=
\frac1G
\sum_i
\frac1{|o_i|}
\sum_tj_{i,t}.
$$

每条 response 等权，意味着长 response 中每个 token 权重更小。DAPO 改成

$$
J_{\mathrm{token}}
=
\frac{
\sum_{i,t}m_{i,t}j_{i,t}
}{
\sum_{i,t}m_{i,t}
},
$$

让 batch 中每个有效 action token 等权。这样高质量长 reasoning 提供更多正 token，低质量长 response 也承受更多负 token。

这项改动不只是数值归一化。它把 response length 变成显式样本权重。若正 reward 与长度高度相关，global token mean 可能增加长答案的总贡献；若长错误答案更多，它也会加大惩罚。需要联合画 reward、advantage 符号与长度，而不是只看平均 response length。

## Overlong reward：截断不是普通错误 {#overlong}

generation 达到 $L_{\max}$ 时，response 可能：

- 推理正确但尚未输出最终答案；
- 已陷入重复或无意义扩展；
- 因预算太短而被系统截断；
- 因基础设施故障不完整。

统一赋一个强负 reward 会把这些原因混在一起。DAPO 先使用 overlong filtering 排除被截断样本的 policy loss，再提出 soft overlong punishment。设缓冲区长度为 $L_{\mathrm{cache}}$：

$$
R_{\mathrm{length}}(y)
=
\begin{cases}
0,
&|y|\le L_{\max}-L_{\mathrm{cache}},\\
\dfrac{
(L_{\max}-L_{\mathrm{cache}})-|y|
}{
L_{\mathrm{cache}}
},
&L_{\max}-L_{\mathrm{cache}}<|y|\le L_{\max},\\
-1,
&|y|>L_{\max}.
\end{cases}
$$

它让接近上限的惩罚逐步增加，减少在硬边界处把相近轨迹标成完全不同 reward 的噪声。代价是 reward 已不再只表示答案正确性；length penalty 会主动塑造策略。对 agent 环境，还必须把 time-limit truncation、任务 terminal 和系统失败分开，见[GAE 的边界语义](../../reinforcement-learning/advantage-estimation-gae.md#boundaries)。

## 最小可执行语义

下面的 reference 同时固定 mixed-group 过滤、population-std advantage、asymmetric PPO 与 global token denominator。它不包含模型采样、分布式队列或生产 verifier。

```python
import torch
def dapo_batch(reward, new_logp, old_logp, mask, low=.2, high=.28):
    mixed = reward.max(1).values > reward.min(1).values
    if not mixed.any():
        raise ValueError("batch contains no mixed group")
    reward, new_logp, old_logp, mask = (
        tensor[mixed] for tensor in (reward, new_logp, old_logp, mask)
    )
    centered = reward - reward.mean(1, keepdim=True)
    std = reward.std(1, keepdim=True, unbiased=False)
    advantage = torch.where(std > 1e-6, centered / std, torch.zeros_like(centered))
    ratio = (new_logp - old_logp).exp()
    raw = ratio * advantage[..., None]
    clipped = ratio.clamp(1 - low, 1 + high) * advantage[..., None]
    token_objective = torch.minimum(raw, clipped)
    return -(token_objective * mask).sum() / mask.sum(), mixed, advantage
reward = torch.tensor([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
old = torch.zeros(4, 2, 3)
ratio = torch.tensor([1., 1.3, 1.])
new = old + ratio.log()
mask = torch.tensor([[[1., 1., 0.], [1., 0., 0.]]] * 4)
loss, mixed, advantage = dapo_batch(reward, new, old, mask)
assert mixed.tolist() == [False, True, False, True]
torch.testing.assert_close(advantage.mean(1), torch.zeros(2))
assert torch.isfinite(loss)
try:
    dapo_batch(torch.ones(2, 2), new[:2], old[:2], mask[:2])
    raise AssertionError("uniform groups must not form an empty learner batch")
except ValueError:
    pass
```

若把最后一行 reduction 改成 per-response mean，就已经改变了 DAPO 的长度权重。对应的更多断言见[手撕 LLM 策略优化](../../practice/llm-policy-optimization.md)。

## 去掉 reference KL 的边界

DAPO 在其长 CoT 数学 RL 设置中移除 direct reference KL，理由是目标允许 policy 明显偏离初始化模型。这个选择不能直接推广到开放式助手、安全对齐或容易 reward hacking 的任务：

- verifier 只覆盖答案正确性时，KL 可能仍保护语言质量与通用能力；
- reference 太强会压制探索，太弱又可能让策略离开 reward 的可信区域；
- 不使用 KL 不等于没有 trust-region 约束，PPO-style clip 仍存在；
- clip 也不等于 reference regularization，二者约束的是不同对象。

old、behavior 与 reference 的区别见[策略身份与概率契约](../../reinforcement-learning/training-inference-discrepancy.md)。

## 实验最窄支持到哪里

论文以 Qwen2.5-32B base、DAPO-Math-17K、每题 16 个 rollout 和 AIME 2024 avg@32 为核心设置，报告 DAPO 达到 50 分，并以约一半 update steps 超过其引用的 DeepSeek-R1-Zero-Qwen-32B 结果。这个证据支持：

- 四项组件在该长 CoT 数学设置中共同形成有效、可复现的训练配方；
- Clip-Higher、dynamic sampling、token-level loss 与 overlong handling 分别对应可观测失败模式；
- 开放代码、数据和训练细节显著扩大了可审计范围。

它不支持：

- DAPO 在任意模型规模、reward 类型或 agent 环境中都优于 PPO/GRPO；
- 50 分可单独归因于某个组件；
- global token mean 在所有目标上都没有长度偏置；
- dynamic sampling 不增加总采样成本；
- 移除 KL 对通用对齐任务普遍安全；
- AIME 改善自动等于更忠实或更可靠的推理。

## 开源边界

[DAPO 官方仓库](https://github.com/BytedTsinghua-SIA/DAPO)公开训练代码、数据、模型与 recipe，并基于 [verl](https://github.com/volcengine/verl)构建。这使研究者能够核对远多于一条 loss：

```text
data transformation
reward/parser
rollout group
dynamic sampling buffer
loss reduction
length handling
distributed trainer
```

仍应固定具体 commit、依赖版本、硬件与配置。当前框架后来加入的算法、默认参数或 bug fix，不应反向写成 2025 年论文原配方。

## 它之后为何还有 VAPO

DAPO 选择无 learned critic 的 group-relative 路线。对每题可以并行采多条、reward 是终局 verifier 的数学任务，这个交换很有吸引力；但超长或多轮 agent trajectory 会让 group barrier、全错组和跨步骤信用重新变得突出。

[VAPO](vapo.md)没有沿着“进一步去掉 value”的方向前进，而是重新设计 value-based PPO：预热 critic、拆分 actor/critic 的 GAE、让 policy $\lambda$ 随长度变化，再吸收 Clip-Higher、token-level loss 与 group sampling。这正说明方法谱系不是排行榜：当任务的物理约束变化，曾经被省掉的 critic 可能重新变得有价值。

## Reference {#reference}

- Yu et al., [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- ByteDance Seed and Tsinghua SIA, [DAPO Official Implementation, Data and Models](https://github.com/BytedTsinghua-SIA/DAPO)
- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- Liu et al., [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
