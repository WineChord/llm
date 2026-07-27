# 语言模型作为策略

把语言模型称为 policy 很容易；真正困难的是确定 state、action、transition 和 reward 分别落在哪个粒度。粒度一旦变化，log-probability、importance ratio、credit assignment 与 batch normalization 都会变化。

## 从自回归分布到策略

给定输入 $x$ 与已生成前缀 $y_{<t}$，语言模型定义

$$
\pi_\theta(y_t\mid x,y_{<t}).
$$

完整 response 的概率为

$$
\pi_\theta(y\mid x)
=\prod_{t=1}^{L}\pi_\theta(y_t\mid x,y_{<t}),
$$

因此

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{L}\log\pi_\theta(y_t\mid x,y_{<t}).
$$

这使已采样 response 的 score-function gradient 可拆到 token：

$$
\nabla_\theta\log\pi_\theta(y\mid x)
=\sum_{t=1}^{L}\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t}).
$$

但“梯度可拆”不代表 reward 自然属于每个 token。若只有一个终局分数 $R(x,y)$，把同一个 advantage 复制到所有 token 是一种 Monte Carlo estimator，而不是逐 token 真值。

## 四种动作尺度

| 尺度 | action | 优点 | 主要代价 |
| --- | --- | --- | --- |
| Token | 单个采样 token | log-prob 精确、mask 清楚 | reward 极稀疏，时间步很长 |
| Response | 整段回答 | 与偏好、verifier 常一致 | sequence ratio 易受长度影响 |
| Turn | 一次 assistant 消息或 tool call | 适合多轮环境 | turn 内信用仍未分解 |
| Episode | 完整任务轨迹 | 终态语义清楚 | 方差最高，难复用局部经验 |

实现经常同时使用多个尺度：environment 在 turn 级转移，policy loss 在 token 级求和，reward 在 response 或 episode 级给出，统计又按 prompt 分组。必须把每个尺度写进数据 schema。

## 状态不是字符串

单轮生成可把 $x$ 视为状态；多轮系统中的 policy 条件更接近

$$
h_t=(m_{\mathrm{system}},u_0,a_0,o_0,\ldots,u_t),
$$

其中 $o_i$ 可能是工具结果、文件状态或外部服务响应。历史字符串只是对真实状态的编码：

- 同样文本可能对应不同权限、文件或数据库状态；
- 截断与摘要会丢失信息；
- tool schema、模板和 tokenizer 版本会改变动作空间；
- 外部状态可能在两次 replay 之间变化。

因此语言 agent 通常更接近 [POMDP](decision-processes.md)。context、memory 与结构化 ledger 都是在构造可用的 belief state，而不是证明 Markov 性已经成立。

## Action mask 是算法的一部分

设序列位置 mask 为 $m_t$。policy objective 的基本 reduction 是

$$
\mathcal L
=-\frac{\sum_t m_t w_t\log\pi_\theta(y_t\mid h_t)}
{\sum_t m_t}.
$$

$m_t=1$ 应只覆盖由 behavior policy 实际采样且允许训练的动作。通常排除：

- system、user 与 prompt token；
- 外部 observation 和 tool result；
- padding；
- 从旧 context 复制进来的历史 token；
- 只作为条件输入的 summary 或 retrieval span。

summary 是否是 action 取决于它是否由当前 policy 采样并参与决策。例如 [CompactionRL](../landscape/works/sao-compactionrl.md#compactionrl)直接训练 summary token；把已有摘要复制到下一段 prompt 时，复制位置不应再次进入 loss。

## Behavior distribution 不只由 logits 决定

若 rollout 使用 temperature $T$，策略分布至少变成

$$
\mu(a\mid h)
=\operatorname{softmax}\left(\frac{z(h)}{T}\right)_a.
$$

top-$k$、top-$p$、grammar mask、repetition penalty 与 constrained decoding 还会再次重归一化。训练时 importance ratio

$$
\rho_t
=\frac{\pi_\theta(a_t\mid h_t)}
{\mu(a_t\mid h_t)}
$$

只有在分母对应实际采样分布时才有校正意义。保存 raw model logits 的 log-softmax，而 rollout 实际从截断分布采样，会把 decoder 差异误写成 policy improvement。

## Length reduction 改变优化偏好

response log-probability 的 token sum

$$
\ell_{\mathrm{sum}}(y)=\sum_t\log\pi(y_t\mid h_t)
$$

与 token mean

$$
\ell_{\mathrm{mean}}(y)=\frac1L\sum_t\log\pi(y_t\mid h_t)
$$

对一条长度固定为 $L$ 的 response，二者当然满足 $\ell_{\mathrm{mean}}=\ell_{\mathrm{sum}}/L$。问题出现在可变长度集合：每条 response 的缩放因子 $1/L_i$ 不同，因此不存在对整个 batch 通用的常数缩放，候选排序与梯度权重都可能改变。

令 $\ell_{i,t}=\log\pi(y_{i,t}\mid h_{i,t})$。对含 $N$ 条 response 的 ragged batch，全局 token mean 与 per-response mean 分别为

$$
\mathcal L_{\mathrm{token}}
=-\frac{\sum_{i=1}^{N}\sum_{t=1}^{L_i}\ell_{i,t}}
{\sum_{i=1}^{N}L_i},
\qquad
\mathcal L_{\mathrm{response}}
=-\frac1N\sum_{i=1}^{N}
\frac1{L_i}\sum_{t=1}^{L_i}\ell_{i,t}.
$$

于是：

- 全局 token mean 让每个有效 token 等权，长 response 因 token 更多而拥有更大的总权重；
- per-response mean 让每条回答拥有相同总权重，短 response 内单个 token 的系数相对更大；
- prompt mean 让每个问题等权；
- episode mean 让长短任务等权。

最后两项仍需说明 prompt 内 response 和 episode 内 segment 怎样先聚合。不存在脱离 estimand 的“正确分母”；必须先决定想让谁在总体目标中等权，再实现分布式 numerator/denominator。

## 终止语义

语言生成的 EOS 只是 policy action；环境终止还可能来自成功、失败、预算或故障：

| 状态 | value bootstrap | 是否算策略结果 |
| --- | --- | --- |
| 成功 / 明确失败 | 通常不 bootstrap | 是 |
| 时间或 token 截断 | 视任务定义，常需 bootstrap | 是，但未完成 |
| 无效格式 / 非法动作 | 按环境规则 | 是 |
| 工具或基础设施故障 | 不应自动当零 reward | 通常否 |

把 timeout 与 terminal 合并，会训练模型偏向短任务或过早结束。完整轨迹字段见[轨迹与策略契约](../agentic-rl/trajectory-contract.md)。

## 从单轮到 Agentic RL

单轮 RLHF 可以把 prompt 固定、response 视为整条 episode。Agentic RL 则让 policy 的动作改变后续观察：

$$
a_t\sim\pi_\theta(\cdot\mid h_t),
\qquad
o_{t+1}\sim\Omega(\cdot\mid s_{t+1},a_t).
$$

这时旧轨迹不只包含旧 token 分布，还包含旧策略诱导的状态访问分布。逐 token ratio 能修正动作概率的一部分差异，却不能神奇地让新策略访问旧数据中从未出现的状态。详见[Off-policy 校正](off-policy-correction.md)与[语言模型信用分配](credit-assignment.md)。

## 最小检查

1. 新旧 policy 相同时，所有有效 action ratio 为 $1$。
2. 修改 prompt、observation 或 padding 的 label，不改变 policy loss。
3. 重建 response 后的 token IDs 与 rollout 保存值逐位相同。
4. 采样 processor 改变时，behavior log-prob 同步改变。
5. 分别报告 token、response、prompt 和 episode denominator。
6. EOS、环境成功、截断和基础设施错误保持不同状态。

对应实现见[手撕强化学习](../practice/reinforcement-learning.md)和[训练目标](../practice/training-objectives.md)。

## Reference {#reference}

- Sutton et al., [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
- Ziegler et al., [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)
- Ouyang et al., [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- Espeholt et al., [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
