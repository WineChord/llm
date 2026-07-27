# 搜索、过程奖励与验证

测试时搜索从同一策略产生多个候选，再用 verifier 选择或扩展；训练则把这些结果变成监督或奖励。搜索收益来自候选多样性与评分器质量的乘积，采样更多不自动产生更好推理。

## Best-of-$N$

对同一输入采样 $N$ 个候选：

$$
y^*=\arg\max_{y_i}v(x,y_i).
$$

若单次成功概率为 $p$ 且样本独立，至少一个成功的理想概率为

$$
1-(1-p)^N.
$$

真实采样高度相关，verifier 也会误排，因此这只是上界直觉。报告结果时应给出总生成 token、验证成本与等预算基线。

## Outcome 与 process verifier

### Outcome verifier

只判断最终答案、测试或目标状态。信号可靠且易自动化，但无法指出失败步骤。

### Process verifier

为中间步骤 $s_t$ 评分：

$$
v_t\approx P(\text{最终成功}\mid s_{\le t}).
$$

它可用于剪枝、排序和过程监督，却更容易学习格式、长度或局部合理性捷径。过程分数应通过“从该状态继续采样的真实成功率”校准，而不只与人工措辞一致。

## Beam 与树搜索

语言 beam search 按累计 log-prob 保留前缀，优化的是模型概率，不是任务正确性。加入 verifier 后，可组合

$$
s(h_t)
=\lambda\log\pi(h_t\mid x)
+(1-\lambda)v(h_t).
$$

$\lambda$ 控制策略先验与验证信号。若 process verifier 不可靠，过早剪枝会永久删除正确路径。

## Monte Carlo Tree Search

树节点表示中间状态，边表示 action span。常见选择分数可写为

$$
\operatorname{PUCT}(s,a)
=Q(s,a)
+cP(a\mid s)
\frac{\sqrt{N(s)}}{1+N(s,a)}.
$$

一次循环包含 selection、expansion、evaluation/rollout 和 backup。用于语言或工具任务时需要额外定义：

- action 是 token、句子、推理步还是工具调用；
- 等价文本怎样合并；
- 环境状态能否复制和回滚；
- verifier 何时调用；
- terminal、timeout 与 invalid action；
- token 与环境调用预算。

若每个自然语言前缀都成为独立节点，分支因表述差异迅速爆炸。结构化动作或语义去重能降低搜索空间，但可能误合并不同状态。

## 过程数据生成

一种路径是对中间步骤做多次 continuation，用最终 outcome 估计该步骤价值：

$$
\hat v(s_t)
=\frac{1}{K}\sum_{k=1}^{K}
\mathbf 1[\text{continuation}_k\text{ succeeds}].
$$

这比单个 judge 标签更接近可达性，但成本高，并依赖 continuation policy。若 policy 很弱，一个本可成功的状态也可能得到低分；若能读取答案或 verifier，估计又会被泄漏污染。

## 从搜索到训练

搜索产物可转为：

- 成功轨迹的 SFT；
- 同一状态下成功/失败 action 的偏好对；
- process reward model 数据；
- policy-gradient 的 outcome/process reward；
- 失败恢复和批判样本。

筛选只保留最优轨迹会缩小策略多样性。保留多条不同成功路径、代表性失败和 verifier 不确定样本，更有利于学习边界。

## Verifier 安全

强 verifier 也可能被攻击：

- 修改测试或评分脚本；
- 读取隐藏答案；
- 触发 timeout 或 parser 异常；
- 输出 NaN、超长文本或特殊编码；
- 完成代理指标而非真实目标；
- 利用缓存键或环境残留。

应隔离 verifier 权限，区分 wrong answer、invalid、timeout 与 infrastructure error，并用隐藏/多重验证器审计奖励突增。

## 何时使用哪种方法

| 条件 | 起点 |
| --- | --- |
| 终局可精确验证，生成便宜 | best-of-$N$ |
| 候选有自然分步 | beam + process verifier |
| 环境可复制、动作结构化 | MCTS |
| verifier 弱或易投机 | 少搜索，先改善评测 |
| 需要长期降低推理成本 | 将搜索结果蒸馏或用于 RL |
| 工具调用昂贵/有副作用 | 模拟器、权限隔离与严格预算 |

## 评测

同时报告：

- pass@1、pass@$N$ 与 chosen accuracy；
- verifier 的 false positive/negative；
- 每题生成与验证 token；
- 环境调用数、wall-clock 和费用；
- 候选间多样性；
- 搜索深度、宽度和截断原因；
- 等总预算的直接采样或更强模型基线；
- 搜索数据用于训练后的 held-out 回归。

策略梯度见[数学与算法](math-algorithms.md)，轨迹字段见[轨迹与策略契约](trajectory-contract.md)，奖励攻击见[评测与安全](evaluation-safety.md)。

## Reference {#reference}

- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)
