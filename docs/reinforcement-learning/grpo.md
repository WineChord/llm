# GRPO：组相对优势、PPO 更新与长度权重

GRPO（Group Relative Policy Optimization）的核心不是“不要 critic”这五个字，而是一套具体交换：对同一 prompt 采样一组 response，用组内 reward 统计构造相对优势，再沿用 PPO-style token ratio 更新 policy。它省去 learned value model，却要求多样化的组采样，并把题目难度、组大小、reward 离散度和 response 长度带进 estimator。

一句话先定位：

```text
GRPO = group-relative advantage estimator
     + PPO-style clipped policy objective
     + 明确的 token/response reduction
     + 可选的 reference KL
```

它不是一种 reward 来源，也不等于 RLVR。原始 [DeepSeekMath](https://arxiv.org/abs/2402.03300)同时讨论 outcome reward model 与 process reward model；后续 reasoning RL 常用 verifier，只说明实验的[反馈制度](feedback-regimes.md)发生了变化。

## 为什么会想去掉 critic

经典 PPO 通常让 critic 估计

$$
V^\pi(h_t)
=
\mathbb E[R\mid h_t],
$$

再用 [GAE](advantage-estimation-gae.md)构造 token 或 step advantage。对单轮、终局 reward 的语言任务，这条路线有三个成本：

1. value model 需要额外参数、显存、前向与优化器状态；
2. response 很长而 reward 只在结尾出现时，早期 prefix 的 value target 很难学习；
3. actor 不断变化，critic 还要追踪非平稳 return distribution。

若同一 prompt 可以便宜地生成 $G$ 个候选，就可以把“这个回答比该题其他回答好多少”作为 baseline。这样不需要跨 prompt 预测绝对 value，却把成本转移为 $G$ 倍 rollout 和 group barrier。

## 原始 outcome GRPO

在原始理想化记号下，对 prompt $x$，冻结的 old policy 采样

$$
y_1,\ldots,y_G
\sim
\pi_{\mathrm{old}}(\cdot\mid x),
$$

得到 sequence reward $R_1,\ldots,R_G$。这里默认 rollout behavior 与 old training policy 一致；当推理引擎、sampling processor 或异步队列破坏该等式时，需要额外记录真实 behavior 并处理分布校正。原始 outcome-supervision 形式构造

$$
\bar R
=
\frac1G\sum_{j=1}^G R_j,
$$

$$
\widehat A_i
=
\frac{
R_i-\bar R
}{
\operatorname{std}(R_1,\ldots,R_G)
}.
$$

同一 response 的 action token 共用 $\widehat A_i$。token ratio 为

$$
\rho_{i,t}(\theta)
=
\frac{
\pi_\theta(y_{i,t}\mid x,y_{i,<t})
}{
\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t})
}.
$$

省略 KL 时，原始论文的 response-mean 目标可写成

$$
J_{\mathrm{GRPO}}
=
\mathbb E
\left[
\frac1G
\sum_{i=1}^G
\frac1{|y_i|}
\sum_{t=1}^{|y_i|}
\min\left(
\rho_{i,t}\widehat A_i,\,
\operatorname{clip}
(\rho_{i,t},1-\epsilon,1+\epsilon)\widehat A_i
\right)
\right].
$$

因此 GRPO 仍继承 PPO clipping 的符号相关几何；“group relative”改变的是 advantage，“response mean”改变的是归约权重。两者应分开分析。

## Standard deviation 必须 pin 约定

`std` 至少有三项实现选择：

| 选择 | 公式 | 小组时的影响 |
| --- | --- | --- |
| population std | $\sqrt{\frac1G\sum_i(R_i-\bar R)^2}$ | DeepSeekMath 公式常按此理解 |
| sample std | $\sqrt{\frac1{G-1}\sum_i(R_i-\bar R)^2}$ | 固定 $G$ 时差一个常数，小 $G$ 更明显 |
| zero-variance fallback | 除 $\sigma+\varepsilon$ 或直接置零 | 决定全同 reward 组是否产生数值噪声 |

以 binary reward

$$
R=[1,1,0,0]
$$

为例，$\bar R=0.5$。population std 为 $0.5$，advantage 是

$$
[1,1,-1,-1].
$$

sample std 为 $\sqrt{1/3}\approx .577$，advantage 变成约

$$
[.866,.866,-.866,-.866].
$$

固定 group size 时二者只差全局常数，可被 learning rate 部分吸收；若 group size、缺失 reward 或采样制度变化，这个结论不再自动成立。页面、配置与 checkpoint metadata 都应记录精确定义。

## Group mean 与 RLOO 的精确关系

RLOO 使用不包含当前样本的 baseline：

$$
\widehat A_i^{\mathrm{RLOO}}
=
R_i-\frac1{G-1}\sum_{j\ne i}R_j.
$$

GRPO 的 self-centered numerator 满足

$$
R_i-\bar R
=
\frac{G-1}{G}
\left(
R_i-\frac1{G-1}\sum_{j\ne i}R_j
\right).
$$

不除 group std 时，self-including mean 只让 RLOO advantage 乘上固定的 $(G-1)/G$。这是一个很重要的细节：不能仅因 baseline 包含当前 reward，就断言梯度方向完全错误。

但 RLOO baseline 条件于其余 rollout，不依赖当前采样 action；它保留经典 action-independent baseline 的无偏语义。GRPO 再除一个依赖整组 reward 的随机标准差后，缩放会随当前 action 与整组难度变化，已不再只是全局常数。

以 $R=[1,1,0,0]$ 的第一个样本为例：

$$
A_1^{\mathrm{RLOO}}
=
1-\frac{1+0+0}{3}
=\frac23,
$$

$$
R_1-\bar R
=\frac12
=\frac34 A_1^{\mathrm{RLOO}}.
$$

固定缩放与随机 std 重权，是两件不同的事。更完整的 baseline theorem 见[无 learned critic 的策略梯度](critic-free-baselines.md)。

## Group std 怎样重权 prompt {#group-std}

对 prompt $x$，GRPO 的有效尺度近似与

$$
\frac1{\operatorname{std}(R\mid x)}
$$

成正比。于是 reward dispersion 小但非零的组可能被放大，dispersion 大的组被缩小。这不是简单的“让不同题量纲一致”，而是在改变 prompt 对 batch gradient 的权重。

binary verifier 下：

- 全对组：centered reward 全零；
- 全错组：centered reward 全零；
- mixed group：才有相对正确性信号；
- 只有一个异类回答的组：其幅度强烈依赖 $G$ 与 std 约定。

若实现用 `(R - mean) / (std + eps)`，全同组的分子理论上仍为零；浮点误差或不同 worker 的归约顺序却可能产生微小非零量。更稳妥的契约是：`std <= threshold` 时优势显式置零。

去掉 group std 的 [Dr. GRPO](https://arxiv.org/abs/2503.20783)正是针对这种 question-level difficulty weighting，而不是简单“少做一次归一化”。

## Response mean 引入的长度权重

原始 GRPO 先对每条 response 的 action token 求平均：

$$
J
\propto
\frac1G\sum_i
\frac1{|y_i|}
\sum_t j_{i,t}.
$$

它让每条 response 近似等权，却使同一 response 内每个 token 的权重为 $1/|y_i|$。从公式可见：

- 正 advantage 的短 response，每 token 获得更大正更新；
- 负 advantage 的长 response，每个错误 token 受到更小惩罚；
- response 变长会改变优化权重，即使 reward 与 ratio 不变。

这不等于“GRPO 一定鼓励变长”或“一定鼓励变短”。最终方向取决于正负 reward 与长度的联合分布。但它说明 response length 不是纯粹的输出统计量，而已进入目标函数。

几种后续 reduction：

### 原始 GRPO：每条 response 等权

$$
\frac1G\sum_i\frac1{|y_i|}\sum_tj_{i,t}.
$$

### DAPO：有效 action token 等权

$$
\frac{
\sum_{i,t}m_{i,t}j_{i,t}
}{
\sum_{i,t}m_{i,t}
}.
$$

### Dr. GRPO：固定生成预算作分母

$$
\frac1{G L_{\max}}
\sum_{i,t}m_{i,t}j_{i,t}.
$$

global token mean 会让长 response 贡献更多 token；固定预算则让未生成位置不改变已有 token 的标度。三者对应不同的经验目标，不能只改一个 trainer 配置名便称为等价实现。

## Reference KL 是另一条轴

DeepSeekMath 的 GRPO 目标还包含相对 reference policy 的 direct KL penalty。常用单动作 estimator 写成

$$
\widehat D_{\mathrm{KL}}
=
\frac{\pi_{\mathrm{ref}}}{\pi_\theta}
-\log\frac{\pi_{\mathrm{ref}}}{\pi_\theta}
-1.
$$

它只有在动作按当前 $\pi_\theta$ 采样时，才对 $D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})$ 无偏。PPO/GRPO 多 epoch 复用的动作通常来自 $\pi_{\mathrm{old}}$ 或真实 behavior $\mu$；直接平均此式一般只是 plug-in/direct KL surrogate，除非再做相应分布校正。

这里至少涉及三种 policy：

- $\pi_\theta$：current learner；
- $\pi_{\mathrm{old}}$：rollout/update ratio 的冻结基准；
- $\pi_{\mathrm{ref}}$：行为锚点。

大规模 rollout 若由另一引擎或陈旧 checkpoint 产生，还要加入真实 behavior $\mu$。old 与 reference 的 log-prob 不能共用字段；完整身份见[训推分布与策略滞后](training-inference-discrepancy.md)。

后续 DAPO 在其 reasoning RL 设置中移除了 reference KL。这是 recipe 与任务假设的选择，不是“GRPO 定义上不需要 KL”。

## Process supervision 版本

原始 DeepSeekMath 还讨论 step-level process rewards。设 response $i$ 的第 $k$ 个 reasoning step 在 token `index(k)` 结束，标准化后的 process reward 为 $\widetilde r_i^{\mathrm{index}(k)}$，则 token advantage 写成

$$
\widehat A_{i,t}
=
\sum_{\mathrm{index}(k)\ge t}
\widetilde r_i^{\mathrm{index}(k)}.
$$

这与把终局 outcome advantage 广播给所有 token 不同：后续 step reward 只回传到它之前的 token。实现必须明确：

- reasoning step 怎样切分；
- process reward 是跨整组、同位置还是同题归一化；
- 一个 step reward 属于结束 token 还是整个 span；
- step 数不同的 response 怎样比较；
- process reward 是否改变了原任务目标。

过程反馈可以缩短信用路径，也可能让模型优化评分器局部模式。其证据与设计边界见[Verifier 与奖励塑形](verifiers-reward-shaping.md)。

## Dynamic sampling 修复了什么 {#dynamic-sampling}

全对或全错组没有 group-relative 信号。DAPO 的 dynamic sampling 持续生成 prompt groups，只把

$$
0
<
\#\{\text{correct responses}\}
<
G
$$

的 mixed group 放入 learner batch，直到有效 batch 填满。

它修复的是 **有效 learner batch 越来越空** 的问题，而不是从全同 reward 中恢复不存在的相对排序。交换来的代价包括：

1. rollout 数量变成随机变量；
2. 容易和极难 prompt 的保留率下降；
3. learner 看到的 prompt distribution 偏向当前策略成功率居中的区域；
4. 同步系统中长尾 rollout 可能掩盖额外采样成本，异步系统中未必如此。

因此必须同时报告 accepted group、rejected group、总生成 token、原始 prompt 难度和 retained distribution。只按 learner step 比较，会把采样预算变化误当作算法增益。

## GRPO 何时合适

较合适的条件是：

- reward 能稳定比较同一 prompt 的不同 response；
- 同题多候选可以并行生成；
- 终局 reward 已足以指导任务；
- learned critic 的成本或误差明显高于 group sampling 成本；
- group barrier 与 rollout 长尾仍可接受。

需要谨慎的条件是：

- $G=1$ 或多候选代价极高；
- 环境一步会改变后续状态，多条 response 不再共享同一个可比较初态；
- reward 极度稀疏，长期全错；
- 多轮任务需要跨 turn/segment bootstrap；
- rollout 长度重尾使最慢候选主导同步时间；
- 异步 learner 让同组候选由明显不同的 behavior policy 产生；
- verifier 的噪声或漏洞比 critic bias 更严重。

“critic-free”不等于“没有 baseline”，也不等于“不做信用分配”。GRPO 用组统计作 baseline，并把 sequence reward 广播或经 process reward 回传；它只是没有 learned $V(h_t)$。

## 一条诊断路径

训练时至少记录：

1. 每个 prompt 的 group mean、std、正确数和缺失 reward；
2. all-correct、all-wrong、mixed group 比例；
3. population/sample std 约定与 zero-variance 数量；
4. raw centered reward 与 std-scaled advantage；
5. response length 和 advantage 符号的联合分布；
6. prompt、response、token 三种 denominator；
7. PPO ratio、KL、clip fraction 与 entropy；
8. 生成但被 dynamic sampling 丢弃的 token；
9. 不同难度、领域、语言和长度 slice 的有效权重；
10. 相同总生成 token 与 wall-clock 下的 RLOO、GRPO、PPO 对照。

最小实现与退化测试见[手撕 LLM 策略优化](../practice/llm-policy-optimization.md#rloo-grpo)。后续方法如何分别修改 estimator、reduction、sampling 和 gate，见[推理 RL 配方地图](reasoning-rl-recipes.md)。

## 历史位置

[DeepSeekMath](https://arxiv.org/abs/2402.03300)在数学语言模型训练中系统描述 GRPO，以 group-relative estimation 省去同规模 critic；[DeepSeek-R1](../landscape/works/deepseek-r1.md)进一步把 group-relative RL 放入可验证 reward、cold start、rejection sampling 与蒸馏的多阶段闭环。

随后工作没有形成一条“后者彻底取代前者”的单线：

- [RLOO](https://arxiv.org/abs/2402.14740)重新强调简单 leave-one-out baseline；
- [Dr. GRPO](https://arxiv.org/abs/2503.20783)分离 group std 与 response-length weighting；
- [DAPO](../landscape/works/dapo.md)组合 asymmetric clip、dynamic sampling、token reduction 与长度处理；
- [VAPO](../landscape/works/vapo.md)回到 value-based PPO，集中处理长 reasoning 的 critic 与 GAE；
- GSPO、SAPO、CISPO 从 ratio 粒度和 gate 形状继续修改更新几何，见[Ratio、Clipping 与 Gate](ratio-clipping-gating.md)。

真正延续的主线不是缩写，而是四个问题：baseline 从哪里来、哪些 rollout 有信息、长度怎样进入分母、分布偏移怎样被约束。

## Reference {#reference}

- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- Ahmadian et al., [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740)
- Liu et al., [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
- Yu et al., [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- Guo et al., [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
