# 推理 RL 配方地图

选择 LLM 强化学习方法时，最危险的问题是“PPO、GRPO、DAPO、VAPO 哪个最好”。这些名字并不处在同一层：GAE 是 advantage estimator，PPO/GRPO 是更新算法，DAPO/VAPO 是多组件 recipe，TIS/DIS 是分布校正，RLVR 则描述 reward provenance。

更可靠的入口是先回答六个问题：

```text
反馈是否可靠
  -> 每个状态能采几条轨迹
  -> 信用需要传播多远
  -> 哪个分布真正产生数据
  -> loss 按什么单位归约
  -> 系统能承受同步等待还是异步偏移
```

本页负责方法选择与互相导航；完整推导分别进入 [GAE](advantage-estimation-gae.md)、[PPO](trust-region-ppo.md)、[GRPO](grpo.md)、[Ratio 与 Gate](ratio-clipping-gating.md)以及[训推分布](training-inference-discrepancy.md)。

## 第一轴：反馈决定可学的目标

先判断 reward 的含义：

| 反馈 | 例子 | 主要风险 |
| --- | --- | --- |
| 确定性 outcome verifier | 数学答案、代码测试、结构化约束 | 规格不完整、parser 错误、测试漏洞 |
| learned reward model | 帮助性、写作质量、安全偏好 | distribution shift、reward hacking |
| process reward | 步骤评分、工具执行状态 | 局部代理改变终局目标 |
| 环境 return | 游戏、浏览、软件工程任务 | 延迟、部分可观测、基础设施失败 |
| 人工或 AI 比较 | chosen/rejected response | 标注分布、judge bias、一致性 |

RLVR 可以与 PPO、RLOO、GRPO 或 value-based actor–critic 组合；RLAIF 也可以产生 pair 或 scalar reward。反馈来源与 optimizer 正交，详见[反馈制度](feedback-regimes.md)和[Verifier 与奖励塑形](verifiers-reward-shaping.md)。

若 reward 自身不可靠，换 optimizer 只会更高效地利用漏洞。应先建立 held-out verifier、人工审查、错误分类和基础设施失败隔离。

## 第二轴：采样结构决定 baseline

### 同一 prompt 多候选便宜

可以优先比较：

- REINFORCE with batch baseline；
- [RLOO](critic-free-baselines.md#rloo)；
- [GRPO](grpo.md)；
- Dr. GRPO 或 DAPO-style sampling/reduction。

组内 baseline 的吸引力在于不训练 critic；代价是每个 prompt 需要 $G\ge2$ 个 rollout，并等待同组完成。若 binary reward 长期全同，组内信号为零。

### 每个状态只能采一条或轨迹很贵

learned critic 更有吸引力：

- PPO + GAE；
- VAPO-style value warmup 与 adaptive GAE；
- SAO-style single-rollout async actor–critic。

critic 可以跨状态共享统计，并在中途 bootstrap；代价是 value bias、额外计算和 actor–critic 非平稳耦合。

### 多轮 environment state 不可复现

同一初始 prompt 的多个 trajectory 未必仍在比较相同状态：工具返回、外部网页、随机环境和其他 agent 会改变后续 observation。此时“组内 response baseline”需要重新定义在 episode、turn 还是 branching state 上，不能直接把单轮 GRPO tensor 扩成长维度。

## 第三轴：信用长度决定 estimator

终局 reward 到 action 的路径可分三类：

| 路径 | 常见选择 | 关键检查 |
| --- | --- | --- |
| 单轮 sequence，reward 广播可接受 | Monte Carlo / group advantage | 长度权重、reward variance |
| 多步轨迹，value 可学 | TD、GAE、V-trace | terminal/truncated、bootstrap |
| 跨工具与 context segment | turn/segment GAE、过程 reward | observation mask、边界、时间单位 |

GAE 使用

$$
\widehat A_t
=
\delta_t
+\gamma\lambda m_t^{\mathrm{trace}}\widehat A_{t+1}.
$$

真正 terminal 令 bootstrap/trace 都停止；有 final observation 的 truncation 仍 bootstrap，却停止 trace。token EOS、assistant turn 结束和 environment terminal 不是同一个事件。

VAPO 的 length-adaptive $\lambda$ 只在“length 与 GAE time step 相同”时有明确语义。Agent trajectory 若按 turn 递推，却用 raw token length 算 $\lambda$，需要额外论证。

## 第四轴：先确认 behavior，再选 correction

至少区分：

$$
\rho
=
\frac{\pi_\theta^{\mathrm{train}}}
{\pi_{\mathrm{old}}^{\mathrm{train}}},
\qquad
\kappa
=
\frac{\pi_{\mathrm{old}}^{\mathrm{train}}}
{\mu_{\mathrm{old}}^{\mathrm{rollout}}},
\qquad
d
=
\frac{\pi_\theta^{\mathrm{train}}}
{\mu_{\mathrm{old}}^{\mathrm{rollout}}}.
$$

- $\rho$ 接近 $1$，等价于 $|\log\rho|$ 较小：current 仍接近 old update policy；
- $\kappa$ 接近 $1$，等价于 $|\log\kappa|$ 较小：train/rollout engine 较一致；
- $d$ 同时包含两种差异；
- reference policy 只进入 KL/behavior anchor，不是上述分母的替代品。

对应方法：

| 问题 | 典型工具 |
| --- | --- |
| current–old 更新过大 | [PPO](trust-region-ppo.md) / [GRPO](grpo.md) clipping、[CISPO](ratio-clipping-gating.md#cispo)、[GSPO](ratio-clipping-gating.md#gspo)、[SAPO](ratio-clipping-gating.md#sapo) |
| train–rollout engine mismatch | [TIS](training-inference-discrepancy.md#tis)、[IcePop](training-inference-discrepancy.md#icepop)、[R3](training-inference-discrepancy.md#r3) |
| 异步 current–behavior 直接偏移 | [DIS](ratio-clipping-gating.md#dis)、[V-trace](off-policy-correction.md#v-trace) 类 correction、staleness filtering |
| context/token contract 不同 | 先修 tokenizer/template/trajectory，ratio 无法补救 |

只看 PPO clip fraction 无法诊断 engine mismatch；只加 TIS 也无法限制 learner 在同一 rollout 上做过多 epoch。

## 第五轴：Reduction 决定谁更重要

设 $j_{i,t}$ 为 per-token surrogate，$m_{i,t}$ 为 action mask。

### Response mean

$$
\frac1B
\sum_i
\frac{\sum_tm_{i,t}j_{i,t}}
{\sum_tm_{i,t}}.
$$

每条 response 近似等权；长 response 的单 token 权重更小。

### Global token mean

$$
\frac{\sum_{i,t}m_{i,t}j_{i,t}}
{\sum_{i,t}m_{i,t}}.
$$

每个 action token 等权；长 response 的总贡献更大。

### Fixed budget

$$
\frac1{B L_{\max}}
\sum_{i,t}m_{i,t}j_{i,t}.
$$

分母不随实际生成长度变化；有效 token 数变化会改变总梯度幅度。

三者都可能合理，但优化的是不同经验权重。实验必须 pin denominator，并把 reward/advantage 与 response length 联合报告。

## 第六轴：系统决定等待还是偏移

同步 group rollout 的主要成本是 barrier：

```text
同一 prompt 的 G 条 response
  -> 等待最慢一条
  -> 才能算 group statistics
  -> 才能进入 learner
```

异步 single-rollout 的主要成本是 staleness：

```text
trajectory 完成立即入队
  -> learner 不等同组
  -> rollout policy 逐渐落后
  -> 需要 behavior log-prob、gate 与选择偏差诊断
```

不存在“异步免费提高吞吐”。它把 idle time 换成 distribution shift、队列选择偏差与版本治理。长尾是否值得消除，应先 profile：

- rollout duration p50/p95/p99；
- group idle fraction；
- learner utilization；
- checkpoint age 与 direct ratio；
- accepted token fraction；
- 相同 wall-clock 下的 held-out improvement。

Agentic 细节见[训练系统](../agentic-rl/training-systems.md)和[SAO 深读](../landscape/works/sao-compactionrl.md#sao)。

## 方法不是互斥选项

一套真实 trainer 通常从各轴选一个组件。

### 经典 RLHF

```text
learned reward
  + value critic
  + GAE
  + PPO clip
  + reference KL
  + synchronous rollout batch
```

### GRPO-style RLVR

```text
binary verifier
  + G responses per prompt
  + group-normalized reward
  + PPO-style token ratio
  + response/token reduction
  + synchronous group barrier
```

### DAPO

```text
binary verifier
  + group advantage
  + Clip-Higher
  + mixed-group dynamic sampling
  + global token mean
  + overlong reward handling
```

### VAPO

```text
verifier reward
  + pretrained value critic
  + decoupled/adaptive GAE
  + Clip-Higher
  + global token mean
  + positive NLL and group sampling
```

### SAO-style Agentic RL

```text
single long trajectory
  + critic / GAE
  + asynchronous queue
  + direct current/behavior ratio
  + double-sided acceptance gate
  + staleness and route diagnostics
```

同一 recipe 中的组件可能来自不同历史分支。命名时最好写出完整组合，而不是用一个流行缩写覆盖所有配置。

## 一张选择表

| 任务条件 | 优先建立的 baseline | 下一步改动 | 不应忽略 |
| --- | --- | --- | --- |
| 单轮、终局 verifier、同题多采便宜 | RLOO / GRPO | DAPO-style sampling/reduction | 全同组、总 rollout token |
| 单轮很长、critic 明显失稳 | PPO + GAE | VAPO-style critic/λ 设计 | value calibration、长度定义 |
| 正确样本极稀少 | group sampling / curriculum | positive NLL、搜索、数据重平衡 | verifier 漏洞、prompt coverage |
| MoE 训推 route 不一致 | 同步小实验 | R3 + probability correction | router metadata、目标偏移 |
| rollout 与 learner 分离 | 记录真实 behavior | TIS/IcePop 或 direct correction | processor、support、staleness |
| 多轮 agent、group barrier 严重 | single-rollout actor–critic | SAO-style async | 选择偏差、critic cold start |
| context 超窗 | segment/turn value baseline | learned compaction | state sufficiency、跨段信用 |

“优先”表示更容易建立可解释 baseline，不表示无需实验比较。

## 按失败现象反推接口

### Reward 上升，held-out 能力下降

先查 verifier coverage、reward hacking、reference drift 与数据污染；不要先调 clip。

### Entropy 快速塌缩

查正 advantage ratio、upper clip、sampling temperature、prompt 难度和重复候选；Clip-Higher 只是可能选项。

### GRPO loss 经常为零

查 all-correct/all-wrong 比例、group size、reward parser、采样多样性；dynamic sampling 只能选择 mixed groups，不能修复错误 verifier。

### Response length 持续漂移

同时查 reward–length correlation、response/token/fixed denominator、truncation reward、EOS 概率与 positive NLL。

### PPO 初期稳定，随后突然崩

查 critic error、reuse epoch、policy staleness、train–rollout log-prob gap、ratio tail 与 optimizer step；mean KL 可能掩盖少数极端 token。

### 异步吞吐提高，样本效率下降

查 checkpoint age、direct ratio、accepted fraction、队列对短轨迹的偏好，以及每生成 token 的能力增益。

## 实验怎样公平

跨算法比较至少固定或报告：

- base checkpoint、tokenizer、chat template；
- prompt distribution 与 verifier 版本；
- sampling temperature、top-$p$、max length；
- **总生成 token**，而不仅是 learner step；
- actor、critic、reference 的计算预算；
- optimizer step、minibatch epoch 与 effective batch；
- response/token/fixed reduction；
- rejected/filtered rollout；
- train–rollout engine 与数值精度；
- wall-clock、GPU-hours 与峰值显存；
- held-out pass@1、pass@$k$、多样性与安全能力；
- 多 seed 曲线和 failure runs。

DAPO dynamic sampling、VAPO group sampling、PPO critic 与 GRPO 多候选使用不同预算形态。只固定 “训练 1000 steps” 并不公平。

## 推荐阅读顺序

1. [语言模型作为策略](language-model-policy.md)与[反馈制度](feedback-regimes.md)：先固定动作、状态与 reward 来源；
2. [Policy Gradient](policy-gradient.md)与 [Actor–Critic](actor-critic.md)：建立 sampled gradient、baseline 与 critic；
3. [Advantage 与 GAE](advantage-estimation-gae.md)：长期信用与双边界；
4. [Trust Region 与 TRPO](trust-region.md)、[PPO](trust-region-ppo.md)：局部更新与 clipped surrogate；
5. [无 critic 的 baseline](critic-free-baselines.md)：REINFORCE、ReMax、RLOO；
6. [GRPO](grpo.md)：组统计、std 与长度分母；
7. [训推分布与策略滞后](training-inference-discrepancy.md)：四策略和三种 ratio；
8. [Ratio、Clipping 与 Gate](ratio-clipping-gating.md)：CISPO、GSPO、SAPO、TIS、IcePop、DIS；
9. [DAPO](../landscape/works/dapo.md)与 [VAPO](../landscape/works/vapo.md)：两套完整 recipe；
10. [推理策略优化谱系](../landscape/lineages/reasoning-policy-optimization.md)：把方法放回历史因果链。

对应代码集中在[手撕 LLM 策略优化](../practice/llm-policy-optimization.md)，避免每个方法页维护一份逐渐漂移的 tensor 实现。

## Reference {#reference}

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- Ahmadian et al., [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740)
- Yu et al., [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- Yue et al., [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)
- Qwen Team, [Stabilizing Reinforcement Learning with LLMs: Formulation and Practices](https://arxiv.org/abs/2512.01374)
- Hou et al., [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508)
