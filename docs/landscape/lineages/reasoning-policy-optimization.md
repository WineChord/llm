# 推理策略优化：从 PPO 到异步 Agentic RL

大模型推理 RL 的演进不是一串 optimizer 缩写。每一轮新方法都在移动同一组物理约束：终局 reward 很稀疏，response 越来越长，critic 难以追踪，组采样会等待最慢候选，推理与训练引擎又未必实现同一个概率分布。

要看清这条历史，先把一个 trainer 拆成六个可以独立变化的对象：

```text
feedback
  -> sampling
  -> advantage / credit
  -> policy-update geometry
  -> loss reduction
  -> rollout-learning system
```

GRPO 主要改变 advantage；DAPO 同时改变 gate、sampling、reduction 与 reward；VAPO 重新设计 critic 和 GAE；GSPO、SAPO、CISPO 改变 ratio/gate；TIS、IcePop、DIS 则处理 rollout distribution 与训练分布的裂缝。方法名字只有放回这些坐标后才有历史意义。

## 2015–2017：先解决“怎样迈一小步”

[GAE](https://arxiv.org/abs/1506.02438)把多步 TD residual 指数加权，给 actor–critic 一条可调的 bias–variance 轴：

$$
\widehat A_t
=
\sum_{\ell\ge0}
(\gamma\lambda)^\ell\delta_{t+\ell}.
$$

[TRPO](https://proceedings.mlr.press/v37/schulman15.html)则从 performance-difference 与 KL 邻域出发，限制新策略离 old policy 太远。[PPO](https://arxiv.org/abs/1707.06347)用 clipped first-order surrogate 换取 minibatch SGD 和更简单的分布式实现。

这三项工作形成后来 LLM RLHF 的经典骨架：

```text
rollout under behavior policy
  -> freeze / recompute old training log-prob
  -> learned reward / environment return
  -> critic + GAE
  -> clipped policy update
  -> optional reference KL
```

其稳定性来自一组联合契约，而不是 clip 一项：old log-prob 必须冻结，critic target 要正确处理 terminal/truncation，同一 rollout 的复用 epoch 不能无限增加，KL、ratio tail 与 value drift 都要观察。完整机制分别见 [Advantage 与 GAE](../../reinforcement-learning/advantage-estimation-gae.md)、[Trust Region 与 TRPO](../../reinforcement-learning/trust-region.md)和[PPO](../../reinforcement-learning/trust-region-ppo.md)。

## 2017–2022：PPO 进入语言模型闭环

人类偏好研究把 reward 从环境函数变成 learned model。[InstructGPT](../works/instructgpt.md)把 demonstrations、pairwise reward model、PPO 与 reference/pretraining constraint 组织成三阶段流程。

语言模型把经典 PPO 推到一个特殊场景：

- 一次“动作”可以按 token、response 或 turn 定义；
- reward 常在 sequence 末端一次给出；
- action space 是巨大词表，但只对 sampled token 计算 surrogate；
- rollout 远比普通反向传播昂贵；
- current、old training、rollout behavior 与 reference policy 同时存在。

这时，value model 的必要性与成本都更加突出。它能把终局 reward 分解为 prefix advantage，却也需要另一份大模型参数或 head、额外优化器状态和不断追踪 current policy 的数据。

## 2024：GRPO 把 baseline 搬到同题候选之间

[DeepSeekMath](https://arxiv.org/abs/2402.03300)提出 GRPO：对同一个数学 prompt 采样 $G$ 个 response，用组均值和标准差构造优势，省去同规模 critic。

$$
\widehat A_i
=
\frac{R_i-\bar R}{\operatorname{std}(R)}.
$$

这个选择把困难从 value learning 转移到 group sampling：

```text
不再训练 critic
  -> 每个 prompt 需要多个 rollout
  -> 必须等待同组候选
  -> 全同 reward 组没有相对信号
  -> group std 与 response length 进入权重
```

与此同时，[Back to Basics](https://arxiv.org/abs/2402.14740)重新评估 REINFORCE、RLOO 与 PPO，提醒研究者：简单的 leave-one-out baseline 在某些 sequence-reward RLHF 设置中已经很强。无 critic 并不是 GRPO 的专属属性，组标准化也不是唯一选择。

GRPO 随 [DeepSeek-R1](../works/deepseek-r1.md)进入更大的公众视野，但 R1 的能力来自规则 reward、cold start、在线 RL、rejection sampling、二次 SFT/RL 与 distillation 的多阶段组合。把这条路线缩写为“GRPO 产生 reasoning”，会把数据与训练闭环全部抹掉。

## 2025 年初：先审计看似无害的分母

[Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783)提出 Dr. GRPO，指出原 GRPO 中两种权重：

1. group std 使不同 prompt 按 reward dispersion 重权；
2. response token mean 使每个 token 的权重与 $1/|y_i|$ 成正比。

Dr. GRPO 去掉 group std，并用固定生成预算作归一化。这项工作推动了一个重要认识：LLM policy loss 的分母不是工程细节，而是经验目标的一部分。

同一 per-token surrogate 可以按三种方式归约：

$$
\frac1G\sum_i\frac1{|y_i|}\sum_tj_{i,t},
$$

$$
\frac{\sum_{i,t}m_{i,t}j_{i,t}}{\sum_{i,t}m_{i,t}},
$$

$$
\frac1{G L_{\max}}\sum_{i,t}m_{i,t}j_{i,t}.
$$

它们分别让 response、有效 token 或固定预算成为权重单位。训练结果若不同，不能只归因于 optimizer 名称。

## 2025 年春：DAPO 把失败模式变成开放配方

[DAPO](../works/dapo.md)从 naive GRPO 的训练失败出发，引入四项联合改动：

- asymmetric Clip-Higher；
- 只保留 mixed group 的 dynamic sampling；
- global token-level loss；
- overlong filtering 与 soft punishment。

它的历史意义有两层。第一，方法不再只发布一个漂亮的 objective，而是公开数据、代码、模型和系统 recipe；第二，四项改动明确落在不同抽象层：

| 失败 | 改动 |
| --- | --- |
| 低概率正 token 很快上裁剪 | Clip-Higher |
| 全对/全错组稀释有效 batch | Dynamic Sampling |
| 长 response 的每 token 权重过小 | Global Token Loss |
| generation limit 制造 noisy negative | Overlong Handling |

DAPO 的成功不能证明 dynamic sampling 本身是新 optimizer，也不能证明 global token mean 没有长度倾向。它说明的是：长 CoT RL 的稳定性必须把算法、采样、reward 与 reduction 一起看。

## 2025 年春：VAPO 没有继续“去 critic”

[VAPO](../works/vapo.md)选择了另一条分支。它观察到 value-based PPO 在长 CoT 中的三个问题：

1. reward-model initialization 对 prefix value 有偏；
2. 固定 $\lambda$ 让终局 reward 在长序列中过度衰减；
3. 稀疏 verifier reward 使正确 response 极其珍贵。

因此 VAPO 先 warm up value model，让 critic target 使用 $\lambda_{\mathrm{critic}}=1$，让 actor 使用

$$
\lambda_{\mathrm{policy}}(l)
=
1-\frac1{\alpha l},
$$

再组合 Clip-Higher、global token loss、positive-example NLL 与 group sampling。

这条分支很有历史感：GRPO 曾以组统计替代 critic，VAPO 又在更长信用路径上把 critic 带回来。不是后一种“退回旧算法”，而是任务约束改变了成本比较：

```text
同题多候选便宜、终局比较充分
  -> group baseline 有吸引力

单轨迹昂贵、跨步骤信用关键
  -> learned value 重新有吸引力
```

## 2025 年中后：ratio 的粒度成为主战场

当 advantage 与 sampling recipe 越来越成熟，研究焦点转向 sampled probability ratio 本身。

### CISPO：clip weight，不把梯度清零

[MiniMax-M1](https://arxiv.org/abs/2506.13585)中的 CISPO 将 clipped importance ratio detach 后乘在 $\log\pi$ 前。越界 token 的权重饱和，但梯度不归零。它与 PPO 的区别不在 clip 数值，而在 **clip 作用于 surrogate 还是 detached gradient coefficient**。

### GSPO：整条 response 共用 ratio

[GSPO](https://arxiv.org/abs/2507.18071)使用

$$
s_i
=
\exp\left(
\frac1{|y_i|}\sum_t\log\rho_{i,t}
\right),
$$

让整条 response 共用一个 sequence-level clip decision。它避免 token 间 update coherence 被打散，却可能因少数异常 token 让整条 response 饱和。$s_i$ 是 trajectory ratio 的长度归一化几何平均，不是普通 trajectory IS weight。

### SAPO：从硬门变成平滑门

[SAPO](https://arxiv.org/abs/2511.20347)用 sigmoid surrogate 让远离 $\rho=1$ 的梯度平滑衰减，并为正负 advantage 采用不同温度。它不在阈值处突然归零，却仍要面对 engine mismatch、policy lag 和 reduction 选择。

这些方法的统一公式、梯度与边界见 [Ratio、Clipping 与 Gate](../../reinforcement-learning/ratio-clipping-gating.md)。

## 2025：训练引擎与推理引擎不再被假定相同

大规模 trainer 常用 vLLM/SGLang 一类推理引擎 rollout，用 FSDP/Megatron 一类训练栈反向传播。即使权重相同，量化、kernel、并行归约、MoE routing 与 sampling processor 都可能让两边 token probability 不同。

于是 PPO ratio

$$
\rho
=
\frac{\pi_\theta^{\mathrm{train}}}
{\pi_{\mathrm{old}}^{\mathrm{train}}}
$$

不再足以描述真实 behavior。还需要

$$
\kappa
=
\frac{\pi_{\mathrm{old}}^{\mathrm{train}}}
{\mu_{\mathrm{old}}^{\mathrm{rollout}}}.
$$

[On the Rollout-Training Mismatch](https://www.opt-ml.org/papers/2025/paper116.pdf)用 TIS 截断 $\kappa$；[Every Step Evolves](https://arxiv.org/abs/2510.18855)中的 IcePop 在可信区间内校正、区间外 mask；[R3](https://arxiv.org/abs/2510.11370)则为 MoE 记录并重放 rollout routing。

它们修的是 engine distribution，不是 PPO trust region。把 TIS cap 当作另一个 PPO epsilon，会混合两条完全不同的分布缝隙。

## 2026：长程 Agent 把时间与空间瓶颈推到前台

单轮 reasoning 的 response 再长，通常仍是一个 prompt 后的自回归生成。Agentic RL 增加：

- 环境 observation 与工具结果；
- 数百轮真实状态转移；
- 极端重尾的 episode 时长；
- rollout 与 learner 的异步队列；
- 超出 context window 的历史。

[SAO 与 CompactionRL](../works/sao-compactionrl.md)分别沿时间、空间两条轴处理这些约束。

SAO 让每个 prompt 只采一条 trajectory，完成后立即进入异步 learner；critic 提供 token/step advantage，DIS 直接比较 current training policy 与 rollout behavior，并用双侧 gate 拒绝 mismatch tail。它用 critic 与分布治理换取不再等待同组候选。

CompactionRL 让 policy 学会生成摘要，并把摘要视作会影响后续状态的 action；训练还要修复 segment 切分导致的 token normalization 与跨段信用。

这时，“PPO 还是 GRPO”已不是完整问题。还要问：

```text
谁产生 behavior probability
哪类 token 是 action
一个 GAE step 对应 token、turn 还是 segment
旧轨迹能滞后多少
context compaction 是否改变环境状态
```

## 一张正交坐标图

这些方法更像从多个接口装配出的配方，而不是一棵后者取代前者的版本树：

```text
feedback      human / AI / reward model / verifier
sampling      single rollout / prompt group / dynamic or filtered sampling
advantage     return / learned V + GAE / RLOO / group statistics
update        PPO clip / Clip-Higher / CISPO / GSPO / SAPO / DIS
reduction     response mean / global token mean / fixed budget
distribution  on-policy / policy lag / engine mismatch
system        synchronous barrier / asynchronous actor–learner
```

因此 [PPO](../../reinforcement-learning/trust-region-ppo.md)常与 learned critic 和 GAE 组合，却不在定义上绑定某个 advantage estimator；[GRPO](../../reinforcement-learning/grpo.md)选择 group-relative advantage，通常仍使用 PPO-style update；[DAPO](../works/dapo.md)同时改 sampling、gate、reduction 与 overlong handling；[VAPO](../works/vapo.md)用 critic/GAE 构造 advantage，又吸收 DAPO 的 Clip-Higher 与 token loss。[CISPO、GSPO 与 SAPO](../../reinforcement-learning/ratio-clipping-gating.md)主要改变 update geometry，论文中常配 group advantage，但不是 GRPO 的算法子孙。

## 怎样读下一篇新论文

面对一个新缩写，先填一张空表：

| 轴 | 必须找到的答案 |
| --- | --- |
| Feedback | reward 来自谁，验证了什么 |
| Sampling | 每 prompt 几条，如何筛选，花了多少 token |
| Advantage | critic、return、RLOO、group std 还是别的 estimator |
| Ratio | current/old、train/rollout 还是 current/behavior |
| Gate | surrogate、detached coefficient、smooth gate 还是 hard mask |
| Reduction | token、response、prompt、trajectory 或固定预算 |
| System | 同步、流水还是异步，staleness 如何控制 |
| Evidence | 哪个模型、数据、budget、baseline 与独立评测 |

若一篇工作只改其中一格，就不应把整个 trainer 收益归给它；若同时改多格，也不能把它压成单一公式。对应的横向选择入口见[推理 RL 配方地图](../../reinforcement-learning/reasoning-rl-recipes.md)，可运行张量语义见[手撕 LLM 策略优化](../../practice/llm-policy-optimization.md)。

## Reference {#reference}

- Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- Schulman et al., [Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- Liu et al., [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)
- Yu et al., [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- Yue et al., [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)
- Zheng et al., [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)
- Gao et al., [Soft Adaptive Policy Optimization](https://arxiv.org/abs/2511.20347)
- Qwen Team, [Stabilizing Reinforcement Learning with LLMs: Formulation and Practices](https://arxiv.org/abs/2512.01374)
- Hou et al., [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508)
