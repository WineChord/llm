# 强化学习

强化学习研究的不是“给 loss 再乘一个 reward”，而是一个策略怎样通过与环境交互，改变未来会看到的数据，并在延迟反馈下改进决策。语言模型把这件事推到了一个特殊尺度：动作空间是 token 序列，状态常由长历史近似，reward 可能来自人类、模型或可执行验证器，采样本身又消耗大量推理资源。

本章沿五层组织知识：

```text
序贯决策对象
  -> 价值估计与信用分配
  -> 策略优化与分布校正
  -> 语言模型后训练
  -> 多轮环境、系统与评测
```

## 先分清三个问题

| 问题 | 主要对象 | 典型方法 |
| --- | --- | --- |
| 怎样预测一个策略会得到什么 | return、$V^\pi$、$Q^\pi$ | Monte Carlo、TD、GAE |
| 怎样改进策略 | advantage、occupancy、trust region | REINFORCE、actor–critic、TRPO、PPO |
| 怎样定义“更好” | reward、preference、verifier、constraint | reward modeling、RLHF、RLVR |

价值估计、策略优化与 reward 设计互相依赖，却不是一件事。一个精确 verifier 不能自动解决长时信用分配；一个低方差 critic 也不能修复错误目标。

## 学习路径

### 序贯决策基础

1. [MDP、POMDP 与回报](decision-processes.md)：状态、历史、转移、终止与时间尺度。
2. [价值函数与 Bellman 递推](values-bellman.md)：预测、最优性与动态规划。
3. [Monte Carlo、TD 与控制](prediction-control.md)：采样回报、bootstrap、SARSA 与 Q-learning。
4. [多步回报、$\lambda$-return 与资格迹](multistep-traces.md)：一步 TD 与完整回报怎样连续插值。
5. [函数逼近与致命三元组](function-approximation.md)：为什么神经网络、bootstrap 与 off-policy 组合会失稳。
6. [探索与最大熵](exploration-entropy.md)：信息获取、随机策略与 entropy regularization。
7. [模型、规划与层级决策](models-planning-hierarchy.md)：Dyna、搜索、Options 与 SMDP。
8. [模仿学习与 Offline RL](offline-imitation.md)：示范、support 与保守策略改进。
9. [约束与多智能体 RL](constraints-multiagent.md)：独立约束、联合动作与群体信用。

### 策略优化

1. [Policy Gradient](policy-gradient.md)：log-derivative、reward-to-go 与 baseline。
2. [Actor–Critic](actor-critic.md)：critic 怎样降低方差，又怎样引入偏差。
3. [Advantage 估计与 GAE](advantage-estimation-gae.md)：双边界 mask、actor/critic target 与 token/turn/segment 时间轴。
4. [Trust Region 与 TRPO](trust-region.md)：performance-difference、natural gradient、Fisher 与 line search。
5. [PPO](trust-region-ppo.md)：clipped surrogate 的精确分段、policy 身份、batch lifecycle 与诊断。
6. [Off-policy 校正](off-policy-correction.md)：importance sampling、Retrace、V-trace 与 policy lag。

### 强化学习与语言模型

1. [语言模型作为策略](language-model-policy.md)：token、response、turn 与 episode 四种动作尺度。
2. [反馈制度](feedback-regimes.md)：分开 RLHF、RLAIF、RLVR、online/offline 与 on/off-policy。
3. [KL 正则化控制](kl-regularized-control.md)：reference policy、隐式 reward 与 soft policy improvement。
4. [RLHF 数据闭环](rlhf-pipeline.md)：示范、偏好、reward model、在线采样与迭代数据。
5. [奖励建模](../training/reward-modeling.md)与[离线偏好优化](../training/offline-preference.md)：先定义反馈，再决定是否需要在线 RL。
6. [无 critic 的 baseline](critic-free-baselines.md)：REINFORCE、ReMax、RLOO 与组内 baseline。
7. [GRPO](grpo.md)：group std、RLOO 关系、长度分母、Dr. GRPO 与 dynamic sampling。
8. [训推分布与策略滞后](training-inference-discrepancy.md)：current、old、behavior、reference 与三种 ratio。
9. [Ratio、Clipping 与 Gate](ratio-clipping-gating.md)：PPO、Clip-Higher、CISPO、GSPO、SAPO、TIS、IcePop 与 DIS。
10. [在线 RL 与可验证奖励](../training/online-rl.md)：rollout、策略更新与异步数据闭环。
11. [RLVR](rlvr.md)与 [Verifier、过程奖励](verifiers-reward-shaping.md)：可验证结果如何成为训练信号。
12. [推理后训练](../training/reasoning-posttraining.md)：搜索、验证、蒸馏与参数更新的闭环。
13. [语言模型信用分配](credit-assignment.md)：sequence reward 怎样落到 token、turn 与 segment。
14. [推理 RL 配方地图](reasoning-rl-recipes.md)：按反馈、采样、信用、分布、归约和系统六轴定位问题。
15. [实验诊断](evaluation-debugging.md)：区分 reward 上升、真实能力、分布漂移和系统故障。

四条方法深读把这条路径放回具体工作：

- [DeepSeek-R1](../landscape/works/deepseek-r1.md)：可验证 reward、cold start、在线 RL 与蒸馏闭环；
- [DAPO](../landscape/works/dapo.md)：Clip-Higher、dynamic sampling、token loss 与 overlong handling；
- [VAPO](../landscape/works/vapo.md)：value warmup、decoupled/adaptive GAE 与稀疏正样本；
- [SAO 与 CompactionRL](../landscape/works/sao-compactionrl.md)：single-rollout 异步更新与跨 segment 信用。

### Agentic RL

[Agentic RL](../agentic-rl/index.md)把单轮生成扩展为有状态环境。这里需要同时维护：

- [轨迹与策略契约](../agentic-rl/trajectory-contract.md)；
- [数据、环境与 verifier](../agentic-rl/data-environments.md)；
- [搜索与过程奖励](../agentic-rl/search-verification.md)；
- [异步训练系统](../agentic-rl/training-systems.md)；
- [长时任务与上下文压缩](../agentic-rl/long-horizon.md)；
- [评测与安全](../agentic-rl/evaluation-safety.md)。

## 一张对象地图

| 层级 | 状态 | 动作 | 反馈 | 最容易错的边界 |
| --- | --- | --- | --- | --- |
| 经典控制 | 环境状态 $s_t$ | 离散或连续动作 | 环境 reward | terminal 与 truncated |
| 单轮语言 RL | prompt 与已生成前缀 | token / response | response reward | prompt token 进入 loss |
| 推理 RL | 问题、草稿、验证状态 | 推理步骤 / 答案 | outcome / process reward | 搜索收益混入训练收益 |
| Agentic RL | 历史、工具与外部状态 | turn / tool call | 过程与终态 | observation 被当作 action |
| 异步 RL | 上述状态 + policy version | 由旧 policy 采样的动作 | 延迟 reward | behavior 与 learner 分布错配 |

## 一张方法坐标

| 方法 | 主要改变 | 没有自动解决 |
| --- | --- | --- |
| [GAE](advantage-estimation-gae.md) | advantage 的时间传播 | reward 定义、policy update |
| [PPO](trust-region-ppo.md) | current–old 的 sampled surrogate | engine mismatch、硬 KL 保证 |
| [RLOO](critic-free-baselines.md#rloo) | leave-one-out baseline | group barrier、长期 bootstrap |
| [GRPO](grpo.md) | group-normalized advantage | verifier、长度权重、全同组 |
| [DAPO](../landscape/works/dapo.md) | gate、采样、归约与长度 recipe | 通用最优性、免费采样 |
| [VAPO](../landscape/works/vapo.md) | critic、GAE 与稀疏正样本 recipe | 多轮状态定义、异步偏移 |
| [TIS](training-inference-discrepancy.md#tis) / [IcePop](training-inference-discrepancy.md#icepop) | train–rollout engine correction | current–old update 过大 |
| [DIS](ratio-clipping-gating.md#dis) | current–behavior direct gate | 队列选择偏差、完整 SAO 系统 |

方法的历史因果与相互继承见[推理策略优化谱系](../landscape/lineages/reasoning-policy-optimization.md)，紧凑实现见[手撕 LLM 策略优化](../practice/llm-policy-optimization.md)。

## 怎样使用公式

每个公式都要回答四个问题：

1. 期望对哪个分布取；
2. 哪些 token、step 或 episode 进入分母；
3. 哪个策略产生数据，哪个策略正在更新；
4. 终止、截断、缺失 reward 和基础设施错误怎样处理。

如果这四项没有固定，同名的 PPO、GRPO 或 RLHF 可能是不同算法。对应的可执行小实现见[手撕强化学习](../practice/reinforcement-learning.md)与[训练目标](../practice/training-objectives.md)。

## 历史入口

[强化学习的演进](history.md)不按模型榜单排列，而是追踪几个反复出现的矛盾：长期回报怎样递推、采样估计怎样控制方差、函数逼近怎样稳定、策略更新怎样限制分布漂移，以及人类或程序反馈怎样成为可学习目标。

从家族案例继续，可分别观察 [DeepSeek](../landscape/families/deepseek.md) 的 GRPO、RLVR 与蒸馏闭环，[Kimi](../landscape/families/kimi.md) 的长轨迹、partial rollout 与多教师路线，以及 [GLM](../landscape/families/glm.md) 的 reasoning / agentic RL、异步系统与跨阶段蒸馏。
