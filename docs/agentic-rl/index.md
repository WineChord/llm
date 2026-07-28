# Agentic RL

Agentic Reinforcement Learning 研究模型如何在多步环境中观察、推理、调用工具、接收反馈并改进策略。它比单轮偏好对齐多出三个核心对象：状态、动作与环境转移。

## 问题定义

可将任务写成部分可观测决策过程

$$
\mathcal M=(\mathcal S,\mathcal A,\mathcal O,T,\Omega,R,\gamma).
$$

在第 $t$ 步，agent 根据历史

$$
h_t=(o_0,a_0,o_1,\ldots,o_t)
$$

选择动作 $a_t\sim\pi_\theta(\cdot\mid h_t)$，环境产生新状态和观察，最终最大化

$$
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}
\left[\sum_{t=0}^{T}\gamma^t r_t\right].
$$

对语言 agent，动作可能是文本 token、结构化工具调用、代码、浏览器操作或终止决定。不同动作粒度会改变 credit assignment 和训练成本。

## 与常见后训练的关系

| 方法 | 数据来源 | 反馈位置 | 是否在线采样环境 |
| --- | --- | --- | --- |
| SFT | 专家示范 | 每个目标 token | 否 |
| DPO 类偏好优化 | 成对或排序样本 | 完整回答 | 通常否 |
| 单轮 RLHF/RLAIF | policy 采样 | 回答级奖励 | 是，但环境简单 |
| Reasoning RL | 可验证问题 | 结果或轨迹级 | 是 |
| Agentic RL | 多步交互 | 过程、状态与结果 | 是，环境有转移 |

Agentic RL 不是一个特定算法。PPO、GRPO、REINFORCE、离线 RL 或 imitation learning 都可能成为训练组件。

从单轮指令与偏好进入在线策略学习的前史见[后训练与对齐](../landscape/lineages/training-alignment.md)；CoT、候选搜索与 verifier 怎样产生训练信号见[推理、搜索与验证](../landscape/lineages/reasoning-verification.md)。本节从二者继续向有状态环境、长时轨迹和终态信用分配推进。

## 系统闭环

```text
task sampler
    -> environment reset
    -> rollout workers
    -> tools / simulators / services
    -> reward and verifier
    -> trajectory store
    -> advantage / target computation
    -> policy update
    -> evaluation and replay
```

任何一层不稳定都会污染学习信号。工具超时可能被误当作策略失败；verifier 漏洞会诱导 reward hacking；环境版本漂移会让离线轨迹失效。

## 六个长期问题

1. **状态表示**：长历史怎样压缩而不丢任务约束？
2. **动作建模**：按 token、消息、工具调用还是语义步骤训练？
3. **信用分配**：最终成功应如何归因到早期决策？
4. **探索**：怎样产生多样轨迹，又不在昂贵环境中浪费预算？
5. **验证**：奖励能否抵抗投机、泄漏和不可重复性？
6. **系统**：rollout、推理服务和训练怎样保持版本一致？

## 推荐阅读路径

1. [强化学习总览](../reinforcement-learning/index.md)：建立 MDP、value、policy optimization 与反馈制度的完整坐标。
2. [从经典 RL 到语言 Agent](rl-foundations.md)：把单轮生成、多轮工具调用与部分可观测环境放在同一建模框架中。
3. [算法决策](math-algorithms.md)：按数据分布、critic、信用粒度与异步程度选择训练方法。
4. [轨迹与策略契约](trajectory-contract.md)：action mask、old log-prob、policy version、终止与异步 lag。
5. [数据与环境](data-environments.md)：任务、工具、verifier 和可复现状态。
6. [搜索、过程奖励与验证](search-verification.md)：best-of-$N$、树搜索、PRM 与 verifier 安全。
7. [训练系统](training-systems.md)：rollout、训练、调度、版本与故障。
8. [长时任务](long-horizon.md)：上下文、层级策略、恢复和跨 episode 学习。
9. [评测与安全](evaluation-safety.md)：能力、成本、污染、奖励攻击和权限边界。
10. [阅读清单](reading-list.md)：按问题组织原始论文与实现。

基础后训练见[后训练与偏好学习](../training/post-training.md)，应用层 agent 见[工具与智能体](../applications/agents.md)。

三条家族案例分别提供不同切面：[DeepSeek](../landscape/families/deepseek.md)连接可验证推理、蒸馏与可恢复长轨迹，[Kimi](../landscape/families/kimi.md)连接 partial rollout、多模态 Agent 与系统状态，[GLM](../landscape/families/glm.md)连接异步训练、环境构造和 Agentic Engineering。它们用于验证本节接口，不替代通用定义。
