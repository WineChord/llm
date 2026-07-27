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

1. [数学与算法](math-algorithms.md)：policy gradient、PPO、GRPO、DPO 与 credit assignment。
2. [数据与环境](data-environments.md)：任务、轨迹、工具、verifier 和可复现状态。
3. [训练系统](training-systems.md)：rollout、训练、调度、版本与故障。
4. [长时任务](long-horizon.md)：上下文、层级策略、恢复和跨 episode 学习。
5. [评测与安全](evaluation-safety.md)：能力、成本、污染、奖励攻击和权限边界。
6. [阅读清单](reading-list.md)：按问题组织原始论文与实现。

基础后训练见[后训练与偏好学习](../training/post-training.md)，应用层 agent 见[工具与智能体](../applications/agents.md)。
