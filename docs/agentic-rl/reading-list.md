# Agentic RL 阅读清单

阅读顺序按问题组织，而不是按发布时间堆叠。每篇材料都应回答“它改变了哪个对象、依赖什么假设、怎样验证”。

## 策略优化与偏好学习

| 材料 | 关注问题 |
| --- | --- |
| [PPO](https://arxiv.org/abs/1707.06347) | clipped policy update 与采样复用 |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | SFT、reward model、PPO 的完整 RLHF 管线 |
| [DPO](https://arxiv.org/abs/2305.18290) | 从偏好对直接优化 policy |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | GRPO 与可验证数学任务 |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | reasoning RL、冷启动与蒸馏 |
| [Dr. GRPO](https://arxiv.org/abs/2503.20783) | 组标准化与长度归一化偏置 |
| [DAPO](https://arxiv.org/abs/2503.14476) | 动态采样、非对称 clipping 与 token-level loss |
| [GSPO](https://arxiv.org/abs/2507.18071) | sequence-level ratio 和 trust region |
| [SAPO](https://arxiv.org/abs/2511.20347) | 平滑、token-adaptive 的 ratio gate |
| [PRIME](https://arxiv.org/abs/2502.01456) | 从 outcome label 学习过程奖励 |

阅读时对齐 policy/reference/behavior model、奖励粒度、KL、采样组和训练阶段。

## 工具与交互

| 材料 | 关注问题 |
| --- | --- |
| [ReAct](https://arxiv.org/abs/2210.03629) | 推理与行动交错的轨迹表示 |
| [Toolformer](https://arxiv.org/abs/2302.04761) | 自监督工具调用数据构造 |
| [WebGPT](https://arxiv.org/abs/2112.09332) | 浏览环境、引用与人类反馈 |
| [Voyager](https://arxiv.org/abs/2305.16291) | 长期技能库与开放式环境 |

重点不是记 prompt 格式，而是观察 action space、环境反馈和训练信号。

## 代码与可验证环境

| 材料 | 关注问题 |
| --- | --- |
| [SWE-bench](https://arxiv.org/abs/2310.06770) | 真实仓库 issue、测试与环境复现 |
| [CodeRL](https://arxiv.org/abs/2207.01780) | 单元测试反馈与代码生成 RL |
| [AlphaCode](https://arxiv.org/abs/2203.07814) | 大规模采样、过滤与聚类 |
| [CodeAct](https://arxiv.org/abs/2402.01030) | 以可执行代码作为统一动作 |

需要区分模型、采样预算、test harness 和检索权限。

## 长时任务与系统

| 材料 | 关注问题 |
| --- | --- |
| [METR time horizons](https://metr.org/time-horizons/) | 固定成功率下的任务时长 |
| [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) | 跨上下文工作周期与可恢复状态 |
| [veRL](https://github.com/volcengine/verl) | rollout、训练角色与分布式执行 |
| [AReaL](https://arxiv.org/abs/2505.24298) | 异步 rollout、policy lag 与吞吐 |
| [SAO](../landscape/works/sao-compactionrl.md#sao) | 单 rollout、critic 与双侧重要性筛选 |
| [CompactionRL](../landscape/works/sao-compactionrl.md#compactionrl) | 学习式摘要、token normalization 与跨段信用 |
| [Agent Lightning](https://arxiv.org/abs/2508.03680) | 任意 agent 轨迹的层级信用分解 |

长时结果应同时记录 checkpoint、harness、工具、预算和外部状态。

## 评测与风险

| 材料 | 关注问题 |
| --- | --- |
| [AgentBench](https://arxiv.org/abs/2308.03688) | 多环境 agent 能力 |
| [ToolBench](https://arxiv.org/abs/2307.16789) | 大规模工具数据与评测 |
| [Do-Not-Answer](https://arxiv.org/abs/2308.13387) | 风险请求与拒答边界 |
| [Sleeper Agents](https://arxiv.org/abs/2401.05566) | 后门行为的持续性 |

单一 leaderboard 无法覆盖污染、成本、权限和恢复能力，应配合[评测与安全](evaluation-safety.md)的检查表。

## 阅读笔记模板

```text
问题：
状态 / 动作 / 奖励：
数据与环境：
核心算法：
系统假设：
对照与预算：
主要结果：
失效模式：
可复现入口：
可迁移结论：
```

完整概念路径从[Agentic RL 总览](index.md)开始。
