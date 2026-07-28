# 采样、搜索与验证

模型回答质量不仅由权重决定，也取决于推理阶段如何分配计算。一次 greedy decode、并行采样多个候选、调用 verifier、在树上搜索以及执行工具，都可能使用同一基础模型，却形成不同的质量—延迟—成本曲线。

本节关注 inference-time policy，而不是把所有“推理能力”归因于更长文字。

从 Chain-of-Thought、self-consistency 到 verifier-guided search，再到用可验证奖励更新策略，关键变化是“计算预算由谁分配、结果由谁判断”。[推理、搜索与验证](../landscape/lineages/reasoning-verification.md)给出这条脉络；[DeepSeek-R1](../landscape/works/deepseek-r1.md) 则用于辨析预训练能力、搜索、强化学习与蒸馏各自贡献了什么。

## 三层对象

1. **候选生成**：temperature、top-$p$、prompt、草稿、分支宽度。
2. **候选评估**：答案一致性、规则检查、执行结果、outcome/process verifier。
3. **预算控制**：哪些问题值得更多样本，何时扩展或停止。

它们可抽象为

$$
\mathcal Y_B
=
\operatorname{Generate}(x;B,\pi),
\qquad
y^\star
=
\operatorname{Select}(\mathcal Y_B;v),
$$

其中 $B$ 是计算预算，$\pi$ 是生成策略，$v$ 是评估器。增加 $B$ 只有在候选具有增量多样性且 $v$ 能区分质量时才有价值。

## 与训练时方法的边界

- Chain-of-thought prompting 改变输入与生成轨迹，不更新权重。
- Best-of-$N$、self-consistency、beam 和树搜索发生在回答阶段。
- 用搜索轨迹训练策略或奖励模型属于后训练。
- Agent 在环境中执行动作时，还需要工具状态、权限和长时任务契约。

因此通用生成、搜索和 verifier 机制放在本节；轨迹怎样反哺策略训练，见 [Agentic RL](../agentic-rl/index.md)。

## 阅读路径

- [推理时计算](test-time-compute.md)：self-consistency、Best-of-$N$、自适应预算与成本曲线。
- [搜索与验证](search-verification.md)：beam、树搜索、过程/结果 verifier 和停止规则。
- [解码](../inference/decoding.md)：logit 处理、sampling、beam 与约束解码的底层语义。
- [测试时计算手撕实现](../practice/test-time-compute.md)：答案归一、候选选择、预算分配与搜索。

## 评测原则

任何 test-time scaling 结果都应同时报告：

- 基础模型、prompt 和 decoding 参数；
- 输入、输出与隐藏/丢弃 token；
- 样本数、并发数、verifier 调用数；
- 首 token、最终答案延迟与总成本；
- pass@1、pass@$k$、selected accuracy；
- 候选相关性和 verifier 选择准确率；
- 相同预算下的基线。

“思考更久”不是实现规范。可比较的对象是明确的计算图、预算和停止条件。

[DeepSeek](../landscape/families/deepseek.md)、[Kimi](../landscape/families/kimi.md) 与 [GLM](../landscape/families/glm.md) 家族包含不同的 reasoning checkpoint、搜索/验证路线与服务模式；比较时必须先分开训练后的策略能力、测试时预算、外部 verifier 和工具系统。
