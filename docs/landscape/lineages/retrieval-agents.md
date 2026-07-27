# 从参数记忆到可行动系统

现代大规模预训练 checkpoint 经常以近似封闭的接口部署：输入 token，输出下一个 token 的分布。规模扩大后，参数确实记住了大量事实，但两个老问题没有随规模消失：知识难以及时更新，答案也很难指出自己依据了什么。检索、工具与智能体的历史，正是模型边界不断向外移动的过程——先让它读取外部记忆，再让它调用外部函数，最后让它在有状态环境里承担连续行动。

## 检索先解决“知识放在哪里”

[REALM](https://arxiv.org/abs/2002.08909)把检索器放进预训练目标，尝试让 masked language modeling 的梯度选择有用文档；[DPR](https://arxiv.org/abs/2004.04906)把开放域问答中的检索变成双 encoder 的稠密向量匹配；[RAG](https://arxiv.org/abs/2005.11401)进一步把检索文档视为生成时的隐变量，在候选文档上边缘化答案概率。

这条路线的重要变化，不只是“在 prompt 前拼文档”。它把系统分成三种可分别更新和验证的状态：

$$
\text{corpus state}\longrightarrow \text{retrieval state}\longrightarrow \text{generation state}.
$$

文档可以更新而不重训模型，检索结果可以单独计算 recall，生成答案也可以检查是否真正受证据支持。与此同时，错误也有了新的传播路径：索引漏掉答案、chunk 切坏语义、reranker 排错、模型忽略证据，最后都会表现成同一种“答错”。[RAG 深读](../works/rag.md)从原论文的 latent-document 目标讲到今天常见的 retrieve–rerank–generate 管线。

## 工具调用改变“模型能做什么”

检索返回文本，工具调用返回的可以是数值、结构化对象或环境状态。[Toolformer](https://arxiv.org/abs/2302.04761)问的是训练问题：能否用少量示例自动标注 API 调用，并保留那些真正降低语言模型损失的调用；[ReAct](https://arxiv.org/abs/2210.03629)问的是推理问题：能否让语言推理与外部行动交错，使观察结果反过来修正下一步计划。

二者经常被并列提及，却不在同一层：

- Toolformer 改变训练数据和 token 目标，让模型学习“何时调用”；
- ReAct 定义一种 trajectory 组织方式，让 observation 进入后续上下文；
- 生产 runtime 还必须独立处理权限、参数校验、超时、幂等、重试和副作用。

这一差别见[ReAct 与 Toolformer 深读](../works/react-toolformer.md)以及[工具调用](../../applications/tool-use.md)。一个模型能生成合法 JSON，只证明语法层通过；它是否应该调用、调用是否成功、结果是否可信，属于不同状态。

## 当一次调用变成一条轨迹

单次问答的损失写作 $\ell(x,y)$；进入环境以后，对象变成

$$
\tau=(o_0,a_0,o_1,a_1,\ldots,o_T),\qquad
P(\tau)=\prod_{t=0}^{T-1}\pi(a_t\mid h_t)P(o_{t+1}\mid o_t,a_t).
$$

这里 $o_t$ 是观察，$a_t$ 是动作，$h_t$ 是截至当前的可见历史。失败可能来自策略，也可能来自工具、环境、权限或观测延迟。于是“模型能力”开始变成系统属性：

$$
\text{success}=f(\text{checkpoint},\text{context},\text{tools},\text{runtime},\text{environment}).
$$

这促使后来的工作把 planning、memory、reflection、search 与 reinforcement learning 接入同一个闭环。可是在真实系统中，语言形式的“思考”不等于可恢复状态，所谓记忆也不能只是不断增长的消息转录。状态契约见[Agent Runtime](../../applications/agent-runtime.md)，长时训练见[Agentic RL](../../agentic-rl/index.md)。

## 三次边界扩张

把这段发展压缩成三个转折，会比罗列 Agent 名称更有用：

1. **参数外记忆**：REALM、DPR、RAG 让知识进入可更新、可检索的外部状态；
2. **生成中调用**：Toolformer 学习插入并消费外部函数结果，ReAct 进一步把 action 与 observation 交错成可继续修订的轨迹；真正改变外部世界还取决于工具权限和 runtime；
3. **参考答案之外的反馈**：verifier 可以只评分最终输出或局部步骤，搜索比较候选路径，Agentic RL 才进一步把环境终态和长时信用纳入策略更新。

每次扩张都增加了能力，也增加了需要被记录和评测的状态。新系统若只展示成功 demo，却没有冻结 corpus version、tool schema、权限、预算、终态和失败分母，就无法判断进步来自模型还是 harness。

## 阅读今天的 Agent 工作

面对一个新的 Agent 论文或框架，可以沿四个问题定位：

- 它改变了 checkpoint，还是只改变 runtime；
- 它新增的是信息、动作、搜索，还是训练反馈；
- 环境终态能否由程序验证，还是依赖另一个模型评分；
- 增长的 token、工具调用和 wall-clock 是否计入同一预算。

这套坐标把“能调用工具”的表面现象还原成可分析的系统。检索侧继续读[检索与索引](../../applications/retrieval-indexing.md)和[有依据生成](../../applications/grounded-generation.md)；行动侧继续读[记忆与规划](../../applications/memory-planning.md)、[Agent 安全](../../applications/agent-security.md)和[Agent 与工具评测](../../evaluation/agent-tool-evaluation.md)。
