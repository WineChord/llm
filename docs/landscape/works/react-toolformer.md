# ReAct 与 Toolformer：模型何时开始行动

让模型输出一个 API 调用并不难，难的是三个连续决定：什么时候值得调用、参数怎样形成、返回结果如何改变后续生成。Toolformer 与 ReAct 分别从训练和推理两侧切入这个问题。前者把有用工具调用筛进语言模型训练数据，后者把 reasoning、action 与 observation 组织成可继续展开的轨迹。

## Toolformer：用语言模型损失筛选调用

[Toolformer](https://arxiv.org/abs/2302.04761)从少量人工示例出发，让模型在普通文本中提出候选 API 调用。执行调用后，把返回值插入文本；若调用能显著降低后续 token 的负对数似然，就保留为自监督训练样本。

设插入调用及结果后的损失为 $L_i^+$。Toolformer 的 baseline 不是只有“不调用”：它取“不插入调用”与“保留调用但隐藏返回结果”两种后续损失的较小者，记为 $L_i^-$。筛选逻辑是：

$$
L_i^- - L_i^+ > \tau.
$$

这让“是否调用”与语言建模收益绑定，而不需要为每个工具人工标注大量轨迹。边界也很清楚：训练时可用的工具、返回格式和分布决定了模型学到什么；损失降低不等同于真实任务正确，更不等同于调用安全。

完整数据闭环包含五步：用少量 demonstrations 教会 API 语法、在普通文本的候选位置采样调用、执行 API、按后续 token loss 过滤、再在保留的调用与结果上继续语言模型训练。任何一步改变都会改变监督分布。尤其“隐藏结果”的 baseline 防止保留那些仅凭调用文本本身就降低 loss、实际返回值没有贡献的样本。

Toolformer 的工具是论文预先定义的一组 API，调用结果可直接插入文本。它没有解决任意动态 schema、写操作、认证或工具发现；把其筛选目标迁移到有副作用的工具前，必须把环境成功与权限加入 verifier。

## ReAct：观察必须进入下一步推理

[ReAct](https://arxiv.org/abs/2210.03629)把轨迹写成交错序列：

```text
Thought -> Action -> Observation -> Thought -> ...
```

行动让模型访问外部信息或环境，观察结果又能修正计划。与“先写完整 chain-of-thought，再执行计划”相比，ReAct 的关键是反馈闭环：中间假设可以被真实 observation 推翻。

一个紧凑 runtime 不需要理解自然语言 Thought，但必须严格处理动作和终态：

```python
def run_agent(policy, tools, question, max_steps=4):
    trace = [{"kind": "question", "value": question}]
    for _ in range(max_steps):
        step = policy(tuple(trace))
        if step["kind"] == "final":
            return step["value"], trace
        if step["kind"] != "action" or step["name"] not in tools:
            raise ValueError("invalid action")
        result = tools[step["name"]](*step.get("args", ()))
        trace.extend([step, {"kind": "observation", "value": result}])
    raise TimeoutError("step budget exhausted")
def policy(trace):
    if not any(x["kind"] == "observation" for x in trace):
        return {"kind": "action", "name": "lookup", "args": ("capital",)}
    value = next(x["value"] for x in reversed(trace) if x["kind"] == "observation")
    return {"kind": "final", "value": value}
answer, trace = run_agent(policy, {"lookup": lambda key: {"capital": "Paris"}[key]}, "France?")
assert answer == "Paris" and [x["kind"] for x in trace] == ["question", "action", "observation"]
```

这个 reference 刻意把 policy 与 runtime 分开。模型只提出 typed action；runtime 决定工具白名单、参数解释和 step budget。真实系统还必须增加 schema validation、权限、幂等键、超时、重试、日志脱敏和副作用确认。

ReAct 论文同时研究知识问答/事实验证和交互环境。在前一类任务里，搜索 observation 帮助更新事实依据；在 ALFWorld、WebShop 等环境里，动作改变外部状态，错误行动未必可逆。共同点是 observation 进入下一决策，差别是环境转移是否有副作用。平均成功率之外，应分别记录无效动作、工具错误、超预算与环境终态。

## 训练信号和运行时闭环并不在同一层

| 层 | Toolformer | ReAct |
| --- | --- | --- |
| 主要问题 | 怎样学会何时调用工具 | 怎样在推理时交错思考与行动 |
| 数据来源 | 候选调用执行后按 LM loss 筛选 | 少量轨迹示例或策略生成 |
| 反馈 | 工具结果改善后续 token 预测 | observation 直接进入下一步上下文 |
| 未解决 | 权限、真实正确性、分布外工具 | 训练、长期信用、安全 runtime |

把 ReAct prompt 用在未训练过工具的模型上，与把 Toolformer 式数据训练进 checkpoint，是两个实验变量。比较时应固定工具、成功判据、调用预算和错误处理。

## 从文本轨迹到系统状态

语言轨迹可读，但不适合作为唯一状态存储。重试可能重复扣款，观察可能过期，工具返回也可能包含不可信指令。成熟 runtime 会把状态拆成：

$$
s_t=(\text{messages},\text{tool state},\text{environment version},
\text{permissions},\text{budget},\text{side effects}).
$$

模型可见的 context 只是 $s_t$ 的投影。checkpoint、prompt 或 summarizer 都不应悄悄覆盖真实环境状态。

当 context 需要压缩时，必须保留 action ID、observation provenance、未完成副作用和 budget；只总结自然语言 Thought 会让 runtime 丢失可恢复状态。工具返回应作为不可信 observation，而不是自动提升为下一轮指令。对应的 typed trajectory 见[轨迹与策略契约](../../agentic-rl/trajectory-contract.md)。

## 从一次闭环走向可训练策略

这两项工作之后，研究沿三条线继续：

- 通过 instruction tuning 或合成轨迹提高工具选择与参数生成；
- 通过 search、reflection 或 verifier 在推理时比较多条轨迹；
- 通过 online RL 根据环境终态优化完整策略。

它们分别对应[工具调用](../../applications/tool-use.md)、[搜索与验证](../../reasoning/search-verification.md)和[Agentic RL](../../agentic-rl/index.md)。完整前后关系见[从参数记忆到可行动系统](../lineages/retrieval-agents.md)，运行时边界见[Agent Runtime](../../applications/agent-runtime.md)。

这条演进不意味着轨迹越长越好。每增加一步都增加错误、延迟和攻击面；只有 observation 能改变后续选择、且最终结果可以验证时，额外行动才可能带来净收益。评测应与无工具、固定检索和等预算 baseline 比较，而不是只展示成功案例。

## Reference {#reference}

- [ReAct 论文](https://arxiv.org/abs/2210.03629)、[ReAct project page](https://react-lm.github.io/)与[ysymyth/ReAct](https://github.com/ysymyth/ReAct)；
- [Toolformer 论文](https://arxiv.org/abs/2302.04761)。论文公开了训练流程与实验，但没有把完整训练系统作为官方仓库发布；复现应明确这一证据边界。
