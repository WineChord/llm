# 工具调用

工具调用把模型输出从自然语言建议变成可校验动作。一个可靠工具既有参数 schema，也有权限、幂等、错误和终态契约。

[ReAct 与 Toolformer](../landscape/works/react-toolformer.md)展示两种不同起点：前者在推理时交错 reasoning、action 与 observation，后者从候选 API 调用中筛选能降低语言模型损失的训练样本。两者在应用系统演进中的位置见[检索、工具与智能体](../landscape/lineages/retrieval-agents.md)。

## 四段边界

$$
\text{intent}
\xrightarrow{\text{select}}
\text{typed request}
\xrightarrow{\text{authorize/execute}}
\text{typed result}
\xrightarrow{\text{observe}}
\text{next state}.
$$

四段应分别记录：

1. 模型为什么选择该工具的可审计理由标签；
2. schema 校验后的规范化参数；
3. 执行身份、幂等键、超时与实际副作用；
4. 结果状态、证据引用和可否安全重试。

模型生成的 JSON 只是 request candidate，不能绕过授权与业务校验。

## Schema

最小工具描述包括：

- 唯一名称和清楚的适用边界；
- 输入类型、必填项、范围、枚举与互斥条件；
- 输出 schema 和稳定的错误结构；
- 读取、写入、通信或破坏性等级；
- 超时、取消和最大结果大小；
- 幂等性与重复调用语义。

受约束解码或 structured outputs 可以保证语法形状。[OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/)与 [Anthropic tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)展示了两类官方接口。结构合法仍不代表：

- 工具选择正确；
- 实体解析无歧义；
- 参数在业务上允许；
- 调用者拥有权限；
- 外部系统最终完成了动作。

这些必须由执行层验证。

## 参数规范化

执行前按确定性顺序处理：

1. schema 与类型；
2. Unicode、日期、时区、单位和路径规范化；
3. 跨字段约束；
4. 资源解析与身份绑定；
5. 权限与策略；
6. 风险分级和必要审批；
7. 生成幂等键；
8. 执行。

不要让模型自行展开模糊资源名后直接执行高风险动作。若“项目 A”可解析到多个对象，应先读取得到候选，再由明确选择消歧。

## 幂等与重试

读取通常可安全重试，写入未必。对可幂等写操作，使用稳定键：

$$
k=\operatorname{Hash}(
\text{actor},\text{tool},\text{canonical args},\text{operation scope}
).
$$

服务端应在同一业务范围内返回第一次操作的结果，而不是重复副作用。若请求超时，状态应标为 unknown，先查询操作状态，再决定是否重试。

指数退避只解决瞬时故障，不解决：

- 参数本身错误；
- 权限不足；
- 配额耗尽；
- 副作用已发生但响应丢失；
- 非幂等操作。

运行时需要按错误类型决定 retry、repair、compensate、ask 或 stop。

## 结果语义

工具结果应把面向模型的摘要与机器状态分开：

```json
{
  "status": "succeeded",
  "operation_id": "op_...",
  "data": {"resource_id": "r_..."},
  "evidence": [{"kind": "api_state", "ref": "r_..."}],
  "retryable": false
}
```

长结果应返回稳定引用、分页 cursor 或摘要加原始对象 ID，不要把无界日志直接塞回上下文。来自网页、邮件或文档的字符串依旧是不可信数据。

## Tool selection

工具描述要区分相近能力，避免多个工具都写“搜索信息”。离线集合至少覆盖：

- 应调用与不应调用；
- 同名实体与缺少参数；
- 多工具先后顺序；
- 工具失败后的替代；
- 明显越权或危险请求；
- 注入内容要求调用工具。

可把选择任务视为带拒绝类的分类：

$$
\hat t=\arg\max_{t\in\mathcal T\cup\{\varnothing\}}
p(t\mid x),
$$

$\varnothing$ 表示无需调用。仅报告“工具名准确率”会忽略过度调用；应同时看 precision、recall 和 no-tool calibration。

## 协议层

工具协议定义互操作，不替代应用安全。

- [Model Context Protocol tools specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)定义工具发现、调用、输入/输出 schema 与错误传递；相应 [schema](https://modelcontextprotocol.io/specification/2025-11-25/schema)给出消息结构。
- [Agent2Agent specification](https://a2a-protocol.org/latest/specification/)面向 agent 之间的任务、消息与产物交换。

这些协议持续演进；本页链接核验于 2026-07-27，实施时应绑定具体版本。协议可描述“怎样交换”，身份、租户、数据分类、审批和业务终态仍由部署系统负责。

## 最小权限

工具服务不要复用模型平台的全局高权限凭据。每次调用应携带：

- 明确主体与委托链；
- 当前任务允许的资源范围；
- 短生命周期凭据；
- 操作类型与风险级别；
- 可审计的请求 ID。

预览与执行最好使用不同接口或明确的 dry-run 标志。审批内容应展示规范化后的真实参数，而不是模型最初的自然语言描述。

## 评测矩阵

| 维度 | 样例 |
| --- | --- |
| 选择 | 正确工具、no-tool、相似工具 |
| 参数 | 类型、单位、时区、边界、歧义 |
| 权限 | 跨租户、越权资源、过期委托 |
| 可靠性 | 超时、限流、部分成功、重复请求 |
| 安全 | 注入、结果伪造、数据外泄 |
| 终态 | 响应成功但业务未完成 |

schema 校验与安全 dispatch 的紧凑实现见[手撕：检索与智能体](../practice/retrieval-agents.md)，多步状态与恢复见[智能体运行时](agent-runtime.md)。

## Reference {#reference}

- [OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/)
- [Anthropic tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)
- [Model Context Protocol tools specification](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)
- [Model Context Protocol schema 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/schema)
- [Agent2Agent specification](https://a2a-protocol.org/latest/specification/)
