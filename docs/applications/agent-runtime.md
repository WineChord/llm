# 智能体运行时

运行时把模型的单步建议变成有界、可恢复、可审计的执行。核心不是反复调用模型，而是显式状态机、事件日志、预算与终态验证。

## 状态机

一个最小任务状态可以写成：

$$
s_t=(g,p_t,o_t,a_t,b_t,r_t),
$$

其中 $g$ 是目标，$p_t$ 是计划，$o_t$ 是最新观察，$a_t$ 是待执行或已执行动作，$b_t$ 是预算，$r_t$ 是资源与权限状态。

典型转移：

```text
created → planning → ready → executing → observing ┐
                  ↑              ↓                 │
                  └──────── replanning ←───────────┘
                                      ↓
                 succeeded | partial | blocked | failed | cancelled
```

终态必须由确定性谓词或外部证据确认。对外部写入，`executing` 超时不能直接转为 `ready` 重试，应先进入 reconciliation。

## Event sourcing

与其反复覆盖一份不可解释的状态，不如追加事件：

```text
TaskCreated
PlanProposed
ToolRequested
ToolAuthorized
ToolStarted
ToolSucceeded / ToolFailed / ToolUnknown
ObservationRecorded
GoalVerified
```

当前状态由事件归约得到。事件至少包含 task ID、step ID、attempt、模型与工具版本、规范化参数摘要、时间、调用身份和证据引用。敏感载荷应加密或只保存受控引用。

事件日志带来三个能力：

- 崩溃后从最后一个已确认事件恢复；
- 解释某个动作为什么执行；
- 用真实轨迹重放评测，但不重放副作用。

## Step contract

每一步应声明：

```text
precondition → action → expected observation → success predicate
                                  ↘ rollback / compensation
```

例如“创建发布版本”与“部署上线”是不同 step。前者返回版本 ID 不代表后者已经可访问。拆分后，失败可以定位在创建、调度、部署或可见性验证。

## Budget

预算是多维向量：

$$
b_t=(n_{\text{steps}},n_{\text{tokens}},t_{\text{wall}},
c_{\text{money}},n_{\text{writes}},r_{\text{risk}}).
$$

运行时在每步执行前检查剩余预算，并预留验证和清理成本。只限制最大迭代数会允许单步无限等待或产生过多外部写入。

预算耗尽不等于失败。若已有有用结果，应进入 partial 并返回完成项、未完成项和恢复点。

## Retry 与 repair

失败处理按原因分流：

| 类型 | 默认策略 |
| --- | --- |
| 瞬时网络错误 | 有界退避后重试 |
| 限流 | 尊重 retry-after，必要时降并发 |
| 参数校验失败 | 修复参数，不重放原请求 |
| 权限不足 | 阻塞并请求授权，不自行扩大权限 |
| 未知副作用 | 查询操作状态或对账 |
| 业务冲突 | 重新读取状态并重规划 |
| 不可逆失败 | 停止、保全证据并上报 |

同一失败重复出现时，模型换一种措辞不算 repair。需要改变可观察变量、调用路径或假设。

## 并发

只有相互独立且资源不冲突的步骤才并行。任务图可以表示为 DAG：

$$
G=(V,E),\qquad
(u,v)\in E\Rightarrow v\ \text{依赖}\ u.
$$

还要处理 DAG 未表达的资源冲突，例如两个步骤同时修改同一文件或消费同一限额。运行时可为资源申请 lease，并在提交前检查版本或 ETag。

并发结果回到模型前应按 step ID 归并，不依赖完成顺序。取消父任务时，应传播取消并等待工具确认，而不是仅停止生成。

## Human-in-the-loop

审批是状态转移，不是聊天提示：

```text
awaiting_approval
  ├─ approved(scope, expiry) → ready
  ├─ edited(new_args)        → validate → ready
  ├─ rejected(reason)        → replanning / cancelled
  └─ expired                 → blocked
```

审批界面展示工具、规范化参数、影响范围、可逆性和凭据主体。批准一次特定动作不应被解释为未来所有同类动作的永久授权。

## Context construction

每次模型调用只注入当前决策所需内容：

- 不变的目标与约束；
- 当前状态与剩余预算；
- 可用工具的最小描述；
- 最近相关观察；
- 经检索的历史证据；
- 明确的输出 schema。

长日志、重复错误和无关工具结果应保存在事件层，通过引用按需取回。上下文压缩策略见[记忆与规划](memory-planning.md)。

## 验证与可观测性

每个 step 记录：

- planning、queue、tool 与 verification latency；
- token、调用数、费用和外部写入数；
- retry、repair、replan 和 approval 次数；
- 输入/输出 schema 版本；
- 成功谓词与证据；
- 安全策略决策。

总体成功率之外，报告 p50/p95 步数和延迟、unknown side-effect rate、恢复成功率以及每个工具的失败分布。

## 恢复演练

至少注入以下故障：

- 模型响应在工具执行前丢失；
- 工具执行成功但响应丢失；
- 事件写入后进程崩溃；
- 审批期间凭据过期；
- 多个 worker 同时领取同一步；
- 取消与成功响应交错；
- 工具 schema 在任务中途升级。

能在 happy path 跑通不是可恢复系统。最小状态机代码见[手撕：检索与智能体](../practice/retrieval-agents.md)，生产 SLO 与事故复盘见[生产可靠性](../evaluation/production-reliability.md)。

## Reference {#reference}

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Temporal durable execution documentation](https://docs.temporal.io/temporal)
- [Making retries safe with idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/)
- [OpenTelemetry Specification](https://opentelemetry.io/docs/specs/otel/)
