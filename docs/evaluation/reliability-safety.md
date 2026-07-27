# 可靠性与安全

可靠性关注系统在正常输入、边界条件、分布漂移和部分故障下是否保持可预测；安全关注它是否跨越权限、隐私和风险边界。一个系统可以稳定地产生危险结果，也可以很安全却因过度拒绝而不可用，因此两者必须分别测量。

本页保留为完整入口。事实与引用见[幻觉与事实性](hallucination.md)，概率与 abstention 见[校准与不确定性](calibration-uncertainty.md)，攻击协议见[安全评测](safety-evaluation.md)，工具副作用见[Agent 与工具评测](agent-tool-evaluation.md)。

## 评测协议

先定义系统边界：

```text
model and policy versions
retrieval / tools / guards / execution layer
trusted instructions and untrusted inputs
tenant, identity and permission model
normal, degraded and adversarial environments
allowed side effects and forbidden outcomes
fallback, retry and human-escalation policy
```

只评模型文本，不能证明含工具系统可靠；只评执行终态，又可能忽略模型给出的危险建议和敏感信息。

## 统计对象

### 可靠性

以 request 或 episode 为单位，状态至少包括：

```text
success
wrong result
abstain
policy refusal
invalid output/action
timeout
dependency or infrastructure failure
partial side effect
unknown final state
```

能力成功率与端到端成功率使用不同分母，但必须并列报告。重试产生相关样本，也会改变成本和副作用；不能把“最终一次成功”当成单次可靠性。

### 安全

以攻击 case 或风险场景为单位，同时记录：

- 是否完成合法用户任务；
- 是否发生目标攻击行为；
- 是否泄露、越权或产生未授权副作用；
- guard 是否检测、阻止或误报；
- 攻击查询、轮次、工具和时间预算。

安全结果必须与 benign utility 配对。一个拒绝全部请求的系统攻击成功率很低，却没有可用性。

## 失败面

```text
request
  -> policy and routing
  -> context / retrieval
  -> model generation
  -> parser / guard
  -> tool and external service
  -> state validation
  -> response and side effect
```

每层都可能：

1. 显式失败；
2. 返回成功状态但语义错误；
3. 产生部分外部动作；
4. 隐藏不确定性；
5. 被不可信输入改变控制流。

错误归因应以可观察状态和版本证据为准，而不是统一称为“模型幻觉”。

## 事实、校准与未知

事实错误可能来自参数知识、检索、证据阅读、工具状态或生成。结构化输出只能约束格式，不能证明字段真实。高风险主张需要：

- 原子主张与证据对应；
- 来源时效和对象版本；
- 计算、代码或数据库的确定性验证；
- calibrated confidence 或 abstention；
- 无法验证时的人工升级。

只降低回答率可以减少错误，却也降低 coverage；应使用 risk–coverage 而非单点准确率。

## 指令层级与不可信数据

网页、文件、邮件、图像文本和工具结果属于数据，不应自动取得控制权限。[OpenAI Instruction Hierarchy](https://openai.com/index/the-instruction-hierarchy/) 研究了让模型区分高低优先级指令的训练方向；系统仍需在模型外实施：

- tool、路径、网络和字段 allowlist；
- 租户、来源和 ACL 隔离；
- 写操作的对象、范围与授权校验；
- 幂等、事务、写后读取与回滚；
- 不可信内容和控制指令的结构分层。

模型拒绝一条注入文本，不代表执行层权限正确；模型遵循正确指令，也不能替代外部授权。

## 隐私

分开评测：

| 层 | 主要问题 |
| --- | --- |
| 训练数据 | 记忆、抽取、删除与 unlearning |
| 上下文 | 不必要敏感数据是否进入 prompt |
| 工具与检索 | 跨租户、ACL 与数据流 |
| 日志 | 原始内容、保留期和调试副本 |
| 输出 | 直接或组合泄露 |

拒绝复述秘密不是唯一防线。系统不应把模型无权使用的数据放进上下文；评测也不应在公开结果中保存真实敏感内容。

## 分布漂移与故障

可靠性集需要覆盖：

- 语言、领域、长度和用户群变化；
- 新旧知识冲突与时间漂移；
- 检索空结果、过期索引和恶意文档；
- 工具超时、部分成功、权限不足和重复回调；
- parser、judge、guard 和模型版本切换；
- 降级模型、缓存和网络隔离。

shadow 与 canary 只能观测实际路由到的分布。若新旧版本接收不同请求，直接比较接受率会混入 selection bias。

## 正确性与攻击失效

- **端到端失败从分母删除**：线上可靠性被高估。
- **重试只算最终成功**：隐藏首次失败、时延与重复副作用。
- **拒答率当安全分**：无害请求受损。
- **guard accuracy 当策略安全**：阈值、攻击适应和执行权限未测。
- **文本拒绝替代状态检查**：未授权动作可能已经发生。
- **静态攻击集替代 adaptive attacker**：防御只记住模板。
- **日志保存完整敏感内容**：观测系统制造新的泄露面。
- **供应商声明替代本地验证**：版本、工具和预算条件不同。

## 何时拆分评测

高风险系统不应追求单一“可靠安全分”。模型能力、guard detection、执行权限、事实验证和业务终态应分别评测，再用明确的 hard constraint 组合。只有单步、无工具、低风险的文本任务，才适合在一张简化报告中合并。

## 报告卡

```text
system boundary and component versions
target population and normal/degraded/adversarial slices
threat model and attacker budget
success/failure/refusal/infra/side-effect taxonomy
capability and end-to-end denominators
ASR, false refusal, utility and confidence intervals
fact support, calibration and coverage
tool permissions, retries and environment revisions
drift windows, canary policy and rollback gates
raw audit trail, privacy handling and known limits
```

统计区间见[统计推断](statistical-inference.md)，上线 SLO 与事故闭环见[生产可靠性](production-reliability.md)，最小工具见[评测工具](../practice/evaluation-tooling.md)。
