# Agent 与工具评测

Agent 评测的对象不是最终一段文本，而是模型在有状态环境中选择动作、调用工具、遵守规则并达到目标的完整 episode。成功必须由真实环境终态验证，未授权副作用必须独立于主任务成功记录。

## 环境协议

一个任务实例至少冻结：

```text
task intent and user-visible constraints
initial environment/database/filesystem state
tool names, schemas, implementations and permissions
policy/rule set and untrusted content
target state and accepted equivalence
forbidden states and side effects
turn/token/tool/time/cost budgets
environment, user-simulator and harness revisions
```

文字声称“已完成”不是终态证据；API 返回 `200` 也不保证业务状态正确。应在 episode 结束后从权威状态重新读取。

## 统计单位与状态

以 task instance 为 cluster、trial 为嵌套重复。逐 trial 至少标记：

```text
success
task failure
policy violation
invalid action / arguments
tool or environment timeout
infrastructure error
user simulator failure
partial state change
unauthorized side effect
unknown final state
```

主任务成功与安全结果是两个轴：

| 主任务 | 未授权副作用 | 解释 |
| --- | --- | --- |
| 成功 | 无 | 完整成功 |
| 成功 | 有 | 能力成功、安全失败 |
| 失败 | 无 | 普通任务失败 |
| 失败 | 有 | 任务与安全双重失败 |

把“成功但越权”的 episode 只记 success 会掩盖最重要风险。

## pass@$k$ 与 pass$^k$

### 至少一次成功

每题生成或运行 $n$ 次，其中 $c$ 次成功，pass@$k$ 的常用无偏估计为

$$
\operatorname{pass@}k
=
1-\frac{\binom{n-c}{k}}{\binom{n}{k}}.
$$

它回答“给 $k$ 次机会，至少一次成功”的能力/搜索问题。

### 连续可靠成功

pass$^k$ 回答“同一任务独立运行 $k$ 次，是否每次都成功”。若单次成功概率固定为 $p$ 且 trial 独立，理想值为

$$
\operatorname{pass}^k=p^k.
$$

实测可对每个 task 的 $k$ 个预注册 trial 取成功乘积，再跨 task 平均。[$\tau$-bench](https://arxiv.org/abs/2406.12045) 提出 pass$^k$ 衡量工具 Agent 多次交互的一致性。

pass@$k$ 随机会增加，pass$^k$ 随 $k$ 增大更严格；二者不能混写。实现还必须冻结随机种子、用户模拟器和“独立 trial”的定义。

## 指标分解

### 任务与过程

- final-state task success；
- 必要子目标完成率；
- invalid tool/action rate；
- recovery after tool error；
- redundant/repeated actions；
- policy/rule compliance；
- user clarification quality。

过程分数不能替代终态。更短轨迹也不必然更好：需要澄清或验证的动作可能提高可靠性。

### 权限与副作用

$$
\operatorname{unauthorized\ rate}
=
\frac{N_{\text{episodes with unauthorized side effect}}}
{N_{\text{all valid episodes}}}.
$$

还应按严重度和对象分层：

- 读到无权数据；
- 将数据发送到错误对象；
- 写入、删除或购买超出授权；
- 跨租户访问；
- 重试产生重复副作用；
- 注入导致攻击者目标完成。

分母不能只用主任务成功 episode。

### 成本与时延

逐 episode 记录：

```text
model tokens and generations
tool calls and retries
retrieval/network bytes
queue/model/tool/user-simulator latency
wall-clock to terminal state
external/API/compute cost
```

时延至少拆成 model thinking/generation、tool execution、environment transition 和 judge/verifier。只报总时延无法定位瓶颈。

## Harness 与状态可重放

可靠 harness 需要：

- 任务间环境隔离与 deterministic reset；
- 每个动作前后 state digest；
- tool request/response 与错误状态；
- 幂等键、重试和取消语义；
- 环境时钟、网络和外部依赖冻结；
- judge 与终态 verifier 分离；
- 失败后保留完整轨迹而不污染下一任务。

[WebArena](https://arxiv.org/abs/2307.13854) 以网站环境评估 Agent，[OSWorld](https://arxiv.org/abs/2404.07972) 使用真实计算机环境，[SWE-bench](https://arxiv.org/abs/2310.06770) 以仓库 issue 和测试评估代码修复。它们的 container、网站、VM、tests 和工具版本都是测量对象的一部分。

[GAIA](https://arxiv.org/abs/2311.12983) 与 [AgentBench](https://arxiv.org/abs/2308.03688) 覆盖更广的工具与环境任务；[BFCL](https://gorilla.cs.berkeley.edu/leaderboard) 聚焦函数调用格式与语义。不同 benchmark 的 success verifier 和外部访问不同，分数不能直接合成通用 Agent 能力。

## 注入与攻击

Agent 会读取网页、邮件、文档、代码注释、图像文字和工具结果。攻击测试应包含：

- direct/indirect prompt injection；
- 伪造 tool result 或高优先级指令；
- 跨轮延迟触发；
- 编码、隐藏文本和多模态载荷；
- 合法任务与攻击任务同时存在；
- adaptive attacker 根据 Agent 行为修改内容。

[AgentDojo](https://arxiv.org/abs/2406.13352) 提供了工具 Agent 间接注入的动态环境。评测要同时测 benign task success、attack success 和防御引入的 utility loss。

## 正确性与失效

- **最终文本代替环境状态**：虚假完成。
- **HTTP 成功码代替业务终态**：部分失败被漏掉。
- **infra error 记模型失败**：能力与系统可靠性混淆。
- **infra error 从分母全删**：端到端可用性被高估。
- **只看平均成功**：同一任务多次不稳定被掩盖。
- **pass@$k$ 与 pass$^k$ 混用**：搜索与可靠性方向相反。
- **主任务成功掩盖越权**：安全失败未计。
- **重试不计成本/副作用**：最终成功被过度美化。
- **环境未 reset**：前一任务状态泄漏。
- **judge 读到攻击文本**：评分器也被注入。

## 何时不用完整 Agent benchmark

若模型只需输出一个无状态、可程序验证的函数参数，函数调用 benchmark 足够；没有工具和外部状态的文本任务不需要模拟复杂环境。完整 Agent harness 的成本应由真实状态、权限和长程交互需求驱动。

## 报告卡

```text
task family and target population
environment/tool/policy/harness revisions
initial/target/forbidden state definitions
model/template/sampling and budgets
task/trial/cluster counts
success/failure/infra/unknown denominators
pass@k and pass^k definitions
unauthorized side effects and severity
tokens/tool calls/latency/cost
injection threat model and adaptive budget
effect/cluster CI and known environment limits
```

安全攻击矩阵见[安全评测](safety-evaluation.md)，统计重采样见[统计推断](statistical-inference.md)，最小状态与 pass 指标实现见[评测工具](../practice/evaluation-tooling.md)。
