# 智能体安全

智能体把不可信文本、模型决策和有权限工具连接在一起，安全问题因此从“生成了不当文本”扩展到真实副作用。核心原则是：数据不能自行升级为指令，模型不能自行扩大权限，工具返回不能自行证明成功。

## 信任边界

```text
user / document / web / email
            ↓ untrusted data
     context builder ── policy
            ↓
           model
            ↓ candidate action
   validator + authorization
            ↓
     sandboxed tool/service
            ↓ untrusted result
    verifier + state update
```

每条边都需要类型、来源和权限。仅在 prompt 中写“忽略恶意指令”不能形成隔离。

## Prompt injection

直接注入来自用户输入，间接注入来自网页、邮件、文档、代码注释或工具结果。[OWASP Prompt Injection](https://owasp.org/www-community/attacks/PromptInjection)总结了基本攻击面；[InjecAgent](https://arxiv.org/abs/2403.02691)与 [AgentDojo](https://arxiv.org/abs/2406.13352)提供了面向工具智能体的攻击与评测环境。

防线应组合：

- 系统指令、工具 schema 与外部内容分区；
- 给外部内容打来源与信任标签；
- 最小化进入上下文的数据；
- 对候选动作做独立策略校验；
- 高风险动作使用规范化参数审批；
- 输出和工具结果做数据泄露检查；
- 对注入回归集持续测试。

任何单一分类器都有漏报和误报；权限边界不能依赖模型先正确识别注入。

## Capability security

工具访问采用能力而非全局万能凭据：

$$
\operatorname{capability}
=(\text{subject},\text{resource},\text{actions},
\text{constraints},\text{expiry}).
$$

能力应短期、最小范围、可撤销并绑定任务。读文件工具不应顺带拥有网络发送能力；浏览器读取会话与提交支付应是不同能力。

工具注册也要最小化。模型看不到的工具通常比“看得到但被提醒不要用”更安全。

## 信息流

为敏感数据标记来源与允许流向：

- public：可公开输出；
- internal：仅当前系统处理；
- confidential：仅指定主体与工具；
- secret：不进入模型或日志，只由受控执行层引用。

候选输出的策略不仅检查内容，还检查流向。例如读取私人邮件后把摘要发送到公共 webhook，是跨边界泄露，即使摘要没有明显密钥。

可将外发判断写成：

$$
\operatorname{allow}(d,\text{sink})
=\bigwedge_i
\operatorname{policy}(\operatorname{label}(d_i),\text{sink}),
$$

其中 $d_i$ 是输出依赖的数据。真实系统可使用 taint tracking、字段级策略或明确的数据引用来近似实现。

## Sandbox

代码、浏览器和文件工具至少限制：

- 文件系统根与写入范围；
- 网络目的地和协议；
- 环境变量与凭据可见性；
- CPU、内存、时间和进程数；
- 系统调用、设备和容器逃逸面；
- 下载文件的类型、大小和后续处理。

沙箱降低影响范围，不证明输入安全。工具链中的解析器、浏览器、编译器和依赖仍需要补丁与供应链治理。

## 高风险动作

删除、支付、发布、外部通信、权限变化和大范围覆盖应经过：

1. 资源重新解析；
2. 规范化参数预览；
3. 当前状态读取；
4. 明确授权或审批；
5. 幂等执行；
6. 独立终态验证；
7. 可行时提供回滚或补偿。

审批必须发生在最后参数确定之后。模型在审批后改变参数，应使批准失效。

## 结果不可信

工具成功响应可能：

- 只代表请求被接收；
- 返回缓存或旧状态；
- 部分对象成功；
- 被中间层伪造或截断；
- 包含来自网页的注入文本。

运行时把结果分成 data 与 control。只有预定义状态字段可以驱动状态机，自由文本永远不能要求提升权限或绕过验证。

## 供应链

工具描述、MCP server、插件、依赖和远程 prompt 模板都是供应链对象。至少记录版本、来源、签名或哈希、权限清单与更新策略。新增工具等同于新增攻击面，应经过静态审查、沙箱测试和最小权限配置。

## 安全评测

[ToolEmu](https://arxiv.org/abs/2309.15817)研究用模拟器测试工具智能体风险。评测集要覆盖：

- 直接与间接 prompt injection；
- 数据窃取和跨租户泄露；
- 权限升级与 confused deputy；
- 高风险动作诱导；
- 工具结果伪造；
- 长任务中的记忆投毒；
- 多智能体消息中的身份冒充；
- 拒绝后换措辞、分步或跨工具绕过。

指标同时报告攻击成功率、正常任务成功率和过度拒绝。只把攻击成功率压低但令正常任务不可用，不是完整方案。

## 事故响应

发现异常时应能：

- 立即撤销能力与停止任务；
- 标记 unknown side effects 并对账；
- 隔离受污染记忆、缓存与索引；
- 保全事件、工具版本和证据；
- 确定受影响主体与数据流向；
- 修复策略后重放无副作用评测。

运行时状态与恢复见[智能体运行时](agent-runtime.md)，工具权限与幂等见[工具调用](tool-use.md)，生产门禁见[可靠性与安全](../evaluation/reliability-safety.md)。
