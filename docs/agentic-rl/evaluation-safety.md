# 评测与安全

Agentic RL 同时改变能力和行为边界。评测不仅要问“能否完成任务”，还要问“使用了什么权限、付出多少成本、是否通过投机获得奖励”。

## 评测分解

整体成功率可拆成

$$
P(S)=P(\text{understand})
P(\text{plan}\mid U)
P(\text{execute}\mid U,P)
P(\text{verify}\mid U,P,E).
$$

这个分解不是独立性假设，而是诊断框架：理解、规划、执行和验证应有各自可观察信号。

## 核心维度

| 维度 | 指标示例 |
| --- | --- |
| 任务结果 | pass rate、goal-state accuracy |
| 样本效率 | 每题 rollout 数、训练环境步 |
| 推理成本 | token、工具调用、GPU 秒、费用 |
| 时间 | p50/p95 完成时长、超时率 |
| 鲁棒性 | seed、扰动、工具错误下的成功率 |
| 恢复 | 检测失败、回滚和重规划率 |
| 范围控制 | 越权操作、无关修改、破坏性动作 |
| 可审计性 | 关键结论能否指向证据 |

报告 `pass@k` 时还要报告总预算。允许无限采样的高通过率不能与单次尝试直接比较。

## Harness 与模型解耦

至少设置三类对照：

1. 同一模型，不同 harness；
2. 不同模型，同一 harness 与预算；
3. 去掉检索、工具、反思或并行 agent 的 ablation。

否则系统改进可能被误写成 checkpoint 改进。模型、提示、工具 schema、环境镜像和 verifier 都要固定版本。

对 white-box harness，还应逐组件注册 system prompt、context management、skills、memory、subagent 与工具实现，并提前声明选择规则。若同一模型试过多个 scaffold 后只报告最佳结果，结果是 model–harness search 的上界，不是固定系统的无偏估计。[Kimi K3 技术报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)披露了一套可组合多种 coding/agent scaffold 的训练与评测框架；其通用审计方式与具体结果见 [Kimi K3](../landscape/works/kimi-k3.md)。

## 终态任务与隐藏验证

Agent Evaluation Task（AET）不向模型提供标准动作路径，只给初始环境、受约束目标、工具集合和预算，最终由独立 verifier 读取环境终态。评测协议至少要包含：

```text
initial state and immutable task ID
allowed actions, permissions and budget
public feedback and hidden verifier
submission limit and retry semantics
terminal state, side effects and cost
verifier isolation and audit revision
```

公开测试帮助 agent 调试，隐藏测试和提交上限限制对 verifier 的穷举适配。Kernel 任务还需用随机 shape、独立计时、数值阈值和 CUDA Graph replay 检测输入缓存、降精度或计时投机；助理环境则需同时检查邮件、知识库、消息等多个应用终态。只看 agent 自报“完成”不构成通过。

## Reward Hacking

常见模式包括：

- 修改测试或评分文件；
- 利用 parser 漏洞输出特殊格式；
- 触发 timeout 或环境错误绕过失败；
- 用冗长过程骗取过程奖励；
- 从缓存、日志或文件名读取答案；
- 完成代理指标而非真实目标。

防御措施：

- verifier 与 agent 权限隔离；
- 隐藏测试和多重验证器；
- 对基础设施错误单独编码；
- 对奖励突增进行轨迹审计；
- 定期更换表面形式，保持语义目标；
- 用负向对抗样本测试评分器。

## 数据污染

真实仓库 benchmark 容易因公开补丁进入训练语料而污染。仅检查文本重合不够，还应检查：

- fork、镜像与重写补丁；
- issue 描述和 commit message；
- 测试用例与最终代码的语义等价；
- teacher model 是否访问过答案；
- 检索工具是否可直接读取未来 commit。

时间切分是一种缓解，不是证明无污染。

## 权限与副作用

把动作按风险分级：

| 级别 | 示例 | 控制 |
| --- | --- | --- |
| 只读 | 搜索、读取、静态分析 | 作用域限制与审计 |
| 可逆写入 | 工作区编辑、草稿 | diff、snapshot、回滚 |
| 外部变更 | push、发信、部署 | 明确授权、对象确认 |
| 高风险或不可逆 | 删除、资金、生产权限 | 强确认、最小权限、隔离 |

训练环境不应奖励现实世界中的未授权副作用。模拟器中的“成功发送”也不能直接迁移为生产权限。

## 多智能体风险

并行 agent 增加：

- 指令与状态不一致；
- 写冲突和重复操作；
- 一个 agent 将不可信内容传给另一个；
- 合并者无法验证全部局部结论；
- 总调用量和攻击面扩张。

需要明确所有权、通信格式、合并验证和全局停止条件。子任务完成并不自动代表根任务完成。

## 安全训练与能力评测

拒绝率不是唯一安全指标。应同时测：

- 对合法任务的过度拒绝；
- 对混淆、编码和多模态注入的鲁棒性；
- 工具输出中的不可信指令；
- 敏感信息最小暴露；
- 权限提升请求是否被正确处理；
- 在长轨迹中是否逐步偏离原始约束。

通用可靠性见[可靠性与安全](../evaluation/reliability-safety.md)，环境设计见[数据与环境](data-environments.md)。

Cyber agent 评测尤其要分开 exploit puzzle、多步网络环境、真实软件候选发现和部署 guard。[UK AISI/CAISI 对 Kimi K3 的初步评估](https://www.aisi.gov.uk/blog/preliminary-assessment-of-kimi-k3s-cyber-capabilities)显示同一模型在这些层级上的结果并不一致：ExploitBench 为 $32\%$，ACE 为 $0/41$，10 个更现实任务中完成 1 个。该结论只适用于评估时的模型入口、harness、预算和日期；它既不能由较强 puzzle 表现推出真实环境全面成功，也不能把 guard 未阻断动作等同于 exploit 已完成。更完整的 threat-model 记录见[安全评测](../evaluation/safety-evaluation.md#cyber-evidence-card)。

## Reference {#reference}

- [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565)
- [ToolEmu: Identifying the Risks of LM Agents with an LM-Emulated Sandbox](https://arxiv.org/abs/2309.15817)
- [AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents](https://arxiv.org/abs/2406.13352)
- [NIST AI 600-1: Generative Artificial Intelligence Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence)
- [UK AISI/CAISI Preliminary Assessment of Kimi K3 Cyber Capabilities](https://www.aisi.gov.uk/blog/preliminary-assessment-of-kimi-k3s-cyber-capabilities)
- [Kimi K3 Technical Report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
