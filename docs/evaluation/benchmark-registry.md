# Benchmark 注册表

Benchmark 注册表不只是链接清单，而是可执行协议的索引。每个条目都应回答测量对象、数据 revision、harness commit、评分器、预算、污染风险和结果可比范围。

## 注册对象

一个 benchmark entry 至少包含：

```text
canonical name and primary source
release date and current dataset revision
task / language / modality / domain scope
statistical unit and grouped split
input/output contract
official or selected harness commit
prompt / few-shot / scoring / parser
tools, retrieval and generation budget
judge / verifier and failure statuses
known contamination and saturation risks
license / access and last verification date
```

论文发布日期与实际运行版本是两个字段。持续更新的 benchmark 不能只写年份；静态 benchmark 也会因 harness 与 parser 更新而改变。

## 基础与静态基准

这些基准适合建立长期 bridge，但容易被训练数据污染或随能力提升饱和：

| 基准 | 初始时间 | 测量对象 | 协议重点 |
| --- | --- | --- | --- |
| [MMLU](https://arxiv.org/abs/2009.03300) | 2020 | 多学科多选 | 标签/选项概率、few-shot |
| [HumanEval](https://arxiv.org/abs/2107.03374) | 2021 | 代码生成 | pass@$k$、测试与超时 |
| [HELM](https://arxiv.org/abs/2211.09110) | 2022 | 多 scenario 评测框架 | adaptation、metric、透明报告 |
| [GPQA](https://arxiv.org/abs/2311.12022) | 2023 | 高难科学问答 | closed/open book、专家难度 |
| [IFEval](https://arxiv.org/abs/2311.07911) | 2023 | 可验证指令遵循 | strict/loose、verifier |
| [PALOMA](https://arxiv.org/abs/2312.10523) | 2023 | 跨域语言模型 fit | tokenizer、PPL、decontamination |
| [MMMU](https://arxiv.org/abs/2311.16502) | 2023 | 多学科多模态理解 | 图片预处理、文本/视觉依赖 |

“静态”只表示题目集合相对冻结，不表示结果永久可比。model template、few-shot、normalization 和 harness task definition 仍需锁定。

HELM 怎样把 scenario、adaptation 与多指标报告组成协议，以及它与后来的模型裁判和动态人类偏好有什么边界，见 [HELM、MT-Bench 与 Chatbot Arena 深读](../landscape/works/helm-arena.md)。

## 动态与时间切分基准

动态 benchmark 试图降低公开答案污染和能力饱和，但版本管理更重要：

| 基准 | 初始时间 | 机制 | 必须冻结 |
| --- | --- | --- | --- |
| [LiveCodeBench](https://arxiv.org/abs/2403.07974) | 2024 | 按时间更新代码题 | release window、测试、语言与时间 |
| [LiveBench](https://arxiv.org/abs/2406.19314) | 2024 | 持续加入新题 | snapshot、类别、judge/verifier |
| [BFCL](https://gorilla.cs.berkeley.edu/leaderboard) | 持续更新 | 函数调用与格式 | data/version、AST/checker、工具 schema |

BFCL 的[官方代码与数据入口](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)应与运行 commit 一起记录。动态更新降低的是某些历史泄漏，不会自动排除 teacher、检索、公开讨论或开发期反复查看造成的污染。

## 代码与软件工程

| 基准 | 初始时间 | 统计单位 | 关键状态 |
| --- | --- | --- | --- |
| [SWE-bench](https://arxiv.org/abs/2310.06770) | 2023 | repository issue | patch apply、tests、timeout、infra |
| [SWE-agent](https://arxiv.org/abs/2405.15793) | 2024 | Agent trajectory / issue | interface、tools、cost、terminal state |

同一 issue 的多个变体、同一 repository 的多个 issue 相关，统计区间应以 repository 或更高 cluster 重采样。容器、依赖、测试补丁和网络策略也是 benchmark revision。

## Agent 与环境

| 基准 | 初始时间 | 主要对象 | 协议重点 |
| --- | --- | --- | --- |
| [GAIA](https://arxiv.org/abs/2311.12983) | 2023 | 工具辅助现实问题 | 工具/网络、答案 verifier |
| [AgentBench](https://arxiv.org/abs/2308.03688) | 2023 | 多环境 Agent | 环境版本、交互预算 |
| [WebArena](https://arxiv.org/abs/2307.13854) | 2023 | 网站任务 | 初始/目标状态、网站 snapshot |
| [OSWorld](https://arxiv.org/abs/2404.07972) | 2024 | 真实电脑环境 | 截图、动作、VM state、side effect |
| [$\tau$-bench](https://arxiv.org/abs/2406.12045) | 2024 | tool–agent–user | 数据库终态、policy、pass$^k$ |

Agent benchmark 的文本回答只是轨迹的一部分。任务是否完成要读取环境终态；未授权写入、发送或删除即使不影响主目标，也必须作为安全结果记录。

## 生成、事实与 Judge

| 基准/方法 | 初始时间 | 对象 | 限制 |
| --- | --- | --- | --- |
| [MT-Bench / Chatbot Arena](https://arxiv.org/abs/2306.05685) | 2023 | 对话与 pairwise 偏好 | judge/人群/对手分布 |
| [TruthfulQA](https://arxiv.org/abs/2109.07958) | 2021 | 常见误解与真实回答 | 固定题集污染 |
| [FActScore](https://arxiv.org/abs/2305.14251) | 2023 | 长文本 atomic support | claim/retrieval/judge 误差 |
| [LongFact / SAFE](https://arxiv.org/abs/2403.18802) | 2024 | 开放域长文本事实性 | 搜索 snapshot 与 evaluator |

这些方法不能合成一个“通用生成质量”。事实支持、完整性、风格和用户偏好属于不同 estimand。

MT-Bench 的模型裁判与 Arena 的在线人类 pairwise 数据也不是同一测量器；二者的桥接关系见 [HELM、MT-Bench 与 Chatbot Arena](../landscape/works/helm-arena.md)。

## 安全与攻击

安全 benchmark 的攻击强度和 benign utility 必须同时注册：

| 基准 | 初始时间 | 主要对象 |
| --- | --- | --- |
| [XSTest](https://arxiv.org/abs/2308.01263) | 2023 | 过度拒绝 |
| [HarmBench](https://arxiv.org/abs/2402.04249) | 2024 | 自动红队与有害行为 |
| [StrongREJECT](https://arxiv.org/abs/2402.10260) | 2024 | jailbreak 响应质量 |
| [JailbreakBench](https://arxiv.org/abs/2404.01318) | 2024 | attack/defense 可复现比较 |
| [AgentDojo](https://arxiv.org/abs/2406.13352) | 2024 | Agent 间接 prompt injection |

安全结果依赖模型、guard、系统 prompt、工具权限、攻击预算和 evaluator revision；裸模型分数不能直接代表部署系统。

## Harness 契约

统一运行记录：

```text
benchmark entry version
dataset revision and asset digests
harness repository commit
task/config/parser/checker digests
model and template revision
environment/container/tool revisions
judge/verifier versions
execution date and dynamic snapshot
```

当官方协议与常用 harness 不同时，选择一种作为主协议，并用 bridge run 同时运行二者。不能把两个协议结果拼进同一历史曲线。

### Harness selection 也是实验因素

Agent 模型常在不同 scaffold 下运行同一 benchmark。除了 harness commit，还应注册：

```text
harness candidate set and selection rule
system prompt, tools, skills and compactor revisions
effort condition and maximum context
temperature, top-p and retry/sample counts
network, package and submission permissions
hardware and environment image
whether the reported result is fixed-harness, best-of-harness or leaderboard
```

若先看测试结果再挑得分最高的 harness，这相当于对 benchmark 调参；应报告所有候选或在独立 development split 选择后冻结。[Kimi K3 技术报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)是一项适合审计的实例：报告通常采用 max effort，单步任务使用 temperature $1$/top-$p=0.95$，agentic 任务使用 temperature $1$/top-$p=1.0$；不同表格混有 Kimi Code、Claude Code、Codex、leaderboard、best-of-harness 和内部结果。完整逐表边界见 [Kimi K3](../landscape/works/kimi-k3.md)。

硬件与 runtime 也可能进入 benchmark definition。报告中的 SWE Marathon 使用为 H20 校准的 pre-final v1.1 branch，并同时列出 PostTrain-H20 与官方 H100 结果；这些数字不应拼成纯 checkpoint 排名。BrowseComp 的 $91.2$（300K compaction）与 $90.4$（完整 1M、无 context management）则说明 compactor 是系统变量：差值不是上下文窗口或模型权重的单因素估计。无法公开重建的数据集应标记 `internal / non-reproducible`，而不是与公开 benchmark 共用证据等级。

## 正确性与失效

- **用论文年份代替数据 revision**：动态集无法重放。
- **榜单名代替 task config**：few-shot、parser 和 budget 不明。
- **静态题集当长期未知测试**：训练与开发污染累积。
- **Agent 只看最终文本**：环境终态和副作用缺失。
- **不同 benchmark 求简单平均**：样本数、难度与目标含义不一致。
- **最新 benchmark 默认更好**：新题透明度、稳定性和统计功效可能不足。
- **官方实现 floating branch**：依赖更新后历史结果改变。

## 何时不应新增 benchmark

若新集合没有独立数据来源、清晰协议、有效 verifier 或相对现有集的新覆盖轴，应优先扩展已有 benchmark 的 slice，而不是制造另一个汇总分。内部回归集可以不公开，但仍需相同的版本、分母和污染治理。

## 注册验收

1. 从空环境按记录 commit 重建一次完整运行。
2. 手工复核随机样本的 prompt、输出、parser 和 verifier。
3. 对 invalid、timeout、infra 和缺失 judgment 做故障注入。
4. 检查题族、repository、环境与用户 cluster。
5. 记录 exact、释义、跨语种和时间污染。
6. 与上一 revision 做 bridge，区分模型变化和 benchmark 变化。

统计聚合见[指标与评测设计](metrics.md)，污染审计见[评测污染](contamination.md)，逐样本记录实现见[评测工具](../practice/evaluation-tooling.md)。

## Reference {#reference}

- [MMLU](https://arxiv.org/abs/2009.03300)
- [Evaluating Large Language Models Trained on Code / HumanEval](https://arxiv.org/abs/2107.03374)
- [Holistic Evaluation of Language Models / HELM](https://arxiv.org/abs/2211.09110)
- [GPQA](https://arxiv.org/abs/2311.12022)
- [Instruction-Following Evaluation for Large Language Models / IFEval](https://arxiv.org/abs/2311.07911)
- [PALOMA: A Benchmark for Evaluating Language Model Fit](https://arxiv.org/abs/2312.10523)
- [MMMU](https://arxiv.org/abs/2311.16502)
- [LiveCodeBench](https://arxiv.org/abs/2403.07974)
- [Kimi K3 Technical Report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
