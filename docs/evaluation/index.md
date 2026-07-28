# 评测与可靠性

评测估计的是“某个可识别系统，在某个冻结分布和协议下的行为”。分数不是模型名的永久属性；checkpoint、模板、工具、推理预算、评分器或数据版本变化后，测量对象已经改变。

评测史反复在“覆盖更广”和“协议更可信”之间摆动。[评测体系的演进](../landscape/lineages/evaluation.md)从 perplexity、静态任务集走到 HELM 的多指标 scenario、动态人类偏好与可执行环境；[HELM、MT-Bench 与 Chatbot Arena](../landscape/works/helm-arena.md) 进一步拆解三个常被混在一起的对象：基准设计、模型裁判和在线成对比较。

## 评测对象

最小对象可写成

$$
\mathcal E
=
(M,D,P,H,J,R,T),
$$

其中：

- $M$：模型权重、tokenizer、adapter、量化与服务版本；
- $D$：数据 revision、split、时间窗和资产 digest；
- $P$：prompt、chat template、few-shot、解码与 parser；
- $H$：harness 代码 commit、依赖和运行环境；
- $J$：judge、verifier、人工 rubric 与阈值；
- $R$：检索、工具、权限和资源预算；
- $T$：执行与核验日期。

只有这一组合可复现，汇总分数才有稳定含义。[HELM](https://arxiv.org/abs/2211.09110) 把 scenario、adaptation、metric 与透明报告结合起来，是协议化评测的代表性框架。

## 四层评测

| 层 | 统计单位 | 主要结果 | 不能替代 |
| --- | --- | --- | --- |
| 模型 | token、题目、回答 | loss、正确、偏好、校准 | 工具与端到端状态 |
| 组件 | query、chunk、tool call | recall、rerank、parser、guard | 完整任务成功 |
| 系统 | request、episode | 终态、成本、时延、副作用 | 真实用户分布 |
| 运行 | 用户、会话、时间窗 | SLO、事故、漂移、业务结果 | 受控因果实验 |

同一请求可在模型层正确、系统层失败，例如答案正确但写入了错误对象；也可模型回答一般、工具系统仍通过确定性验证完成任务。

## 分母与失败状态

统一状态词表：

```text
correct / success
wrong / task failure
abstain
policy refusal
invalid output or action
timeout
environment failure
infrastructure error
unauthorized side effect
missing judgment
```

至少报告两个分母：

1. **能力口径**：在有效输入、环境和评分成功的样本上估计能力；
2. **端到端口径**：所有进入系统的样本都计入，反映真实可用性。

只排除超时和 infra error 会高估端到端表现；把环境故障全部算模型错误又无法诊断能力。正确做法是保留完整状态并同时报告条件指标与全量指标。

## 评测生命周期

1. **决策问题**：选模型、验证改动、设置回归门槛，还是估计线上风险。
2. **Estimand**：定义要估计的总体、统计单位、权重与指标。
3. **数据冻结**：记录版本、时间、污染风险和切分。
4. **协议冻结**：模型、模板、harness、工具、judge 与预算全部绑定 revision。
5. **预注册分析**：主指标、slice、缺失值、重复次数和比较方法先确定。
6. **执行与保存**：保留逐样本输入摘要、原始输出、状态、分数和时延。
7. **统计推断**：报告 effect、置信区间、功效与多重比较处理。
8. **错误审计**：抽查争议、失败、攻击和分布漂移。
9. **冻结报告卡**：使结果可重算、可桥接、可复核。

## 专题入口

### 协议与统计

- [语言模型评测协议](language-model-evaluation.md)：PPL、多选、生成、harness 和统一报告卡。
- [指标与评测设计](metrics.md)：estimand、分母、聚合与质量–成本边界。
- [统计推断](statistical-inference.md)：paired/cluster bootstrap、effect、power 和 multiple comparison。
- [校准与不确定性](calibration-uncertainty.md)：NLL、Brier、ECE、risk–coverage 和 semantic uncertainty。
- [Benchmark 注册表](benchmark-registry.md)：静态、动态、可执行和 Agent benchmark 的版本化入口。

### 生成、事实与污染

- [生成式评测与 LLM Judge](generative-judges.md)：rubric、pairwise、swap、BT 图与注入。
- [幻觉与事实性](hallucination.md)：atomic claim、support、completeness、freshness 与 abstention。
- [评测污染](contamination.md)：exact、释义、跨语种、时间窗和工具泄漏。
- [多模态评测](multimodal-evaluation.md)：感知、grounding、生成、Agent 与时延拆分。

### Agent、可靠性与安全

- [Agent 与工具评测](agent-tool-evaluation.md)：环境终态、pass@$k$/pass$^k$、成本和未授权副作用。
- [安全评测](safety-evaluation.md)：threat model、攻击成功、过度拒绝与效用前沿。
- [可靠性与安全](reliability-safety.md)：稳定入口与跨层失败地图。
- [指令遵循](instruction-following.md)：组合约束、优先级、格式与作用域。
- [生产可靠性](production-reliability.md)：SLO、降级、观测、canary 与事故闭环。

## 实现契约

推荐保存机器可读的逐样本记录：

```text
run ID and evaluation-card digest
item / cluster / slice IDs
model and harness revisions
input asset digests and protocol
raw output and parsed output
judge / verifier raw response and version
status, score, latency, tokens and tool cost
retry / seed / trial index
```

汇总表应由这些不可变记录重算，而不是只保存最终 CSV。评测工具的紧凑实现见[评测工具](../practice/evaluation-tooling.md)。

## 典型失效

- **榜单数值脱离协议**：同名 benchmark 在不同 harness 中并不可比。
- **题目行当成独立样本**：同一仓库、题族或用户的相关性使区间过窄。
- **只报显著性不报 effect**：统计可辨识但实际无意义。
- **反复看测试集调 prompt**：测试集退化为开发集。
- **judge 与候选顺序绑定**：位置偏置被误当模型差异。
- **安全分数只看拒绝**：无害任务全部拒绝也可能看似安全。
- **Agent 只评文本**：环境已发生未授权副作用却被漏掉。
- **动态 benchmark 不锁 revision**：历史结果无法重放。

## 最小报告卡

```text
decision question and target population
model/checkpoint/tokenizer/template/adapters
dataset revision, split, time window and asset digests
harness commit, dependencies and execution date
few-shot, decoding, tools, retrieval and budgets
parser/judge/verifier versions and thresholds
statistical unit, denominator and missing-value policy
sample/trial/cluster counts and slice weights
effect estimate, confidence interval and multiplicity handling
contamination and threat-model audit
raw-record location and known limitations
```

[DeepSeek](../landscape/families/deepseek.md)、[Kimi](../landscape/families/kimi.md) 与 [GLM](../landscape/families/glm.md) 家族页把模型、API、harness、工具预算与作者报告值分列，适合作为版本化评测的具体案例；跨家族结论仍须回到同一报告卡重跑。
