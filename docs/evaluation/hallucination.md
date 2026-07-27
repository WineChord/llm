# 幻觉与事实性

“幻觉”通常指输出流畅，却与输入证据、可验证事实、任务约束或真实工具状态不一致。它不是单一故障：事实错误、引用错误、时间过期、推理错误和状态误报需要不同分母、证据与修复。

## 评测协议

先冻结判定参照：

| 类型 | 参照 | 示例 |
| --- | --- | --- |
| 内在一致性 | 给定上下文 | 摘要改变原文数字 |
| 外部事实性 | 指定权威来源集合 | 编造不存在的论文 |
| 引用支持 | 被引用页面与主张 | 链接存在但不支持结论 |
| 完整性 | 题目要求或参考事实集 | 漏掉关键限制 |
| 时效性 | 查询时间与对象版本 | 用旧任职信息回答当前问题 |
| 推理/计算 | 规则、程序或 verifier | 中间计算与答案矛盾 |
| 工具状态 | 实际环境终态 | 声称已保存但对象未变化 |
| 多模态 grounding | 图片、音频、视频 | 将画面外对象写进描述 |

开放域评测还需冻结检索源、索引时间、网络访问和证据质量规则。若来源集合本身不完整，结果只能表示“在该证据范围内未找到支持”，不能直接判定世界事实为假。

## 统计对象：原子主张

将回答拆成原子主张

$$
C(y)=\{c_1,\ldots,c_m\}.
$$

每个 $c_i$ 关联零个或多个证据，并标为：

```text
supported
contradicted
not supported
unknown / insufficient evidence
stale for the requested time
not factual / not applicable
```

[FActScore](https://arxiv.org/abs/2305.14251) 提出将长文本分解为 atomic facts，并计算可靠知识源支持的比例。[LongFact / SAFE](https://arxiv.org/abs/2403.18802) 进一步研究搜索增强的长文本事实评测。两者提供可复用框架，但 claim extraction、检索和 judge 都是可能出错的测量组件。

## 分母与指标

设 $S$、$U$、$K$ 分别为 supported、unsupported/contradicted 与 unknown 的可判定计数，$A=S+U+K$ 为全部事实主张。

### Support precision

$$
\operatorname{support\ precision}
=\frac{S}{S+U}.
$$

unknown 不应静默从报告中消失，因此同时报告可判定率：

$$
\operatorname{decidable\ rate}
=\frac{S+U}{A}.
$$

若将 unknown 一律算 unsupported，会把检索覆盖不足混入生成事实性；若全部排除，又会奖励无法验证的主张。

### Completeness

有独立参考需求或事实集 $G=\{g_1,\ldots,g_q\}$ 时：

$$
\operatorname{completeness}
=
\frac{|\{g\in G:\text{answer covers }g\}|}{|G|}.
$$

没有近似完整的 $G$ 时，不应把回答长度或 claim 数称为 recall。支持率高的短回答可能极不完整，因此 support precision 与 completeness 必须分开。

### Freshness

对有时间要求的主张，记录证据时间 $t_e$、对象版本 $v_e$ 与查询截止 $t_q$。freshness 不是“网页能打开”，而是证据是否在目标时间窗内支持同一对象和版本。至少报告：

- 有时间要求的 claim 数；
- 具备可核验证据时间的比例；
- stale claim 比例；
- 未披露更新时间的 unknown 比例。

### 引用正确性

引用评测分开：

1. citation validity：目标可访问且对象正确；
2. citation entailment：证据支持主张；
3. citation completeness：需要证据的 claim 是否都覆盖；
4. source quality：来源是否适合该主张；
5. source freshness：时间与版本是否匹配。

“有引用”不等于“引用支持”，引用多也不等于事实完整。

## 检测机制

### 有参考或 verifier

exact match、结构化比较、计算器、编译器、测试和数据库状态适合确定性任务。parser 应保留 `invalid`、`timeout` 和 `infra error`，不能全部归入事实错误。

### 多次采样

[SelfCheckGPT](https://arxiv.org/abs/2303.08896) 用多次采样的一致性检测可能不可靠的陈述。直觉为

$$
\operatorname{risk}(c)\uparrow
\quad\text{when}\quad
\operatorname{semantic\ disagreement}
(c_1,\ldots,c_k)\uparrow.
$$

表面字符串差异不是语义冲突；多个样本也可能稳定复述同一错误。一致性是 uncertainty signal，不是事实 verifier。

### 检索增强

[RAG](https://arxiv.org/abs/2005.11401) 将外部文档带入生成。端到端错误可按阶段审计：

$$
P(E)
\le
P(E_{\text{retrieve}})
+P(E_{\text{read}}\mid \neg E_{\text{retrieve}})
+P(E_{\text{generate}}\mid \text{usable evidence}).
$$

这是错误上界式的诊断分解，不假设事件独立。应分别测 retrieval recall、evidence selection、entailment 和有证据时的 generation。

## 实现契约

逐 claim 保存：

```text
answer and claim IDs
claim extractor/version/span
query and retrieved evidence IDs
source URL/object/version/timestamps
entailment label, judge/verifier version and confidence
supported/contradicted/unknown/stale status
human audit and disagreement
```

网页内容会变化，证据需保存 digest 或合法可重放 snapshot。judge 升级后应在同一 claim/evidence 上 bridge，而不是覆盖旧标签。

## 缓解分层

### 数据与训练

- 去重、时间标注与来源质量；
- 反例和“证据不足”行为；
- 引用、工具状态与长上下文监督；
- 区分事实、推断和未知。

### 推理与工具

- 先拆主张，再逐条检索和验证；
- 数字、代码和状态使用确定性工具；
- 允许澄清、abstain 和人工升级；
- 写操作后读取真实对象，不以文本声明为完成证据。

### 系统

- 来源、时间和 ACL 随 chunk 传播；
- 检索空结果与冲突证据显式进入模型状态；
- 高风险输出需要独立 verifier；
- 监控支持率、unknown、stale、拒答和 coverage，而不是只看接受率。

## 正确性与攻击失效

- **claim 拆得越细分数越高/低**：原子化规则改变分母。
- **只检索支持证据**：confirmation bias，未搜索反证。
- **unknown 被删除**：检索失败样本从分母消失。
- **引用外观欺骗 judge**：链接格式被当作支持。
- **旧网页覆盖新事实**：freshness 与对象版本未核对。
- **同一模型生成并评判**：共享偏差和记忆。
- **检索文档含 prompt injection**：评测器或 Agent 被不可信文本劫持。
- **无限拒答降低错误率**：coverage 和 usefulness 崩溃。
- **工具返回成功码即视为成功**：业务终态未重读。

## 何时不用自动事实分数

法律、医疗、金融、实时政治、复杂科学争议和高影响状态变化，不能只依赖自动 claim/judge 总分。应使用领域专家、权威原始来源和明确时间边界。纯创作、意见或不可证伪陈述也不应被强行转换为事实 claim。

## 验证与报告卡

1. 在人工双盲子集上测 claim extraction、retrieval 和 entailment 各自误差。
2. 按 claim 数、领域、来源、时间、回答长度和可回答性分层。
3. 同时报 support precision、decidable rate、completeness、freshness 和 refusal。
4. 对 supporting/contradicting evidence 顺序交换，检查 judge 稳定性。
5. 用过期、无关、伪引用和注入文档做攻击测试。
6. 记录检索和 judge 预算，避免更大搜索成本被写成模型事实能力。

```text
target domain and reference/source policy
model, retrieval index and execution date
claim extraction and atomicity protocol
evidence snapshot and source-quality rules
support/unknown/stale denominators
completeness reference set
judge/verifier/human audit and confidence intervals
coverage, refusal, latency and cost
known source gaps and contamination
```

概率与 abstention 见[校准与不确定性](calibration-uncertainty.md)，引用与 judge 攻击见[生成式评测与 LLM Judge](generative-judges.md)，实现入口见[评测工具](../practice/evaluation-tooling.md)。
