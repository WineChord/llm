# 生成式评测与 LLM Judge

开放式回答没有唯一参考答案。LLM judge 能按 rubric 扩展评分或 pairwise 比较，但它是一个有偏、可被攻击、会随版本变化的测量器，不是事实真值。

[HELM、MT-Bench 与 Chatbot Arena](../landscape/works/helm-arena.md) 区分多场景 benchmark、离线模型裁判与在线人类 pairwise 数据；[评测体系的演进](../landscape/lineages/evaluation.md)则解释评测为何从单一准确率逐步转向多指标、动态协议与可执行环境。

## 评测协议

先确定 judge 的任务：

| 类型 | 输出 | 适合 | 风险 |
| --- | --- | --- | --- |
| Pointwise | 单回答分项/总分 | 绝对 rubric | 分数尺度漂移 |
| Pairwise | A/B/tie | 模型比较 | 位置、对手和图结构 |
| Reference-guided | 与参考比对 | 有明确要点 | 参考不完整 |
| Evidence-grounded | claim–evidence 标签 | 事实与引用 | 检索和注入 |
| Critique | 错误类型与理由 | 调试 | 理由流畅不等于标签正确 |

冻结：

```text
judge model and exact revision
judge prompt/rubric/examples
candidate formatting and anonymization
temperature, seed and retries
reference/evidence/tool access
parser and invalid-output policy
human calibration set and execution date
```

[MT-Bench 与 Chatbot Arena](https://arxiv.org/abs/2306.05685) 展示了 LLM judge 与人类 pairwise 评测的代表性方案。其结论和 prompt 属于特定模型与时期，后续 judge 不能沿用历史一致性假设。

## 统计对象与分母

逐 judgment 保存：

```text
A win / B win / tie
swap-consistent / swap-inconsistent
invalid judge output
judge refusal
timeout / infrastructure error
human disagreement
```

至少报告：

- 有效 judgment coverage；
- win/tie/loss 分母；
- swap consistency；
- judge–human agreement；
- 各 rubric 维度；
- 按长度、格式、语言、模型家族和难度的 slice。

删除 swap-inconsistent 或 judge-invalid 样本会选择性排除难例；它们应单列，并同时给出 end-to-end 与有效子集结果。

## 候选顺序交换

同一 pair 运行两次：

```text
run 1: candidate A, candidate B
run 2: candidate B, candidate A
```

若第一次判 A 优、交换后仍判原 A 优，则一致。若两次都偏好“第一个位置”，就是 position inconsistency。聚合策略需预先定义：

- 只有一致时记 win，其余记 inconsistent；
- 两次一胜一平按软计数；
- 交给人工复核；
- 用多 judge 或重复采样投票。

不能看到模型结果后选择最有利规则。[Judging the Judges](https://arxiv.org/abs/2406.07791) 系统研究了 pairwise judge 的位置偏置与顺序一致性问题。

swap 聚合器必须先把“第一个/第二个”映射回原候选身份，再判断一致性。下面约定两次调用的输出只允许 `first`、`second` 或 `tie`，并保留 inconsistent 状态而不是静默删除。

```python
def aggregate_swapped_judgment(run_ab, run_ba):
    allowed = {"first", "second", "tie"}
    if run_ab not in allowed or run_ba not in allowed:
        raise ValueError("invalid judge label")
    map_ab = {"first": "A", "second": "B", "tie": "tie"}
    map_ba = {"first": "B", "second": "A", "tie": "tie"}
    original_ab = map_ab[run_ab]
    original_ba = map_ba[run_ba]
    if original_ab == original_ba:
        return {"status": "consistent", "winner": original_ab}
    return {"status": "inconsistent", "votes": (original_ab, original_ba)}
stable_a = aggregate_swapped_judgment("first", "second")
position_bias = aggregate_swapped_judgment("first", "first")
stable_tie = aggregate_swapped_judgment("tie", "tie")
assert stable_a == {"status": "consistent", "winner": "A"}
assert position_bias["status"] == "inconsistent" and position_bias["votes"] == ("A", "B")
assert stable_tie["winner"] == "tie"
```

这段代码不替系统决定如何给 inconsistent case 计分；人工复核、软计数或多 judge 投票必须预注册。生产评测还要保存 rubric、候选匿名化、随机种子和原始两次判定，并按长度、格式、模型家族与任务 cluster 报告 swap consistency。

## Pairwise 聚合与 BT 图

将模型 $i$ 的潜在 score 记为 $s_i$：

$$
P(i\succ j)=\sigma(s_i-s_j).
$$

Bradley–Terry 只识别 score 差值；所有 $s_i$ 加同一常数不改变概率。更重要的是，比较图必须连通：节点是模型，边是实际比较过的 pair。若图分成多个连通分量，各分量之间的 score offset 任意，无法得到全局排名。

聚合应报告：

- 每条边的样本数与任务分布；
- pairing graph 是否连通；
- 对手选择是否均衡；
- score/差值的 cluster bootstrap 区间；
- tie 模型与处理；
- 新模型加入后的 bridge edges。

Elo 等在线更新还依赖顺序和更新规则，不能只给最终数值而不保留全部 pair。

## Rubric 与人工校准

rubric 应把可分离维度拆开，例如：

```text
task correctness
factual support
completeness
instruction compliance
clarity / style
safety
```

先建立人工双盲子集，测：

- judge–human agreement；
- 人类之间 agreement；
- 各分项 confusion；
- 高/低分与 margin 区域；
- 长度、格式、语言和模型家族偏差。

人工 disagreement 不是必须消除的噪声；它可能表示 rubric 不清、任务主观或答案真正难分。此时报告分歧比分配一个伪精确标签更诚实。

## Judge 注入与对抗

候选回答、引用、网页和工具输出都可能包含：

- 要求 judge 忽略 rubric 的文本；
- 伪造“参考答案”或评分指令；
- 隐藏字符、编码和超长填充；
- 引用外观或自称高质量；
- 针对已知 judge 模板优化的措辞。

防护与评测：

1. 候选作为结构化不可信字段，而非拼接控制文本；
2. rubric 与候选边界使用固定 schema；
3. 对注入、顺序、身份、长度和格式做扰动；
4. 关键事实用独立 verifier，不让 judge 自行补证据；
5. 高影响样本人工复核；
6. 将攻击成功与普通评分准确分开报告。

即使 prompt isolation 改善，也不能假设 judge 对未知攻击鲁棒。

## 实现契约

```text
item/pair/cluster IDs
candidate generator/model revisions
raw and normalized candidates
order randomization and swap linkage
judge/rubric/parser revisions
raw judgment, rationale and parsed label
tie/invalid/inconsistent status
human audit and evidence
tokens, latency and cost
```

judge rationale 便于调试，但不能用自身理由证明标签正确。历史原始输出应保留，升级 parser 或 judge 后做 bridge。

## 正确性与失效

- **只跑一个顺序**：位置偏置无法发现。
- **图不连通仍给全局排名**：score offset 不可辨识。
- **对手分布不同直接比 win rate**：强弱对手混入模型分数。
- **judge 与候选同家族**：自偏好和风格偏好未审计。
- **长答案看似完整**：长度 shortcut。
- **引用数量当事实性**：未做 entailment。
- **无效 judgment 从分母删除**：难例被排除。
- **judge 升级覆盖旧结果**：历史曲线断裂。
- **候选注入控制 judge**：评分器成为攻击目标。
- **自动分数替代人工金标**：测量误差未知。

## 何时不用 LLM Judge

有可执行 verifier、精确数据库终态、形式规则或高风险领域专家判断时，应优先使用它们。样本较小、差异细微且会影响重大决策时，人工双盲通常比堆叠多个自动 judge 更可解释。主观创作可以用人类偏好，不必伪装成客观绝对分。

## 报告卡

```text
judge task and rubric dimensions
judge model/prompt/parser revisions and date
candidate anonymization and formatting
pairing design and graph connectivity
swap/repeat protocol and inconsistency policy
valid/tie/invalid/missing denominators
human calibration and agreement
effect/CI by cluster and critical slices
injection/length/style/family attacks
tokens, latency, cost and known limits
```

事实 claim 的 judge 口径见[幻觉与事实性](hallucination.md)，统计聚合见[统计推断](statistical-inference.md)，最小 swap/BT 工具见[评测工具](../practice/evaluation-tooling.md)。

## GLM-5：judge 也必须固定为评测依赖 {#glm-judge}

GLM-5 的搜索评测发现 BrowseComp 对 judge prompt 与 judge model 敏感，最终统一采用 OpenAI 官方 prompt 与 o3-mini。这个选择提高同一实验内部的一致性，却不能让分数跨 judge 自动可比；judge 是 harness 的一部分，必须与模型、采样预算和页面快照一起冻结。

CC-Bench-V2 又展示了另一种 judge：模型主动运行网页、检查 DOM 与视觉结果。它比只读截图拥有更多证据，也多了浏览器版本、工具权限、交互顺序与错误恢复等变量。报告的内部一致率和排序相关性应作为校准证据保存，不能替代公开任务、样本级 verdict 和人工复核。

## Reference {#reference}

- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [Judging the Judges](https://arxiv.org/abs/2406.07791)
- [BrowseComp](https://arxiv.org/abs/2504.12516)
- [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763)
