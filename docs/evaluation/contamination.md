# 评测污染

评测污染发生在模型、teacher、检索、工具、prompt 开发或评分器接触了本应独立的测试信息。内部去重只能减少训练语料重复，不能证明 benchmark、答案和同题变体没有泄漏。

## 协议边界

先列出所有可能访问测试资产的组件：

```text
pretraining / SFT / preference / synthetic data
teacher and data-generation models
prompt, few-shot and checkpoint selection
retrieval corpus and search engine
tools, public repositories and solution pages
harness, parser, tests and hidden-state verifier
judge / reward / safety evaluator
human annotators and debugging logs
```

污染不是只有“权重见过题目”。如果 Agent 能搜索公开答案、teacher 用测试题生成训练解释，或开发者反复查看失败样本调 prompt，最终分数都不再是原先的独立估计。

## 统计对象与状态

以 benchmark item family 为单位，记录匹配层级：

```text
no detected match
exact text/code match
near duplicate
paraphrase / structural variant candidate
cross-lingual candidate
answer / explanation / test leakage
source or time overlap only
unknown because source inventory is incomplete
```

`no detected match` 只表示在给定索引和方法下未找到，不等于证明无污染。语义候选也不是污染真值，需要来源、时间和人工复核。

## Exact 与 near match

### Exact

对明确 normalization $f$：

$$
h(d)=H(f(d)).
$$

规范化版本必须记录。删除所有标点、数字或变量名可能把不同事实和代码错误地视作相同。

### Shingle 与 Jaccard

将文本映射为 token/shingle 集合 $S(d)$：

$$
J(a,b)
=
\frac{|S(a)\cap S(b)|}
{|S(a)\cup S(b)|}.
$$

MinHash/LSH 可召回候选，再用精确 Jaccard、长度比和结构确认。[Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) 研究了语料重复、记忆与评测的关系，并提供[官方实现](https://github.com/google-research/deduplicate-text-datasets)。

代码还需比较：

- 去注释和格式后的 token；
- identifier rename；
- AST 或测试结构；
- problem statement 与 solution；
- repository commit 和生成补丁。

Near match 阈值应按自然语言、代码、数学和多模态 caption 分开校准。

## 释义、结构与跨语种

污染可能保留解法结构而改变表面：

- 题干改写、选项重排；
- 数字、实体或变量替换；
- 翻译与回译；
- 解释先泄漏，题干后生成；
- 同源题库或模板实例；
- 图像中的同题文字与文字版；
- 代码测试不变而 issue 描述改写。

[Investigating Data Contamination in Modern Benchmarks](https://arxiv.org/abs/2311.04850) 研究了释义后的 benchmark 内容对评测的影响。embedding 或 LLM 相似度适合召回候选，不应单独作为删除或污染判决；跨语种还需要对齐实体、答案与题目结构。

## 答案、解释与 harness 泄漏

只搜索题干会漏掉：

- 参考答案与解析；
- 多选选项和标签；
- unit tests、hidden test 名称或期望输出；
- benchmark repository issue、commit 和 README；
- parser 正则与 scorer loophole；
- judge rubric 中的答案要点。

Agent 评测尤其要冻结网络和文件访问。如果工具能读取 tests、solution branch 或目标数据库状态，任务已从推理变成检索或特权访问。

## Teacher 与合成数据

teacher 可能在其预训练中见过测试答案，也可能通过联网检索访问。使用 benchmark prompt 让 teacher 生成 rationale、变体或偏好数据，会把测试信息显式带入 student。

合成数据需记录：

```text
parent item family
teacher/version/access policy
generation time and specification
verifier/test access
derived prompts/answers/explanations
train/dev/test destination
```

先按题族和时间冻结 split，再在 train side 生成。先生成变体后随机切分会产生跨 split 泄漏。

## 时间窗

动态 benchmark 的核心不是“最新”，而是明确时间因果：

```text
item creation time
first public availability
data/model training cutoff
teacher and retrieval snapshot
evaluation execution time
```

一个模型训练截止早于题目公开时间，仍可能通过更新的检索索引或工具看到答案。反之，网页发布时间晚也不能证明底层题目此前从未出现。

[LiveCodeBench](https://arxiv.org/abs/2403.07974) 和 [LiveBench](https://arxiv.org/abs/2406.19314) 使用持续更新或时间相关数据减少部分静态污染风险。结果必须绑定具体 release/snapshot；动态集不是永久无污染。

## 分母与报告

对 $N$ 个 item family，至少报告：

$$
\operatorname{flagged\ rate}
=
\frac{N_{\text{exact}}+N_{\text{near}}+N_{\text{semantic confirmed}}}
{N}.
$$

同时保留 candidate、confirmed、unknown 和 removed。若只在未标记子集报告分数，应给：

- 全量结果；
- clean-confirmed 子集；
- unknown 子集；
- 各匹配层级；
- 子集大小与能力/难度分布。

删除污染题可能使剩余题更难或更偏，不能把分数变化全部归因记忆。

## 实现契约

```text
benchmark/data/teacher/retrieval revisions
item-family and source IDs
normalizers, shingles and match thresholds
semantic/cross-lingual retriever versions
candidate score and confirmation evidence
first-public and cutoff timestamps
answer/explanation/test/harness access
reviewer decision and uncertainty
```

索引需覆盖训练数据的 raw、parsed、dedup cluster 和 synthetic derivatives。只有最终 token shard 时，来源与时间复核会受限。

## 正确性与攻击失效

- **内部去重当 decontamination**：单次出现的测试答案仍泄漏。
- **exact-only**：释义、翻译和结构变体漏掉。
- **embedding 相似即污染**：合理同主题样本被误删。
- **只查题干**：答案、解释和 tests 泄漏。
- **只看模型训练 cutoff**：teacher、检索和工具仍可能访问。
- **删除标记题不报分布变化**：clean 子集选择偏差。
- **测试失败用于调 prompt**：开发过程污染未记录。
- **动态 benchmark 用浮动版本**：历史比较不再是同一数据。
- **judge 见过参考答案外观**：评分器偏向特定措辞。

## 何时不能宣称“无污染”

闭源预训练数据、第三方 teacher、实时搜索和公开网络使完全证明无污染通常不可行。此时应写明已检查的来源、方法、时间和剩余 unknown，而不是给绝对保证。高价值结论可用私有新题、独立专家出题和 post-cutoff 时间窗做补充。

## 验证与报告卡

1. 分别索引题干、选项、答案、解释、测试和同题元数据。
2. 在 exact、near、释义、翻译和结构变体对上校准召回/误报。
3. 抽样人工复核阈值上下和 semantic candidates。
4. 冻结 teacher、检索、工具和执行时间。
5. 对全量、clean-confirmed 和 unknown 分别报告 effect 与区间。
6. 保存被移除题的难度、来源和模型差异，检查选择偏差。

```text
benchmark snapshot and item-family definition
all data/teacher/tool access surfaces
normalization and matching versions
exact/near/paraphrase/cross-lingual/time results
answer/explanation/test leakage
confirmed/candidate/unknown denominators
clean-subset shift and confidence intervals
execution date, cutoffs and remaining uncertainty
```

训练语料的去重机制见[过滤、去重与污染](../data/filtering-dedup.md)，逐项审计工具见[评测工具](../practice/evaluation-tooling.md)。

## Reference {#reference}

- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- [google-research/deduplicate-text-datasets](https://github.com/google-research/deduplicate-text-datasets)
- [Investigating Data Contamination in Modern Benchmarks](https://arxiv.org/abs/2311.04850)
- [LiveCodeBench](https://arxiv.org/abs/2403.07974)
- [LiveBench](https://arxiv.org/abs/2406.19314)
