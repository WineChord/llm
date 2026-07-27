# 过滤、去重与污染

过滤回答“内容是否适合进入候选语料”，去重回答“多少内容提供了重复信号”，污染检查回答“训练数据是否泄漏了评测对象或答案”。三者可以共享索引与相似度工具，但目标、阈值和错误成本不同。

## 问题分解

| 操作 | 比较对象 | 主要风险 |
| --- | --- | --- |
| 规则过滤 | 单文档与明确约束 | 规则过宽、语言偏置 |
| 质量评分 | 文档与训练目标 | 评分器风格偏好、分布漂移 |
| Exact dedup | 规范化后完全相同内容 | 规范化过强或过弱 |
| Near dedup | 局部改写、模板、镜像 | 阈值误删合理相似内容 |
| Decontamination | 训练候选与冻结评测资产 | 改写、翻译、答案与解析泄漏 |

安全过滤也不等于模型安全训练。移除明确不允许的内容是数据边界；研究模型如何处理风险输入，则需要另行设计受控训练集与评测集。

## 数据契约

每个过滤决定应保存：

```text
document_id
input representation and normalizer version
rule / model / index version
score and threshold
decision and reason codes
dedup cluster and representative
review status
```

不得只保存最终 `keep/drop`。阈值变化、评分器升级或误删申诉都需要原始分数和原因重放。

近重复与污染至少保留两种表示：

- 适合复核的原始结构和文本；
- 用于匹配的规范化 token、shingle 或 fingerprint。

匹配表示不能覆盖原文，否则无法判断“高相似”来自公共模板、代码样板还是实质重复。

## 过滤机制

### 硬规则

编码错误、空页、模板占比、极端重复、非法控制字符和明确不在范围内的来源适合用确定性规则处理。规则应返回可解释原因，并在每种语言和文档类型上测量保留率。

### 学习式质量评分

分类器或语言模型可以学习比规则更细的质量信号，但其训练集定义了“质量”的偏好。常见失效包括：

- 偏好百科式或自身生成风格；
- 把短文本、方言、低资源语言或专业符号误判为噪声；
- 使用与下游 benchmark 重叠的正例；
- 在新 crawl 或新域上分数失准。

阈值应根据保留预算和 slice-level precision/recall 选择，而不是追求全局分数最大。[DataComp-LM](https://arxiv.org/abs/2406.11794) 通过固定训练预算比较不同数据筛选方案，说明过滤器的价值最终要由受控训练结果验证。

## 去重机制

### Exact dedup

在明确的规范化函数 $f$ 下计算

$$
h_d=H(f(d)).
$$

相同 $h_d$ 的文档进入同一 cluster。规范化通常可统一换行、Unicode 和确定的空白，但不应默认删除所有标点、数字或代码格式；过强规范化会把不同事实和程序合并。

### Near dedup

把文档映射为 $k$-shingle 集合 $S(d)$，Jaccard 相似度为

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

MinHash 满足单个最小哈希碰撞概率等于 Jaccard：

$$
P[h_{\min}(A)=h_{\min}(B)]=J(A,B).
$$

将 signature 分成 $b$ 个 band、每个 $r$ 行时，候选对概率近似为

$$
P_{\text{candidate}}(s)
=1-(1-s^r)^b.
$$

LSH 只负责召回候选，最终仍应使用精确相似度、长度比和结构信息确认。[Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) 展示了训练语料重复与记忆、评测之间的联系，并提供了[官方实现](https://github.com/google-research/deduplicate-text-datasets)。

### 代表样本

cluster 代表不应简单取第一条。可按来源使用条件、解析完整度、时间、结构保留和文本质量排序。cluster 成员及理由必须保留，否则删除或来源撤回时无法重选代表。

## 污染检查

benchmark decontamination 的匹配对象至少包括：

- 题干、选项、参考答案和解释；
- 同源题、模板题、翻译与改写；
- 公开 solution、代码测试、仓库 issue 和 harness；
- teacher 或检索系统可访问的答案；
- 开发阶段用于选模板、阈值和 checkpoint 的样本。

只做 exact string match 会漏掉改写污染。[Investigating Data Contamination in Modern Benchmarks](https://arxiv.org/abs/2311.04850) 表明，经过释义的 benchmark 内容也可能影响测量。匹配结果应分成 `exact`、`near`、`semantic candidate` 和 `unknown`，语义候选必须抽样人工复核，不能把模型相似度直接当删除真值。

训练内部去重不能替代评测污染检查：一个答案在训练集中只出现一次，仍然足以使对应测试题失去独立性。

## 正确性与失效

- **先切分后去重**：同一文档族跨越 train/test，形成泄漏。
- **全语料一个阈值**：代码模板、法规、诗歌和网页正文的合理重复结构不同。
- **只看全局保留率**：低资源语言可能被几乎清空。
- **保留 cluster 却丢成员**：无法处理来源撤回和阈值回溯。
- **质量分数参与 benchmark 选题**：评分器可能把答案外观当质量信号。
- **过度去重**：删除有意义的多来源证据与语言变体，降低覆盖。
- **去重不足**：重复数据抬高某些模式的梯度权重，增加逐字记忆。

## 何时不应使用近重复删除

当重复本身是任务信号，例如日志频率建模、固定格式解析、拼写变体研究或代码模板学习时，不应直接删除全部近重复。可改为 cluster-level weighting、限制每簇上限或在评测切分时分组隔离。小型高质量集合也可先 exact dedup，再用人工 review 处理 near duplicates。

## 验证

1. 构造 exact、局部复制、模板相同、语义相同但事实不同的对照对。
2. 按语言、领域、长度、来源和文档类型报告候选率、确认率与删除率。
3. 对阈值两侧样本做盲审，估计 false positive 与 false negative。
4. 比较去重前后的 token 频率、cluster size、验证损失与记忆化切片。
5. 用冻结 benchmark 的题干、答案、解释和释义版本分别检索。
6. 在新快照上复用相同规则时检查分数漂移，而不是只比较通过率。

过滤后的真实训练权重见[数据混合与课程](mixtures-curricula.md)，记忆与抽取风险见[记忆化、隐私与删除](memorization-privacy.md)，污染统计实现见[评测工具](../practice/evaluation-tooling.md)。
