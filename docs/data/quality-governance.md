# 数据质量与治理

数据治理的核心是可追溯性：知道一条样本来自哪里、经过哪些变换、为什么被保留，以及如何删除或重建。

一条样本不应只有“最终文本”。更稳妥的记录是不可变来源对象与一串可重放变换：

$$
r_i=(\text{source URI},\text{snapshot},\text{content hash},
\text{license},\text{policy tags}),
$$

$$
x_i^{(k+1)}=f_k\!\left(x_i^{(k)};\,\text{code revision},\text{config}\right).
$$

最终训练清单引用稳定 sample ID、版本与权重，而不是把网页内容、清洗结果和采样决策压成一个无法追责的文件夹。来源快照怎样固定见[来源与溯源](sources-provenance.md)。

## 质量不是单一分数

至少分开评估：

- **可读性**：编码、语法、结构是否完整。
- **信息密度**：是否包含重复模板、广告或导航噪声。
- **正确性**：事实与推理是否可靠；这通常最难自动判断。
- **多样性**：语言、领域、体裁、难度和观点是否覆盖目标分布。
- **可训练性**：长度、格式和上下文是否适合当前目标。
- **合规性**：许可、隐私、地域与用途限制是否满足。

一个综合分数会隐藏维度冲突。更稳妥的做法是保留分项特征，在采样或课程阶段组合。

过滤器也不应只报告“删掉多少”。若规则 $f$ 决定保留样本，应按来源、语言、领域和长度比较：

$$
\operatorname{retention}_{g}
=\frac{\sum_i\mathbf 1[g_i=g]\,\mathbf 1[f(x_i)=1]}
{\sum_i\mathbf 1[g_i=g]}.
$$

总体保留率正常，某个低资源语言或特定体裁仍可能被几乎清空。对阈值型 quality classifier，还应人工抽检阈值两侧和高置信极端样本；否则分类器偏见会在大规模过滤后被放大。

## 去重与污染

精确哈希只能发现完全相同文本；MinHash、局部敏感哈希或 embedding 相似度可发现近重复。阈值必须按数据类型校准：代码模板、法律条款和百科定义本来就高度相似。

[Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) 展示了去重对记忆和评测的影响。污染检查应同时扫描 benchmark 题干、选项、答案、解释和常见变体，并保留时间戳证据。

去重单位决定删除语义。文档级 dedup 可能保留被复制到长页面中的段落，段落级 dedup 又可能误删合法的标准条款和代码样板。实践中应保存 cluster ID、代表样本选择规则与成员列表；删除代表样本时，cluster 仍能被重新选主。

污染也不能只做一次字符串匹配。发布 benchmark 后出现的网页镜像、解题讨论和合成改写，会让时间边界与语义近邻同样重要。可靠报告应区分 exact match、near-duplicate、答案泄漏和同源但不含答案的背景材料，见[过滤、去重与污染](filtering-dedup.md)与[评测污染](../evaluation/contamination.md)。

## 隐私与删除

- 采集前定义允许来源和禁止字段。
- 在原始层、处理层和训练清单中维护稳定样本 ID。
- 将删除请求设计成可重放的 denylist，而不是手工修改某个快照。
- 训练后删除并不等价于权重中遗忘；需要单独评估记忆与可能的 unlearning 方法。
- 公布数据统计时避免泄露可逆的稀有样本。

删除必须穿过派生图：原始对象、解析文本、去重 cluster、token shard、训练 manifest、检索索引和缓存都有各自副本。仅从下一版训练清单移除样本，是“未来不再使用”，不是已经从历史 checkpoint 中消失。二者应在数据卡和处置记录中分开表述。

## 合成数据

合成数据适合扩充格式、覆盖可验证任务或生成对比样本，但要记录生成器版本、prompt 模板、采样参数、过滤器和验证器。模型自举若缺少外部真值，容易发生错误循环、风格坍缩与多样性下降。

合成器、judge 和训练模型若来自同一家族，错误可能高度相关。可程序验证的数学、代码或结构任务应保存 verifier 结果；开放任务则需要独立抽检、来源多样性与真实数据 holdout。合成比例上升时，应观察的不只是训练 loss，还包括输出风格、事实错误、重复模板和下游切片的变化。

## 发布前最后一公里

每个发布数据版本至少包含：来源与许可、时间范围、语言和领域分布、处理步骤、去重方法、已知污染、敏感信息策略、删除机制、统计口径、限制与版本哈希。

还应从 manifest 随机抽样并完整重放到训练张量，确认 tokenizer、模板、mask、sample weight 和 packing 没有绕过治理决策；只检查清洗后的文本不够。进入张量前的接口见[序列构造与打包](sequence-construction.md)，混合权重见[数据混合与课程](mixtures-curricula.md)，偏好和环境轨迹的额外字段见[偏好、过程与轨迹数据](feedback-trajectories.md)。

## Reference {#reference}

- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116)
- [DataComp-LM: In Search of the Next Generation of Language Model Pretraining Datasets](https://arxiv.org/abs/2406.11794)
