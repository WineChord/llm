# 证据约束生成

Grounded generation 要求答案中的可核验陈述由给定证据支持，并能定位到相应来源。它比“附上若干链接”更严格：引用必须与 claim 对齐，来源存在也不代表它支持当前句子。

## Claim–evidence 图

把答案拆成原子陈述 $C=\{c_i\}$，证据片段为 $E=\{e_j\}$，引用关系构成二部图：

$$
G=(C,E,A),\qquad
A_{ij}=1\ \text{表示答案声称}\ e_j\ \text{支持}\ c_i.
$$

至少区分：

- **citation completeness**：需要证据的 claim 有多少被引用；
- **citation correctness**：引用是否真的支持对应 claim；
- **source quality**：证据本身是否权威、当前且适用；
- **answer correctness**：答案是否正确，不能由前三项自动推出。

若 $N$ 个可核验 claim 中 $M$ 个至少有一条引用：

$$
\operatorname{completeness}=\frac{M}{N}.
$$

若一共给出 $K$ 条 claim–citation 边，其中 $S$ 条经判定支持：

$$
\operatorname{precision}=\frac{S}{K}.
$$

统计口径必须说明哪些句子需要外部证据，以及部分支持、冲突和来源间接转述如何计分。

## 生成策略

三种常见路线：

### Context-first

先固定证据集合，再要求模型只根据证据回答。实现简单，但模型仍可能混入参数记忆，或把文档中的命令当成指令。

### Plan claims first

先列出待回答的子问题和所需证据，再逐项检索、写作和引用。它适合多约束任务，但需要防止计划阶段未经证据就锁定结论。

### Retrieve while generating

生成过程中判断是否需要检索并回到证据。[Self-RAG](https://arxiv.org/abs/2310.11511)研究了检索与自反思 token。动态检索能节省简单问题的成本，也把检索决策本身变成需要评测的模型行为。

无论路线如何，最终答案都应由结构化 claim 与 evidence ID 组装，而非事后随机把链接贴到段尾。

## 不可回答

拒答不是单一阈值，而是多种状态：

- 语料没有相关证据；
- 有相关材料，但不足以推出答案；
- 来源相互冲突；
- 证据已经过期或适用范围不明；
- 用户无权访问必要证据；
- 问题本身缺少关键参数。

可以把回答决策写成：

$$
\text{answer if }
\operatorname{coverage}\ge\tau_c,\quad
\operatorname{support}\ge\tau_s,\quad
\operatorname{risk}\le\tau_r.
$$

这些阈值必须按任务风险校准。模型的自报置信度不是天然概率；应使用有标签的可回答/不可回答集合测 precision–recall 与选择性风险。

## 冲突与时间

证据冲突时不要通过“多数片段”自动投票。先比较：

- 是否讨论同一对象、版本和时间；
- 一手来源与转述来源；
- 发布日期与事实生效日期；
- 规范、实测和推断是否被混为一谈；
- 某来源是否已经撤回或被替代。

若仍无法消解，应展示冲突和缺口，不生成虚假的单一结论。

## 引用生成

稳定做法是先让模型输出结构化草稿：

```json
{
  "claims": [
    {"text": "…", "evidence_ids": ["E17", "E22"]}
  ]
}
```

系统随后验证 evidence ID 存在、权限仍有效、span 未过期，再渲染为自然语言。JSON schema 只能限制结构，不能证明证据蕴含 claim。

## 验证层

验证可以逐层增强：

1. ID、URI 和 span 的存在性；
2. claim 与引用之间的词项或实体一致性；
3. NLI 或专用 entailment 模型；
4. 独立 LLM judge，隐藏原答案措辞以减少偏差；
5. 高风险场景的人审或确定性规则。

judge 也会受提示、顺序和模型家族影响。必须保留人工校准集，并报告 judge 与人工的一致率。

## 端到端失效

| 现象 | 可能根因 |
| --- | --- |
| 答案正确但引用错 | 参数记忆作答，事后引用匹配 |
| 引用正确但答案错 | 误读限定词、计算错误、组合推理失败 |
| 每句都有引用但内容空泛 | 优化了引用覆盖，未优化任务效用 |
| 总是拒答 | 检索阈值、风险阈值或 judge 过严 |
| 旧事实反复出现 | 版本过滤、缓存或生效时间错误 |
| 引用链接可开但无法定位 | chunk 没有稳定 span，页面已漂移 |

## 最小评测集

评测集应包含直接事实、多证据组合、数字计算、否定条件、冲突来源、时间变化、不可回答与对抗注入。每个样本保存 gold claims、允许的证据集合、不可回答原因和风险等级。

引用质量与幻觉边界见[幻觉与事实性](../evaluation/hallucination.md)，检索前半段见[索引与召回](retrieval-indexing.md)，精度/召回代码见[手撕：评测工具](../practice/evaluation-tooling.md)。

## Reference {#reference}

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Self-RAG](https://arxiv.org/abs/2310.11511)
