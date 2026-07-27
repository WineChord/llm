# 重排与上下文工程

召回追求“不要漏”，重排和上下文组装追求“在有限预算里留下最能支持答案的证据”。这一层同时控制相关性、覆盖、多样性、顺序与可信度。

## Query transformation

原查询未必适合检索。常见变换包括：

- 规范化实体、日期、产品版本和缩写；
- 把多约束问题拆为可独立检索的子问题；
- 用对话状态补全省略指代；
- 生成假设文档或多个视角查询；
- 根据第一轮证据继续检索缺失环节。

[HyDE](https://arxiv.org/abs/2212.10496)先生成假设文档再编码检索。它可能缩小 query–document 表达差距，也可能把生成器的错误假设注入检索。查询变换必须保留原问题和变换轨迹，并分别评估原查询、变换查询与二者融合。

## Reranker

双编码器先独立编码 query 与 document，适合大规模召回；cross-encoder 联合编码二者：

$$
s_i=f_\theta([\text{query};\text{document}_i]),
$$

能利用细粒度交互，但成本随候选数与文本长度增长。两阶段系统的常见预算是：

$$
N_{\text{corpus}}
\xrightarrow{\text{retriever}} k_1
\xrightarrow{\text{reranker}} k_2
\xrightarrow{\text{context}} k_3,
\qquad k_1\gg k_2\ge k_3.
$$

reranker 的训练标签应接近最终任务：“主题相关”不一定代表“足以支持答案”。长文档被截断时，还应明确打分的是标题、窗口还是完整章节。

## 覆盖与去重

只按相关性取 top-$k$ 容易获得多段重复证据。Maximal Marginal Relevance 在相关性与新颖性之间权衡：

$$
d^*=\arg\max_{d\in R\setminus S}
\left[
\lambda\,s(q,d)
-(1-\lambda)\max_{d'\in S}s(d,d')
\right].
$$

$R$ 是候选集，$S$ 是已选择集合。MMR 的原始工作见 [Carbonell 与 Goldstein](https://aclanthology.org/X98-1025/)。相似度并不等于信息重复：两个片段可能文字相似但分别包含前提和结论，因此应在真实多证据任务上调 $\lambda$。

多跳问题还可显式维护尚未覆盖的 claim 或实体集合：

$$
\Delta(d\mid S)
=\operatorname{new\_evidence}(d,S)
-\alpha\operatorname{redundancy}(d,S)
-\beta\operatorname{cost}(d).
$$

这不是统一标准，而是一种把选择目标写清楚的工程模板。

## 上下文顺序

[Lost in the Middle](https://arxiv.org/abs/2307.03172)表明，长上下文模型对信息位置可能敏感。组装时应：

- 把全局说明与具体证据分区；
- 保留文档标题、版本、时间和稳定引用 ID；
- 邻接放置需要联合理解的片段；
- 避免把重复模板占满开头与结尾；
- 对顺序置换做敏感性测试。

“支持更长窗口”不等于“所有位置同样可用”。窗口扩大还会增加 prefill 成本，并稀释指令和证据的相对注意力。

## 压缩

压缩有三种粒度：

1. **抽取**：选择原文句子或 span，可保持精确引用；
2. **摘要**：重写内容，节省更多 token，但可能改变事实；
3. **表示压缩**：删除低信息 token 或使用模型内部表示，验证更困难。

[LLMLingua](https://arxiv.org/abs/2310.05736)研究 prompt 压缩。任何压缩都应在 token 节省之外检查答案、数字、否定词、限定条件和引用 span；高风险事实优先保留原文。

## 层级与迭代检索

[RAPTOR](https://arxiv.org/abs/2401.18059)把文本片段递归聚类和摘要成树，用不同抽象层检索。层级索引适合“先找主题，再找细节”，但摘要节点不是原始证据，最终引用仍应回到叶子或可核验来源。

迭代检索允许模型根据已见证据提出下一跳查询。它提升组合能力，也带来查询漂移、循环和成本不确定。运行时至少需要：

- 最大轮数与总候选预算；
- 已查询内容去重；
- 缺失证据的显式状态；
- 每轮新增覆盖量；
- 无增益终止条件。

## 上下文契约

送入生成器的每个证据块建议使用结构化字段：

```text
[E17]
title: ...
source: ...
version: ...
effective_at: ...
content: ...
```

标识只用于关联引用，不暗示可信等级。系统指令与证据必须分区，文档中的命令式文本仍属于数据。

## 评测

分别测：

- candidate recall 与 gold evidence coverage；
- reranker nDCG、MRR 和 pairwise accuracy；
- 选择后冗余率与 token 利用率；
- 顺序置换、删减和对抗片段下的答案稳定性；
- 压缩前后 claim 保留与引用可追溯性；
- 端到端答案正确性和引用支持性。

最终输出怎样绑定证据见[证据约束生成](grounded-generation.md)，核心融合与 MMR 实现见[手撕：检索与智能体](../practice/retrieval-agents.md)。
