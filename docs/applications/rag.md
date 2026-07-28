# Retrieval-Augmented Generation

RAG 把外部语料作为可检索的非参数记忆，在生成前选择证据。[Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) 给出了参数记忆与检索记忆联合建模的经典框架；工程系统还要处理索引、权限、新鲜度、引用和拒答。

[RAG 深读](../landscape/works/rag.md)回到论文中的 RAG-Sequence、RAG-Token 与 latent-document marginalization，并用小型精确检索器展示概率融合；从 REALM 到工具智能体的更长脉络见[检索、工具与智能体](../landscape/lineages/retrieval-agents.md)。

## 闭环而非组件

$$
\text{source}
\rightarrow \text{parse}
\rightarrow \text{index}
\rightarrow \text{retrieve}
\rightarrow \text{rerank}
\rightarrow \text{assemble}
\rightarrow \text{generate}
\rightarrow \text{verify}.
$$

这条链可以分成三个控制面：

| 控制面 | 核心问题 | 专题 |
| --- | --- | --- |
| 候选集合 | 正确证据有没有被解析、切分并召回 | [索引与召回](retrieval-indexing.md) |
| 有限上下文 | 哪些候选值得保留，怎样避免重复与位置偏差 | [重排与上下文工程](reranking-context.md) |
| 输出契约 | 哪些陈述有证据，引用是否支持，何时不回答 | [证据约束生成](grounded-generation.md) |

把所有失败都归因于 embedding 会导致无效优化。解析丢表格、ACL 过滤过早、查询改写漂移、reranker 截断或引用错配，都可能产生相同的“没有答对”表象。

## 两种概率视角

经典 RAG-Sequence 对潜在文档 $z$ 求和：

$$
p(y\mid x)=\sum_{z\in\mathcal Z_k(x)}
p_\eta(z\mid x)\,p_\theta(y\mid x,z),
$$

其中 $p_\eta$ 是检索器，$p_\theta$ 是生成器，$\mathcal Z_k(x)$ 是 top-$k$ 文档集合。生产系统通常不联合训练这两个分布，却仍应保留这一分解：答案错可能来自候选集，也可能来自条件生成。

若证据由多个片段组成，还需评估集合覆盖：

$$
\operatorname{coverage}(E,G)
=\frac{|E\cap G|}{|G|},
$$

$G$ 是完成答案所需的 gold evidence，$E$ 是进入上下文的证据。单个相关片段的 Recall@k 不能代表多跳问题已经可解。

## 数据契约

每个可检索单元至少携带：

- 稳定文档与片段标识；
- 标题、章节路径和相邻结构；
- 来源 URI、版本、抓取与生效时间；
- 内容哈希与解析器版本；
- 租户、权限标签和保留策略；
- 文本在原文中的 span 或页码；
- 适合展示的引用标签。

索引是源数据的派生物，不是事实来源。删除、权限变化和新版本必须能传播到倒排索引、向量索引、缓存和引用层。

## 离线与在线分离

离线评测回答“固定语料与查询上哪种方法更好”；在线系统还需回答：

- 增量更新多久可见；
- 索引版本怎样与生成请求绑定；
- 查询高峰是否降低 ANN 搜索深度；
- reranker 超时怎样降级；
- 证据被删除后缓存怎样失效；
- 权限服务不可用时是 fail closed 还是继续返回。

这些行为必须进入请求追踪，否则离线 Recall@k 很高也无法解释线上错误。

## 基线顺序

一个可审计的升级顺序通常是：

1. 精确关键词检索与明确 metadata filter；
2. dense retrieval，与关键词结果分开测；
3. rank fusion，而不是直接相加不可比的原始分数；
4. reranker 与上下文去重；
5. 查询分解、迭代检索或结构化索引；
6. 自反思或检索策略学习。

后一步只有在前一步的失败已被数据证明时才值得引入。复杂流水线会同时增加延迟、调参空间和不可解释的耦合。

## 安全边界

检索内容属于不可信数据，不因进入 prompt 就升级为指令。至少隔离：

- 文档中的 prompt injection 与伪造系统消息；
- 跨租户或跨权限组召回；
- 恶意文件、解析器漏洞和索引投毒；
- PII、机密数据和删除请求；
- 过期事实与相互冲突的版本；
- 显示链接与真实证据 span 不一致。

RAG 不自动消除幻觉；它只提供了让陈述可核验的机会。引用支持性见[证据约束生成](grounded-generation.md)，端到端攻击面见[智能体安全](agent-security.md)。

## 最小验收

- 构造“能答、缺证据、证据冲突、无权限、已删除、注入文本”六类查询；
- 分别记录解析、过滤、召回、重排、组装、生成和引用结果；
- 对正确答案检查引用是否真的蕴含相应陈述；
- 对拒答检查是合理识别证据不足，而非单纯检索超时；
- 对每次实验固定语料快照、索引版本、模型版本与参数。

核心检索与融合代码见[手撕：检索与智能体](../practice/retrieval-agents.md)，统计评测见[手撕：评测工具](../practice/evaluation-tooling.md)。

## Reference {#reference}

- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)
