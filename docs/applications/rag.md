# Retrieval-Augmented Generation

RAG 在生成前检索外部语料，使答案可以引用更新的、可审计的证据。[Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) 给出参数记忆与非参数记忆结合的经典框架。

## 流水线

$$
\text{query}
\rightarrow \text{rewrite}
\rightarrow \text{retrieve}
\rightarrow \text{rerank}
\rightarrow \text{assemble context}
\rightarrow \text{generate}
\rightarrow \text{verify}
$$

每一步都可能成为瓶颈，不能只评 embedding 模型。

## 索引

- chunk 应保留语义、标题、层级和来源边界。
- overlap 增加召回，也增加重复与上下文浪费。
- dense retrieval 擅长语义近似，sparse retrieval 擅长精确词与稀有实体。
- hybrid retrieval 常比单一路线稳健，但需要分数校准。
- metadata filter 应在授权与检索阶段同时生效。

## Reranking 与上下文

cross-encoder 或 LLM reranker 能提高排序质量，但增加延迟。上下文组装需要去重、保留引用、控制顺序，并防止低可信文档覆盖高可信来源。

## 评测拆分

- **检索**：Recall@k、MRR、nDCG、证据覆盖。
- **生成**：答案正确性、完整性、引用精度和引用召回。
- **系统**：端到端成功率、延迟、成本、权限与新鲜度。

一个正确答案可能引用错误证据；一个检索命中的系统也可能在生成阶段忽略证据，因此必须分别打分。

## 安全

检索文档是外部数据，不是系统指令。需要隔离 prompt injection、租户权限、恶意文件、PII、索引投毒和过期内容。引用链接只能证明来源存在，不能自动证明模型陈述被来源支持。
