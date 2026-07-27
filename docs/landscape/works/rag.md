# RAG：把知识从参数里拆出来

RAG 经常被描述成“检索几段文档，再拼进 prompt”。这概括了今天常见的工程形态，却掩盖了原始工作的关键思想：检索文档是生成模型中的隐变量，答案概率需要在候选证据上边缘化。[RAG 论文](https://arxiv.org/abs/2005.11401)把稠密检索器与预训练 seq2seq 生成器放进同一微调框架，使可更新的非参数记忆参与条件生成。

## 前一站：从参数知识到可检索记忆

预训练模型能在参数中保存事实，但更新一个事实通常意味着继续训练，答案也没有天然 provenance。[REALM](https://arxiv.org/abs/2002.08909)在预训练中引入 latent retriever，[DPR](https://arxiv.org/abs/2004.04906)用独立 question/passage encoder 与 in-batch negatives 训练开放域问答检索器。

DPR 的分数通常写成

$$
s(q,z)=E_Q(q)^\top E_P(z),
\qquad
p_\eta(z\mid q)=\frac{e^{s(q,z)}}{\sum_{z'}e^{s(q,z')}}.
$$

索引把文档表示持久化，query encoder 在请求时产生向量。这带来可更新性，也引入 embedding version、chunk version 与 index version 的一致性问题。

## RAG 的隐变量目标

给定输入 $x$、检索文档 $z$ 与输出 $y$，RAG-Sequence 在整条输出上共享文档：

$$
p(y\mid x)
\approx\sum_{z\in\operatorname{top}K(x)}
p_\eta(z\mid x)\prod_t p_\theta(y_t\mid x,z,y_{<t}).
$$

RAG-Token 允许不同 token 对文档重新边缘化：

$$
p(y\mid x)
\approx\prod_t\sum_z p_\eta(z\mid x)
p_\theta(y_t\mid x,z,y_{<t}).
$$

两者不是普通的实现开关。前者鼓励整段答案依赖同一证据集合，后者表达力更强但计算和解释都更复杂。

```python
import torch
def rag_sequence_nll(retrieval_logits, token_logp):
    log_pz = retrieval_logits.log_softmax(-1)
    doc_logp = token_logp.sum(-1)
    return -torch.logsumexp(log_pz + doc_logp, dim=-1).mean()
retrieval = torch.tensor([[2.0, 0.0], [0.0, 1.0]], requires_grad=True)
token_logp = torch.log(torch.tensor([[[.8, .7], [.2, .3]], [[.4, .5], [.9, .8]]]))
loss = rag_sequence_nll(retrieval, token_logp)
loss.backward()
assert loss.ndim == 0 and retrieval.grad.shape == retrieval.shape
```

代码中的 `token_logp[b,k,t]` 是在第 $k$ 篇文档条件下真实目标 token 的 log probability。它没有实现 top-$K$ 检索器；只固定了“先在 token 轴累加，再在文档轴做 log-sum-exp”的语义。

## 从联合模型到工程管线

后来的生产 RAG 常把训练耦合拆开：

```text
ingest -> parse -> chunk -> embed -> index
query -> retrieve -> rerank -> pack -> generate -> cite -> verify
```

拆开后，每层可以独立替换和观测，也失去了原始端到端目标的自动协调。retriever 的相似度最高文档未必最能帮助 generator；reranker 的离线 NDCG 提升也未必转化为最终正确率。因此必须同时保存：

- corpus、parser、chunker、embedding 与 index 版本；
- recall@k、rerank 指标与答案支持率；
- 实际送进模型的文档 ID、顺序和截断位置；
- 无证据、冲突证据和检索失败时的策略。

## FiD 与“让生成器自己融合”

[Fusion-in-Decoder](https://arxiv.org/abs/2007.01282)分别编码多个 passage，再让 decoder 在 cross-attention 中融合。它避免把所有 passage 在 encoder 输入端完全拼接，却把计算压力转移到 decoder 可见的 encoder states。增加候选文档可能提高 retrieval coverage，也会近似线性增加 encoder 工作与 decoder cross-attention memory；若额外候选主要是噪声，最终答案质量并不随数量单调提高。

“塞更多上下文”从来不是免费的正确性策略。无关文档会稀释注意力，冲突文档需要时间与来源判断，长上下文还可能把答案放进模型不敏感的位置。

## RAG 没有自动提供 provenance

检索到一篇文档不代表答案受它支持。可靠系统至少区分：

1. **retrieval relevance**：文档与问题相关；
2. **answer support**：文档蕴含具体答案；
3. **citation correctness**：引用位置与陈述对应；
4. **corpus authority**：来源在当前场景可被信任。

生成器可能凭参数记忆答对，却引用无关文档；也可能忠实复述一篇过时来源。验证方法见[有依据生成](../../applications/grounded-generation.md)，完整检索评测见[检索与索引](../../applications/retrieval-indexing.md)。

## 它留下的方向

RAG 把“模型知道什么”改写成“模型、索引和运行时共同能访问什么”。随后工具调用把外部状态从只读文档扩展到可执行动作，Agent 又把一次检索扩展为多步轨迹。下一篇[ReAct 与 Toolformer](react-toolformer.md)正从这里开始；整条历史见[从参数记忆到可行动系统](../lineages/retrieval-agents.md)。
