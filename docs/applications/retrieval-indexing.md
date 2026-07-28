# 索引与召回

召回阶段的目标不是选出最终答案，而是在可接受成本下构造高覆盖候选集。关键词、稠密向量和近似最近邻解决不同问题；它们首先受数据边界和过滤顺序约束。

## 从文档到检索单元

切分函数可以写成：

$$
\mathcal D
\xrightarrow{\text{parse}}
\{(c_i,m_i,s_i)\}_{i=1}^{N},
$$

其中 $c_i$ 是内容，$m_i$ 是权限、时间和结构 metadata，$s_i$ 是回到原文的 span。好的 chunk 必须同时满足：

- 单独取出时语义基本完整；
- 标题、列表、表格和代码边界没有被随意截断；
- 能回到唯一原文位置；
- 不把不同权限或版本混在同一单元；
- 长度适合 retriever 与 reranker，而非只适合生成窗口。

固定 token 长度加 overlap 是基线，不是最终答案。层级文档可先按章节结构切分，再对过长节点递归切分；表格、代码、对话和扫描 PDF 需要各自的解析契约。

### 保留 span 的窗口基线 {#span-preserving-chunking-reference}

输入是带稳定顺序的 token 序列、窗口大小与 overlap；输出保留半开原文区间 `[start,end)` 及其内容，因此任何候选都能精确回到同一 token snapshot。

```python
def chunk_with_spans(tokens, size, overlap=0):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("require 0 <= overlap < size")
    chunks, step = [], size - overlap
    for start in range(0, len(tokens), step):
        end = min(start + size, len(tokens))
        chunks.append({"start": start, "end": end, "tokens": tokens[start:end]})
        if end == len(tokens):
            break
    return chunks

tokens = list("abcdefghijk")
chunks = chunk_with_spans(tokens, size=5, overlap=2)
assert chunks[0] == {"start": 0, "end": 5, "tokens": list("abcde")}
assert chunks[-1]["end"] == len(tokens)
assert all(tokens[c["start"]:c["end"]] == c["tokens"] for c in chunks)
assert all(a["end"] - b["start"] == 2 for a, b in zip(chunks, chunks[1:]))
```

不变量是 span 与内容可逆、最后一块不越界、相邻窗口共享明确的 overlap。生产解析还要把文档版本、字符 / byte offset、标题路径、权限和结构边界写入 chunk identity；结构化切分与边界测试见[手撕：检索与智能体 · 带 span 的切分](../practice/retrieval-agents.md#span-preserving-chunking-reference)。

## Sparse retrieval

BM25 对查询 $q$ 和文档 $d$ 的常用形式为：

$$
\operatorname{BM25}(q,d)=
\sum_{t\in q}
\operatorname{IDF}(t)
\frac{f(t,d)(k_1+1)}
{f(t,d)+k_1\left(1-b+b\frac{|d|}{\operatorname{avgdl}}\right)}.
$$

$f(t,d)$ 是词频，$b$ 控制长度归一化，$k_1$ 控制词频饱和。它对产品编号、报错文本、人名和稀有术语通常很有价值。BM25 的概率相关性脉络可从 Robertson 与 Spärck Jones 的[技术报告](https://www.microsoft.com/en-us/research/publication/simple-proven-approaches-to-text-retrieval/)回溯。

关键词方法依赖分词、字段权重和查询语言。代码符号、中文粒度、同义词扩展与停用词规则都会改变结果，必须把 analyzer 版本视为索引版本的一部分。

BM25 的原子实现应接收已经由同一 analyzer 处理的 token，而不是在评分函数里偷偷选择分词规则。下面返回一条 query 对每篇文档的分数；`df` 与 `avgdl` 必须来自同一索引 snapshot。

```python
import math
from collections import Counter
def bm25(query, documents, k1=1.2, b=.75):
    if not documents:
        return []
    counts = [Counter(doc) for doc in documents]
    avgdl = sum(map(len, documents)) / len(documents)
    if avgdl == 0:
        return [0.] * len(documents)
    df = {term: sum(term in c for c in counts) for term in set(query)}
    scores = []
    for doc, tf in zip(documents, counts):
        score = 0.
        for term in query:
            freq = tf[term]
            if not freq:
                continue
            idf = math.log(1 + (len(documents) - df[term] + .5) / (df[term] + .5))
            score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * len(doc) / avgdl))
        scores.append(score)
    return scores
docs = [["red", "apple"], ["green", "apple", "tree"], ["banana"]]
scores = bm25(["green", "apple"], docs)
assert scores[1] > scores[0] > scores[2]
assert bm25(["missing"], docs) == [0., 0., 0.]
assert bm25(["apple"], []) == []
assert bm25(["apple"], [[], []]) == [0., 0.]
```

这里没有字段权重、权限过滤或索引增量；生产召回必须先把租户、版本与可见范围绑定到候选集合。重复 query term 是否重复计权也是 analyzer/query-language 契约的一部分，不能在不同服务中各自猜测。

## Dense retrieval

双编码器分别编码 query 与 passage：

$$
s(q,d)=
\frac{f_q(q)^\top f_d(d)}
{\lVert f_q(q)\rVert_2\lVert f_d(d)\rVert_2}.
$$

[DPR](https://arxiv.org/abs/2004.04906) 使用 in-batch negatives 训练开放域问答检索器。对 batch 中第 $i$ 个正例，常见损失为：

$$
\mathcal L_i=-\log
\frac{\exp(s(q_i,d_i^+)/\tau)}
{\sum_j \exp(s(q_i,d_j)/\tau)}.
$$

负例决定模型学会区分什么。随机负例过易，hard negative 过度集中又可能包含未标注正例。训练、离线评测和线上语料之间的领域偏移，也会让高相似度不再等于可回答。

### 精确 cosine top-$k$ {#exact-cosine-topk-reference}

输入 query `[D]`、document matrix `[N,D]` 与 $k$，输出完整扫描后的相似度和 document row ID，作为 ANN 的 correctness oracle。零向量的 cosine 未定义，因此 reference 明确拒绝，而不是静默产生一个伪分数。

```python
import torch

def exact_cosine_topk(query, documents, k):
    if query.ndim != 1 or documents.ndim != 2:
        raise ValueError("expected query [D] and documents [N,D]")
    if query.numel() != documents.shape[1] or not 0 < k <= documents.shape[0]:
        raise ValueError("incompatible shape or k")
    query, documents = query.float(), documents.float()
    query_norm = query.norm()
    document_norm = documents.norm(dim=-1)
    if query_norm == 0 or torch.any(document_norm == 0):
        raise ValueError("cosine similarity requires non-zero vectors")
    score = (documents @ query) / (document_norm * query_norm)
    index = torch.argsort(score, descending=True, stable=True)[:k]
    return score[index], index

query = torch.tensor([1., 0.])
documents = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [1., 1.]])
score, index = exact_cosine_topk(query, documents, k=2)
assert index.tolist() == [0, 3]
torch.testing.assert_close(score, torch.tensor([1., 2 ** -0.5]))
assert torch.all(score[:-1] >= score[1:])
```

分数必须由同一 embedding revision、同一归一化规则和同一可见候选集产生；稳定排序只定义相同分数时的 row-order tie-break。生产检索应先做权限约束，再用这个 oracle 测 ANN recall–latency 曲线，而不是在线全扫；更完整的零向量与候选对照见[手撕：检索与智能体 · 精确 dense retrieval](../practice/retrieval-agents.md#exact-dense-retrieval-reference)。

[ColBERT](https://arxiv.org/abs/2004.12832) 保留 token 级表示并用 late interaction：

$$
s(q,d)=\sum_{i\in q}\max_{j\in d}Q_i^\top D_j.
$$

它比单向量保存更多细粒度匹配信息，代价是索引和检索计算更大。[ColBERTv2](https://arxiv.org/abs/2112.01488) 进一步研究了压缩与训练。

## Hybrid retrieval

稀疏与稠密分数通常不在同一尺度。直接线性相加前必须校准；更稳健的起点是基于名次的 Reciprocal Rank Fusion：

$$
\operatorname{RRF}(d)=
\sum_{r\in\mathcal R}\frac{1}{k+\operatorname{rank}_r(d)}.
$$

原始 [RRF 论文](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/)展示了这种简单融合。$k$ 不是普适常数，应结合候选深度验证；不同检索器若高度相关，增加一路并不会带来等量新信息。

## 近似最近邻

精确扫描 $N$ 个 $d$ 维向量的代价近似 $O(Nd)$。大规模服务常使用 ANN，以少量召回损失换延迟和内存。

[HNSW](https://arxiv.org/abs/1603.09320) 建立分层近邻图：上层稀疏图负责远距离导航，底层稠密图负责局部搜索。关键参数至少包括：

- 构图连接数，影响内存和图质量；
- 构图搜索宽度，影响写入成本；
- 查询搜索宽度，影响 recall–latency 曲线；
- 距离度量和向量归一化；
- 删除、更新和增量插入策略。

ANN 只能近似给定 embedding 空间中的近邻。它不能修复 embedding 语义错误，也不能代替权限过滤。

## 过滤顺序

过滤有两种基本路线：

- **pre-filter**：先限定可见集合，再做相似度搜索；安全边界清楚，但小过滤集可能破坏索引效率；
- **post-filter**：先取 ANN 候选再过滤；实现简单，但过滤后可能不足 $k$，且绝不能把未授权候选送入后续模型。

生产实现常使用可感知过滤的索引、迭代扩大候选集或按租户分片。无论方案如何，授权必须在返回证据前强制成立，不能依赖 prompt 要求模型忽略不可见内容。

## 新鲜度与版本

一次请求应能回答：

- 使用哪一版原始语料、解析器、embedding 与索引；
- 新文档何时完成全链写入；
- 更新是否留下旧版本重复候选；
- 删除是否同步清除缓存和派生索引；
- 回滚时 query encoder 与 document embedding 是否兼容。

双编码器升级若只重算一侧，会让空间失配。灰度期间应绑定兼容版本，并用相同查询集比较 exact search 与 ANN 的差额。

## 诊断矩阵

| 现象 | 优先检查 |
| --- | --- |
| 专有名词搜不到 | analyzer、字段、BM25 与原始文本 |
| 语义近似但实体错 | hard negative、metadata filter、reranker |
| exact 命中而 ANN 未命中 | 搜索宽度、构图质量、度量与归一化 |
| 更新后新旧答案混合 | 文档版本、增量索引、缓存键 |
| 有权限用户偶发空结果 | pre-filter 选择性与候选扩展 |
| 离线好、线上差 | 查询分布、语料新鲜度、解析失败 |

候选如何精排和压缩见[重排与上下文工程](reranking-context.md)，BM25、RRF 与精确向量检索代码见[手撕：检索与智能体](../practice/retrieval-agents.md)。

## Reference {#reference}

- [Simple and Proven Approaches to Text Retrieval](https://www.microsoft.com/en-us/research/publication/simple-proven-approaches-to-text-retrieval/)
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [ColBERT](https://arxiv.org/abs/2004.12832)
- [ColBERTv2](https://arxiv.org/abs/2112.01488)
- [RRF 原论文](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/)
- [HNSW](https://arxiv.org/abs/1603.09320)
