# 手撕：检索与智能体

本页实现从候选召回到工具执行的最短闭环。检索代码保留分数定义，智能体代码把 schema、权限、幂等和终态放在模型之外。

## 带 span 的切分 {#span-preserving-chunking-reference}

chunk 必须能回到原文位置。以下函数在 token 序列上滑窗，并保留半开区间 $[start,end)$：

```python
from collections import Counter
from dataclasses import dataclass, field
import math
import torch

def chunk_tokens(tokens, size, overlap=0):
    """tokens:[N] -> [{"start","end","tokens"}]."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("require 0 <= overlap < size")
    chunks, step = [], size - overlap
    for start in range(0, len(tokens), step):
        end = min(start + size, len(tokens))
        chunks.append({"start": start, "end": end, "tokens": tokens[start:end]})
        if end == len(tokens):
            break
    return chunks
```

```python
tokens = list(range(11))
chunks = chunk_tokens(tokens, size=5, overlap=2)
assert chunks[0]["tokens"] == [0, 1, 2, 3, 4]
assert chunks[-1]["end"] == len(tokens)
assert all(a["end"] - b["start"] == 2 for a, b in zip(chunks, chunks[1:]))
```

生产解析应优先保留标题、表格、代码和权限边界；固定窗口只是可审计基线。

## BM25

```python
def bm25_scores(query, documents, k1=1.2, b=0.75):
    """query:list[str], documents:list[list[str]] -> scores:[N]."""
    if not documents:
        return []
    counts = [Counter(doc) for doc in documents]
    avgdl = sum(map(len, documents)) / len(documents)
    df = Counter(term for terms in map(set, documents) for term in terms)
    scores = []
    for doc, tf in zip(documents, counts):
        score = 0.0
        for term in query:
            freq = tf[term]
            if not freq:
                continue
            idf = math.log(1 + (len(documents) - df[term] + 0.5) / (df[term] + 0.5))
            norm = freq + k1 * (1 - b + b * len(doc) / max(avgdl, 1e-12))
            score += idf * freq * (k1 + 1) / norm
        scores.append(score)
    return scores
```

```python
docs = [["red", "apple"], ["green", "apple", "apple"], ["blue", "berry"]]
scores = bm25_scores(["apple"], docs)
assert scores[1] > scores[0] > scores[2]
```

分词、字段权重、同义词和语言分析器都是 BM25 配置的一部分。[索引与召回](../applications/retrieval-indexing.md)给出公式与边界。

## 精确 dense retrieval {#exact-dense-retrieval-reference}

```python
def cosine_topk(query, documents, k):
    """query:[D], documents:[N,D] -> values, indices."""
    if query.ndim != 1 or documents.ndim != 2 or query.numel() != documents.size(1):
        raise ValueError("incompatible embedding shapes")
    q = query.float() / query.float().norm().clamp_min(1e-12)
    d = documents.float() / documents.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (d @ q).topk(min(k, documents.size(0)))
```

它适合核对 ANN 的 recall loss。向量为零时归一化为零相似度，而不是产生 NaN；业务上是否允许零向量应另行决定。

## Reciprocal Rank Fusion

RRF 只使用名次，不直接相加不可比分数：

```python
def reciprocal_rank_fusion(rankings, k=60):
    """rankings:list[list[hashable]] -> document IDs from high to low."""
    score = Counter()
    for ranking in rankings:
        seen = set()
        for rank, doc_id in enumerate(ranking, 1):
            if doc_id in seen:
                continue
            score[doc_id] += 1 / (k + rank)
            seen.add(doc_id)
    return [doc_id for doc_id, _ in score.most_common()]
```

同一 ranking 内重复 ID 只计一次。[RRF 原论文](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/)使用这种简单融合；$k$ 与候选深度仍需在目标数据上验证。

## Maximal Marginal Relevance

```python
def mmr(query_score, pair_similarity, limit, diversity=0.5):
    """query_score:[N], pair_similarity:[N,N] -> selected indices."""
    if not 0 <= diversity <= 1 or pair_similarity.shape != (query_score.numel(),) * 2:
        raise ValueError("invalid MMR inputs")
    selected, remaining = [], set(range(query_score.numel()))
    while remaining and len(selected) < limit:
        if not selected:
            choice = max(remaining, key=lambda i: (query_score[i].item(), -i))
            selected.append(choice)
            remaining.remove(choice)
            continue
        def score(i):
            redundancy = max((pair_similarity[i, j].item() for j in selected), default=0.0)
            return diversity * query_score[i].item() - (1 - diversity) * redundancy
        choice = max(remaining, key=lambda i: (score(i), -i))
        selected.append(choice)
        remaining.remove(choice)
    return selected
```

```python
q = torch.tensor([1.0, 0.95, 0.8])
sim = torch.tensor([[1.0, 0.99, 0.1], [0.99, 1.0, 0.2], [0.1, 0.2, 1.0]])
assert mmr(q, sim, limit=2, diversity=0.5) == [0, 2]
```

文字相似不总等于证据重复；多跳任务应检查 gold evidence coverage，而非只优化多样性。

## 引用聚合

```python
def citation_report(claims):
    """claims contain requires_citation and citation labels: supported/unsupported/unknown."""
    required = [c for c in claims if c["requires_citation"]]
    cited = [c for c in required if c["citations"]]
    labels = [label for c in cited for label in c["citations"]]
    decidable = [label for label in labels if label != "unknown"]
    return {
        "completeness": len(cited) / len(required) if required else 1.0,
        "support_precision": decidable.count("supported") / len(decidable) if decidable else None,
        "decidable_rate": len(decidable) / len(labels) if labels else None,
    }
```

unknown 不能静默从报告消失，因此同时给 decidable rate。claim–evidence 设计见[证据约束生成](../applications/grounded-generation.md)。

## Tool schema

下面只实现足以展示边界的类型、必填项和未知字段检查：

```python
TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}

def validate_arguments(arguments, schema):
    """Validate a strict object schema and return a shallow canonical copy."""
    if not isinstance(arguments, dict) or schema.get("type") != "object":
        raise ValueError("tool input must be an object")
    properties = schema.get("properties", {})
    missing = set(schema.get("required", [])) - arguments.keys()
    extra = arguments.keys() - properties.keys()
    if missing or extra:
        raise ValueError(f"missing={sorted(missing)}, extra={sorted(extra)}")
    for name, value in arguments.items():
        expected = TYPE_MAP.get(properties[name].get("type"))
        if expected is None or isinstance(value, bool) and properties[name]["type"] != "boolean":
            raise ValueError(f"unsupported or ambiguous type for {name}")
        if not isinstance(value, expected):
            raise ValueError(f"wrong type for {name}")
        if "enum" in properties[name] and value not in properties[name]["enum"]:
            raise ValueError(f"invalid enum for {name}")
    return dict(arguments)
```

这不是完整 JSON Schema 实现，生产系统应使用经过测试的 validator。schema 合法仍不代表实体、权限或业务约束正确。

## 有权限、幂等的 dispatch

<details class="code-disclosure">
<summary id="durable-tool-dispatch-reference">可恢复的 tool dispatch <span class="code-disclosure__meta">Python · 49 行</span></summary>
<div class="code-disclosure__body" markdown="1">

```python
@dataclass
class Tool:
    name: str
    schema: dict
    risk: int
    execute: object

@dataclass
class ToolRuntime:
    tools: dict
    max_risk: int
    operations: dict = field(default_factory=dict)

    def dispatch(self, name, arguments, operation_id):
        if name not in self.tools:
            raise ValueError("unknown tool")
        tool = self.tools[name]
        if tool.risk > self.max_risk:
            raise PermissionError("tool exceeds delegated risk")
        args = validate_arguments(arguments, tool.schema)
        if operation_id in self.operations:
            entry = self.operations[operation_id]
            if entry["tool"] != name or entry["arguments"] != args:
                raise ValueError("operation_id was reused for a different request")
            if entry["status"] in {"succeeded", "failed"}:
                return dict(entry["result"])
            entry["status"] = "needs_reconcile"
            return {"status": "needs_reconcile", "operation_id": operation_id}
        self.operations[operation_id] = {
            "tool": name,
            "arguments": args,
            "status": "in_flight",
        }
        try:
            result = tool.execute(**args)
            if not isinstance(result, dict) or result.get("status") not in {
                "succeeded", "failed", "unknown",
            }:
                raise ValueError("tool returned an invalid status")
        except Exception:
            self.operations[operation_id]["status"] = "needs_reconcile"
            raise
        if result["status"] == "unknown":
            self.operations[operation_id].update(
                status="needs_reconcile", result=dict(result)
            )
            return {"status": "needs_reconcile", "operation_id": operation_id}
        self.operations[operation_id].update(
            status=result["status"], result=dict(result)
        )
        return dict(result)
```

</div>
</details>

下面分别模拟执行结果未知与调用抛出异常；两条路径都只能执行一次，后续同 ID 请求必须转入对账。

```python
uncertain_calls = []
def uncertain_write(project):
    uncertain_calls.append(project)
    return {"status": "unknown"}
write_schema = {
    "type": "object",
    "properties": {"project": {"type": "string"}},
    "required": ["project"],
}
uncertain_runtime = ToolRuntime(
    {"write": Tool("write", write_schema, 1, uncertain_write)}, max_risk=1
)
first = uncertain_runtime.dispatch("write", {"project": "alpha"}, "op-write-1")
second = uncertain_runtime.dispatch("write", {"project": "alpha"}, "op-write-1")
assert first["status"] == second["status"] == "needs_reconcile"
assert uncertain_calls == ["alpha"]

failed_calls = []
def interrupted_write(project):
    failed_calls.append(project)
    raise TimeoutError("response was lost")
failed_runtime = ToolRuntime(
    {"write": Tool("write", write_schema, 1, interrupted_write)}, max_risk=1
)
try:
    failed_runtime.dispatch("write", {"project": "alpha"}, "op-write-2")
except TimeoutError:
    pass
else:
    raise AssertionError("execution exception must propagate")
retry = failed_runtime.dispatch("write", {"project": "alpha"}, "op-write-2")
assert retry["status"] == "needs_reconcile" and failed_calls == ["alpha"]
```

`operation_id` 应由主体、工具、规范化参数与业务范围共同生成。runtime 在调用前先持久化 `in_flight`；timeout、异常或未知结果都会转入 `needs_reconcile`。同一 ID 此后只允许读取终态或进入对账，不能再次触发外部写入。

## 事件归约

持久化结构化事件，不保存自由形式的长篇内部推理：

```python
def reduce_task(state, event):
    """Pure reducer: old state + typed event -> new state."""
    state = dict(state)
    kind = event["type"]
    if state["phase"] in {"succeeded", "failed", "cancelled"}:
        if state["phase"] == "cancelled" and kind == "cancelled":
            return state
        raise ValueError("terminal task state cannot transition")
    if kind == "tool_requested":
        if state["phase"] not in {"planning", "observing"}:
            raise ValueError("tool request is not allowed now")
        state.update(phase="executing", operation_id=event["operation_id"])
    elif kind == "tool_succeeded":
        if state["phase"] != "executing":
            raise ValueError("tool result arrived outside execution")
        if state.get("operation_id") != event["operation_id"]:
            raise ValueError("stale tool result")
        state.update(
            phase="observing",
            evidence=event["evidence"],
            last_operation_id=state.pop("operation_id"),
        )
    elif kind == "goal_verified":
        if state["phase"] != "observing":
            raise ValueError("goal needs an observation")
        state.update(phase="succeeded")
    elif kind == "cancelled":
        state.update(phase="cancelled")
    else:
        raise ValueError(f"unknown event: {kind}")
    return state
```

```python
state = {"phase": "planning"}
state = reduce_task(state, {"type": "tool_requested", "operation_id": "op1"})
state = reduce_task(state, {"type": "tool_succeeded", "operation_id": "op1", "evidence": ["r7"]})
state = reduce_task(state, {"type": "goal_verified"})
assert state["phase"] == "succeeded"
try:
    reduce_task(state, {"type": "cancelled"})
except ValueError:
    pass
else:
    raise AssertionError("terminal state accepted cancellation")
try:
    reduce_task(
        {"phase": "planning", "operation_id": "op1"},
        {"type": "tool_succeeded", "operation_id": "op1", "evidence": []},
    )
except ValueError:
    pass
else:
    raise AssertionError("out-of-order tool result was accepted")
```

## 有界 agent loop

策略只返回候选动作；运行时执行、记录和验证：

```python
def run_agent(policy, runtime, initial_state, verify, max_steps):
    """policy(state)->answer|tool candidate; verify(state)->bool."""
    state = dict(initial_state)
    events = []
    for _ in range(max_steps):
        if verify(state):
            return {"status": "succeeded", "state": state, "events": events}
        candidate = policy(state)
        if candidate["kind"] == "answer":
            return {"status": "answered", "answer": candidate["text"], "events": events}
        if candidate["kind"] != "tool":
            raise ValueError("policy returned an unknown action")
        result = runtime.dispatch(
            candidate["name"], candidate["arguments"], candidate["operation_id"]
        )
        events.append({"candidate": candidate, "result": result})
        state["last_result"] = result
        status = result.get("status")
        if status not in {"succeeded", "failed", "unknown", "needs_reconcile"}:
            raise ValueError("tool returned an invalid status")
        if status in {"unknown", "needs_reconcile"}:
            return {
                "status": "needs_reconcile",
                "state": state,
                "events": events,
            }
    return {"status": "budget_exhausted", "state": state, "events": events}
```

回归用三步预算施压：第一次结果未知后，loop 必须立即退出；即使随后再次提交同一候选，也不能发生第二次外部执行。

```python
loop_calls = []
def uncertain_loop_write(project):
    loop_calls.append(project)
    return {"status": "unknown"}
loop_runtime = ToolRuntime(
    {"write": Tool("write", write_schema, 1, uncertain_loop_write)}, max_risk=1
)
candidate = {
    "kind": "tool", "name": "write", "arguments": {"project": "alpha"},
    "operation_id": "op-loop-1",
}
outcome = run_agent(lambda state: candidate, loop_runtime, {}, lambda state: False, 3)
assert outcome["status"] == "needs_reconcile"
assert len(outcome["events"]) == 1 and loop_calls == ["alpha"]
assert loop_runtime.dispatch(
    "write", {"project": "alpha"}, "op-loop-1"
)["status"] == "needs_reconcile"
assert loop_calls == ["alpha"]
```

真实运行时还要处理 approval、取消、重试分类、并发、敏感日志和崩溃恢复。最大步数只是预算的一维，不应替代时间、费用、写入和风险预算。

## 验证矩阵

- retrieval：空文档、稀有词、零向量、重复 ID、gold evidence coverage；
- context：去重、顺序置换、压缩后数字与否定条件；
- tool：no-tool、未知字段、布尔值冒充整数、越权、重复 operation ID；
- runtime：成功响应丢失、unknown side effect、取消与完成交错；
- memory：过期、冲突、跨租户和注入写入；
- security：不可信文档诱导外发或扩大权限。

机制说明见 [RAG](../applications/rag.md)、[工具调用](../applications/tool-use.md)、[运行时](../applications/agent-runtime.md)和[智能体安全](../applications/agent-security.md)。

## Reference {#reference}

- [RRF 原论文](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
