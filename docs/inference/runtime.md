# 推理运行时

推理引擎把一组长度不同、动态到达的请求映射为 GPU batch。核心状态不是一个输入 tensor，而是请求生命周期、KV block、调度预算、采样器和流式输出的组合。

[推理运行时与服务](../landscape/lineages/inference-serving.md)给出从 Orca 到 PagedAttention、chunked prefill 与 P/D 分离的因果链；[vLLM / PagedAttention](../landscape/works/vllm-pagedattention.md) 进一步把逻辑 token、物理 block、block table 与 copy-on-write 落成可执行模型。

## 请求状态机

一个请求至少经历：

```text
waiting
-> admitted
-> prefilling
-> decoding
-> finished / cancelled / failed
```

被抢占的请求还可能进入 `swapped` 或 `recompute` 状态。每次转移都要同时更新队列、KV 所有权、token 数、采样状态和客户端流。

### 核心转移不变量 {#request-transition-reference}

最小转移函数接收当前状态、目标状态和当前 KV block 数，返回新的状态与资源数；任何终态都释放请求所有权，非法边直接失败。

```python
ALLOWED = {
    "waiting": {"admitted", "cancelled", "failed"},
    "admitted": {"prefilling", "cancelled", "failed"},
    "prefilling": {"decoding", "cancelled", "failed"},
    "decoding": {"finished", "cancelled", "failed"},
}
TERMINAL = {"finished", "cancelled", "failed"}

def apply_transition(state, target, kv_blocks):
    if not isinstance(kv_blocks, int) or kv_blocks < 0:
        raise ValueError("kv_blocks must be a non-negative integer")
    if target not in ALLOWED.get(state, set()):
        raise ValueError(f"illegal transition: {state} -> {target}")
    return target, 0 if target in TERMINAL else kv_blocks

state, blocks = apply_transition("waiting", "admitted", 0)
state, blocks = apply_transition(state, "prefilling", 2)
state, blocks = apply_transition(state, "cancelled", blocks)
assert state == "cancelled" and blocks == 0
assert "decoding" not in ALLOWED.get(state, set())
try:
    apply_transition("waiting", "admitted", -1)
except ValueError:
    pass
else:
    raise AssertionError("negative KV ownership must be rejected")
```

状态机的输入是当前状态、目标状态与资源变化，输出是一个新的可枚举状态；终态必须同步清空 KV 所有权。完整请求类比单个转移函数更长，因此默认折叠，但其中的状态集合、合法边和终态回收仍是正文语义的一部分。

<details class="code-disclosure">
<summary id="request-state-machine">请求状态转移与资源回滚 <span class="code-disclosure__meta">Python · 51 行</span></summary>
<div class="code-disclosure__body" markdown="1">

```python
from dataclasses import dataclass
from enum import Enum, auto
class State(Enum):
    WAITING = auto()
    ADMITTED = auto()
    PREFILLING = auto()
    DECODING = auto()
    FINISHED = auto()
    CANCELLED = auto()
    FAILED = auto()
TERMINAL = {State.FINISHED, State.CANCELLED, State.FAILED}
ALLOWED = {
    State.WAITING: {State.ADMITTED, State.CANCELLED, State.FAILED},
    State.ADMITTED: {State.PREFILLING, State.CANCELLED, State.FAILED},
    State.PREFILLING: {State.DECODING, State.CANCELLED, State.FAILED},
    State.DECODING: TERMINAL,
}
@dataclass
class Request:
    request_id: str
    state: State = State.WAITING
    kv_blocks: int = 0
    computed_tokens: int = 0
    def transition(self, target):
        if target not in ALLOWED.get(self.state, set()):
            raise ValueError(f"illegal transition: {self.state} -> {target}")
        self.state = target
        if target in TERMINAL:
            self.kv_blocks = 0
    def reserve_blocks(self, count):
        if self.state not in {State.ADMITTED, State.PREFILLING, State.DECODING}:
            raise ValueError("request does not own runtime resources")
        if not isinstance(count, int) or count <= 0:
            raise ValueError("reservation count must be a positive integer")
        self.kv_blocks += count
r = Request("r0")
r.transition(State.ADMITTED)
r.reserve_blocks(2)
try: r.reserve_blocks(0)
except ValueError: pass
else: raise AssertionError("non-positive reservation must be rejected")
r.transition(State.PREFILLING)
r.transition(State.DECODING)
r.computed_tokens = 8
r.transition(State.FINISHED)
assert r.kv_blocks == 0 and r.computed_tokens == 8
try:
    r.transition(State.DECODING)
    raise AssertionError("terminal request was revived")
except ValueError:
    pass
```

</div>
</details>

reference 把非法跳转和终态复活变为显式错误，并保持 `computed_tokens` 作为已提交历史。生产引擎还需让队列移动、block 分配、采样状态和客户端事件在同一事务边界提交；尤其不能把“字段已改”误当作 GPU 使用已经结束。block table、连续批处理与回滚的组合实现见[手撕：推理引擎](../practice/inference-engine.md)。

最小请求对象包括：

```text
request and tenant id
prompt / generated token ids
priority, deadline and arrival time
sampling and grammar state
model, adapter and tokenizer version
logical block table
computed-token count
finish reason and cancellation token
```

## Continuous batching

静态 batch 等所有序列结束再换批，短请求会等待最长请求。continuous batching 在每个模型迭代边界移除已完成请求、加入新请求。[Orca](https://www.usenix.org/conference/osdi22/presentation/yu) 给出了迭代级调度的代表性设计。

调度器每轮决定：

- 哪些 waiting 请求做 prefill；
- 哪些 running 请求做一个或多个 decode token；
- 本轮总 token budget；
- 是否抢占或换出低优先级请求；
- 是否有足够 KV block；
- 哪些请求因 SLO 或公平性优先。

“batch size”因此是动态的；性能报告应同时给出 scheduled sequences 与 scheduled tokens。

### `build_batch` baseline {#continuous-batching-build-batch-reference}

输入是一轮开始时的请求快照，以及 token、序列和 prefill chunk 三个预算；输出为 `(request_id, phase, scheduled_tokens)` 列表。baseline 先给每个 decode 请求一个 token，再用剩余预算放入 waiting / prefill 请求。

```python
def build_batch(requests, token_budget, max_sequences, prefill_chunk):
    if min(token_budget, max_sequences, prefill_chunk) <= 0:
        raise ValueError("scheduler budgets must be positive")
    request_ids = [request["id"] for request in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request ids must be unique within a scheduling round")
    batch, used = [], 0
    decode = [r for r in requests if r["phase"] == "decode"]
    prefill = [r for r in requests if r["phase"] in {"waiting", "prefill"}]
    for request in decode + prefill:
        if len(batch) == max_sequences or used == token_budget:
            break
        available = token_budget - used
        tokens = 1 if request["phase"] == "decode" else min(
            request["prompt_left"], prefill_chunk, available
        )
        if tokens > 0 and tokens <= available:
            batch.append((request["id"], request["phase"], tokens))
            used += tokens
    return batch

requests = [{"id": "d0", "phase": "decode", "prompt_left": 0},
            {"id": "p0", "phase": "prefill", "prompt_left": 6},
            {"id": "d1", "phase": "decode", "prompt_left": 0}]
batch = build_batch(requests, token_budget=5, max_sequences=3, prefill_chunk=3)
assert [item[0] for item in batch] == ["d0", "d1", "p0"]
assert sum(item[2] for item in batch) == 5
assert all(tokens == 1 for _, phase, tokens in batch if phase == "decode")
try:
    build_batch(requests + [requests[0]], 5, 4, 3)
except ValueError:
    pass
else:
    raise AssertionError("one request cannot be scheduled twice in a round")
```

每个请求在一轮中最多出现一次，decode 恰好消费一个 token，且两个预算都不得越界。函数不修改请求状态，也不承诺公平性；生产调度还必须联合 KV / workspace reservation、age、deadline、cache affinity、抢占和原子提交。状态推进与更多预算断言见[手撕：推理引擎 · Continuous batching](../practice/inference-engine.md#continuous-batching-reference)。

## 分页 KV

连续为每个请求预留最大上下文会浪费显存。分页运行时把逻辑 token 区间映射到固定大小物理 block：

$$
\text{logical block }j
\longrightarrow
\text{physical block }b_j.
$$

block table 让一个请求的 KV 不必物理连续。[PagedAttention](https://arxiv.org/abs/2309.06180) 进一步让 attention kernel 按映射读取 K/V。

### Block 大小

- 大 block：映射少、访问规整，但最后一块内部碎片大；
- 小 block：碎片小，但元数据、分配和 kernel 查表更多。

选择必须使用真实 prompt/output 长度与并发分布，而不是只测固定长度。

## Prefix cache 与 copy-on-write

共享 system prompt、文档前缀或 beam 分支可指向相同只读 block。当前缀末尾的 block 尚未填满，新分支继续写入前必须 copy-on-write，否则会修改其他请求的历史。

cache key 至少包含：

```text
exact token ids
model and weights
adapter
position and rope configuration
attention/cache dtype
tenant or permission boundary
```

字符串相同但 tokenization、adapter 或权限不同都不能复用。跨租户共享即使内容碰巧相同，也要经过明确的数据隔离设计。

## Chunked prefill

长 prompt 的一次 prefill 可能占据整个 GPU，阻塞交互式 decode。chunked prefill 将其分为多个 token 块，与 decode 请求拼成每轮预算：

$$
N_{\text{scheduled}}
=N_{\text{decode}}+N_{\text{prefill chunk}}.
$$

[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) 展示了用 chunked prefill 控制 throughput–latency 权衡的路线。chunk 越大，prefill GEMM 越高效；越小，decode stall 越低但 launch 和重复调度更多。

## 抢占

显存不足时有两种常见选择：

- **recompute**：释放 KV，恢复时重新 prefill 已处理 token；
- **swap**：把 KV 移到 CPU 或更低层存储，恢复时传回。

recompute 消耗计算，swap 消耗链路带宽和主机内存。选择取决于上下文长度、带宽、优先级和预期等待时间。无论哪种方式，都必须保留 sampler、grammar 与 position 状态。

## Prefill 与 decode 输入

运行时常把不同请求的 token 展平成一条 token array，并用 metadata 描述每个序列的 query range、context length 和 block table。kernel 正确性依赖这些 offset：

- prefill 请求可能一次有多个 query token；
- decode 请求通常只有一个新 query；
- chunked prefill 已有历史 cache；
- prefix hit 使“prompt 长度”不等于“本轮计算长度”；
- speculative verify 一次有多个候选 query。

只根据 tensor shape 推断语义容易在混合 batch 中出错。

## Streaming 与 detokenization

token 生成完成不等于客户端已收到文本。运行时还要：

- 增量解码 byte/subword；
- 缓冲潜在 stop sequence；
- 处理 Unicode 未完成片段；
- 发送 backpressure；
- 客户端断开后及时取消；
- 保留 finish reason；
- 避免慢客户端阻塞 GPU 调度线程。

tokenizer 和网络发送通常放在独立线程或进程，但状态顺序必须与请求 token 序列一致。

## 引擎正确性

最小回归矩阵包括：

1. 单请求 prefill + decode 对齐框架 reference；
2. 不同长度 continuous batch；
3. prefix hit/miss 与 copy-on-write；
4. block 分配、回收和抢占；
5. chunked prefill 与 decode 混排；
6. beam/speculative 分支；
7. adapter、量化和不同 cache dtype；
8. 取消、超时、OOM 与进程重启；
9. 固定 seed 下的可接受重放边界。

策略与过载控制见[调度与服务](serving.md)，跨集群阶段拆分见 [Prefill–Decode 分离](disaggregation.md)。

## Reference {#reference}

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)
