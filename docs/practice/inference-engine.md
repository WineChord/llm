# 手撕：推理引擎

推理引擎把逻辑 token 序列映射到物理 KV、把不同阶段请求组成 batch，并在取消、复用和推测解码时维护状态。本页实现元数据与精确语义，不模拟 GPU kernel。

## KV 容量

$L$ 层、batch $B$、上下文 $T$、KV head 数 $A_{\mathrm{kv}}$、head dimension $D$、元素字节数 $s$：

$$
M_{\mathrm{KV}}=2LBTA_{\mathrm{kv}}Ds.
$$

```python
import math
from dataclasses import dataclass
from enum import Enum
import torch

def kv_bytes(layers, batch, tokens, kv_heads, head_dim, element_bytes):
    values = (layers, batch, tokens, kv_heads, head_dim, element_bytes)
    if any(v <= 0 for v in values):
        raise ValueError("KV dimensions must be positive")
    return 2 * layers * batch * tokens * kv_heads * head_dim * element_bytes
```

只有 cache 实际按 TP rank 分片时，per-rank 大小才能再除以 TP size。

## Page table

page 大小为 $q$，逻辑位置 $t$ 映射到：

$$
b_{\mathrm{logical}}=\lfloor t/q\rfloor,\qquad
\operatorname{slot}=q\cdot
\operatorname{table}[b_{\mathrm{logical}}]+(t\bmod q).
$$

```python
def physical_slot(block_table, block_size, token_pos):
    """block_table:[logical blocks] -> one physical token slot."""
    if block_size <= 0 or token_pos < 0:
        raise ValueError("invalid block size or token position")
    logical, offset = divmod(token_pos, block_size)
    if logical >= len(block_table):
        raise IndexError("logical block is not allocated")
    return block_table[logical] * block_size + offset
```

page table 允许逻辑序列连续而物理块离散。它不自动实现 prefix sharing 或 eviction。

## Refcount 与 copy-on-write

一个物理块只能是 free、exclusive 或 shared。共享块发生部分写入前必须 COW：

```python
class BlockAllocator:
    def __init__(self, blocks):
        if blocks <= 0:
            raise ValueError("allocator needs physical blocks")
        self.refcount = [0] * blocks
        self.free = set(range(blocks))

    def allocate(self):
        if not self.free:
            raise MemoryError("KV blocks exhausted")
        block = self.free.pop()
        self.refcount[block] = 1
        return block

    def share(self, table):
        table = list(table)
        if len(set(table)) != len(table):
            raise ValueError("one block table cannot contain duplicate blocks")
        if any(block < 0 or block >= len(self.refcount) for block in table):
            raise IndexError("physical block is out of range")
        if any(self.refcount[block] <= 0 for block in table):
            raise ValueError("cannot share a free block")
        for block in table:
            self.refcount[block] += 1
        return table

    def cow(self, table, logical_block):
        table = list(table)
        if not 0 <= logical_block < len(table):
            raise IndexError("logical block is out of range")
        old = table[logical_block]
        if not 0 <= old < len(self.refcount) or self.refcount[old] <= 0:
            raise ValueError("cannot copy an invalid block")
        if self.refcount[old] == 1:
            return table, None
        new = self.allocate()
        self.refcount[old] -= 1
        table[logical_block] = new
        return table, (old, new)

    def release(self, table):
        table = list(table)
        if len(set(table)) != len(table):
            raise ValueError("one block table cannot contain duplicate blocks")
        if any(block < 0 or block >= len(self.refcount) for block in table):
            raise IndexError("physical block is out of range")
        if any(self.refcount[block] <= 0 for block in table):
            raise ValueError("cannot release a free block")
        for block in table:
            self.refcount[block] -= 1
            if self.refcount[block] == 0:
                self.free.add(block)
```

```python
allocator = BlockAllocator(4)
parent = [allocator.allocate(), allocator.allocate()]
child = allocator.share(parent)
child, copy = allocator.cow(child, 1)
assert copy == (parent[1], child[1])
assert allocator.refcount[parent[0]] == 2
allocator.release(parent)
allocator.release(child)
assert len(allocator.free) == 4
full = BlockAllocator(1)
shared = full.share([full.allocate()])
try:
    full.cow(shared, 0)
except MemoryError:
    pass
assert full.refcount[shared[0]] == 2
probe = BlockAllocator(2)
live = probe.allocate()
free_block = next(iter(probe.free))
before = list(probe.refcount)
for operation in (probe.share, probe.release):
    try:
        operation([live, free_block])
    except ValueError:
        pass
    assert probe.refcount == before
```

`cow` 返回的 `(old,new)` 要驱动实际 KV payload copy；GPU event 完成前，旧块或新块都不可错误复用。

## Prefix reuse

cache key 不能只有自然语言字符串。至少绑定精确 token IDs、model/weight、tokenizer/template、adapter、RoPE、dtype/layout/quant schema 与 security domain。

下面是故意使用线性扫描的 reference：

```python
class PrefixCache:
    def __init__(self):
        self.entries = {}

    def put(self, scope, token_ids, block_table):
        """scope is an immutable compatibility tuple."""
        key = (tuple(scope), tuple(token_ids))
        self.entries[key] = tuple(block_table)

    def longest(self, scope, token_ids):
        scope, token_ids = tuple(scope), tuple(token_ids)
        best = (0, ())
        for (entry_scope, prefix), table in self.entries.items():
            if entry_scope == scope and len(prefix) > best[0] and token_ids[:len(prefix)] == prefix:
                best = (len(prefix), table)
        return best
```

生产实现可用 radix tree，不能改变 scope 精确匹配。前缀命中只有在 lookup、load 与 install 成本小于重算节省时才有收益。

## Request state machine

请求任一时刻只能属于一个状态；finish 与 cancel 应幂等：

```python
class Phase(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"

ALLOWED = {
    Phase.WAITING: {Phase.PREFILL, Phase.CANCELLED, Phase.FAILED},
    Phase.PREFILL: {Phase.DECODE, Phase.CANCELLED, Phase.FAILED},
    Phase.DECODE: {Phase.FINISHED, Phase.CANCELLED, Phase.FAILED},
}

@dataclass
class Request:
    request_id: str
    prompt_left: int
    generated: int = 0
    phase: Phase = Phase.WAITING

    def transition(self, target):
        if self.phase == target and target in {Phase.FINISHED, Phase.CANCELLED}:
            return
        if target not in ALLOWED.get(self.phase, set()):
            raise ValueError(f"illegal transition: {self.phase.value} -> {target.value}")
        self.phase = target
```

运行时还要保存 block table、RNG、grammar、adapter、priority、deadline 与 finish reason。状态转移发生后再崩溃，恢复必须能判断对应资源是否已释放。

## Continuous batching

decode 每个请求通常只需一个 token，prefill 可以切成 chunk。下面先服务 decode，再用剩余 token budget 放 prefill：

```python
def build_batch(requests, token_budget, max_sequences, prefill_chunk):
    """Return [(request_id, phase, scheduled_tokens)] within both budgets."""
    if min(token_budget, max_sequences, prefill_chunk) <= 0:
        raise ValueError("scheduler budgets must be positive")
    batch, used = [], 0
    decode = [r for r in requests if r.phase == Phase.DECODE]
    prefill = [r for r in requests if r.phase in {Phase.WAITING, Phase.PREFILL}]
    for request in decode + prefill:
        if len(batch) == max_sequences or used == token_budget:
            break
        tokens = 1 if request.phase == Phase.DECODE else min(
            request.prompt_left, prefill_chunk, token_budget - used
        )
        if tokens > 0 and used + tokens <= token_budget:
            batch.append((request.request_id, request.phase, tokens))
            used += tokens
    return batch
```

这只是 decode-priority baseline，可能饿死长 prefill。生产调度还要加入 admission、age/deadline、公平性、KV 预算、cache affinity 与 overload 策略。

## Group quantization

对每组对称 $b$ bit 量化：

$$
s_g=\frac{\max|x_g|}{2^{b-1}-1},\qquad
q_g=\operatorname{round}(x_g/s_g).
$$

```python
def quantize_groups(x, bits=4, group_size=32):
    """x:[...,D] -> integer q:[...,D], scale:[...,D/group,1]."""
    if bits < 2 or x.size(-1) % group_size:
        raise ValueError("invalid bits or non-divisible group size")
    qmax = 2 ** (bits - 1) - 1
    shape = (*x.shape[:-1], x.size(-1) // group_size, group_size)
    group = x.float().reshape(shape)
    scale = (group.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
    q = (group / scale).round().clamp(-qmax, qmax).to(torch.int8)
    return q.reshape_as(x), scale

def dequantize_groups(q, scale, group_size):
    shape = (*q.shape[:-1], q.size(-1) // group_size, group_size)
    return (q.reshape(shape).float() * scale).reshape_as(q)

def effective_bits(bits, group_size, scale_bits=16, zero_bits=0):
    return bits + (scale_bits + zero_bits) / group_size
```

```python
x = torch.randn(3, 64)
q, scale = quantize_groups(x, bits=4, group_size=16)
x_hat = dequantize_groups(q, scale, 16)
assert q.dtype == torch.int8 and x_hat.shape == x.shape
assert effective_bits(4, 16) == 5
```

checkpoint 位宽、运行时权重格式、activation dtype、accumulator dtype 和实际 GEMM kernel 必须分别报告。存成 4 bit 不代表执行了低比特 GEMM。

## Exact speculative step

draft 分布 $q$ 提议 token $x$，target 分布 $p$ 以

$$
\alpha(x)=\min\left(1,\frac{p(x)}{q(x)}\right)
$$

接受；拒绝后从正残差 $[p-q]_+$ 采样：

```python
def speculative_step(p, q, proposal, generator=None):
    """p/q:[V] normalized distributions -> token, accepted."""
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("p and q must share one token space")
    p, q = p.double(), q.double()
    p, q = p / p.sum(), q / q.sum()
    qx = q[proposal]
    accept = 1.0 if qx == 0 else min(1.0, (p[proposal] / qx).item())
    if torch.rand((), generator=generator).item() < accept:
        return int(proposal), True
    residual = (p - q).clamp_min(0)
    residual = p if residual.sum() == 0 else residual / residual.sum()
    return int(torch.multinomial(residual, 1, generator=generator)), False
```

draft 与 target 必须使用一致的 logit processor、grammar 与 token space。拒绝时还要回滚 KV、RNG 和 grammar state；近似 variant 应明确标注分布已改变。

## P/D cache descriptor

decode worker 安装远端 KV 前，校验不可变布局描述：

```python
REQUIRED_KV_FIELDS = {
    "request_id", "model_version", "token_ids_hash", "layers",
    "kv_heads", "head_dim", "dtype", "layout", "rope", "security_domain",
}

def validate_kv_descriptor(descriptor, expected):
    missing = REQUIRED_KV_FIELDS - descriptor.keys()
    if missing:
        raise ValueError(f"missing KV fields: {sorted(missing)}")
    mismatch = {
        key: (descriptor.get(key), value)
        for key, value in expected.items()
        if descriptor.get(key) != value
    }
    if mismatch:
        raise ValueError(f"incompatible KV descriptor: {mismatch}")
    return descriptor["request_id"]
```

install 还应绑定 operation ID 并保持幂等。传输完成、checksum 通过和 block table 提交之前，请求不能进入 decode。

## Goodput

goodput 只统计同时满足 TTFT、TPOT、E2E 与质量 SLO 的请求：

```python
def goodput(records, window_seconds, slo):
    """records contain ttft, tpot, e2e, quality_ok, status."""
    if window_seconds <= 0:
        raise ValueError("window must be positive")
    ok = [
        r for r in records
        if r["status"] == "finished"
        and r["quality_ok"]
        and r["ttft"] <= slo["ttft"]
        and r["tpot"] <= slo["tpot"]
        and r["e2e"] <= slo["e2e"]
    ]
    return len(ok) / window_seconds
```

吞吐峰值不等于可交付容量。还要单列 admission reject、cancel、OOM、preemption、cache miss、compile 与 unknown failure。

## 不变量与故障注入

- refcount 等于 live owner 数；
- shared block 在部分写前 COW；
- finish/cancel/resource release 幂等；
- 每个请求只在一个 queue 或执行状态；
- token counter 单调；
- cache key 跨版本与安全域不误命中；
- speculative reject 后 KV/RNG/grammar 一致回滚；
- P/D install 不接受 partial、stale 或不兼容 descriptor；
- admission 为输出上限预留 KV，过载行为明确。

机制与成本见 [KV Cache](../inference/kv-cache.md)、[调度与服务](../inference/serving.md)、[量化](../inference/quantization.md)和[推测解码](../inference/speculative-decoding.md)。

## Reference {#reference}

- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
