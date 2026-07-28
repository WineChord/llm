# vLLM 与 PagedAttention：把 KV Cache 变成分页状态

自回归服务的 KV Cache 会随每个请求逐 token 增长，结束时又突然释放。静态连续 tensor 很适合 kernel，却不适合长度未知、动态到达的请求。PagedAttention 的核心是把逻辑序列连续性与物理 KV 连续性解耦，使运行时可以按需分配、共享和回收固定大小 block。

## 为什么连续预留会压低 batch

对 $L$ 层、batch 中第 $i$ 条序列长度 $T_i$、KV head 数 $H_{\text{kv}}$、head dimension $d_h$、每元素 $b$ 字节：

$$
M_{\text{KV}}
=2LbH_{\text{kv}}d_h\sum_iT_i.
$$

若每条请求都按上限 $T_{\max}$ 预留，则浪费

$$
M_{\text{reserved}}-M_{\text{used}}
=2LbH_{\text{kv}}d_h
\sum_i(T_{\max}-T_i).
$$

请求长度差异、取消和 beam 分叉还会造成外部碎片。Continuous batching 虽然不断填补计算 slot，却会让更多不同生命周期的 KV 同时驻留；显存管理因此成为吞吐的前置条件。

## 页表与内部碎片

[PagedAttention 论文](https://arxiv.org/abs/2309.06180)把每条序列的逻辑 block 映射到物理 block：

$$
(\text{request},j)\mapsto b_j.
$$

block size 为 $P_b$ 个 token 时，长度为 $T$ 的序列需要

$$
N_{\text{block}}=\left\lceil\frac{T}{P_b}\right\rceil,
$$

末块内部浪费满足

$$
0\le
N_{\text{block}}P_b-T<P_b.
$$

小 block 减少尾部浪费，却增加 page-table metadata、allocator 操作和 kernel 查表；大 block 访问更规整，但短请求和共享尾部的机会成本更高。选择应基于真实长度与并发分布。

## Copy-on-write 是共享的正确性边界

多个请求可以引用相同的只读 prefix block。若共享的最后一块尚未填满，新分支不能直接续写；它必须复制该 block，再把新 token 写入私有副本。

### 最小分页与 COW 模拟

下面用标准库模拟 block table、refcount 和尾块 copy-on-write。它没有存储真实 K/V tensor，只固定所有权与逻辑 token 序列。

```python
class BlockPool:
    def __init__(self, blocks, block_size):
        self.block_size = block_size
        self.free = list(range(blocks))
        self.data = {}
        self.refs = {}
    def allocate(self, values=()):
        if not self.free:
            raise MemoryError("KV pool exhausted")
        block = self.free.pop()
        self.data[block] = list(values)
        self.refs[block] = 1
        return block
    def fork(self, table):
        copy = list(table)
        for block in copy:
            self.refs[block] += 1
        return copy
    def append(self, table, token):
        table = list(table)
        if not table or len(self.data[table[-1]]) == self.block_size:
            table.append(self.allocate())
        elif self.refs[table[-1]] > 1:
            shared = table[-1]
            self.refs[shared] -= 1
            table[-1] = self.allocate(self.data[shared])
        self.data[table[-1]].append(token)
        return table
    def read(self, table):
        return [token for block in table for token in self.data[block]]
pool = BlockPool(blocks=5, block_size=2)
original = []
for token in (10, 20, 30):
    original = pool.append(original, token)
branch = pool.fork(original)
branch = pool.append(branch, 40)
assert pool.read(original) == [10, 20, 30]
assert pool.read(branch) == [10, 20, 30, 40]
assert original[-1] != branch[-1]
assert pool.refs[original[0]] == 2 and pool.refs[original[-1]] == 1
```

生产 allocator 还必须处理 release、GPU event、并发更新、OOM rollback、preemption 和跨 worker install。紧凑 tensor/状态 reference 见[手撕：推理引擎](../../practice/inference-engine.md)。

## Attention kernel 如何读取分页 KV

逻辑 token 位置 $t$ 对应

$$
j=\left\lfloor\frac{t}{P_b}\right\rfloor,
\qquad
o=t\bmod P_b,
$$

再由 page table 找到物理 block $b_j$，访问其中 offset $o$ 的 K/V。间接寻址牺牲部分连续性和预取机会，却避免大规模连续预留与迁移。

Kernel metadata 必须同时描述：

- 每条序列的 query range 与 context length；
- logical-to-physical block table；
- layer/head/block layout；
- prefill、decode 或 verify 语义；
- causal、sliding-window、prefix 或 branch mask；
- KV dtype、quantization scales 与 schema version。

只根据输入 tensor shape 推断这些语义，会在 mixed batch、chunked prefill 或 speculative verify 中产生静默错误。Kernel 侧权衡见 [Attention Kernel](../../systems/attention-kernels.md)。

## Continuous batching 与 paging 互相成就

Iteration-level scheduler 每轮移除完成请求并接纳新请求。分页 allocator 使这种接纳不再要求找到一块足够大的连续区间；更高的可用 batch 又提高 decode 对权重读取的摊销。

但调度器必须在执行前预留本轮可能增长的 block：

$$
\sum_i q_i\le K_{\text{token}},
\qquad
\sum_i
\left\lceil
\frac{T_i+q_i}{P_b}
\right\rceil
\le N_{\text{block available+owned}}.
$$

只检查当前 free blocks 会忽略已有请求的增长承诺。输出长度未知时，admission 还需使用上限、预测或抢占策略，并明确 OOM 时是 recompute、swap 还是拒绝。

调度状态机见[推理运行时](../../inference/runtime.md)，容量和 SLO 见[调度与 Goodput](../../inference/scheduling-goodput.md)。

## Prefix、beam 与 speculative 都是 block 所有权问题

- Prefix cache：多个请求共享只读历史，命中后增加 refcount。
- Beam search：分叉前共享，续写非满尾块前 COW。
- Speculative decoding：候选 token 写入 provisional blocks，接受后 commit，拒绝后回收。
- Preemption：释放或迁移物理 block，但保留逻辑 token、sampler 与 grammar state。
- Cancel：队列、page table、cache reference 和 GPU event 全部完成后才能回收。

Prefix cache key 至少绑定 exact token IDs、model/adapter、tokenizer/template、position/RoPE、KV dtype/layout 和安全域。字符串相同不足以证明 KV 兼容。完整契约见[缓存复用](../../inference/cache-reuse.md)和[推测解码](../../inference/speculative-decoding.md)。

## 分页没有消除所有浪费

PagedAttention 主要减少连续预留、外部碎片和重复复制，但仍存在：

- 每条序列最后一块的内部碎片；
- block table、refcount 与 allocator metadata；
- cache 中冷 prefix 的显存机会成本；
- beam/speculative 的临时分支；
- kernel workspace 与 graph capture 预留；
- TP/PP worker 之间重复或分片的 KV；
- 量化 scale 和对齐 padding。

因此论文中的 near-zero waste 指向特定管理基线与实验条件，不应解释为运行时没有任何 KV 开销。

## 从单机分页到分布式状态

Prefill–Decode 分离时，prefill worker 必须把逻辑 KV 描述与物理数据传给 decode worker。若两端 block size、TP degree 或 head layout 不同，需要 reshard 和重新安装：

$$
\text{layout}_{P}\longrightarrow\text{layout}_{D}.
$$

传输只有在 model、adapter、position、dtype 和 cache schema 完全兼容时才能复用。安装应幂等，同一请求重试不能生成两份 owner。跨 worker 的数据路径见[推理服务谱系](../lineages/inference-serving.md)和 [Prefill–Decode 分离](../../inference/disaggregation.md)。

## Reference {#reference}

- [PagedAttention/vLLM 论文](https://arxiv.org/abs/2309.06180)是一手算法与系统评测来源。
- [vLLM 官方仓库](https://github.com/vllm-project/vllm)持续演进 scheduler、KV manager、prefix caching、distributed serving 与 kernel；当前类名和配置不能反向当作论文接口。
- 论文中的 block 管理思想可以由不同 kernel 和 allocator 实现；“分页 KV”不保证采用 vLLM 的具体内存布局。
- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) 是一手 iteration-level scheduling 来源。Continuous batching 与 PagedAttention 相互补充，但不是同一项工作。

评测至少报告 block size、KV dtype、模型 head layout、prompt/output 分布、并发、token budget、内部浪费、prefix hit、TTFT、TPOT、goodput 与峰值显存。只比较静态同长度 batch 会绕过分页系统真正解决的动态生命周期问题。
