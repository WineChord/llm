# KV Cache：容量、布局与生命周期

自回归 decode 在第 $t$ 步只产生少量新 query，却要访问此前所有 token 的 key 与 value。缓存历史 K/V 可以避免重复执行 projection，也把推理的主要约束从计算转向显存容量、读带宽和状态管理。长上下文服务中，KV 往往比单个请求的 activation 大得多；如果只计算“每 token 字节数”而忽略分页、所有权和临时 buffer，容量规划仍会失真。

## 容量模型

设模型有 $L$ 层，batch 中共有 $B$ 条等长序列，每条缓存 $T$ 个 token；query head 数为 $H_q$，K/V head 数为 $H_{kv}$，head dimension 为 $d_h$，每个缓存元素占 $s$ 字节。未压缩 KV 的主体大小为

$$
M_{\mathrm{KV}}
=2LBTH_{kv}d_hs.
$$

系数 $2$ 对应 K 和 V。单 token、单请求的增量为

$$
m_{\mathrm{token}}
=2LH_{kv}d_hs.
$$

对 MHA，$H_{kv}=H_q$；GQA 与 MQA 通过减小 $H_{kv}$ 降低缓存和读取量。这个收益不等于 attention FLOPs 按同一比例下降，因为每个 query head 仍要计算自己的 score 和 value 加权。

真实峰值应写成显存账本：

$$
M_{\mathrm{peak}}
=M_{\mathrm{weights}}
+M_{\mathrm{KV\ pool}}
+M_{\mathrm{workspace}}
+M_{\mathrm{graph}}
+M_{\mathrm{allocator}}
+M_{\mathrm{transient}}.
$$

只有 cache 确实按 head 或其他维度分片时，才能把 $M_{\mathrm{KV}}$ 除以 tensor-parallel size；复制式布局不能这样估算。量化 scale、block table、对齐、copy-on-write 和通信 staging 也必须计入。

## 连续布局为什么会失效

若为每条请求一次性预留最大长度 $T_{\max}$，长度为 $T_i$ 的请求会浪费 $T_{\max}-T_i$ 个 slot；请求增长、取消和不同长度混排还会造成外部碎片。[PagedAttention](https://arxiv.org/abs/2309.06180)把逻辑 token 区间映射到固定大小物理 block，使逻辑连续与物理连续解耦：

$$
\text{logical block }j
\longrightarrow
\text{physical block }b_j.
$$

若一个 block 容纳 $q$ 个 token，$n$ 条活跃序列的尾块内部碎片上界为

$$
F_{\max}=n(q-1).
$$

当尾部余数近似均匀时，期望浪费约为

$$
\mathbb E[F]\approx\frac{n(q-1)}{2}.
$$

小 block 降低尾部浪费，却增加 block table、分配频率和 kernel 查表；大 block 的访问更规整，却让短请求和频繁抢占更昂贵。应使用真实长度分布比较“有效 KV 字节 / 已分配 KV 字节”，而不是只比较理论容量。

[vAttention](https://arxiv.org/abs/2405.04437)展示了另一条精确路线：利用虚拟内存让逻辑 KV 保持连续，并按需映射物理页。分页张量和虚拟内存都不改变模型语义，但依赖不同的 allocator、kernel 和运行时边界；不能只凭论文中的单机结果断言某种布局对所有 GPU、页大小和负载更优。

## 物理布局

逻辑上常写成

$$
K,V\in
\mathbb R^{L\times B\times H_{kv}\times T\times d_h},
$$

生产实现却可能按 layer、block、head、token 或 vector width 重排。布局设计需要同时回答：

- decode kernel 沿哪个维度连续读取；
- GQA 中多个 query head 如何映射到一个 KV head；
- TP / CP 是否分片 cache，跨 rank attention 怎样获得全局上下文；
- block size、vector width 与量化 group 是否对齐；
- P/D 分离时怎样描述 shard、dtype、scale 和字节序；
- CUDA Graph 捕获后物理地址能否稳定复用。

布局 schema 是持久状态的一部分。仅凭 tensor shape 相同，不能认定两个 cache 可互换。

## 生命周期与所有权

物理 block 至少有三种状态：

```text
free
exclusive
shared-read-only
```

最小不变量是：

1. `refcount` 等于仍持有该 block 的逻辑序列数；
2. shared block 被部分续写前必须 copy-on-write；
3. 最后一次 GPU 读写完成前不得进入 free list；
4. cancel、finish、抢占和错误回滚只能释放一次；
5. block table 更新与 computed-token count 同步提交；
6. 未初始化 slot 永远不能被 attention 读取。

最后一块尤其危险：两个请求可能共享完整前缀，却各自继续写入同一个未填满 block。若没有 copy-on-write，错误通常只在并发或特定长度下出现，很难由单请求 logits 回归发现。

运行时状态机、block table 与抢占见[推理运行时](runtime.md)；跨 worker 的布局转换与安装见[Prefill–Decode 分离](disaggregation.md)。

## 精确节省与有损压缩

需要把不同方法分开评价：

| 方法 | 是否保持原模型语义 | 主要收益 | 主要代价 |
| --- | --- | --- | --- |
| MQA / GQA | 架构定义内精确 | 减少 KV head | 需模型原生支持或重新训练 |
| 分页、虚拟内存、COW | 精确 | 降低预留与复制 | 元数据和 kernel 复杂度 |
| Prefix reuse | 命中条件满足时精确 | 跳过重复 prefill | 查找、隔离与一致性 |
| KV 量化 | 通常近似 | 减少容量和带宽 | 误差、scale 与反量化开销 |
| Window / eviction | 近似 | 将容量限制在窗口内 | 丢失历史可见性 |
| 摘要、低秩、token merge | 近似 | 更激进压缩 | 改变信息和模型行为 |

[KIVI](https://arxiv.org/abs/2402.02750)按 key 和 value 的不同统计特征设计非对称低比特量化；[KVQuant](https://arxiv.org/abs/2401.18079)研究逐通道量化、pre-RoPE key 等设计。它们是质量—容量权衡，不能与精确分页混称为“无损 KV 优化”。[StreamingLLM](https://arxiv.org/abs/2309.17453)、[H2O](https://arxiv.org/abs/2306.14048)与 [PyramidKV](https://arxiv.org/abs/2406.02069)进一步改变保留 token 的集合，必须以任务和长上下文质量验证。

量化格式、scale 粒度和真实 kernel 路径见[量化](quantization.md)；前缀复用、缓存键和多层缓存见[缓存复用](cache-reuse.md)。

## 正确性契约

cache descriptor 至少绑定：

```text
exact token ids and computed-token count
model and weight version
tokenizer and prompt-template version
adapter and multimodal-input identity
RoPE and position configuration
layer, head, block and shard layout
storage dtype, quantization schema and scales
tenant and permission boundary
```

读取 cache 前要验证 descriptor；版本不兼容应明确 miss 或拒绝，不能“尽力解释”。量化 cache 还要定义 scale 的轴、group、更新时机和 accumulator dtype。跨设备传输必须在校验、DMA 完成和目标端 block 安装后才能让 decode 可见。

## 何时不应复杂化

- 单请求、短上下文、显存充足时，连续 cache 可能更简单且更快；
- block table 查找或 page fault 已进入关键路径时，应先测 layout 和页粒度；
- prefix 重复率低时，维护索引与逐出策略可能得不偿失；
- 质量敏感且缺少代表性长上下文评测时，不应贸然量化或淘汰 KV；
- 网络不足以隐藏 cache 迁移时，不应仅为资源解耦而引入远程层。

## 验证

功能回归至少覆盖：

1. 连续 reference 与分页 / 虚拟内存路径的 logits 对齐；
2. block 边界、空序列、最大长度和非整除 head；
3. prefix hit / miss、共享尾块与 copy-on-write；
4. cancel、beam 分叉、speculative 拒绝、抢占和 OOM 回滚；
5. TP / CP 分片与 P/D 布局转换；
6. 量化 cache 的逐层误差和端到端任务质量；
7. 版本切换、租户隔离与过期 block 回收。

性能报告至少包含有效 KV 容量、碎片率、读带宽、分配延迟、page / block 大小、命中率、TTFT、TPOT、goodput 和 p99。只报告“最大上下文长度”无法说明系统在并发服务中的可用性。

block table、引用计数、copy-on-write 与调度预算的可审计实现见[手撕：推理引擎](../practice/inference-engine.md)。
