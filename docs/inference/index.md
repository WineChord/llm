# 推理与服务

自回归推理由两个不同工作负载组成：

- **prefill**：并行处理 prompt，产生首 token 并写入 KV；
- **decode**：每轮新增少量 token，反复读取权重和历史 KV。

prefill 更接近大矩阵计算，decode 常受权重/KV 带宽、batch shape 和调度限制。任何优化都应说明影响哪一段。

## 请求链

$$
\text{admit}
\rightarrow \text{tokenize}
\rightarrow \text{prefill}
\rightarrow \text{install KV}
\rightarrow \text{decode/sample}
\rightarrow \text{stream/finish}.
$$

每个请求还携带 model、tokenizer、template、adapter、RNG、grammar、priority、deadline、block table 与 finish reason。它们共同构成可恢复状态。

## 指标

| 指标 | 含义与边界 |
| --- | --- |
| TTFT | queue、tokenize、prefill、KV install、sample 与网络之和 |
| TPOT / ITL | decode token 间隔；说明是否含 streaming |
| E2E | 完整请求时间，受输出长度影响 |
| tokens/s | 输入、输出或总 token，必须标口径 |
| goodput | 单位时间内同时满足质量和全部 SLO 的请求 |
| p95/p99 | 长度与负载 slice 下的尾延迟 |
| memory/request | KV、临时 buffer、碎片与共享摊销 |
| failure rate | reject、timeout、OOM、cancel、unknown 与错误输出 |

峰值 throughput 不等于可交付容量；低并发 latency 也不代表过载稳定。

## 机制地图

### 输出分布

[解码](decoding.md)覆盖 greedy、sampling、beam、grammar、停止与基础 speculative 语义。

### KV 与复用

- [KV Cache](kv-cache.md)：容量、page、layout、有损压缩与淘汰分类；
- [缓存复用](cache-reuse.md)：prefix key、radix、COW、安全域与收益条件；
- [推理运行时](runtime.md)：请求状态、block table、chunked prefill、抢占与 streaming；
- [手撕推理引擎](../practice/inference-engine.md)：allocator、调度、量化与状态断言。

精确物理管理、精确 prefix reuse、有损 KV 表示和有损淘汰是四种不同问题。

### 调度

[调度与服务](serving.md)给出资源账本和容量边界，[Goodput 与公平调度](scheduling-goodput.md)进一步覆盖 admission、Little’s law、deadline、fairness、cache affinity 与 overload。

### 加速

- [量化](quantization.md)：W8A8、W4A16、W4A8、FP8/FP4 与真实 kernel；
- [推测解码](speculative-decoding.md)：draft、接受/残差分布、tree verification 与事务回滚；
- [加速总览](acceleration.md)：kernel、graph、编译、路由与端到端决策。

低比特 checkpoint 不证明执行了低比特 GEMM；高接受率也不保证推测解码加速，必须计 draft、verify 与状态成本。

### 集群

[Prefill–Decode 分离](disaggregation.md)解释 KV 传输、布局兼容、背压和失败；[基准与可靠性](benchmarking-reliability.md)固定长度分布、负载、SLO、故障注入和报告卡。

## 成本下界

KV bytes：

$$
M_{\mathrm{KV}}=2LBTA_{\mathrm{kv}}d_hs.
$$

P/D 传输：

$$
T_{\mathrm{transfer}}\ge
\frac{M_{\mathrm{KV}}}{B_{\mathrm{effective}}}.
$$

Little’s law：

$$
L_q=\lambda W.
$$

这些只是量级起点。真实结果还受 page 碎片、量化 metadata、网络拥塞、batch 组成和 kernel 影响。

## 正确性契约

- full、chunked prefill 与逐 token logits 对齐；
- token counter、position、RNG 和 grammar 单调一致；
- shared KV 部分写前 COW；
- GPU event 完成前 block 不复用；
- finish、cancel 与 install 幂等；
- cache key 绑定精确 token 与所有兼容版本；
- speculative reject 回滚 KV/RNG/grammar；
- overload、admission reject 与 unknown side effect 明确；
- P/D 只有在 transfer、checksum、install 完成后进入 decode。

## 推荐顺序

先读[解码](decoding.md)→[KV Cache](kv-cache.md)→[运行时](runtime.md)→[调度](serving.md)，再按瓶颈进入量化、推测解码或 P/D 分离。用[性能模型](../systems/performance-model.md)算成本，用[推理可靠性](benchmarking-reliability.md)验证端到端结果。
