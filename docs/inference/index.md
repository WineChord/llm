# 推理与服务

自回归推理包含两个性质不同的阶段：

- **prefill**：并行处理输入 token，生成首个输出并填充 KV Cache。
- **decode**：每轮只新增少量 token，重复读取权重与历史 KV。

prefill 常有较大的矩阵乘法，decode 更容易受内存带宽、批处理形状和调度影响。

## 核心指标

| 指标 | 含义 |
| --- | --- |
| TTFT | 请求到首 token 的时间 |
| ITL / TPOT | 相邻输出 token 的延迟 |
| E2E latency | 完整请求时间 |
| tokens/s | 输出或总 token 吞吐，必须说明口径 |
| goodput | 满足 SLO 的有效吞吐 |
| tail latency | P95/P99 等尾部延迟 |
| memory/request | 权重、KV、临时 buffer 与碎片 |

平均吞吐不能代表交互体验；低并发 latency 也不能代表高负载稳定性。

## 系统链条

请求经过 tokenizer、admission control、scheduler、模型执行、采样、streaming 与后处理。若存在 prefix cache、tool call、RAG 或安全分类器，端到端延迟还包含这些组件。

## 阅读路径

1. [解码](decoding.md)：greedy、sampling、beam、grammar、停止与 speculative decoding。
2. [KV Cache](kv-cache.md)：容量、分页、prefix 与压缩。
3. [推理运行时](runtime.md)：请求状态、continuous batching、block table、chunked prefill 与 streaming。
4. [调度与服务](serving.md)：SLO、容量、过载与公平。
5. [Prefill–Decode 分离](disaggregation.md)：跨 worker 的 KV 传输、路由与恢复。
6. [加速与量化](acceleration.md)：kernel、低比特、图与路由。
