# 调度与服务

在线服务面对长度未知、到达随机、优先级不同的请求。调度器的任务是在显存约束下组合 batch，并控制交互延迟与吞吐。

## Continuous Batching

静态 batch 等最慢请求结束后再换批，会产生空闲槽。continuous batching 在 token step 边界移除完成请求、加入新请求。[Orca](https://arxiv.org/abs/2206.02658) 描述了迭代级调度与选择性 batching。

## Prefill 与 Decode 干扰

长 prefill 会占用大量计算，阻塞短 decode，造成 ITL 尖峰。常见策略：

- 限制单轮 prefill token；
- chunked prefill，把长输入拆成片段与 decode 交错；
- prefill/decode 分离到不同 worker；
- 按 SLO、长度或租户设优先级；
- admission control 防止 KV 超卖。

[Sarathi-Serve](https://arxiv.org/abs/2403.02310) 研究 chunked prefill 与 stall-free batching。

## 调度状态

一个请求至少包含 prompt 长度、已生成长度、KV block、采样器状态、优先级、截止时间、adapter、租户和取消状态。streaming 客户端断开后应及时回收计算与缓存。

## 容量规划

压测矩阵至少覆盖：

- prompt/output 长度分布；
- 并发与到达过程；
- greedy、sampling、beam 或多候选；
- 单模型、多 adapter 与多租户；
- prefix cache 冷/热状态；
- P50/P95/P99 TTFT、ITL 和 E2E；
- OOM、抢占、取消与过载降级。

## 过载行为

系统应优先明确拒绝、排队或降级，而不是在不可控队列中让所有请求超时。goodput 比峰值 tokens/s 更能反映是否满足服务目标。
