# 系统与基础设施

大模型系统把一个全局计算图映射到具体 tensor layout、加速器、网络与存储。统一主线是：

$$
\text{shape}
\rightarrow \text{FLOPs/bytes}
\rightarrow \text{placement/communication}
\rightarrow \text{kernel}
\rightarrow \text{schedule}
\rightarrow \text{correctness and SLO}.
$$

系统机制也有清楚的因果脉络。[分布式训练系统](../landscape/lineages/distributed-training-systems.md)从数据并行的全量复制，走到 Megatron 的算子切分、ZeRO 的状态分片与多维并行；[推理运行时与服务](../landscape/lineages/inference-serving.md)则从静态 batching 走到迭代级调度、分页 KV 与 P/D 分离。对应实现可深入 [Megatron 与 ZeRO](../landscape/works/megatron-zero.md)、[FlashAttention](../landscape/works/flashattention.md) 和 [vLLM / PagedAttention](../landscape/works/vllm-pagedattention.md)。

## 定量起点

[性能模型](performance-model.md)先计算：

- 参数与激活 shape；
- forward/backward FLOPs；
- HBM、SRAM 与网络 bytes；
- arithmetic intensity 与 roofline 上界；
- 权重、梯度、optimizer、activation 和 transient 峰值；
- MFU/HFU 与有效 token 吞吐。

训练 step 的各项并非简单相加：

$$
T_{\mathrm{step}}\ne
T_{\mathrm{compute}}+T_{\mathrm{memory}}+T_{\mathrm{communication}},
$$

因为通信、访存与计算可以部分重叠。目标是缩短关键路径，而非独立 microbenchmark 的峰值。

## 单设备执行

| 层 | 主要对象 | 入口 |
| --- | --- | --- |
| 数值 | storage、compute、accumulator、scale、rounding | [数值与低精度](precision-numerics.md) |
| 硬件 | SM、warp、register、shared memory、TMA、graph | [GPU 执行](gpu-execution.md) |
| 通用 kernel | GEMM、norm、activation、fusion、benchmark | [Kernel 与性能](kernels-performance.md) |
| Attention | online softmax、prefill/decode、paged/ragged | [Attention Kernel](attention-kernels.md) |
| MoE | route、permute、all-to-all、grouped GEMM、combine | [MoE 系统](moe-systems.md) |

理论 $O(T)$ 或更低位宽不等于 wall-clock 更快。真实收益取决于 shape、layout、metadata、kernel 和硬件。

## 分布式训练

[并行训练总览](parallelism.md)建立 DP、TP、PP、CP、SP 与 EP 坐标；随后：

1. [集合通信与状态分片](collectives-sharding.md)：all-reduce、reduce-scatter、all-gather、DDP、ZeRO 与 FSDP；
2. [模型并行](model-parallelism.md)：每种 placement 的 forward/backward collective 与通信量；
3. [内存、数值与硬件总览](memory-numerics-hardware.md)：旧入口与跨层资源账本；
4. [分布式手撕](../practice/distributed-systems.md)：token loss、layout、MoE permutation 与 manifest。

每个 tensor 都应能回答 global shape、local shape、process mesh 和 placement。每个 collective 都应能回答参与 rank、顺序、count、dtype 与 buffer 生命周期。

## Checkpoint 与韧性

[检查点与容错](checkpointing.md)定义 immutable shard、manifest、committed marker、异步 staging 与跨 topology 恢复；[系统韧性与可观测性](resilience-observability.md)覆盖：

- worker、GPU、NIC、存储与 control plane 故障；
- straggler、collective timeout 与 silent corruption；
- health、trace、fault injection 与恢复 SLO；
- gray release、schema 兼容与事故回归。

一次训练成功不代表可以严格 resume。weights-only 恢复属于 warm start；optimizer、scheduler、scaler、RNG、global tokens、data cursor、mixture 与版本缺一不可重放。

## 资源层次

| 层次 | 关键问题 |
| --- | --- |
| 算子 | tile、fusion、occupancy 和数值路径 |
| 单卡 | HBM 峰值、allocator、graph 与 host sync |
| 节点内 | NVLink/NVSwitch、PCIe、NUMA 与 CPU staging |
| 节点间 | topology、channel、拥塞和 collective |
| 存储 | shard、吞吐、原子提交和校验 |
| 编排 | rank、lease、弹性、版本与故障恢复 |

## 评价原则

- 峰值 FLOPs 不等于有效训练吞吐；
- GPU utilization 可能包含低效 kernel 或等待；
- 单卡 microbenchmark 不代替多节点扩展；
- 更大 global batch 会改变优化轨迹；
- 低精度要同时报告格式、scale、accumulator 与质量；
- 平均 step time 会掩盖 straggler 与 checkpoint 抖动；
- 优化后的实现必须与慢 reference、backward 和真实 shape 对齐。

学习时先走[性能模型](performance-model.md)→[并行总览](parallelism.md)→[Kernel](kernels-performance.md)→[检查点](checkpointing.md)；排错时从[调试手册](../practice/debugging.md)进入首个分叉点。

开放系统组件与模型版本的对应关系可从 [DeepSeek](../landscape/families/deepseek.md)、[Kimi](../landscape/families/kimi.md) 与 [GLM](../landscape/families/glm.md) 家族页反查；组织名或仓库名本身不证明某个组件已经进入生产 checkpoint。
