# 系统与基础设施

大模型系统的核心矛盾是：单个加速器放不下或算不完，而分布式切分会引入通信、同步、空泡、容错和可复现性成本。

## 资源模型

训练 step 可以粗略拆为：

$$
T_{\text{step}}
=T_{\text{compute}}+T_{\text{memory}}+T_{\text{communication}}
+T_{\text{input}}+T_{\text{bubble}}+T_{\text{checkpoint}}
$$

这些项会重叠，因此不能把 profiler 中各阶段时间简单相加。优化目标是关键路径，而不是单个 kernel 的峰值。

## 系统层次

| 层次 | 关键问题 |
| --- | --- |
| 算子与 kernel | GEMM、attention、norm、通信能否融合和饱和硬件 |
| 单设备内存 | 权重、激活、梯度、optimizer state 与临时 buffer |
| 节点内互联 | GPU 拓扑、NVLink/NVSwitch、NUMA 与 CPU pinned memory |
| 节点间网络 | 带宽、时延、拥塞、collective 算法与拓扑映射 |
| 训练编排 | rank、数据游标、弹性、checkpoint、故障诊断 |
| 存储与数据 | shard、预取、缓存、元数据、吞吐与小文件问题 |

## 评价原则

- 峰值 FLOPs 不是有效训练吞吐。
- GPU utilization 可能包含等待或低效 kernel，需结合 MFU、带宽和通信 trace。
- 单卡 microbenchmark 不能代替多节点强/弱扩展测试。
- 更大的 global batch 可能提高硬件效率，却改变优化轨迹和收敛 token。
- 训练成功一次不代表可恢复、可重复或可长期运行。

继续阅读[并行训练](parallelism.md)、[内存、数值与硬件](memory-numerics-hardware.md)及[检查点与容错](checkpointing.md)。
