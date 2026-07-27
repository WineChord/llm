# 内存、数值与硬件

训练是否能运行由峰值内存决定，运行是否高效由计算、内存带宽和通信的关键路径决定。

## 显存账本

设参数量为 $N$。以 BF16 权重、BF16 梯度、FP32 master weight 和 Adam FP32 一二阶矩为例，仅静态模型状态就可能接近：

$$
2N+2N+4N+8N=16N\ \text{bytes}
$$

实际实现可能没有 master weight，或使用分片、低精度 optimizer；还需加入激活、KV、中间 buffer、通信 bucket、allocator 碎片和 kernel workspace。

## Activation

激活随 batch、序列长度、层数和 hidden size 增长。activation checkpointing 只保存部分边界，反向时重算中间结果，以计算换内存。选择 checkpoint 粒度时要测 wall-clock，而不只看理论重算 FLOPs。

## Roofline 视角

算术强度：

$$
I=\frac{\text{FLOPs}}{\text{bytes moved}}
$$

当 $I$ 低时更可能受内存带宽限制；高时更可能受计算峰值限制。prefill 的大 GEMM 往往更接近 compute-bound，逐 token decode 的小 batch 与权重/KV 读取往往更接近 bandwidth-bound。

## 数值路径

- softmax 需要减去行最大值以避免指数溢出。
- 大规模归约的顺序会改变浮点舍入，分布式结果不必 bitwise 相同。
- loss scaling 只能缓解 FP16 梯度下溢，不能修复错误 loss 或坏数据。
- FP8 训练需追踪每类张量的缩放与 amax，不应只报告“开启 FP8”。
- optimizer state、norm 与 residual 路径常保留更高精度。

## 硬件与网络

需要联合记录：

- 加速器型号、显存、计算格式与实际频率；
- 节点内拓扑、PCIe、NVLink/NVSwitch；
- 节点间 NIC 数、带宽、RDMA、交换结构与超额订阅；
- CPU、NUMA、内存、数据盘与远端对象存储；
- 驱动、runtime、collective library、kernel 与框架版本。

只说“用了多少张 GPU”无法复现性能。训练框架的代表性参考包括 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 与 [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)。状态怎样切分见[集合通信与状态分片](collectives-sharding.md)，算术强度怎样落到 kernel 见[Kernel 与性能](kernels-performance.md)。

## Reference {#reference}

- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [Megatron Core](https://github.com/NVIDIA/Megatron-LM)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
