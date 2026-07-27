# 集合通信与状态分片

分布式训练首先是一个数据布局问题：每个 rank 持有什么张量，何时需要完整值，计算后结果应落在哪里。collective 只是实现布局变换的工具，不能脱离 tensor shape 和依赖关系记忆。

## 通信成本模型

对大小为 $n$ 字节的消息，可用

$$
T\approx \alpha+\frac{n}{\beta}
$$

粗略描述通信时间，其中 $\alpha$ 是启动时延，$\beta$ 是有效带宽。小消息更受时延影响，大消息更受带宽影响；真实 collective 还取决于 rank 数、ring/tree 算法、拓扑、拥塞和并发流。

## 常见 collective

| 操作 | 输入布局 | 输出布局 | 典型用途 |
| --- | --- | --- | --- |
| broadcast | 一份完整值 | 每 rank 完整值 | 参数或控制状态分发 |
| all-reduce | 每 rank 部分贡献 | 每 rank 完整归约值 | DDP 梯度同步 |
| reduce-scatter | 每 rank 部分贡献 | 每 rank 一份归约 shard | 分片梯度 |
| all-gather | 每 rank 一份 shard | 每 rank 完整拼接值 | 分片参数临时物化 |
| all-to-all | 每 rank 多个目标 shard | 每 rank 收到重排 shard | expert/token dispatch |
| point-to-point | 指定发送与接收 | 指定张量 | pipeline stage 传递 |

all-reduce 常可理解为 reduce-scatter 加 all-gather，但具体库可能选择不同算法。语义等价不代表性能相同。

## Distributed Data Parallel

DDP 在每个 rank 保存完整模型，处理不同 microbatch，并归约梯度：

$$
g=\frac{1}{N_{\text{DP}}}
\sum_{r=1}^{N_{\text{DP}}}g_r.
$$

实现会把参数梯度按 bucket 分组，在反向过程中一旦 bucket 就绪便启动 all-reduce，从而与尚未完成的反向计算重叠。bucket 太小增加启动开销，太大则延后通信。

梯度累积时，前若干 microbatch 可以不通信，最后一次再归约。必须保证 loss scale 与最终平均口径正确；`no_sync` 只改变通信时机，不应改变有效 global batch。

## ZeRO 的三个阶段

[ZeRO](https://arxiv.org/abs/1910.02054)逐步消除数据并行副本中的冗余状态：

| 阶段 | 分片对象 | 前向需要的主要额外通信 |
| --- | --- | --- |
| Stage 1 | optimizer state | 参数仍完整复制 |
| Stage 2 | optimizer state + gradients | 参数仍完整复制 |
| Stage 3 | optimizer state + gradients + parameters | 使用前 all-gather 参数 shard |

若每个参数的权重、梯度和 Adam states 分别占 $b_p,b_g,b_o$ 字节，忽略 buffer 后，单 rank 静态状态近似为

$$
M_{\text{S2}}
\approx Nb_p+\frac{N(b_g+b_o)}{P},
$$

$$
M_{\text{S3}}
\approx \frac{N(b_p+b_g+b_o)}{P},
$$

其中 $P$ 是数据并行度。峰值还包含正在 all-gather 的参数、activation、通信 bucket 和 allocator 碎片。

## FSDP 的执行语义

Fully Sharded Data Parallel 在模块使用前物化完整参数，计算后重新释放或 reshard，反向重复相应过程。关键选择包括：

- shard group 的大小；
- wrap 粒度；
- prefetch 顺序；
- forward 后是否立即 reshard；
- mixed-precision 参数、归约与 buffer dtype；
- CPU offload；
- full、sharded 或 local checkpoint。

wrap 太细会产生大量小 all-gather，太粗则峰值显存高且重叠不足。应以模块执行顺序和实际 bucket trace 调整。

## 梯度归约的正确性

分布式平均最容易在可变长度数据上出错。设 rank $r$ 有效 token 数为 $n_r$、loss 总和为 $S_r$，正确全局 token 均值是

$$
\bar L=\frac{\sum_rS_r}{\sum_rn_r},
$$

而不是

$$
\frac{1}{P}\sum_r\frac{S_r}{n_r}.
$$

当 $n_r$ 不同时，两者梯度不同。packed sequence、动态长度和过滤空目标会让这种差异经常出现。

### 全局 token 归一化 {#global-token-reduction-reference}

`local_loss_sums[r]` 是 rank $r$ 上尚未归一化的 token loss 总和，`local_token_counts[r]` 是相应有效 token 数。reference 模拟对两个标量分别做 sum reduction，再计算一个全局 mean；返回值是所有 rank 应共同使用的标量目标。

```python
import torch

def global_token_mean(local_loss_sums, local_token_counts):
    loss_sum = torch.stack(local_loss_sums).sum()
    token_count = torch.stack(local_token_counts).sum()
    if token_count <= 0:
        raise ValueError("global batch has no supervised token")
    return loss_sum / token_count

sums = [torch.tensor(2.), torch.tensor(12.)]
counts = [torch.tensor(1.), torch.tensor(3.)]
correct = global_token_mean(sums, counts)
local_mean_average = torch.stack([s / n for s, n in zip(sums, counts)]).mean()
assert torch.allclose(correct, torch.tensor(3.5))
assert not torch.allclose(correct, local_mean_average)
```

不变量是分子和分母使用相同 rank 集合，且除法只在全局归约后发生。真实 DDP/FSDP 路径还要根据框架的梯度平均语义补偿 world size，并保证所有 rank 以相同顺序调用 collective；这段代码不模拟通信或 autograd hook。分片 linear、通信体积与多 rank shape 对照见[手撕：分布式与容错](../practice/distributed-systems.md)。

## 通信与计算重叠

只有异步 collective 在关键依赖之前完成，才算真正重叠。常见失效包括：

- 默认 stream 的隐式同步；
- bucket 生成过晚；
- 通信和 GEMM 竞争同一资源；
- 多个并行维度同时压同一链路；
- host 侧 launch 不及时；
- profiler 把排队时间误认为 overlap。

应画出依赖 DAG，并比较 exposed communication，而不是把 collective 总时长直接从 step time 中相减。

## 非逐元素优化器的分片边界 {#matrix-wise-optimizer-sharding}

ZeRO 的经典内存分析默认优化器状态可以按元素切分；Muon 一类需要对完整二维更新矩阵做全局归一化、矩阵乘法或正交化的优化器打破了这个前提。若把矩阵 $G\in\mathbb R^{m\times n}$ 沿元素任意切到多个 rank，再让每个 rank 独立计算

$$
G\left(G^\top G\right)
$$

或 Newton–Schulz 迭代，得到的不是完整矩阵上的同一算子，除非每一步额外重建全局 Gram 量或完整矩阵。分片单位因此必须跟随算子语义，而不能只追求字节均匀。

[DeepSeek-V4](../landscape/works/deepseek-v4.md#muon)采用的折中是把每个逻辑矩阵完整分配给一个 data-parallel rank，再以 knapsack 近似平衡状态容量；报告中的 dense 路径把每个 rank 限制在至多五个矩阵，padding 通常低于 10%。对 MoE 参数，系统按 down、up、gate projection 跨层拼接相同角色的矩阵，但不切断任一逻辑矩阵；相同 shape 才组成 batch。梯度同步使用 all-to-all 搬运 shard，再在拥有该矩阵的 rank 上以 FP32 求和，避免低精度 ring/tree reduction 改变结果。

这不是“Muon 必须这样实现”的定理，而是三种成本的明确交换：

- **完整矩阵归属** 保住优化器算子的数学语义；
- **负载装箱与有限冗余** 避免单个大矩阵制造严重的 rank imbalance；
- **all-to-all + rank-local reduction** 减少低精度归约误差，却改变通信拓扑与峰值 buffer。

设计新的 matrix-wise optimizer 时，应先写出它跨哪些轴做 reduction，再决定可合法切分的边界；只有 element-wise 状态才天然适合沿任意参数 shard 更新。

## Checkpoint 与 reshard

分片训练的 checkpoint 必须保存全局 shape、shard offset、参数名、dtype、优化器分组与并行拓扑。恢复到不同 world size 时需要 reshard；若只按旧 rank 文件编号读取，可能静默错位。

安全流程是：

1. 写入临时目录；
2. 每个 shard 完成 checksum；
3. 生成全局 manifest；
4. 原子写入完成标记；
5. 用独立进程读取并做 shape/dtype 检查；
6. 定期验证跨并行度转换。

更完整的恢复状态见[检查点与容错](checkpointing.md)，层内与层间切分见[模型并行](model-parallelism.md)。

## Reference {#reference}

- [ZeRO](https://arxiv.org/abs/1910.02054)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)
- [NVIDIA NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348)
