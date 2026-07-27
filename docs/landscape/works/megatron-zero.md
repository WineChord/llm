# Megatron-LM 与 ZeRO：切计算还是切状态

Megatron-LM 与 ZeRO 常被同时列为“大模型并行技术”，但它们解决的第一问题不同：Megatron 的经典 tensor parallel 切开一层的矩阵计算，ZeRO 切开 data-parallel ranks 之间重复保存的训练状态。二者可以组合，也会通过 collective、GEMM shape 和峰值物化彼此影响。

## 两种“放不下”

设模型有 $P$ 个参数。经典 FP16 Adam 配方可能包含：

| 状态 | 典型字节/参数 |
| --- | ---: |
| FP16 参数 | 2 |
| FP16 梯度 | 2 |
| FP32 master weight | 4 |
| FP32 一阶、二阶矩 | 8 |

这给出约 $16P$ bytes 的常用估计，但并非协议常数。BF16、低精度 optimizer、梯度 dtype、master weight 是否存在都会改变结果。更安全的写法是

$$
M_{\text{state}}
=P(b_w+b_g+b_{\text{master}}+b_m+b_v).
$$

若总状态过大但单层仍能计算，优先分片 data-parallel redundancy；若一个 layer 的权重、activation 或 workspace 已超过单卡容量，则需要 tensor parallel。完整决策链见[分布式训练系统谱系](../lineages/distributed-training-systems.md)。

## Megatron 的相邻切分

[Megatron-LM 论文](https://arxiv.org/abs/1909.08053)针对 Transformer 中成对出现的线性层设计 column/row parallel。设 $Y=XW$。

Column parallel 沿输出维切权重：

$$
W=[W_1,\ldots,W_T],\qquad Y_i=XW_i.
$$

每个 rank 得到不同输出列。若激活函数逐元素作用，各 rank 可以直接计算，不必先 all-gather。

下一层可沿输入维切分：

$$
X=[X_1,\ldots,X_T],\qquad
W=
\begin{bmatrix}
W_1\\
\vdots\\
W_T
\end{bmatrix},
$$

$$
Y=\sum_iX_iW_i.
$$

这一步需要合并 partial sums。相邻安排把通信集中在少数边界，而不是每个局部算子后都重建完整 tensor。Attention 的 Q/K/V、output projection 与 MLP 使用相同原则，但具体 shard 方向还受 head 布局和 checkpoint 格式约束。

### 最小语义 reference

下面只在 CPU 上顺序模拟 shard；它验证切分后的数学结果与 dense linear 一致，同时比较 ZeRO 静态状态，不模拟真实 collective 性能。

```python
def matmul(left, right):
    columns = list(zip(*right))
    return [[sum(a * b for a, b in zip(row, column)) for column in columns]
            for row in left]
def column_parallel(x, weight, parts):
    width = len(weight[0]) // parts
    outputs = []
    for start in range(0, len(weight[0]), width):
        shard = [row[start:start + width] for row in weight]
        outputs.append(matmul(x, shard))
    return [sum((output[row] for output in outputs), []) for row in range(len(x))]
def row_parallel(x, weight, parts):
    width = len(weight) // parts
    result = [[0.0] * len(weight[0]) for _ in x]
    for start in range(0, len(weight), width):
        local = matmul([row[start:start + width] for row in x],
                       weight[start:start + width])
        result = [[a + b for a, b in zip(old, new)]
                  for old, new in zip(result, local)]
    return result
def zero_bytes(parameters, data_parallel, stage, weight=2, grad=2, optimizer=12):
    if stage == 0:
        per_parameter = weight + grad + optimizer
    elif stage == 1:
        per_parameter = weight + grad + optimizer / data_parallel
    elif stage == 2:
        per_parameter = weight + (grad + optimizer) / data_parallel
    elif stage == 3:
        per_parameter = (weight + grad + optimizer) / data_parallel
    else:
        raise ValueError("stage must be 0, 1, 2, or 3")
    return parameters * per_parameter
def close(left, right, tolerance=1e-12):
    return all(abs(a - b) <= tolerance
               for rows in zip(left, right) for a, b in zip(*rows))
x = [[1.0, 2.0, -1.0, 0.5], [0.0, 3.0, 2.0, -2.0]]
w = [[1.0, 0.0, 2.0, -1.0], [2.0, 1.0, 0.0, 3.0],
     [-1.0, 2.0, 1.0, 0.0], [0.5, -2.0, 3.0, 1.0]]
dense = matmul(x, w)
assert close(column_parallel(x, w, 2), dense)
assert close(row_parallel(x, w, 2), dense)
memory = [zero_bytes(1_000_000, 4, stage) for stage in range(4)]
assert memory[3] < memory[2] < memory[1] < memory[0]
```

真实分布式实现还要验证 process group、collective 顺序、bias 所有权、dropout RNG 和全局 loss 归一化。对应 reference 见[手撕：分布式与容错](../../practice/distributed-systems.md)。

## TP 的真实成本是 activation 通信

Tensor parallel 并不是“参数除以卡数”这么简单。若一轮边界 collective 的 tensor 有 $N_{\text{tok}}$ 个 token、hidden width 为 $d$、元素宽度为 $b$，单次 payload 为

$$
S_{\text{act}}=N_{\text{tok}}db.
$$

对 $T$ 个 TP ranks，ring all-reduce 每 rank 搬运量近似

$$
V_{\text{AR,rank}}
\approx2\frac{T-1}{T}S_{\text{act}}.
$$

TP degree 增大时，单卡参数与计算下降，但每层 collective 更频繁，本地 GEMM 也更窄。若 TP 跨越低带宽节点，通信可能完全暴露；若 batch tokens 太少，GEMM 与 collective 都会受启动开销主导。因此 TP 通常映射到节点内高速互联。

## ZeRO：逐步消除副本

[ZeRO](https://arxiv.org/abs/1910.02054)从 optimizer states 开始，再分片 gradients 和 parameters。设权重、梯度、optimizer states 总字节为 $W,G,O$，数据并行度为 $D$：

$$
M_{\text{S1}}\approx W+G+\frac{O}{D},
$$

$$
M_{\text{S2}}\approx W+\frac{G+O}{D},
$$

$$
M_{\text{S3,static}}\approx\frac{W+G+O}{D}.
$$

Stage 1/2 仍复制参数；Stage 3 在使用模块前 all-gather 参数，计算后 reshard，并以 reduce-scatter 落下梯度 shard。峰值应写成

$$
M_{\text{S3,peak}}
\approx M_{\text{S3,static}}
+M_{\text{largest materialized unit}}
+M_{\text{bucket}}+M_{\text{allocator}}.
$$

wrap 太细会制造大量小 all-gather；wrap 太粗则临时完整参数过大。Prefetch 可以隐藏网络，却可能同时物化多个模块，使理论静态节省无法转化为可用显存。

## 组合以后，拓扑决定收益

Megatron 的后续扩展研究组合 tensor、pipeline 与 data parallel：

$$
N_{\text{GPU}}=D_{\text{TP}}D_{\text{PP}}D_{\text{DP}}.
$$

常见映射是 TP 在节点内、PP 沿稳定点对点链路、DP 跨更大范围。但这只是起点：

- ZeRO/FSDP 的 shard group 不一定应覆盖所有 DP ranks；
- TP collective 与 FSDP all-gather 可能争用同一 fabric；
- pipeline microbatch 改变每次 TP payload 和 activation 峰值；
- sequence/context parallel 会再引入布局转换；
- checkpoint 必须保存 global tensor identity，而不能把旧 rank 文件名当语义。

组合并行的目标是最小化关键路径上的 exposed communication，而不是让每个并行度都尽可能大。完整的 collective 和布局推导见[集合通信与状态分片](../../systems/collectives-sharding.md)与[模型并行](../../systems/model-parallelism.md)。

## 原始工作与实现边界

- [Megatron-LM 原始论文](https://arxiv.org/abs/1909.08053)固定了经典 Transformer tensor-parallel 切分。
- [Megatron 大规模训练论文](https://arxiv.org/abs/2104.04473)研究 TP、PP、DP 的组合与 interleaved pipeline。
- [NVIDIA Megatron-LM 官方仓库](https://github.com/NVIDIA/Megatron-LM)持续加入 sequence/context/expert parallel、distributed optimizer 与新硬件路径；当前代码不能反向代表 2019 论文的全部实验条件。
- [ZeRO 原始论文](https://arxiv.org/abs/1910.02054)给出三阶段冗余消除与通信分析。
- [DeepSpeed ZeRO 官方指南](https://www.deepspeed.ai/tutorials/zero/)描述一种生产实现及其配置；offload、quantization 和新 runtime 特性需要按具体版本核验。
- [PyTorch FSDP 论文](https://arxiv.org/abs/2304.11277)与[官方文档](https://docs.pytorch.org/docs/stable/fsdp.html)提供另一套 fully sharded 执行与 API 语义，不应把 FSDP 配置名机械映射成某个 DeepSpeed stage。

评测必须同时报告模型 shape、TP/PP/DP 网格、物理拓扑、microbatch、dtype、checkpoint/recompute、峰值显存、collective 暴露和有效 token throughput。只报告 GPU 数量无法复现实验。
