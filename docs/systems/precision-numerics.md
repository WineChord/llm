# 数值精度与低精度计算

低精度不是把一个配置开关从 BF16 改成 FP8。一个完整数值路径至少包含：

- 参数和激活的存储格式；
- GEMM 或 attention 的输入格式；
- 乘法、累加与 reduction 精度；
- scale、amax、zero point 等量化状态；
- optimizer、residual、norm 和 loss 的保留精度；
- 溢出、下溢、舍入与异常值策略。

只有这些语义都确定，才能讨论显存、吞吐和质量。

## 浮点格式解决什么问题

浮点数可抽象为

$$
x=(-1)^s\,m\,2^e.
$$

指数位主要决定动态范围，尾数位主要决定相对精度。FP16 有较多尾数但指数范围较窄；BF16 保留与 FP32 相同数量的指数位，因此更不容易溢出，但单位舍入误差更大。

| 格式 | 常见角色 | 主要收益 | 主要风险 |
| --- | --- | --- | --- |
| FP32 | reduction、optimizer、reference | 动态范围和精度高 | 带宽、显存和算力成本高 |
| TF32 | NVIDIA 上的 FP32 GEMM 路径 | 保留 FP32 范围并使用 Tensor Core | 乘法有效精度低于 FP32 |
| FP16 | 训练和推理输入 | 硬件成熟、占用低 | 指数范围窄，常需 loss scaling |
| BF16 | 主流混合精度训练 | FP32 级指数范围 | 尾数较短，细小更新可能舍入掉 |
| FP8 E4M3 | forward 激活与权重 | 更低带宽和更高吞吐 | 范围较小，强依赖 scaling |
| FP8 E5M2 | 梯度等高动态范围张量 | 指数范围更大 | 精度低于 E4M3 |
| INT8 / INT4 | 推理权重或激活 | 压缩与低比特 kernel | 需要 calibration、metadata 与真实硬件路径 |

[Transformer Engine 的 FP8 说明](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)给出了 E4M3、E5M2 的典型分工和 scaling 机制。它描述的是受支持硬件与软件栈中的实现语义，不应外推为所有加速器的统一行为。

## Storage、compute 与 accumulator

一个算子应写成四元组：

$$
(\text{storage},\text{input},\text{accumulator},\text{output}).
$$

例如“BF16 训练”可能实际表示：

- 参数和激活以 BF16 存储；
- Tensor Core 读取 BF16；
- 点积在 FP32 accumulator 中累加；
- 输出再舍入到 BF16；
- norm、loss 与 optimizer state 保持 FP32。

只写一个 dtype 会隐藏最重要的稳定性信息。特别是 reduction：求和长度为 $n$ 时，舍入误差会随求和顺序和 $n$ 境况变化；分布式树形 reduction 与单卡顺序 reduction 不保证 bitwise 一致。

## 舍入、溢出与下溢

设目标格式在当前指数区间的相邻可表示数间距为 $\operatorname{ulp}(x)$。round-to-nearest 的局部绝对误差通常受

$$
|\widehat x-x|
\lesssim \frac{1}{2}\operatorname{ulp}(x)
$$

约束，但连续多次舍入、消减和非结合求和会累积误差。

三类问题需要分开：

- **overflow**：数值超出最大有限值，产生 Inf 或饱和；
- **underflow**：数值落入 subnormal 或被舍入为零；
- **cancellation**：相近大数相减后有效位大量丢失。

softmax 通过减去行最大值避免指数上溢：

$$
\operatorname{softmax}(x_i)
=\frac{\exp(x_i-m)}
{\sum_j\exp(x_j-m)},
\qquad
m=\max_j x_j.
$$

这并不能修复全 mask 行、输入 NaN 或错误的 mask sentinel；这些情况必须由算子契约单独定义。

## Loss scaling

FP16 训练中，将 loss 乘以 $g$ 可把小梯度搬到可表示范围：

$$
\widetilde{\mathcal L}=g\mathcal L,
\qquad
\nabla\mathcal L
=\frac{1}{g}\nabla\widetilde{\mathcal L}.
$$

动态 loss scaling 在检测到 Inf / NaN 时降低 $g$，稳定一段时间后再增大。它只缓解梯度下溢，不会修复错误的目标函数、坏数据、过大学习率或 forward 中已经发生的溢出。BF16 通常不因指数范围而需要 loss scaling，但仍可能出现其他数值问题。

## FP8 scaling

对张量 $x$ 和目标格式最大有限值 $q_{\max}$，一种基本缩放是

$$
s=\frac{a_{\max}}{q_{\max}},
\qquad
q=\operatorname{cast}_{\mathrm{FP8}}\left(\frac{x}{s}\right),
\qquad
\widehat x=sq.
$$

$a_{\max}$ 可来自当前张量，也可来自历史窗口：

- **current scaling** 使用当前统计，响应快，但统计本身可能增加同步与 kernel；
- **delayed scaling** 使用历史 amax，执行更容易流水化，但分布漂移时可能短暂失配；
- **block scaling** 对较小连续分块单独缩放，减少单个 outlier 对整张量精度的影响，同时增加 scale metadata 和 layout 约束。

scale 不是可随意丢弃的临时量。训练恢复、权重导出、跨节点通信和推理部署都需要知道它的轴、粒度、格式与版本。

## MXFP8 与 NVFP4

[MXFP8](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html)把连续小块元素与块级 E8M0 scale 绑定，使不同幅度区域不必共享一个 tensor-wide scale。其收益依赖特定数据布局、转置策略和硬件支持；块边界改变会改变量化结果。

[NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)进一步使用 FP4 数据和分层 scale，并在训练路径中结合随机舍入等机制。FP4 的有效表示能力不能只由“4 bit”判断：

$$
b_{\mathrm{effective}}
=b_{\mathrm{data}}
+\frac{b_{\mathrm{local\ scale}}}{g}
+\frac{b_{\mathrm{global\ scale}}}{G},
$$

其中 $g$、$G$ 是相应 metadata 的摊销粒度。

截至 2026 年，这些格式属于硬件和软件共同定义的快速演进路径。文档应把格式数学、官方实现现状和特定芯片 benchmark 分层陈述，不把单个平台上的收敛或加速结果写成跨架构保证。

## 随机舍入

确定性 round-to-nearest 可能持续把小于半个 ULP 的同向更新舍掉。随机舍入令相邻两个可表示值的选择概率保持期望无偏：

$$
\mathbb E[\operatorname{SR}(x)]=x.
$$

它可以改善极低精度训练中的微小更新保留，但引入新的 RNG 状态和复现要求。分布式 resume 若没有恢复舍入 RNG，就不再是严格轨迹恢复。

## Norm、residual 与 optimizer

以下路径通常比大 GEMM 更敏感：

- LayerNorm / RMSNorm 的平方和与均值；
- residual 多层累加；
- softmax 最大值和指数和；
- cross entropy 的 log-sum-exp；
- Adam 的一阶、二阶矩与参数更新；
- global gradient norm。

常见策略是低精度读取、FP32 reduction、再低精度写回。是否保留 FP32 master weight 取决于 optimizer 与训练格式，不能从“BF16 参数”直接推断。

对 RMSNorm，若输入为 $x\in\mathbb R^H$，

$$
\operatorname{RMSNorm}(x)
=\frac{x}
{\sqrt{\frac{1}{H}\sum_{i=1}^{H}x_i^2+\epsilon}}
\odot w.
$$

平方和应在足够精度中累加；$\epsilon$ 的位置和大小属于算子语义，不能在融合时悄然改变。

## 数值正确性契约

每个低精度算子或训练配置至少记录：

- 输入、输出、accumulator 和 reduction dtype；
- scale 的轴、group、更新频率和保存格式；
- zero point、clipping、rounding 与 saturation 语义；
- NaN、Inf、subnormal 和全 mask 输入行为；
- stochastic rounding、dropout 与 sampling RNG 状态；
- 哪些路径强制 FP32；
- 分布式 reduction 和 checkpoint 后允许的误差边界；
- 可接受误差是逐元素、loss、梯度、收敛还是任务质量。

精确 resume 要保存 scaler、amax history、scale、optimizer state 和所有相关 RNG。只恢复低精度权重通常只是 warm start。

## 何时不要降低精度

- reference 实现、梯度检查和故障定位阶段；
- reduction 极长、动态范围极大而误差预算未知；
- 输出概率尾部、排序或约束解码对微小差异高度敏感；
- calibration 数据与线上分布不一致；
- 硬件没有对应低精度 kernel，只会在运行时反量化；
- shape 太小，metadata、cast 和 launch 开销超过收益；
- 低精度节省被通信、输入或调度瓶颈完全遮蔽。

若收益只来自 checkpoint 体积，而运行时仍执行高精度 GEMM，应明确称为存储压缩，而不是低精度计算加速。

## 验证方法

1. 用 FP64 或稳定 FP32 路径建立小 shape reference。
2. 覆盖零、极小值、最大有限值、outlier、NaN、Inf 和全 mask 行。
3. 分别比较绝对误差、相对误差、ULP、cosine similarity 和 top-$k$ 稳定性。
4. 对 forward、backward、optimizer update 和 checkpoint resume 分层验证。
5. 在代表性训练区间观察 loss、gradient norm、overflow 次数和 scale 分布。
6. 对推理验证 perplexity 之外的长上下文、结构化输出和目标任务质量。
7. 使用实际 kernel 和真实 shape 测显存、带宽与端到端吞吐；把 cast、scale 和 metadata 计入成本。

Google 对 [bfloat16](https://docs.cloud.google.com/tpu/docs/bfloat16) 的说明展示了指数范围与尾数精度的设计取舍；具体 GPU kernel 和数据布局则应结合 [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)及目标运行时验证。

低比特表示进入部署后的校准与 kernel 约束见[量化](../inference/quantization.md)，训练中发现 overflow、NaN 与续训漂移的顺序见[调试手册](../practice/debugging.md)。

## Reference {#reference}

- [NVIDIA Transformer Engine 的 FP8 说明](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [MXFP8](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html)
- [NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- [bfloat16](https://docs.cloud.google.com/tpu/docs/bfloat16)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
