# 模型结构

模型结构规定信息在 token、时间、通道、层、专家和模态之间怎样流动。比较结构时要同时追踪表达能力、训练稳定性、状态容量、硬件执行和评测失效。

结构史更适合按“旧瓶颈怎样逼出新接口”来读：[从递归到注意力](../landscape/lineages/transduction-to-attention.md)串起 seq2seq bottleneck、可微对齐与 self-attention；[条件计算](../landscape/lineages/conditional-compute.md)解释稀疏专家为何同时是模型问题和通信问题；[线性时间序列模型](../landscape/lineages/linear-time-sequence-models.md)追踪卷积、状态空间与选择性递推如何重新处理长序列。

## Decoder 主干

典型 decoder-only block：

$$
y_l=x_l+\mathcal S_l(\operatorname{Norm}(x_l)),\qquad
x_{l+1}=y_l+\mathcal C_l(\operatorname{Norm}(y_l)),
$$

$\mathcal S_l$ 是序列混合器，$\mathcal C_l$ 是 dense MLP 或稀疏 expert。主干对象包括：

1. token embedding 与输出词表；
2. [位置编码](position-encoding.md)；
3. norm、residual 与深度方向信号；
4. attention、递推或卷积序列混合；
5. SwiGLU 等通道混合；
6. dense 或 MoE 参数容量；
7. 训练与增量推理状态。

完整计算图见 [Transformer](transformer.md)与 [Decoder Block](decoder-block.md)。

## 序列混合家族

| 家族 | 历史状态 | 主要优势 | 主要边界 |
| --- | --- | --- | --- |
| Full attention | 精确 KV | 任意 token pair 直接交互 | 二次 prefill、线性增长 KV |
| Local/sparse attention | 选定 KV | 降低计算或内存 | 稀疏图决定可达性 |
| Linear attention | 固定或低秩 state | 递推与线性长度 | 状态容量、核函数与数值漂移 |
| State-space model | recurrent state | 并行 scan 与增量常量状态 | associative recall 与 kernel 成熟度 |
| Convolution/long filter | 卷积状态 | 长程感受野 | 内容依赖选择性有限 |
| Hybrid | 多种状态 | 精确路径与低成本路径互补 | 配比、实现与调度更复杂 |

[注意力家族](attention-variants.md)解释 MHA/GQA/MLA，[状态空间与线性注意力](state-space-linear-attention.md)解释 scan、SSD 与 delta rule，[架构比较](moe-alternatives.md)把复杂度与有效能力放在同一张表中。

## 位置与长度

位置机制决定 token 顺序怎样进入分数或表示；扩长方法改变位置分布，却不能单独解决计算、KV 与有效信息利用：

- [注意力与位置总览](attention-position.md)：二者的接口；
- [位置编码](position-encoding.md)：RoPE、ALiBi、相对位置与多维位置；
- [长上下文](long-context.md)：训练长度、位置扩展、稀疏计算、cache 与评测。

最大可接受长度、训练过的长度、可计算长度和有效长度必须分开。

## 参数稀疏

[Mixture of Experts](moe.md)通过 token-dependent routing 激活部分 expert：

$$
y_t=\sum_{i\in\operatorname{TopK}(p_t)}p_{t,i}E_i(x_t).
$$

它增加总参数而控制激活计算，但引入 router、capacity、permutation、all-to-all、grouped GEMM 与负载失衡。[MoE 系统](../systems/moe-systems.md)负责执行路径；结构页负责路由目标与容量语义。

## 记忆

[记忆架构](memory-architectures.md)区分：

- 精确局部 KV；
- 段级递归与压缩历史；
- 外部非参数检索；
- 固定递推 state；
- 测试时可塑参数或 fast weight。

这些方案对“记住”的定义不同。请求间状态隔离、reset 与过期治理属于正确性契约，不是附加功能。

## 多模态接口

视觉、音频和视频可通过 projector、resampler、cross-attention、统一离散 token 或连续生成头接入。[多模态融合](../multimodal/architecture-training.md)比较接口，[理解与生成统一](../multimodal/unified-understanding-generation.md)讨论目标冲突。

## 结构选择问题

评价新结构时逐项回答：

1. 改变的是序列混合、通道混合、深度路由还是状态；
2. forward、backward 与增量 step 的 shape；
3. 参数、FLOPs、bytes、KV 与通信；
4. recurrent、parallel 与 chunked 形式是否等价；
5. 需要哪些新 kernel、并行布局与 checkpoint state；
6. copy、recall、推理、长上下文和真实任务各自怎样评测；
7. 收益来自结构、数据、训练时长还是更大预算；
8. 哪些结果只有作者报告，哪些已有开放实现或复现。

最小代码从[张量原语](../practice/tensor-primitives.md)、[完整 Transformer](../practice/transformer-from-scratch.md)与[递推记忆](../practice/sequence-models.md)进入。
