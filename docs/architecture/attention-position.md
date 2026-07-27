# 注意力与位置

注意力变体通常在三个目标间取舍：表达能力、KV Cache 体积和硬件效率。理论 FLOPs 相同并不意味着 wall-clock 相同，kernel 融合、内存访问和张量形状往往更关键。

本页保留为稳定总览。MHA、MQA、GQA 与 MLA 的计算和 cache 形态见[注意力家族](attention-variants.md)；RoPE 插值、YaRN、滑动窗口与长序列评测见[长上下文](long-context.md)；online softmax 与 FlashAttention 实现见[Kernel 与性能](../systems/kernels-performance.md)。

## MHA、MQA 与 GQA

- **Multi-Head Attention**：每个 query head 有独立 K/V head，容量高但 KV Cache 大。
- **Multi-Query Attention**：所有 query head 共享一组 K/V，显著减少缓存和读取带宽。
- **Grouped-Query Attention**：若干 query head 共享一组 K/V，在质量与效率间折中。

设 batch 为 $B$，缓存长度为 $T$，K/V head 数为 $H_{kv}$，head dimension 为 $d_h$，元素字节数为 $s$，单层 KV Cache 近似：

$$
M_{\text{KV,layer}}=2BTH_{kv}d_hs
$$

因此减少 $H_{kv}$ 会直接降低 decode 阶段缓存带宽。[MQA](https://arxiv.org/abs/1911.02150) 与 [GQA](https://arxiv.org/abs/2305.13245) 给出了代表性设计。

## 高效精确注意力

[FlashAttention](https://arxiv.org/abs/2205.14135) 通过 tiling 与 online softmax 减少 HBM 读写，不是把精确注意力改成近似线性注意力。[FlashAttention-2](https://arxiv.org/abs/2307.08691) 进一步改善并行划分与工作分配。判断收益时要核对 head dimension、mask、dropout、序列长度与硬件支持。

## 位置表示

### RoPE

[Rotary Position Embedding](https://arxiv.org/abs/2104.09864) 对 Q/K 的二维子空间施加与位置相关的旋转，使点积自然包含相对位置信息。频率基数、缩放策略和训练长度会影响外推；简单扩大最大 position 参数并不等于模型学会长上下文。

### ALiBi

[ALiBi](https://arxiv.org/abs/2108.12409) 在 attention score 上加入随相对距离变化的线性偏置，不增加位置 embedding。其简洁性有利于长度外推，但实际质量仍依赖模型与训练设置。

## 长上下文不是单一指标

必须分别测试：

- 能否在长输入中定位证据；
- 证据位置变化时是否稳定；
- 多跳整合与干扰项下是否可靠；
- prefill 时间、显存与并发容量；
- 训练长度之外的数值稳定性；
- 生成长度与输入长度同时增长时的退化。

位置外推、稀疏注意力、检索和状态压缩解决的是不同问题，不能只用“支持多少 token”概括。
