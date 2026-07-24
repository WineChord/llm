# 预训练

预训练把大量 token 转化为参数中的统计规律。其难点既是优化大模型，也是让数据顺序、损失和系统状态在数周运行中保持一致、可恢复、可解释。

## 目标函数

decoder-only 模型主要使用 causal language modeling。encoder 或 encoder-decoder 模型也常用 masked language modeling、span corruption 和 sequence-to-sequence denoising。目标选择决定可见上下文与生成方式，不能仅靠推理模板弥补。

## Batch 与 token

global batch 通常由以下量组成：

\[
B_{\text{global}}
=B_{\text{micro}}\times N_{\text{data}}\times N_{\text{accum}}
\]

真正决定学习率和数据进度的常是每步有效 token 数，而非样本数；动态长度和 packing 会使两者差异明显。

## 课程与持续预训练

课程可以按质量、难度、领域、语言或上下文长度改变采样。后期提高高质量数据权重可能改善能力，但也可能降低覆盖或造成重复。持续预训练适合领域适配，需要：

- 较小学习率与稳定 warmup；
- 混入通用 replay 数据；
- 分别评估领域增益和通用能力遗忘；
- 保留原始与新 tokenizer/checkpoint 的兼容记录。

## 长上下文训练

扩大训练长度会同时改变 attention 成本、batch、位置分布和数据拼接。常见策略包括分阶段扩长、长度混合、序列并行和 context parallel。只在短数据上插值位置编码，不能证明长上下文推理能力。

## 监控

至少持续记录：训练/验证损失、学习率、梯度范数、参数范数、tokens/s、MFU、数据源比例、padding 比、overflow、NaN/Inf、通信时间、checkpoint 时间与硬件故障。

缩放预算见[缩放与计算](../foundations/scaling.md)，系统实现见[并行训练](../systems/parallelism.md)。数据质量的影响可参考 [DataComp-LM](https://arxiv.org/abs/2406.11794)。
