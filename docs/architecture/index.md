# 模型结构

模型结构决定信息如何在 token、层、专家和模态之间流动。比较结构时应同时看表达能力、训练稳定性、硬件利用率、缓存形态与长序列复杂度。

## 主干组件

典型 decoder-only 模型由以下部分重复堆叠：

1. token embedding 与位置信息；
2. normalization；
3. self-attention 或其他序列混合器；
4. residual connection；
5. feed-forward network，可能替换为 MoE；
6. 最终 normalization 与 language-model head。

## 主要设计轴

| 设计轴 | 常见选择 | 主要影响 |
| --- | --- | --- |
| 序列混合 | MHA、GQA、线性注意力、SSM、卷积 | 上下文建模、复杂度、缓存 |
| 位置表示 | 绝对位置、RoPE、ALiBi、相对偏置 | 长度泛化与外推 |
| 通道混合 | GELU FFN、SwiGLU、MoE | 参数容量与计算密度 |
| 归一化 | LayerNorm、RMSNorm，pre/post norm | 数值稳定性与深度 |
| 稀疏性 | dense、专家路由、稀疏注意力 | 激活计算与通信 |
| 模态接口 | projector、cross-attention、统一 token | 对齐成本与生成范围 |

## 阅读顺序

先掌握 [Transformer](transformer.md)，再研究[注意力与位置](attention-position.md)。[稀疏与替代架构](moe-alternatives.md)只有放在训练与服务约束下比较才有意义；多模态结构见[多模态融合](../multimodal/architecture-training.md)。
