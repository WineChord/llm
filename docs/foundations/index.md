# 基础知识地图

一个生成式语言模型可以沿以下链条理解：

$$
\text{原始数据}
\rightarrow \text{token 序列}
\rightarrow \text{向量表示}
\rightarrow \text{上下文混合}
\rightarrow \text{条件分布}
\rightarrow \text{解码与行动}
$$

## 关键对象

- **tokenizer** 决定模型看到的离散单位，也改变序列长度与计算量。
- **embedding** 把离散 token 映射到连续空间。
- **backbone** 在上下文中变换表示；Transformer 只是其中最主流的一类。
- **language-model head** 产生词表上的 logits。
- **objective** 决定哪些预测误差会被优化。
- **decoder** 把概率分布变成具体序列，温度与截断策略会改变输出分布。

## 三个不要混淆

1. **参数知识与上下文信息**：前者写在权重中，后者只在当前计算图或缓存中。
2. **训练目标与下游能力**：低语言建模损失是基础，不自动保证事实性、指令遵循或工具使用。
3. **模型能力与系统能力**：检索、权限、重试、验证器和外部工具往往决定最终系统是否可靠。

## 推荐顺序

先读[语言建模](language-modeling.md)，再用[概率、损失与梯度](probability-objectives.md)把 softmax、交叉熵、困惑度和数值稳定性落到实现；之后阅读[分词与表示](tokenization.md)和[缩放与计算](scaling.md)，再进入[模型结构](../architecture/index.md)。
