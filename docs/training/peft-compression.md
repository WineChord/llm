# 参数高效训练与压缩

参数高效微调和模型压缩解决不同问题：前者减少训练时可更新参数与优化器状态，后者减少部署时计算、内存或带宽。

## LoRA

[LoRA](https://arxiv.org/abs/2106.09685) 冻结原权重 \(W\)，学习低秩增量：

\[
W'=W+\Delta W,\quad \Delta W=BA,\quad \operatorname{rank}(BA)=r
\]

显存节省主要来自不为全部参数保存梯度和 optimizer state；前向仍需原模型权重。rank、目标层、缩放、dropout 与数据共同决定效果。多个 adapter 可切换或合并，但合并后的干扰需要重新评测。

## QLoRA

[QLoRA](https://arxiv.org/abs/2305.14314) 将冻结基座量化为 4-bit，并通过其计算图训练 LoRA adapter。NF4、double quantization 与 paged optimizer 是其配方组成。它降低微调显存，不代表训练或服务中的所有算子都以 4-bit 执行。

## 蒸馏

知识蒸馏让 student 拟合 teacher 的 logits、表示、生成轨迹或偏好。[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) 给出经典温度蒸馏。生成式模型还要防止 teacher 错误被系统性复制，并区分能力蒸馏、风格模仿与合成数据训练。

## 剪枝与稀疏

非结构化稀疏只有在硬件与 kernel 能利用时才带来实际加速；结构化剪枝更易部署，但容量损失更直接。剪枝后通常需要校准或微调，并对长上下文、少数语言和安全行为做回归。

## 量化

- PTQ 在训练后校准权重或激活。
- QAT 在训练中模拟量化误差。
- weight-only 量化主要减少权重带宽；长 batch decode 还可能受 KV 和激活限制。

[GPTQ](https://arxiv.org/abs/2210.17323)、[AWQ](https://arxiv.org/abs/2306.00978) 与 [SmoothQuant](https://arxiv.org/abs/2211.10438) 代表不同 PTQ 路线。比较时应报告量化粒度、校准集、kernel、真实延迟、吞吐、显存和逐任务质量，而不只写 bit 数。
