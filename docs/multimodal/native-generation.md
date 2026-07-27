# 原生多模态与生成

“原生多模态”没有统一技术定义。一个系统能接收图片或调用生成工具，并不能说明哪些模态真正共享表示、参数和训练目标。

本页保留为稳定入口。系统化内容已拆分为：

- [多模态融合、位置与训练](architecture-training.md)：encoder、projector、resampler、cross-attention 与 token budget；
- [理解与生成统一](unified-understanding-generation.md)：连续/离散表示、共享主干、attention mask 与多目标冲突；
- [图像生成](generative-modeling.md)：VQ、diffusion、flow matching 与采样；
- [音频与语音](audio-language-models.md)：codec token、流式和全双工；
- [视频与世界模型](video-world-models.md)：时空 token、未来预测和长时一致性。

## 三种系统边界

| 形态 | 计算路径 | 能说明什么 | 不能据此说明 |
| --- | --- | --- | --- |
| 工具旁路 | LLM 调用 OCR、ASR 或生成器 | 能组合成熟组件 | 主干已联合学习该模态 |
| 适配器融合 | encoder + projector/resampler + LLM | 模态进入语言隐藏空间 | 理解与生成使用同一表示 |
| 联合建模 | 共享 token、主干或目标 | 多模态参数存在直接耦合 | 所有模态能力均衡或无冲突 |

这些形态可以组合。统一模型仍可能调用 OCR；共享主干也可以保留独立理解 encoder 与生成 tokenizer。

## 判断清单

面对“统一”“原生”或“端到端”声明，应核对：

1. 各模态输入由什么 encoder/tokenizer 产生；
2. 哪些层和参数共享；
3. 哪些模态参与预训练、指令和偏好目标；
4. 输出由共享 softmax、独立 decoder 还是外部工具生成；
5. 模态 token、位置和 mask 如何构造；
6. 是否公开训练数据、checkpoint、实现和可复现实验；
7. 理解、生成、grounding 与系统成本是否分别评测。

案例页面可说明具体模型的组合方式，但通用机制以以上主题页为准。

## Reference {#reference}

- [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818)
- [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/abs/2408.11039)
