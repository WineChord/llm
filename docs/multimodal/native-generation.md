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

## “统一”至少有四个层级

1. **接口统一**：所有任务都通过同一消息或 token API 表达；
2. **序列统一**：不同模态进入同一 Transformer 上下文；
3. **参数统一**：主干或输出层在模态间共享；
4. **目标统一**：同一 loss 与概率语义覆盖不同模态。

前一层不推出后一层。Chameleon 将离散图像 token 与文本 token 交错做 early fusion，自回归目标在 token 空间统一；Transfusion 则让文本使用 next-token loss、连续图像 patch 使用 diffusion loss，统一主干但保留不同生成语义。两者都比“LLM 调用画图工具”耦合得深，却也不是同一种原生性。

混合目标可抽象为

$$
\mathcal L
=\lambda_{\text{text}}\mathcal L_{\text{AR}}
+\lambda_{\text{image}}\mathcal L_{\text{diff}}
+\lambda_{\text{align}}\mathcal L_{\text{align}}.
$$

$\lambda$ 不只是超参数：它与模态 batch 比例、token 数和 loss reduction 共同决定实际梯度。图像样本每条有上千 patch，文本样本每条有数百 token 时，按 token 平均与按 sample 平均会形成不同课程。

## 序列布局定义了信息流

统一序列必须说明：

- 图像/音频是离散 token 还是连续 latent；
- modality、position 与 time embedding 怎样叠加；
- 文本能否读取未来图像 latent，图像生成能否读取完整文本；
- understanding target 与 generation target 哪些位置参与 loss；
- 模态切换、多个媒体对象和 packing 边界如何编码；
- decode 时保存 KV、diffusion state 还是外部生成器状态。

只发布一张架构图而不发布 attention mask 与 loss mask，无法重建训练目标。对应张量接口见[理解与生成统一](unified-understanding-generation.md)和[多模态融合、位置与训练](architecture-training.md)。

## 共享主干会产生梯度竞争

语义理解希望表示对纹理、像素扰动更不敏感，图像生成却要恢复局部细节；语音识别可丢掉说话人音色，语音生成又需要保留它。共享参数可能产生正迁移，也可能让一个高频模态压制另一个。应至少报告：

- 单模态与混合训练的 paired ablation；
- 各模态 gradient norm、方向相似度与采样比例；
- 同一 checkpoint 的理解、生成和跨模态任务；
- 关闭某个模态数据后，其他能力是否恢复；
- 不同媒体长度下的容量与延迟。

这也是解耦视觉 encoder、独立 tokenizer 或 adapter 仍然有价值的原因：结构分离有时是在显式管理目标冲突，而不是“不够统一”。

## 用可审计问题替代标签

面对“统一”“原生”或“端到端”声明，应核对：

1. 各模态输入由什么 encoder/tokenizer 产生；
2. 哪些层和参数共享；
3. 哪些模态参与预训练、指令和偏好目标；
4. 输出由共享 softmax、独立 decoder 还是外部工具生成；
5. 模态 token、位置和 mask 如何构造；
6. 是否公开训练数据、checkpoint、实现和可复现实验；
7. 理解、生成、grounding 与系统成本是否分别评测。

还要区分作者披露、可下载 checkpoint 和可复现训练代码。开放权重只能验证前向行为，不能自动验证数据混合、目标权重或训练稳定性。案例页面可说明具体模型的组合方式，但通用机制以以上主题页为准；图像生成的离散与连续路径见[多模态生成模型](generative-modeling.md)，音视频状态见[音频与视频](audio-video.md)。

## Reference {#reference}

- [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818)
- [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://arxiv.org/abs/2408.11039)
