# 多模态

多模态模型把文本、图像、文档、音频、视频、传感器状态与动作映射到可共同计算的表示。核心问题不是模型能接受多少输入类型，而是信息怎样被编码、对齐、压缩、融合、监督和验证。

[多模态理解与生成](../landscape/lineages/multimodal-generation.md)沿两条最初相对独立的路线展开：一条从对比式视觉—文本对齐走向可对话的视觉语言模型，另一条从离散表征和 diffusion 走向可扩展生成。关键接口可分别深读 [CLIP](../landscape/works/clip.md)、[Flamingo、BLIP-2 与 LLaVA](../landscape/works/visual-language-bridges.md)，以及 [DDPM、DiT 与 Flow Matching](../landscape/works/diffusion-dit-flow.md)。

## 统一计算图

一个通用系统可写成

$$
z_m=E_m(x_m),
\qquad
h_m=P_m(z_m),
\qquad
y=G(h_{\text{text}},h_{\text{image}},h_{\text{audio}},\ldots),
$$

其中：

- $E_m$ 是模态 encoder 或 tokenizer；
- $P_m$ 是 projector、resampler 或离散化接口；
- $G$ 是共享语言/生成主干；
- 输出可能是文本、离散媒体 token、连续 latent 或动作。

模型名称不能替代计算图。判断能力时，应追踪哪些参数共享、哪些表示可逆、哪些模态真正参与 loss，以及推理中是否调用外部工具。

## 能力层次

| 层次 | 目标 | 关键瓶颈 |
| --- | --- | --- |
| 感知 | 从信号提取局部与全局特征 | 分辨率、采样率、encoder |
| 对齐 | 建立跨模态语义对应 | 对比数据、负样本、粒度 |
| 融合 | 让语言或共享主干读取模态 | token budget、位置、接口 |
| Grounding | 把输出绑定到区域、时间或元素 | 坐标、时序、可验证性 |
| 生成 | 从条件生成图像、音频或视频 | 表示可逆性、采样成本 |
| 行动 | 把感知和语言映射到环境动作 | 状态、动力学、安全 |

Caption、VQA、OCR、grounding、生成和行动不可互相替代。一个模型可能擅长整体描述，却无法读小字、定位证据或执行可靠动作。

## 阅读路径

- [视觉语言模型](vision-language.md)：ViT、CLIP、视觉 token 与基本接入方式。
- [融合、位置与训练](architecture-training.md)：projector、resampler、cross-attention、动态分辨率和训练契约。
- [文档、图表、GUI 与 Grounding](document-gui-grounding.md)：版面、OCR、坐标和界面交互。
- [理解与生成统一](unified-understanding-generation.md)：连续/离散表示、共享主干与多目标冲突。
- [图像生成](generative-modeling.md)：VQ、diffusion、flow matching 与 conditioning。
- [音频与语音](audio-language-models.md)：codec、语义/声学 token、流式与全双工。
- [视频与世界模型](video-world-models.md)：时空 token、未来预测、长时一致性。
- [具身智能与动作](embodied-agents.md)：VLA、动作表示、闭环控制与安全。

旧的[原生多模态与生成](native-generation.md)和[音频与视频](audio-video.md)页面继续作为稳定分流入口。

## Token budget

图像 patch 数近似

$$
N_{\text{image}}
=
\frac{H}{P_h}\frac{W}{P_w},
$$

视频 tubelet 数近似

$$
N_{\text{video}}
=
\frac{T}{P_t}\frac{H}{P_h}\frac{W}{P_w},
$$

音频若每秒产生 $r$ 个 token，时长为 $\tau$：

$$
N_{\text{audio}}=r\tau.
$$

这些 token 与文本共享上下文、prefill、显存和训练 loss。分辨率、帧率和采样率提高时，必须同时说明压缩、位置和成本。

## 统一评测轴

1. **证据依赖**：答案是否真正依赖输入模态。
2. **定位**：能否指向正确区域、时间段或界面元素。
3. **组合推理**：能否整合多个模态与多处证据。
4. **生成忠实度**：语义、属性、文字、身份与局部编辑。
5. **鲁棒性**：裁切、压缩、噪声、缺失模态和输入攻击。
6. **系统成本**：原始输入规模、模态 token、prefill、显存、首包与费用。
7. **安全边界**：不可信媒体内容不能改变高权限控制意图。

自动综合分数应与可解释的能力切片、人工对照和系统指标并列。

## 案例与概念

[Kimi-VL 深读](kimi-vl.md)从原生分辨率、视觉—语言桥、MoE decoder 与长视觉上下文展开一个具体模型；[Kimi 家族案例](kimi.md)和[DeepSeek 案例](deepseek.md)则用于观察技术路线怎样跨版本组合通用机制。概念结论仍以本节各主题页和一手论文为准，厂商时间线不承担通用分类。

紧凑的 patchify、对比损失、resampler、坐标、模态 mask、RVQ 与时空 attention 练习见[多模态手撕实现](../practice/multimodal.md)。
