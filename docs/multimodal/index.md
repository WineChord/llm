# 多模态与生成

多模态模型把文本、图像、文档、音频与视频映射到可共同计算的表示，也可以把共享状态重新解码为媒体。核心问题不是输入类型的数量，而是信息怎样被采样、编码、对齐、压缩、融合、生成和验证。

[多模态的历史](history.md)沿四条水流展开：可学习感知、语言对齐、可逆生成和环境交互。[从“看懂”到“生成”](../landscape/lineages/multimodal-generation.md)保留较短的工作谱系；[CLIP](../landscape/works/clip.md)、[视觉语言桥](../landscape/works/visual-language-bridges.md)与 [DDPM、DiT、Flow](../landscape/works/diffusion-dit-flow.md)则深入关键转折。

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

<figure class="concept-figure" id="multimodal-field-map" markdown="1">

![图像、音频、视频与身体状态经过采样和表示，进入语义推理、生成或世界模型，输出语言、媒体与动作](../assets/diagrams/multimodal-computing-map.svg)

<figcaption>统一接口位于中间，而不是起点或终点。采样与表示决定模型获得哪些证据；媒体 decoder、规划器和运行时决定输出能否可靠回到真实世界。</figcaption>

</figure>

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

### 先建立共同接口

1. [信号、表示与 Token 化](foundations/signals-tokenization.md)：像素、波形、帧和状态怎样进入有限序列；
2. [对齐、桥接与融合](foundations/alignment-fusion.md)：共享空间、projector、resampler、cross-attention 与 early fusion；
3. [空间、时间、位置与 Mask](foundations/position-time-masks.md)：多轴位置、真实时间戳、同步和信息流；
4. [多模态数据、训练与系统](foundations/data-training-systems.md)：数据 mixture、动态 shape、loss、并行与服务成本。

### 再沿能力分流

- **视觉理解**：[视觉表示、感知与 Grounding](vision/representation-grounding.md)先区分全局语义与空间证据，再进入[视觉语言模型](vision-language.md)和[文档、图表、GUI](document-gui-grounding.md)。
- **图像生成**：[生成建模总览](generative-modeling.md)比较离散自回归、diffusion 与 flow，再沿 autoencoder、score、DiT 和编辑专题深读。
- **声音**：[音频表示、Codec 与理解](audio/representations-understanding.md)从波形与事件开始，[音频生成、语音交互与流式](audio/generation-streaming.md)继续到 TTS、音乐、通用声音与全双工。
- **视频**：[视频理解与长程记忆](video/understanding-long-context.md)处理采样、事件和证据，[视频生成](video/generation.md)处理运动、镜头、音画与长时一致性。
- **全模态**：[理解与生成统一](unified-understanding-generation.md)研究共享表示和混合目标，[Any-to-Any 系统](omni/any-to-any.md)研究多输入、多输出和流式状态。
- **世界与行动**：[世界模型](../world-models/index.md)要求动作条件与规划验证，[具身智能](../embodied/index.md)进一步落到动作协议、控制频率和物理安全。

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

[Kimi-VL 深读](kimi-vl.md)从原生分辨率、视觉—语言桥、MoE decoder 与长视觉上下文展开一个具体模型；[Kimi 多模态分支](kimi.md)、[DeepSeek 多模态案例](deepseek.md)与 [GLM 多模态分支](glm.md)用于观察技术路线怎样跨版本组合通用机制。完整模型边界分别见 [Kimi](../landscape/families/kimi.md)、[DeepSeek](../landscape/families/deepseek.md)和 [GLM](../landscape/families/glm.md) 家族总览；概念结论仍以本节各主题页和一手论文为准。

紧凑的 patchify、对比损失、resampler、坐标、模态 mask、RVQ 与时空 attention 练习见[多模态手撕实现](../practice/multimodal.md)。
