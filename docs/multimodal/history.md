# 多模态的四条汇流

今天的多模态模型看起来像一棵枝叶繁密的大树：有的负责读图，有的生成声音，有的预测下一帧，有的直接控制机器人。若只按模型名称记忆，很快会被版本淹没；更稳定的理解方式，是追踪四条原本相对独立、后来逐步汇合的技术水流：

1. **感知**：把连续信号变成可计算的表示；
2. **语义对齐**：让图像、声音、视频与语言进入可比较的空间；
3. **可逆生成**：从紧凑表示重新合成像素、波形与运动；
4. **环境交互**：用观察预测未来，并把预测转化为动作。

这四条路线并非依次替代。一个现代系统往往同时包含感知 encoder、语言主干、媒体 tokenizer、生成 decoder、记忆与控制器；所谓“原生”或“统一”，只有落到这些接口怎样训练和共享参数上才有意义。

<figure class="concept-figure" id="multimodal-computing-map" markdown="1">

![从图像、音频、视频与身体状态，经模态表示、共享语义、生成和世界模型，输出语言、媒体与动作的多模态计算图](../assets/diagrams/multimodal-computing-map.svg)

<figcaption>同一个 Transformer 接口并不会抹平模态差异。信号采样决定可见证据，表示决定可逆性，共享主干决定信息怎样交互，输出协议则决定系统能否回到媒体或真实环境。</figcaption>

</figure>

## 第一条水流：从手工特征到可学习感知

早期视觉和语音系统通常把表示设计与任务学习分开：视觉依赖边缘、角点和局部描述子，语音依赖频谱、声学模型与语言模型。它们建立了两个至今仍然重要的观念：

- 原始信号必须先被采样和压缩；
- 不同不变性需要不同归纳偏置，例如平移、局部时间连续性和尺度变化。

[LeNet](https://ieeexplore.ieee.org/document/726791) 展示了卷积、共享权重与端到端梯度训练如何结合；[AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) 则把大数据、GPU 与深层卷积网络推到同一条扩展曲线上。随后，检测与分割把“这是什么”推进到“它在哪里”，为后来的 grounding 留下坐标、区域和密集预测接口。

语音侧经历了相似变化。频谱仍是有效的信号坐标系，但表示逐渐由模型从波形中学习。[wav2vec 2.0](https://arxiv.org/abs/2006.11477) 和 [HuBERT](https://arxiv.org/abs/2106.07447) 通过遮蔽预测与离散伪标签，从大量无标注语音中学习上下文表示。这里产生的表示适合识别，却未必足以重建音色和相位；“理解表示”和“生成表示”从一开始就不是同一个目标。

Transformer 把这种演进带到视觉和视频。[Vision Transformer](https://arxiv.org/abs/2010.11929) 把图像切成 patch 序列，[TimeSformer](https://arxiv.org/abs/2102.05095) 与 [VideoMAE](https://arxiv.org/abs/2203.12602) 继续处理时空 token。统一的序列接口降低了跨模态架构复用成本，却没有消除采样率、二维几何和时间别名等模态差异。

## 第二条水流：语言成为开放语义接口

图像描述曾经主要是“视觉 encoder + 文本 decoder”的单向翻译任务。[Show and Tell](https://arxiv.org/abs/1411.4555) 是这一范式的代表：图像被压成一个条件向量，语言模型据此生成 caption。它能够描述，却难以把任意语言概念反向变成视觉检索与分类接口。

[CLIP](https://arxiv.org/abs/2103.00020) 改变了接口：不再为每个封闭标签集训练分类头，而是把图像和文本映射到同一个对比空间。自然语言由输出形式变成了开放词表查询。随后几条路线解决“如何让语言模型持续读取视觉证据”：

- [Flamingo](https://arxiv.org/abs/2204.14198) 用 resampler 与 gated cross-attention，把视觉条件反复注入冻结语言模型；
- [BLIP-2](https://arxiv.org/abs/2301.12597) 用 Q-Former 在冻结视觉 encoder 与语言模型之间学习信息瓶颈；
- [LLaVA](https://arxiv.org/abs/2304.08485) 说明简单 projector 配合视觉指令数据也能形成有效基线。

这一步的实质不是“图片变成文字”，而是建立三层契约：

$$
x_{\text{media}}
\xrightarrow{E_m}
z_m
\xrightarrow{P_m}
h_m
\xrightarrow{G}
y.
$$

$E_m$ 决定保留什么感知信息，$P_m$ 决定以多少 token、什么位置和尺度接入，$G$ 决定语言推理怎样访问证据。模型可能拥有很强的语言先验，却因低分辨率、过强压缩或错误坐标变换而看不清输入；因此 caption、OCR、grounding 与跨帧推理必须分开验证。

音频和视频也逐渐进入语言接口。[Whisper](https://arxiv.org/abs/2212.04356) 把多任务语音识别、翻译与时间戳统一为序列预测；[CLAP](https://arxiv.org/abs/2206.04769) 把声音与语言映射到共享空间；视频—语言模型则需要额外处理采样、镜头边界和事件顺序。语言提供了统一的任务描述，却不能代替模态内部的时空结构。

## 第三条水流：表示开始可逆

理解模型追求任务充分的表示；生成模型还要求表示能够恢复信号。两者的分叉可以用一个问题区分：

> 从表示中能否重建那些对当前语义任务不重要、但对感知质量重要的细节？

[VAE](https://arxiv.org/abs/1312.6114) 把数据映射到连续概率 latent，[GAN](https://arxiv.org/abs/1406.2661) 通过生成器与判别器博弈学习样本分布，[PixelRNN/PixelCNN](https://arxiv.org/abs/1601.06759) 则直接分解像素条件概率。三者分别强调可推断 latent、对抗式感知质量与精确似然分解，也暴露出模糊、训练不稳定和长序列采样等不同代价。

[VQ-VAE](https://arxiv.org/abs/1711.00937) 把图像压成离散码，[VQGAN](https://arxiv.org/abs/2012.09841) 进一步用感知和对抗目标改善重建。于是图像可以像文本一样交给自回归或 masked-token 模型处理。音频侧的 [SoundStream](https://arxiv.org/abs/2107.03312) 与 [EnCodec](https://arxiv.org/abs/2210.13438) 用多级量化码本在码率、语义和音质之间取舍；视频 tokenizer 还要同时压缩空间与时间。

另一条生成路线从连续扰动出发。[DDPM](https://arxiv.org/abs/2006.11239) 学习逆转加噪过程，[Latent Diffusion](https://arxiv.org/abs/2112.10752) 把计算移入压缩 latent，[DiT](https://arxiv.org/abs/2212.09748) 让 Transformer 成为可扩展去噪 backbone。[Flow Matching](https://arxiv.org/abs/2210.02747) 则把问题写成沿概率路径学习速度场。它们共享“从简单分布运输到数据分布”的几何，但训练参数化、时间方向、采样器和少步误差并不相同。

这条水流后来扩展到声音和视频：

- 波形生成从逐样本自回归走向 codec token、并行声码器与连续生成；
- 图像生成从单次文生图走向编辑、布局控制、身份保持与多轮交互；
- 视频生成从短片段外观逼真走向运动、镜头、音画同步和较长时间一致性。

媒体质量提高后，评测反而更难：像素距离不能代替语义忠实，人类偏好不能定位时间错误，单个展示样例也不能证明分布覆盖。

## 第四条水流：从生成下一帧到选择下一步

“生成未来”不自动等于“理解环境”。世界模型至少要说明：

1. 状态包含哪些与行动有关的信息；
2. 动作如何改变状态；
3. 预测能否在闭环中支持规划；
4. 误差在长时 rollout 中怎样累积。

[World Models](https://arxiv.org/abs/1803.10122) 把视觉压缩、潜在动力学和控制器拆开；[PlaNet](https://arxiv.org/abs/1811.04551) 与 [Dreamer](https://arxiv.org/abs/1912.01603) 在 latent imagination 中学习策略；[MuZero](https://www.nature.com/articles/s41586-020-03051-4) 表明模型不必重建全部观察，也可以学习支持价值与策略预测的动力学。

另一支路线减少对像素重建的依赖。[I-JEPA](https://arxiv.org/abs/2301.08243) 与 [V-JEPA](https://arxiv.org/abs/2404.08471) 在表示空间预测被遮蔽的图像或视频区域；[V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/) 进一步把视频预训练与机器人动作条件模型连接起来。这里的关键不是生成一张漂亮未来帧，而是预测对下游决策足够稳定的状态。

大规模交互式视频模型又把视觉生成与环境响应拉到一起。[Genie](https://arxiv.org/abs/2402.15391) 从无标注视频学习可控环境，[Genie 2](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/) 和 [Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) 继续强调交互、场景一致性和在线动作响应。它们提供了新的训练环境与模拟接口，但不能仅凭画面连贯就推断出精确物理、可靠因果或可迁移控制。

具身模型把接口推到最后一步。RT-1、RT-2、OpenVLA、$\pi_0$、Gemini Robotics 与 GR00T 等路线共同探索视觉、语言、状态与动作怎样共享主干。真正的困难落在闭环而不是单步输出：

$$
o_t
\xrightarrow{\pi_\theta(\,\cdot\mid g\,)}
a_{t:t+H-1}
\xrightarrow{\text{environment}}
o_{t+1}.
$$

动作 chunk 能降低推理频率，却会减弱突发事件下的响应；互联网视觉语言知识能改善语义泛化，却不能直接补足接触动力学；离线成功率也不能代替真实部署中的恢复、安全和延迟。

## 两次真正的汇流

第一次汇流发生在<strong>共享语义空间</strong>：视觉、声音与视频可以被语言查询，语言模型也能基于外部证据推理。

第二次汇流发生在<strong>共享生成与行动主干</strong>：同一系统既读取媒体，也输出媒体 token、连续 latent 或动作。此时“统一”不再只是输入格式，而涉及：

- 理解表示与重建表示是否共享；
- 自回归、diffusion 与 flow 目标怎样组合；
- 不同模态的 token 数和 loss 怎样平衡；
- 位置、时间、相机与机器人坐标系怎样贯通；
- 训练时的 teacher forcing 怎样过渡到真实闭环。

这也是现代多模态系统最容易混淆的地方：能力表面趋同，内部契约却可能完全不同。

## 怎样阅读接下来的章节

先读[信号、表示与 Token 化](foundations/signals-tokenization.md)，理解连续世界如何进入有限上下文；再按任务进入[视觉表示与 Grounding](vision/representation-grounding.md)、[音频表示与理解](audio/representations-understanding.md)和[视频理解与长程记忆](video/understanding-long-context.md)。生成路线从[生成建模总览](generative-modeling.md)分流到离散生成、diffusion、flow、声音和视频专题；环境交互则沿[世界模型](../world-models/index.md)与[具身智能](../embodied/index.md)继续。

若要判断一项新工作真正改变了什么，可以依次问：

1. 它改变的是表示、目标、数据、采样器，还是系统接口？
2. 改进发生在单模态感知、跨模态对齐、生成还是闭环行动？
3. 报告的指标能否定位改进来自哪里？
4. 公开证据支持的是机制结论、实验结论，还是产品演示？

这样，新模型会落入一条可解释的历史河道，而不是变成另一串孤立名字。

## Reference {#reference}

- [LeCun et al., Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791)
- [Krizhevsky et al., ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
- [Radford et al., Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Alayrac et al., Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
- [van den Oord et al., Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
- [Ho et al., Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Lipman et al., Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Ha and Schmidhuber, World Models](https://arxiv.org/abs/1803.10122)
- [Assran et al., Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
- [Bruce et al., Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)
