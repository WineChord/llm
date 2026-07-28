# DeepSeek 多模态案例

DeepSeek 的 VL、VL2、Janus 与 OCR 系列不是同一模型的简单迭代，而是对视觉理解、统一生成和文档压缩三个问题的不同回答。本页重在比较计算路径。

## 四条路线

| 路线 | 主要目标 | 表示与主干 | 入口 |
| --- | --- | --- | --- |
| DeepSeek-VL | 通用视觉语言理解 | vision encoder + adapter + language model | [论文](https://arxiv.org/abs/2403.05525) |
| DeepSeek-VL2 | 动态分辨率、多图理解 | dynamic tiling + MoE language backbone | [论文](https://arxiv.org/abs/2412.10302) |
| Janus / Janus-Pro | 理解与图像生成统一 | 解耦视觉编码 + 统一 autoregressive Transformer | [Janus](https://arxiv.org/abs/2410.13848)、[Janus-Pro](https://arxiv.org/abs/2501.17811) |
| DeepSeek-OCR | 文档与文字的视觉压缩 | 高分辨率视觉编码 + token 压缩 | [官方仓库](https://github.com/deepseek-ai/DeepSeek-OCR) |

这四条路线的发布对象也不同。VL/VL2 主要研究视觉语言理解，Janus 把理解与图像生成放进统一自回归主干，OCR 则把高分辨率文档压成可供语言模型读取的视觉表示。它们共享部分研究动机，不应把后一条路线的能力或训练配方倒写进前一模型。

## VL：先对齐，再联合训练

典型流水线是

$$
I\xrightarrow{E_v}Z_v\xrightarrow{P}H_v
\xrightarrow{\text{LLM}}Y.
$$

第一阶段冻结大部分主干，训练 projector 对齐视觉与语言空间；后续阶段加入视觉语言预训练与指令数据。优点是复用成熟组件，风险是接口成为信息瓶颈。

DeepSeek-VL 的公开说明强调 hybrid vision encoder：一条路径保留语义，一条路径处理高分辨率细节，再送入 adapter。其训练并非“先完成视觉再碰语言”的简单串联；语言数据从视觉语言预训练阶段就参与，以降低多模态训练对既有语言能力的侵蚀。复现时要把冻结策略、两路 feature shape、projector 与语言数据比例写进同一配方。

评估时要区分：

- encoder 的输入分辨率与裁切；
- 视觉 token 数量；
- projector 类型；
- LLM 是否全量更新；
- caption、OCR、grounding 和对话数据的比例。

## VL2：动态切片与稀疏主干

高分辨率图像若直接缩放，小字和局部结构会丢失；若全部切成 patch，序列又迅速膨胀。动态切片通常保留全局缩略图，再选择若干局部 tile：

$$
N_v=N_{\text{global}}+\sum_{j=1}^{m}N_{\text{tile},j}.
$$

tile 数 $m$ 控制细节与成本。多图场景还要在图片之间分配预算。VL2 将这类视觉预算与 MoE 语言主干结合，说明视觉 prefill、expert routing 和 KV cache 不能分开评估。

动态分辨率还改变位置语义：局部 tile 需要知道自己在原图中的坐标，全局缩略图则提供跨 tile 关系。只把所有 patch 顺序拼接，会让相邻 token 与相邻像素不再等价。评测应按图像宽高比、tile 数、小目标/OCR 尺寸和多图数量切片，并同时报告视觉 token 数。

## Janus：解耦表示，共享推理主干

理解倾向语义抽象，生成倾向局部重建。Janus 使用不同视觉编码路径得到各自适合的表示，再映射到统一序列交给 Transformer：

$$
H=
\begin{cases}
P_u(E_u(I)), & \text{understanding},\\
P_g(E_g(I)), & \text{generation}.
\end{cases}
$$

这种设计保留统一自回归建模的接口，同时避免强迫同一 encoder 同时优化相反目标。它也带来新的问题：任务路由、模态 token 竞争和两个编码空间的一致性。

“共享 Transformer”也不意味着输出头完全相同。理解路径把连续语义 feature 映射进语言空间，生成路径则处理离散视觉 code 并由图像 tokenizer 解码。需要分别验证：

1. 理解输入不会误走生成 encoder；
2. modality token 与 attention mask 能隔离不合法序列；
3. 图像 code 的采样与文本 EOS 不会混淆；
4. 文本能力比较使用相同模板和语言 checkpoint；
5. 图像质量、理解准确率与跨模态一致性分别报告。

Janus 的价值在于把“表示是否共享”与“推理主干是否共享”拆成两个设计轴，而不是证明解耦一定优于所有 early-fusion 方案。

## OCR：压缩不是识别率的同义词

文档模型既要看清字符，也要理解版面。视觉 token 压缩率可写成

$$
\rho=\frac{N_{\text{pixels or patches}}}{N_{\text{visual tokens}}},
$$

但更高 $\rho$ 只有在下游文字、表格、公式和阅读顺序仍可恢复时才有价值。OCR 系统应至少分开评价：

- 字符与词错误率；
- 阅读顺序和段落结构；
- 表格单元格关系；
- 数学公式结构；
- 多页上下文；
- token、时延和显存。

把 OCR 输出作为纯文本送给 LLM 可降低视觉成本，但会丢失字体、位置、图形和不确定性；端到端视觉模型保留结构，却更难审计识别错误。

压缩率还必须绑定输入分辨率和下游协议。相同视觉 token 数处理一张单栏扫描件与一张密集表格，信息损失并不相同；如果 benchmark 只问页面主题，可能掩盖字符级和结构级错误。文档与 GUI 的坐标、阅读顺序和结构化评测见[文档、图表、GUI 与 Grounding](document-gui-grounding.md)。

## 把选择落到计算与错误类型

| 问题 | VL/VL2 | Janus | OCR 路线 |
| --- | --- | --- | --- |
| 主要输出 | 文本 | 文本与图像 token | 文本或结构化文档 |
| 视觉表示 | 连续语义特征 | 理解/生成解耦 | 高密度文档特征 |
| 成本核心 | 高分辨率 prefill | 长图像序列生成 | 压缩率与结构恢复 |
| 关键失效 | 细节遗漏、语言先验 | 模态干扰、生成一致性 | 阅读顺序、表格与公式 |

还应在相同硬件上报告 image preprocessing、vision encoder、projector 与 LLM prefill 的分段时延；否则“视觉 token 更少”不一定转化成更低端到端延迟。原理见[多模态融合与训练](architecture-training.md)、[原生多模态与生成](native-generation.md)和[多模态评测](../evaluation/multimodal-evaluation.md)，家族主干见[DeepSeek 演化案例](../landscape/deepseek-timeline.md)。

## Reference {#reference}

- [DeepSeek-VL: Towards Real-World Vision-Language Understanding](https://arxiv.org/abs/2403.05525)
- [DeepSeek-VL2: Mixture-of-Experts Vision-Language Models](https://arxiv.org/abs/2412.10302)
- [Janus](https://arxiv.org/abs/2410.13848)
- [Janus-Pro](https://arxiv.org/abs/2501.17811)
- [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
