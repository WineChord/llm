# 多模态

VLM、MLLM 与通用多模态模型都试图让文本模型读取或生成图像、视频、音频、语音、文档布局、传感器状态或动作。关键不在名称，而在模态如何表示、对齐、融合和监督。

## 四个层次

1. **感知编码**：把图像 patch、音频帧或其他信号变成特征。
2. **模态接口**：projector、resampler、Q-Former、cross-attention 或统一 tokenizer。
3. **语言与推理主干**：接收模态 token，并产生文本、离散动作或其他 token。
4. **生成头或解码器**：输出图像、音频、视频或连续控制信号。

## 任务并不等价

- caption 衡量描述能力，不等于细粒度视觉定位。
- VQA 可能依赖语言先验，不等于真正读取图像。
- OCR、图表、文档和 GUI 需要高分辨率与空间结构。
- 视频增加时间维、长上下文与采样策略问题。
- 语音系统还涉及流式延迟、说话人、韵律与声学条件。
- 具身模型的动作输出必须满足动力学和安全约束。

## 评估轴

同时考察感知、grounding、组合推理、跨模态一致性、幻觉、时序理解、生成质量、延迟、输入分辨率和模态缺失鲁棒性。只用一个综合分数会隐藏模型到底是在“看”、在猜语言先验，还是依赖测试集模式。

## 阅读路径

- [融合与训练](architecture-training.md)：projector、resampler、cross-attention、分辨率与训练阶段。
- [原生多模态与生成](native-generation.md)：连续/离散表示、自回归/扩散与理解—生成冲突。
- [Kimi 案例](kimi.md)：MoE、视觉、长上下文与 agent 能力如何组合。
- [DeepSeek 案例](deepseek.md)：VL、VL2、Janus 与 OCR 路线怎样分化。

代表性基础包括 [CLIP](https://arxiv.org/abs/2103.00020)、[Flamingo](https://arxiv.org/abs/2204.14198)、[BLIP-2](https://arxiv.org/abs/2301.12597) 和 [LLaVA](https://arxiv.org/abs/2304.08485)。
