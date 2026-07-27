# 更新日志

## 2026-07-27

- 重做[知识架构](guide/architecture.md)，将概览、机制、案例、实践与索引分层；新增结构完整性门禁，确保页面进入导航并形成可维护的交叉引用网络。
- 补齐[概率与损失](foundations/probability-objectives.md)、[序列构造](data/sequence-construction.md)、[偏好与轨迹数据](data/feedback-trajectories.md)、[Decoder Block](architecture/decoder-block.md)、[注意力家族](architecture/attention-variants.md)、[长上下文](architecture/long-context.md)和 [MoE](architecture/moe.md) 的机制链。
- 将训练主线展开为 [SFT](training/supervised-finetuning.md)、[奖励建模与 DPO/IPO/KTO](training/reward-preference.md)、[优化器家族](training/optimizer-families.md)与[在线策略训练](agentic-rl/math-algorithms.md)，统一数据、目标和评测契约。
- 将 AI Infra 主线展开为 [collective/ZeRO/FSDP](systems/collectives-sharding.md)、[TP/PP/CP/EP](systems/model-parallelism.md)、[kernel](systems/kernels-performance.md)、[解码](inference/decoding.md)、[分页运行时与 chunked prefill](inference/runtime.md)及 [P/D 分离](inference/disaggregation.md)。
- 扩展[视觉语言](multimodal/vision-language.md)、[自回归/diffusion/flow matching](multimodal/generative-modeling.md)、[音频视频](multimodal/audio-video.md)、[经典 RL](agentic-rl/rl-foundations.md)、[轨迹版本](agentic-rl/trajectory-contract.md)、[过程奖励与搜索验证](agentic-rl/search-verification.md)及[语言模型评测协议](evaluation/language-model-evaluation.md)。
- 加入可审计的[最小实现](practice/minimal-implementations.md)，并补全[一手论文](references.md)、[术语](glossary.md)、[学习路线](guide/roadmap.md)和跨章节链接。
- 重构[知识地图](foundations/index.md)，新增[模型谱系](landscape/index.md)、[训练 token 口径](landscape/training-tokens.md)和 [DeepSeek 演化案例](landscape/deepseek-timeline.md)，分开记录论文、权重、API、产品与未知项。
- 扩展[多模态主线](multimodal/index.md)，覆盖连续与离散表示、理解与生成，以及 [Kimi](multimodal/kimi.md) 和 [DeepSeek](multimodal/deepseek.md) 的视觉、生成与 OCR 路线。
- 建立 [Agentic RL](agentic-rl/index.md) 专章，贯通[策略优化](agentic-rl/math-algorithms.md)、[轨迹数据与环境](agentic-rl/data-environments.md)、[训练系统](agentic-rl/training-systems.md)、[长时任务](agentic-rl/long-horizon.md)和[安全评测](agentic-rl/evaluation-safety.md)。
- 增加 [Coding Agent](applications/coding-agents.md)、[幻觉与事实性](evaluation/hallucination.md)、[指令遵循](evaluation/instruction-following.md)、[生产可靠性](evaluation/production-reliability.md)专题，并补全交叉链接。
- 加入[证据卡、事实口径和研究闭环](guide/evidence.md)，统一时效性结论、推断与未知的表达。
- 扩充[原始论文与官方实现](references.md)、[模型谱系资料](landscape/index.md)和[术语索引](glossary.md)。

## 2026-07-24

- 统一[公式与 Markdown](guide/method.md) 分隔符并固定 MathJax 版本；发布前校对源码、生成 HTML 与桌面/移动浏览器渲染。
- 建立从[基础](foundations/index.md)、[数据](data/index.md)、[模型结构](architecture/index.md)到[训练](training/index.md)、[系统](systems/index.md)、[推理](inference/index.md)、[应用](applications/index.md)与[可靠性](evaluation/index.md)的知识架构。
- 发布首批 39 个页面，加入公式、实现示例、[术语表](glossary.md)和[原始文献索引](references.md)。
- 使用原生 [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) 布局、浅色/深色主题、全文搜索和苹方字体。
- 加入内容、[内部链接](guide/architecture.md#cross-links)、敏感信息模式、Python 代码与严格构建检查。
- 核心章节覆盖[语言建模](foundations/language-modeling.md)、[分词](foundations/tokenization.md)、[缩放](foundations/scaling.md)、[数据治理](data/quality-governance.md)、[Transformer](architecture/transformer.md)、[注意力](architecture/attention-position.md)、[MoE 与替代架构](architecture/moe-alternatives.md)、[多模态](multimodal/index.md)、[预训练](training/pretraining.md)、[偏好学习](training/post-training.md)、[模型压缩](training/peft-compression.md)、[分布式训练](systems/parallelism.md)、[KV Cache](inference/kv-cache.md)、[在线调度](inference/serving.md)、[RAG](applications/rag.md)、[智能体](applications/agents.md)和[评测](evaluation/metrics.md)。
- 所有时效性工程资料按具体版本使用；[文献导航](references.md)优先链接原始论文和官方实现。
