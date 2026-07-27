# LLM

从目标函数到生产系统。

## 阅读

- [学习路线](guide/roadmap.md)：从共同基础进入模型、训练、系统、推理、智能体或多模态。
- [技术谱系](landscape/index.md)：沿问题、转折和关键工作理解路线怎样形成。
- [知识架构](guide/architecture.md)：理解各主题的边界与依赖。
- [覆盖地图](guide/coverage.md)：在机制、实现、系统、评测和证据之间导航。
- [手撕实现](practice/index.md)：用最小代码固定张量、状态与算法语义。

## 主题

| 领域 | 核心问题 |
| --- | --- |
| [基础](foundations/index.md) | token、概率、目标、上下文学习与缩放 |
| [数据](data/index.md) | 来源、去重、混合、合成数据与训练序列 |
| [模型结构](architecture/index.md) | Transformer、位置、注意力、MoE、递推与记忆 |
| [多模态](multimodal/index.md) | 视觉、文档、GUI、音频、视频与生成 |
| [训练与对齐](training/index.md) | 预训练、SFT、蒸馏、PEFT 与优化稳定性 |
| [系统](systems/index.md) | 数值、GPU、kernel、并行、MoE 与容错 |
| [推理与服务](inference/index.md) | 解码、KV、量化、调度、缓存与 P/D 分离 |
| [检索与智能体](applications/index.md) | 索引、重排、证据生成、工具、记忆与运行时 |
| [推理时计算](reasoning/index.md) | 采样、搜索、验证与预算分配 |
| [强化学习](reinforcement-learning/index.md) | 序贯决策、策略优化、语言模型反馈与 Agentic RL |
| [评测与可靠性](evaluation/index.md) | 协议、统计、校准、事实性、安全与生产门禁 |
| [技术谱系](landscape/index.md) | 历史转折、关键工作、实现传承与证据边界 |

## 索引

[术语表](glossary.md) · [精选文献](references.md) · [证据与研究方法](guide/evidence.md) · [更新日志](changelog.md)

[DeepSeek-V4](landscape/works/deepseek-v4.md)从 mHC、CSA / HCA、Muon 一直追到 MegaMoE、异构 KV、全词表 OPD 与可恢复 rollout；四篇[机制深读](landscape/index.md#deepseek-v4-system)和[103 项引用图谱](landscape/deepseek-v4-reference-map.md)把公式、代码、系统接口与评测口径逐层拆开。

[Kimi K3](landscape/works/kimi-k3.md)则沿架构、预训练、后训练、Agentic RL、3T 训练、百万 token 推理与评测协议展开；[Kimi 演化](landscape/kimi-timeline.md)和[K3 引用图谱](landscape/kimi-k3-reference-map.md)分别补足时间脉络与 150 项一手来源。

[GLM-5](landscape/works/glm-5.md)把 744B MoE、MLA-256、DSA、Muon Split、Shared MTP、28.5T 数据课程、slime 异步 Agentic RL、可执行环境和异构芯片部署放进同一闭环；[架构深读](landscape/works/glm-5-architecture.md)、[Agentic Engineering](landscape/works/glm-agentic-engineering.md)、[GLM 演化](landscape/glm-timeline.md)与[引用图谱](landscape/glm-5-reference-map.md)分别固定机制、系统、版本和证据边界。
