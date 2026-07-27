# LLM

大语言模型知识库。

## 从哪里开始

- [学习路线](guide/roadmap.md)：按研究、训练系统、推理服务或应用方向组织阅读顺序。
- [知识地图](foundations/index.md)：理解各层如何连接。
- [模型谱系](landscape/index.md)：分开记录架构、训练、能力与发布事件。
- [证据与研究方法](guide/evidence.md)：校准事实、推断、时效与未知。
- [术语表](glossary.md)：查阅缩写、对象与常用指标。
- [文献](references.md)：按主题查找原始论文与官方资料。

## 知识地图

| 层次 | 核心问题 | 入口 |
| --- | --- | --- |
| 表示与目标 | 模型究竟学习什么 | [基础](foundations/index.md) |
| 演化与证据 | 模型版本怎样比较、结论如何核验 | [模型谱系](landscape/index.md) |
| 数据 | 训练信号从哪里来、是否可信 | [数据工程](data/index.md) |
| 结构 | 信息如何混合、路由和记忆 | [模型结构](architecture/index.md) |
| 模态 | 视觉、音频与语言如何联合表示 | [多模态](multimodal/index.md) |
| 学习 | 参数如何被优化并形成行为 | [训练与对齐](training/index.md) |
| 系统 | 计算、内存和通信如何协同 | [基础设施](systems/index.md) |
| 服务 | 如何控制吞吐、时延与成本 | [推理与服务](inference/index.md) |
| 行动 | 如何检索、调用工具并从环境反馈学习 | [Agentic RL](agentic-rl/index.md) |
| 能力边界 | 如何评测、校准并稳定运行 | [评测与可靠性](evaluation/index.md) |

内容按原理、机制、实现、权衡和失效模式组织；时效性结论会标明核验范围。
