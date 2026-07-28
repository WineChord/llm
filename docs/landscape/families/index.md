# 模型家族

模型家族不是一串越来越大的版本号。一次发布可能只更新服务端 checkpoint，另一次可能公开权重却没有新论文；代码、数学、视觉和 Agent 分支也常常从同一主干分叉，再在后续版本重新汇合。若把这些事件压成一条“模型越来越强”的直线，架构继承、训练变化和证据边界都会消失。

这里用四层结构阅读持续演进的模型：

| 层级 | 回答的问题 | 页面职责 |
| --- | --- | --- |
| 家族总览 | 这条家族有哪些主干与分支，应该从哪里进入 | 建立分支图、公开产物账本与主题入口 |
| 时间线 | 什么在何时公开，前后版本究竟改变了什么 | 分开 paper、weights、code、API、product 与 license |
| 工作深读 | 一次关键发布怎样把模型、训练和系统接起来 | 解释特有机制、实验口径、实现接口与未知项 |
| Canonical 主题 | 其中哪些知识可以迁移到别的模型 | 维护通用公式、最小实现、系统约束、评测与失效模式 |

同一机制只在 canonical 页面完整解释；家族页保留它为何在该版本出现、公开证据到哪里为止，并提供双向链接。这样既不会把站点写成厂商目录，也不会让具体模型失去历史位置。

## 三条家族线

| 家族 | 最适合观察的主线 | 从这里开始 |
| --- | --- | --- |
| DeepSeek | 专门模型怎样汇入 MoE 主干，架构效率、推理后训练、多模态与训练系统怎样相互牵引 | [家族总览](deepseek.md) · [演化时间线](../deepseek-timeline.md) |
| Kimi | 长上下文、长程 RL、稀疏模型、线性注意力、原生多模态与 Agent 系统怎样逐步合流 | [家族总览](kimi.md) · [技术谱系](../kimi-timeline.md) |
| GLM | Blank infilling 怎样演变为对话与工具协议，再进入模型—训练—Agent 系统共设计 | [家族总览](glm.md) · [演化时间线](../glm-timeline.md) |

这三条线不是质量排名。它们公开材料的粒度、模型可得性、评测协议和产品边界不同，跨家族比较前必须先固定 checkpoint、chat template、推理预算、工具权限、上下文长度和评测日期。

## 发布账本怎样读

每条事件至少拆成以下对象：

```text
paper / report revision
checkpoint / weights / model card
training and inference code
dataset / recipe / evaluation harness
API / product
license
```

“有论文”不等于“有权重”，“权重开放”不等于训练数据与完整配方开放，“API 同名”也不保证服务端 checkpoint 没有变化。家族页因此同时记录来源类型与未知项，而不是用相邻版本的数字补空白。

发布日期也应按对象解释。论文首次提交、修订、仓库创建、权重上线、API 切换与许可证更新可以是六个日期；需要比较训练规模时，再进入[训练 token 口径](../training-tokens.md)统一分母。

## 从家族回到通用问题

沿家族阅读的价值，在于让具体设计重新接回一般问题：

| 观察轴 | 先问什么 | Canonical 入口 |
| --- | --- | --- |
| 表示与目标 | 训练信号改变了信息流，还是只改变数据分布 | [语言建模](../../foundations/language-modeling.md) · [概率与目标](../../foundations/probability-objectives.md) |
| 架构 | 改变参数容量、激活计算、序列状态还是残差路径 | [注意力家族](../../architecture/attention-variants.md) · [MoE](../../architecture/moe.md) · [线性注意力](../../architecture/state-space-linear-attention.md) |
| 长上下文 | 声明的最大长度怎样转化为有效记忆与服务成本 | [长上下文](../../architecture/long-context.md) · [KV Cache](../../inference/kv-cache.md) |
| 后训练 | 能力来自 SFT、偏好数据、可验证奖励、蒸馏还是搜索 | [推理后训练](../../training/reasoning-posttraining.md) · [LLM 强化学习](../../reinforcement-learning/index.md) |
| 多模态 | 视觉、音频或视频是外接适配器、共享主干还是统一生成接口 | [多模态与生成](../../multimodal/index.md) |
| 系统 | 理论计算下降是否落到 kernel、通信、缓存与调度 | [训练系统](../../systems/index.md) · [推理服务](../../inference/index.md) |
| Agent | 模型能力、工具协议、运行时与环境成功率能否分开归因 | [检索与智能体](../../applications/index.md) · [Agentic RL](../../agentic-rl/index.md) |
| 评测 | 分数是否使用同一数据、预算、judge、工具和时间窗 | [评测与可靠性](../../evaluation/index.md) |

阅读某一发布时，可以先在家族页确认对象，再进入时间线理解继承关系，随后在工作页查看特有细节，最后回到 canonical 页面比较可迁移的机制。反向阅读也同样重要：从一个通用问题出发，家族案例能说明它在真实模型中怎样受到数据、硬件和产品接口约束。

## 覆盖边界

家族总览追踪已经公开且能够由一手来源确认的对象；没有公开的训练数据配比、算力、完整后训练配方、安全评估或服务端变更保持未知。新发布进入站点时，先更新事件账本与继承关系，再判断是否需要工作深读或 canonical 修订。仅有名称变化或营销表述而没有新的可验证技术信息时，不制造重复页面。

## Reference {#reference}

- [DeepSeek official GitHub organization](https://github.com/deepseek-ai)
- [DeepSeek official model collection](https://huggingface.co/deepseek-ai)
- [Moonshot AI official GitHub organization](https://github.com/MoonshotAI)
- [Moonshot AI official model collection](https://huggingface.co/moonshotai)
- [THUDM official GitHub organization](https://github.com/THUDM)
- [Original GLM implementation](https://github.com/THUDM/GLM)
- [Z.ai official GitHub organization](https://github.com/zai-org)
- [Z.ai official model collection](https://huggingface.co/zai-org)
