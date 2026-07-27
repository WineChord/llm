# 训练 token 口径

训练 token 是最常被比较、也最容易误解的模型数字之一。它只有在 tokenizer、阶段、采样方式和累计规则明确时才有意义。

## 四个不同对象

### 原始语料规模

抓取或采购的数据在清洗、去重和过滤前的规模。它描述候选池，不等于模型实际看过的数据。

### 去重后的语料池

可供采样的数据总量。重复采样、课程学习和模态混合都会使实际处理量与语料池大小不同。

### 训练处理量

每个 batch 中进入模型的 token 之和：

$$
D=\sum_{t=1}^{S}B_tL_t,
$$

其中 $B_t$ 是第 $t$ 步的全局 batch size，$L_t$ 是有效序列长度。padding、packing、丢弃样本和动态长度会影响精确值。

### 跨阶段累计暴露量

如果预训练、持续预训练和多个后训练阶段都披露了处理量，可以写成

$$
D_{\text{exposure}}=D_{\text{pre}}+D_{\text{cpt}}+D_{\text{post}},
$$

但这只是 token 暴露量的工程统计，不代表互不重复的数据规模，也不能把 RL 环境轨迹简单视为与预训练 token 等价的学习信号。

## 推荐标注

| 标签 | 含义 |
| --- | --- |
| `[D]` | 原始来源直接披露 |
| `[C]` | 由公开配置可确定换算 |
| `[I]` | 由多个来源推断 |
| `[U]` | 公开资料不足 |

表格中的每个数字都应附带阶段和标签。例如“约 15T `[D]`，预训练”比“训练了 15T”更完整。

## 代表性公开口径

下表只用于展示口径，不构成完整模型排名；核验截至 2026-07-23。

| 家族 | 公开处理量 | 阶段 | 证据与说明 |
| --- | ---: | --- | --- |
| DeepSeek LLM | 2T | 预训练 | `[D]` [技术报告](https://arxiv.org/abs/2401.02954) |
| DeepSeek-V2 | 8.1T | 预训练 | `[D]` [技术报告](https://arxiv.org/abs/2405.04434) |
| DeepSeek-V3 | 14.8T | 预训练 | `[D]` [技术报告](https://arxiv.org/abs/2412.19437) |
| Kimi K2 | 15.5T | 预训练 | `[D]` [技术报告仓库](https://github.com/MoonshotAI/Kimi-K2) |
| Qwen2.5 | 最多 18T | 预训练 | `[D]` [官方博客](https://qwenlm.github.io/blog/qwen2.5-llm/)；不同尺寸口径需看模型卡 |
| Qwen3 | 约 36T | 预训练 | `[D]` [官方博客](https://qwenlm.github.io/blog/qwen3/) |
| GLM-5 | 28.55T | 预训练 | `[D]` [技术报告](https://arxiv.org/abs/2602.15763) |

这里的“公开”不意味着可直接横向排名。不同 tokenizer 对同一文本产生不同 token 数；代码、数学、中文、图像 token 的比例也改变单位 token 的信息结构。

## 与计算量的关系

对 dense Transformer，常用粗略估算为

$$
C\approx 6ND,
$$

其中 $N$ 为参数量，$D$ 为训练 token 数，常数包含前向与反向计算。MoE 应区分总参数与激活参数，并额外考虑路由、通信和共享模块；混合精度、稀疏性和重计算也使实际 FLOPs 偏离简单公式。

[Chinchilla](https://arxiv.org/abs/2203.15556)讨论给定计算预算下参数与数据的配比，但其最优关系受数据质量、重复训练、架构和目标函数影响，不能机械套到所有现代训练配方。

## 常见错误

- 把模型的 128K 或 1M 上下文写成训练数据量；
- 将“数据池有 10T token”改写成“模型训练了 10T token”；
- 把多阶段数字相加后称为“独立数据规模”；
- 从训练步数和 batch size 换算时忽略 packing 与有效长度；
- 把图像 patch、离散图像 token 与文本 token 当作同质单位；
- 只比较 token 数，不比较质量过滤、课程、重复率和计算预算。

## 维护一张可审计台账

建议保留以下字段：

| 字段 | 用途 |
| --- | --- |
| model/version | 防止家族名覆盖具体 checkpoint |
| tokenizer | 说明单位定义 |
| stage | pretrain、CPT、SFT、RL 等 |
| amount | 原始数字与单位 |
| evidence | `[D]/[C]/[I]/[U]` |
| source/date | 可回溯原始材料 |
| overlap | 是否可能与其他阶段重复 |
| notes | 多模态、采样或换算条件 |

研究方法见[证据与研究方法](../guide/evidence.md)，结构与数据配比见[缩放与计算](../foundations/scaling.md)。

## Reference {#reference}

- [DeepSeek LLM](https://arxiv.org/abs/2401.02954)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [MoonshotAI/Kimi-K2 technical report and weights](https://github.com/MoonshotAI/Kimi-K2)
- [Qwen2.5 LLM official blog](https://qwenlm.github.io/blog/qwen2.5-llm/)
- [Qwen3 official blog](https://qwenlm.github.io/blog/qwen3/)
- [GLM-5 Technical Report](https://arxiv.org/abs/2602.15763)
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
