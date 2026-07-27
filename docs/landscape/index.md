# 模型谱系

模型谱系不是发布日期列表，而是一套回答“什么发生了变化”的坐标系。只有把架构、训练、能力、开放程度与产品形态分开，模型之间的继承关系才不会被命名和营销口径遮蔽。

## 五条并行轴

| 轴 | 观察对象 | 不能替代它的指标 |
| --- | --- | --- |
| 架构 | dense/MoE、注意力、路由、归一化、多模态接口 | 参数量 |
| 训练 | 数据口径、token 暴露、优化器、阶段与目标 | 上下文长度 |
| 后训练 | SFT、偏好优化、RL、工具与环境反馈 | base benchmark |
| 系统 | 训练并行、精度、推理缓存、服务调度 | 单卡速度 |
| 发布 | 论文、权重、API、产品和许可证 | 单一“发布日期” |

例如同一模型家族可以先发布技术报告，随后开放 API，再发布较小权重；后续产品版本也可能复用名称但更换 checkpoint。记录谱系时应把事件拆开。

## 最小模型卡

```text
模型与精确版本：
对象类型：base / instruct / reasoning / multimodal / system
参数：总参数 / 激活参数 / 未知
输入与输出模态：
上下文与最大生成：
预训练与后训练口径：
核心结构变化：
权重、代码、许可证：
论文 / API / 产品事件：
核验日期与未知项：
```

若某字段没有公开证据，保留“未知”比从相邻版本外推更准确。

## 典型演化模式

### 从规模扩张到稀疏激活

MoE 把模型容量与单 token 计算部分解耦：

$$
y=\sum_{i\in \operatorname{TopK}(g(x))}p_i(x)E_i(x).
$$

总参数量反映容量上限，激活参数量更接近每 token 的专家计算；通信、负载均衡和专家缓存又会改变实际系统成本。见[稀疏与替代架构](../architecture/moe-alternatives.md)。

### 从语言模型到可行动系统

能力边界逐渐包含长上下文、工具调用、代码执行、环境反馈和多轮状态。此时 benchmark 分数不再只由 checkpoint 决定：

$$
\text{system quality}=f(\text{model},\text{context},\text{tools},\text{policy},\text{harness}).
$$

模型谱系与 agent 系统谱系应分别记录，详见[工具与智能体](../applications/agents.md)和[Coding Agent](../applications/coding-agents.md)。

### 从外接视觉到原生多模态

演化路径包括 projector 对齐、cross-attention、统一 token、自回归理解与生成，以及视觉工具旁路。所谓“原生”必须落到训练目标和计算图，不应只依据产品支持图片输入。见[多模态](../multimodal/index.md)。

## 两张互补表

- [训练 token 口径](training-tokens.md)关注数据暴露、阶段与可比性。
- [DeepSeek 演化案例](deepseek-timeline.md)展示怎样把论文、权重、API 与产品事件拆开。

具体家族页面适合作为案例，不应替代[基础](../foundations/index.md)、[模型结构](../architecture/index.md)和[训练](../training/index.md)中的通用机制。
