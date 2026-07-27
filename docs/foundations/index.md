# 基础知识地图

生成式模型把离散或连续观测编码为序列表示，根据上下文产生条件分布，再由训练目标与解码策略塑造行为。

$$
\text{raw observations}
\rightarrow \text{tokens}
\rightarrow \text{representations}
\rightarrow \text{context mixing}
\rightarrow p_\theta(x_t\mid x_{<t})
\rightarrow \text{generation or action}.
$$

## 六个基础对象

| 对象 | 关键问题 | 入口 |
| --- | --- | --- |
| 随机变量 | 联合、条件与期望怎样定义 | [概率、损失与梯度](probability-objectives.md) |
| token | 文本怎样变成可逆离散序列 | [分词与表示](tokenization.md) |
| 表示 | embedding 与 hidden state 保存什么 | [语言建模](language-modeling.md) |
| 上下文 | 当前输入如何改变预测而不改参数 | [上下文学习](in-context-learning.md) |
| 目标 | 哪些 token、偏好或奖励产生梯度 | [训练与对齐](../training/index.md) |
| 规模 | 参数、数据、计算与生命周期成本怎样配比 | [缩放与计算](scaling.md) |

## 语言建模链

token 序列 $x_{1:T}$ 的自回归分解：

$$
p_\theta(x_{1:T})=
\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

训练通常最小化有效 token 上的负对数似然；推理则从同一条件分布经过 temperature、top-$k$、top-$p$、grammar 或搜索生成具体序列。目标相同不代表解码行为相同。

## 三种状态

### 参数

预训练和后训练写入权重，更新慢，容量有限，难以逐条追溯来源。

### 上下文与 cache

当前请求的 token、hidden state 与 KV cache，只在 forward 或会话生命周期内存在。它们可以携带新事实，也可能携带注入或错误。

### 外部状态

检索库、工具、数据库、任务日志和长期记忆由系统维护，有权限、版本和终态。它们不等同于模型“知道了”某件事。

把三者分开，才能定位过期知识、上下文冲突、cache 泄漏与工具失败。

## 不要混淆

1. **训练目标与下游能力**：低 language-model loss 不自动保证事实性、推理、指令遵循或工具使用。
2. **概率与置信**：高 token probability 不自动是事实正确概率。
3. **最大上下文与有效上下文**：接口接受长度不代表信息在所有位置可用。
4. **参数量与计算量**：MoE 总参数、激活参数、训练 FLOPs 与推理 bytes 是不同口径。
5. **模型能力与系统成功**：检索、权限、重试、验证器和服务状态会改变端到端结果。
6. **作者结果与普适规律**：经验拟合依赖数据、架构、优化器与评测。

## 推荐顺序

1. 从[语言建模](language-modeling.md)理解条件概率、训练和生成；
2. 用[概率、损失与梯度](probability-objectives.md)推导 softmax、交叉熵和数值稳定；
3. 手写 [Tokenizer](../practice/tokenizers.md)并检查 Unicode、special token 与 round-trip；
4. 学习[上下文学习](in-context-learning.md)，区分 forward-pass adaptation 与参数更新；
5. 用[缩放与计算](scaling.md)建立参数、token、FLOPs 与生命周期成本；
6. 进入[模型结构](../architecture/index.md)，再连接数据、训练、系统与评测。

每个概念至少经过公式、最小实现、失败样例和评测协议四层。全站覆盖关系见[覆盖地图](../guide/coverage.md)。
