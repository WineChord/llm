# 上下文学习

上下文学习（in-context learning, ICL）指模型在权重不更新的情况下，根据当前输入中的示例、规则或反馈改变预测。它发生在一次前向计算中：

$$
p_\theta(y_q\mid x_1,y_1,\ldots,x_k,y_k,x_q),
$$

其中参数 $\theta$ 固定，变化的是上下文及由它诱导的隐藏状态。

ICL 不等于检索、微调或长期记忆。检索负责选择上下文，ICL 描述模型如何利用已给出的上下文；微调更新参数；跨请求记忆还需要额外状态与生命周期契约。

ICL 被系统研究之前，语言模型已经经历了从扩大参数到同时扩大数据与训练计算的转折；这段背景和“上下文能力不等于上下文长度”的边界见[规模、数据与上下文](../landscape/lineages/scaling-and-context.md)。

## 从零样本到多样本

- **Zero-shot**：只给任务描述和查询。
- **One-shot / few-shot**：加入少量输入—输出示例。
- **In-context instruction induction**：从示例中推断隐含规则。
- **In-context adaptation**：在上下文中给纠错、偏好或中间反馈。

[GPT-3](https://arxiv.org/abs/2005.14165)系统展示了规模增大后广泛的 few-shot 行为，但“示例更多”不保证单调提升。示例质量、顺序、覆盖范围和格式可能比数量更重要。

## 一个可计算的视角

在线性回归玩具任务中，给定

$$
X=[x_1^\top;\ldots;x_k^\top],
\qquad
y=[y_1,\ldots,y_k]^\top,
$$

从初值 $w_0$ 做一步梯度下降：

$$
w_1=w_0-\eta X^\top(Xw_0-y),
\qquad
\hat y_q=x_q^\top w_1.
$$

[Transformers Learn In-Context by Gradient Descent](https://arxiv.org/abs/2212.07677)说明 Transformer 在受控任务中可以学出与学习算法相似的前向计算。这个类比提供了可检验机制，但不能推出所有自然语言 ICL 都是在隐式执行梯度下降。

## Induction heads

若上下文出现模式

$$
\ldots[A][B]\ldots[A],
$$

某些 attention head 会从第二个 $A$ 寻找先前 $A$，再复制其后继 $B$ 的信息。其抽象路径是：

1. 一个前置 head 把“前一个 token”信息写入当前位置；
2. induction head 以当前 token 为 query，匹配历史中的同 token；
3. value 携带匹配位置之后的 token。

[In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895)把该机制与受控复制行为联系起来。它解释局部模式补全很有价值，但不应被扩张为对所有推理、任务识别和知识调用的唯一解释。

## 示例选择

一个 demonstration 不只有语义内容，还包含任务格式、标签先验与局部分布。选择示例时应分别控制：

- **相关性**：与查询的语义或操作结构接近；
- **覆盖性**：覆盖不同标签、边界情况和失败类型；
- **正确性**：答案与中间过程可靠；
- **多样性**：避免重复近邻造成错误置信；
- **长度成本**：保留足够查询和输出预算；
- **顺序**：近因效应、首因效应与标签聚集。

检索相似度只是候选生成器。最终组合需要去重、类别平衡、长度约束和冲突检测。

## 顺序与标签语义

设示例序列为 $D=(d_1,\ldots,d_k)$。一般没有

$$
p(y\mid D,x)=p(y\mid \pi(D),x)
$$

对任意排列 $\pi$ 成立。因果模型的位置、注意力衰减和最后若干示例会让顺序改变输出。

标签名也会注入预训练语义。把 `positive/negative` 换成无意义符号可以分离“任务规则学习”与“标签词先验”；交换标签映射可以测试模型是否真正读取 demonstrations。

## 冲突与优先级

上下文可能与参数知识、system instruction、工具结果或其他示例冲突。模型行为取决于训练所得的优先级，而不是形式逻辑自动保证。评测至少覆盖：

- 示例内部互相矛盾；
- 示例与任务说明矛盾；
- 检索文档与参数知识矛盾；
- 图像文字与外部指令矛盾；
- 多轮中较早规则与最新纠错矛盾。

对生产系统，可信数据应带来源与边界；不可信检索内容不能与控制指令共享同一权限。

## 位置与容量

ICL 受有效上下文而不是声明的最大长度限制。长上下文中的示例会竞争注意力与 KV Cache，且模型可能对头尾位置敏感。样本数量增加同时带来：

$$
T
=
T_{\text{instruction}}
+
\sum_{i=1}^{k}T_{d_i}
+
T_{\text{query}}
+
T_{\text{output budget}}.
$$

当预算固定时，增加 demonstration 可能挤压问题、证据或输出。应比较相同总 token 预算下的方案，而不只比较相同示例数。

## 实现契约

可复现实验应固定：

1. tokenizer 与 chat template；
2. demonstration 的原始顺序和分隔符；
3. 检索器、候选池与去重方式；
4. 最大上下文和各部分截断顺序；
5. decoding 参数与随机种子范围；
6. answer extractor 与等价答案归一化；
7. 是否允许模型看到任务标签说明；
8. 数据集与预训练污染检查。

合成任务的最小实现可使用线性回归、键值复制和 multi-query associative recall。相关序列实验见[序列模型手撕实现](../practice/sequence-models.md)。

## 失效模式

- **顺序敏感**：相同示例重排后答案大幅变化。
- **Recency bias**：最后一个 demonstration 压倒整体证据。
- **Label bias**：模型依赖标签词语义而非示例映射。
- **格式脆弱**：分隔符、空格或角色模板改变结果。
- **错误放大**：一个错误示例被模型归纳为规则。
- **上下文冲突**：低可信文本覆盖更高可信约束。
- **伪 ICL**：任务或测试样本已在训练数据中出现。
- **容量饱和**：示例更多但有效利用率下降。
- **位置偏置**：证据移到中部后性能显著降低。

## 验证设计

| 问题 | 最小对照 |
| --- | --- |
| 模型是否利用示例 | 无示例、正确示例、随机示例 |
| 是否学习标签映射 | 无意义标签、交换标签 |
| 是否依赖顺序 | 多次随机排列并报告方差 |
| 是否只是语义近邻 | 同任务远样本与异任务近样本 |
| 是否真正读取全部示例 | 删除单例、加入冲突例 |
| 是否受位置影响 | 同一证据在头、中、尾移动 |
| 是否受污染 | 新合成任务、时间切分、可控随机标签 |
| 是否值得成本 | 相同总 token 与延迟预算比较 |

长位置效应见[长上下文](../architecture/long-context.md)，检索侧的候选构造见[RAG](../applications/rag.md)，参数更新式适配见[参数高效与压缩](../training/peft-compression.md)。

## Reference {#reference}

- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Transformers Learn In-Context by Gradient Descent](https://arxiv.org/abs/2212.07677)
- [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895)
