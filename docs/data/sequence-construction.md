# 序列构造与打包

原始样本只有经过模板、分词、截断、拼接和 mask 才会成为训练张量。序列构造是数据与模型之间的接口；接口错误往往不会触发异常，只会让模型稳定地学习错误目标。

## 三组边界

一条训练序列同时包含三种边界：

| 边界 | 约束对象 | 典型错误 |
| --- | --- | --- |
| 样本边界 | 两条独立文档或对话 | 跨样本 attention 泄漏 |
| 消息边界 | system、user、assistant、tool | role token 与模板错位 |
| 目标边界 | 哪些 token 进入 loss | prompt、padding 或工具结果被误训 |

attention mask 决定“能看到什么”，loss mask 决定“优化什么”，position IDs 决定“位于哪里”。三者不能互相替代。

## Chat template

对话样本应先序列化为唯一的规范形式：

```text
<bos><system>...</system><user>...</user><assistant>...</assistant><eos>
```

上面只是抽象示意，不是通用 token 字符串。真正契约由 tokenizer 的特殊 token、role 分隔符、generation prompt 和 tool schema 共同决定。训练和服务若使用不同模板，模型看到的条件分布就不同。

模板审计至少覆盖：

- 空 system message、连续同角色消息与多轮工具结果；
- BOS/EOS 是否自动添加，是否被重复添加；
- assistant 起始标记是否进入输入，结束标记是否进入目标；
- 文本中出现与控制 token 相同的字面内容时如何转义；
- tool call 的 JSON、结果和错误状态怎样序列化；
- 截断是否切断结构化消息或把答案全部移除。

## Causal LM 的 label shift

token 序列为 $x_0,\ldots,x_{T-1}$ 时，输入位置 $t$ 预测目标 $x_{t+1}$。一种清晰的张量契约是

```text
input_ids: x[0 : T - 1]
labels:    x[1 : T]
```

许多框架允许传入同一长度的 `input_ids` 与 `labels`，由模型内部完成 shift。两种方式只能选一种；重复 shift 会漏掉 token，完全不 shift 则变成复制目标。

只训练 assistant response 时，prompt 位置的 label 设为 ignore index：

$$
m_t=
\begin{cases}
1,&x_{t+1}\text{ 属于目标回复},\\
0,&\text{其他}.
\end{cases}
$$

role marker、EOS 和工具结果是否参与训练是建模选择，应显式记录。

## Padding 与 packing

### Padding

同 batch 序列补齐到共同长度。右 padding 对训练直观；左 padding 常用于批量生成，使最后一个非 padding token 对齐。无论哪种方式，都要同时校正 attention mask、position IDs 和 KV Cache 起点。

### Packing

将多条短样本拼进固定长度块可提高有效 token 比：

$$
\eta_{\text{token}}
=\frac{N_{\text{有效 token}}}{N_{\text{张量 token}}}.
$$

packing 有两种不同语义：

1. **连续语料流**：文档可以互相可见，常用于大规模预训练；
2. **独立样本拼箱**：每条样本必须有 block-diagonal causal mask，不能跨边界读取。

只在边界插入 EOS 不一定能阻止 attention 穿透；如果任务要求独立，就需要显式分段 mask 或支持 packed metadata 的 kernel。

## 截断与长度采样

固定从尾部截断会系统性删除答案结尾，从头部截断则可能删除任务条件。可按任务采用：

- 保留开头与结尾，中间裁剪；
- 在文档内随机窗口采样；
- 对多轮对话保留 system 与最近若干轮；
- 先按消息或结构单元裁剪，再分词；
- 对超长样本单独进入长上下文 batch。

必须统计“截断后仍有多少目标 token”。一条只剩 prompt、没有监督目标的 SFT 样本不应悄悄进入训练。

## 多源混合的真实权重

数据源的样本概率不等于它对梯度的权重。若源 $i$ 的平均有效长度为 $\bar L_i$，其近似 token 占比为

$$
q_i=
\frac{p_i\bar L_i}
{\sum_j p_j\bar L_j}.
$$

长推理轨迹、代码和短对话即使样本数相同，贡献的 token 数也完全不同。若 loss 先按样本平均，权重又会变化。因此同时报告样本占比、有效 token 占比和 loss 归一化方式。

## 最小可重放记录

每个数据版本应能重建：

```text
raw sample id
template and tokenizer version
normalization and filtering version
truncation decision
pack group and segment offsets
input_ids / labels / attention metadata hashes
source mixture and random seed
```

无需永久保存所有中间张量，但必须能定位“某个 checkpoint 看到的具体序列为何如此”。

## 验证样例

发布训练前固定一组金丝雀样本，打印 token、解码文本、role、segment、position、attention 可见区间和 label。至少包含短文本、空字段、多轮工具、Unicode、代码、超长截断与 packed 多样本。再用一个极小 batch 过拟合，确认 loss 能下降且生成模板闭合。

下一步可读[监督微调](../training/supervised-finetuning.md)和[轨迹与反馈数据](feedback-trajectories.md)。
