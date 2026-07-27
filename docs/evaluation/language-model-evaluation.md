# 语言模型评测协议

同一个 checkpoint 可以因模板、打分方式、few-shot、解码和答案解析不同得到明显不同的分数。评测的基本单位不是“模型名”，而是模型、协议、数据版本与执行环境的组合。

## Loss 与困惑度

对有效 token 集 $\mathcal T$：

$$
\bar{\mathcal L}
=-\frac{1}{|\mathcal T|}
\sum_{t\in\mathcal T}
\log p_\theta(x_t\mid x_{<t}),
\qquad
\operatorname{PPL}=e^{\bar{\mathcal L}}.
$$

必须固定：

- tokenizer 与 normalizer；
- BOS/EOS 和文档拼接；
- context window 与滑动 stride；
- 每个 token 是否只计分一次；
- padding、跨文档边界和 ignore mask；
- 空间是自然对数还是其他底数。

不同 tokenizer 的 token 单位不同，PPL 不宜直接横向比较。可补充 bits-per-byte 或 byte-normalized 指标，但同样要说明文本编码。

## 多项选择

常见协议至少有三种：

### 选项标签概率

prompt 末尾要求输出 `A/B/C/D`，比较对应单 token 或 token 序列概率。结果高度依赖标签 tokenization 与提示格式。

### 选项文本条件概率

对每个候选 $c_i$ 计算

$$
s_i=\sum_{t=1}^{|c_i|}
\log p(c_{i,t}\mid x,c_{i,<t}).
$$

总和偏向短选项；平均 log-prob 又可能偏向长而常见的表述。是否做长度归一化必须作为协议固定，而不能看到结果后选择。

### 自由生成后解析

让模型生成解释和答案，再用 parser 提取。它更接近交互行为，却混入指令遵循、解码和 parser 失败。应同时报告原始正确率与可解析率。

[MMLU](https://arxiv.org/abs/2009.03300)等基准在不同 harness 中常因这些选择出现差异；比较前先对齐官方或明确指定的协议。

## Few-shot 与 in-context learning

示例选择、顺序、分隔符和答案格式都会改变结果。few-shot 评测应：

- 固定示例来源与随机种子；
- 防止测试题或近重复进入示例；
- 记录是否按类别平衡；
- 多个顺序重复并报告方差；
- 保证上下文截断没有删除早期示例；
- 比较 zero-shot 与相同模板的基线。

只报告最佳示例顺序会产生选择偏差。

## 生成式任务

### 确定性任务

数学、代码、结构化答案优先使用 executable verifier。解析层应区分：

```text
correct
wrong
invalid format
timeout
infrastructure error
```

把 infra error 计作模型错误会低估能力，排除所有 invalid 又会高估端到端可用性；两种口径应分别报告。

### 开放式任务

摘要、对话与创作没有唯一答案。可组合：

- reference-based metric；
- 人类 pairwise；
- rubric-based judge；
- 事实或引用 verifier；
- 长度、风格和安全切片。

单个相似度指标不足以覆盖事实、完整性与可读性。

## pass@$k$

若每题生成 $n$ 个候选，其中 $c$ 个通过，常用无偏估计为

$$
\operatorname{pass@}k
=1-
\frac{\binom{n-c}{k}}{\binom{n}{k}},
\qquad n-c\ge k.
$$

若 $n-c<k$，该题估计为 $1$。必须报告 $n$、$k$、temperature、总 token 与 verifier；pass@10 不能与单次 greedy 直接比较。

## LLM-as-a-judge

judge 可按 rubric 输出分项或 pairwise 选择，但常受：

- 位置顺序；
- 回答长度与格式；
- 自身模型家族偏好；
- 引用外观；
- prompt injection；
- 评分温度与版本；
- 难例上的不稳定。

校准流程：

1. 建立人工双盲子集；
2. 测 judge–human 一致与分项混淆；
3. 交换候选顺序；
4. 去除模型身份与无关格式；
5. 插入明显错误和正确对照；
6. 对低 margin 样本人工复核；
7. judge 升级时做桥接评测。

judge 分数是一个测量通道，不是事实真值。

## Guard 与安全分类器

安全评测要区分输入风险识别、输出风险识别、拒答策略和合法任务过度拒绝。分类器阈值改变 precision/recall，不能只报告 accuracy。多轮、编码、工具输出和图像内文本需要独立切片。

guard 本身也会被版本、语言和上下文长度影响。生产策略应保留模型判断之外的权限与执行约束。

## 污染

检查不只做题干文本匹配，还包括：

- 选项、答案和解释；
- 翻译、改写和同源题；
- 公开 solution、commit 与 benchmark harness；
- teacher model 是否可访问答案；
- 检索或工具能否读到隐藏测试。

时间切分可以降低已公开答案污染，但依赖、网页和检索索引也要冻结。

## 评测卡

```text
model checkpoint and weights digest
tokenizer, template and system prompt
dataset version and split
few-shot examples and order
scoring / decoding / parser
tools, retrieval and external access
judge/verifier version
hardware and inference precision
sample count, repetitions and confidence interval
invalid/timeout/infra-error policy
contamination audit
```

统计比较见[评测设计](metrics.md)，事实性见[幻觉与事实性](hallucination.md)，多步系统见[Agentic RL 评测与安全](../agentic-rl/evaluation-safety.md)。
