# 校准与不确定性

模型给出答案，还需要知道何时值得相信、何时应请求更多证据或 abstain。校准评测比较“预测置信度”与“经验正确率”；它不等于提高准确率，也不保证分布外风险被识别。

## 协议

先定义概率对应的事件：

```text
single-label answer is correct
generated answer passes verifier
atomic claim is supported
tool action reaches target state
safety classifier label is correct
```

置信度来源可以是 class probability、sequence score、verbalized probability、采样一致性、judge 分数或独立 calibrator。它们不是同一量，不能混在一条 calibration curve 中。

评测冻结 model/checkpoint、prompt、temperature、candidate set、parser、verifier 和数据时间窗；若正确性标签有噪声，校准上限也受 evaluator 限制。

## 分母与缺失

每个预测保存 $(p_i,y_i)$，其中 $p_i\in[0,1]$，$y_i\in\{0,1\}$。还要保留：

```text
abstain before prediction
invalid confidence
invalid answer
verifier / judge missing
timeout / infrastructure error
```

只在“模型愿意给置信度且 parser 成功”的样本上画曲线会产生选择偏差。至少报告 confidence coverage 和端到端 coverage。

## Proper scoring rules

### Negative log-likelihood

二元 NLL 为

$$
\operatorname{NLL}
=-\frac{1}{n}\sum_{i=1}^n
\left[
y_i\log p_i+(1-y_i)\log(1-p_i)
\right].
$$

NLL 对高置信错误惩罚很强。实现需对 $p=0,1$ 做数值裁剪，但报告裁剪阈值；过强裁剪会隐藏极端错误。

### Brier score

$$
\operatorname{Brier}
=\frac{1}{n}\sum_{i=1}^n(p_i-y_i)^2.
$$

Brier 有界、易解释，同时混合 calibration 与 discrimination。模型排序能力更强但未校准时，单个 Brier 不能说明问题来源。

## Expected Calibration Error

将样本按置信度分箱 $B_1,\ldots,B_K$：

$$
\operatorname{ECE}
=
\sum_{k=1}^K
\frac{|B_k|}{n}
\left|
\operatorname{acc}(B_k)
-\operatorname{conf}(B_k)
\right|.
$$

ECE 简单但不是 proper scoring rule，并且对 bin 数、边界、样本量和空 bin 敏感。报告时需给出：

- equal-width 或 equal-mass；
- $K$ 与空 bin 处理；
- 每箱样本数和置信区间；
- overall 与关键 slice；
- reliability diagram，而不只一个数。

不同实现的 ECE 不宜直接比较。

## Selective prediction

阈值 $\tau$ 以上才回答：

$$
C(\tau)=P(p\ge\tau),
$$

$$
R(\tau)
=
\mathbb E[\ell(Y,\hat Y)\mid p\ge\tau].
$$

$C$ 是 coverage，$R$ 是 retained risk。画完整 risk–coverage curve，或报告预先定义 coverage 下的 risk；只报告回答子集 accuracy 会奖励无限 abstention。

高风险系统还应把人工升级、额外检索和工具验证作为不同 action，而不是只有 answer/refuse 二元选择。

## 生成式不确定性

### Verbalized confidence

模型可以直接输出概率或置信等级。[Teaching Models to Express Their Uncertainty in Words](https://arxiv.org/abs/2205.14334) 研究了 verbalized uncertainty；[Language Models Mostly Know What They Know](https://arxiv.org/abs/2207.05221) 研究了模型对自身知识的判断。

口头概率受 prompt、数字偏好和社会性措辞影响。需要：

- 冻结 elicitation prompt；
- 检查概率 parser 与非法值；
- 与同一事件的经验正确率校准；
- 测试改写、顺序与语言；
- 不把流畅的“不确定”当成真正概率。

### Semantic uncertainty

开放生成存在多个语义等价文本。先把输出聚为语义类 $c$，再聚合类概率

$$
P(c\mid x)
=\sum_{y\in c}P(y\mid x),
$$

语义熵为

$$
H_{\text{sem}}(x)
=-\sum_cP(c\mid x)\log P(c\mid x).
$$

[Semantic Uncertainty](https://arxiv.org/abs/2302.09664) 研究了用语义等价类区分表面多样与含义不确定。聚类器、采样预算和类概率估计本身会引入误差；多个样本稳定复述同一错误时，低语义熵仍不代表正确。

## Calibration under shift

校准依赖分布。应按时间、语言、领域、长度、难度、工具状态和数据来源切片，并在以下变化后重评：

- checkpoint 或 adapter；
- 量化、温度与解码；
- prompt、template 或 few-shot；
- verifier/judge 与阈值；
- 检索索引和工具；
- 数据时间窗。

在开发集拟合温度或 isotonic calibrator 后，必须用独立测试集报告结果。在线重新校准也要防止 delayed label 和 selection bias。

## 实现契约

```text
event definition and correctness evaluator
confidence source and extraction
model/protocol/data revisions
invalid/missing/abstain policy
NLL clipping and Brier definition
ECE binning and slice weights
risk/coverage threshold policy
semantic samples, clustering and budget
calibrator training/test split
```

## 正确性与攻击失效

- **token probability当回答概率**：多 token、自由生成和工具任务语义不同。
- **ECE 单值无 bin 信息**：实现不可比较。
- **只在回答样本校准**：abstention coverage 被隐藏。
- **开发集拟合后仍在开发集报告**：过拟合。
- **多次采样字符串不一致当风险**：忽略语义等价。
- **低熵当正确**：模型可能稳定错误。
- **judge confidence 当事实概率**：judge 自身偏差未校准。
- **攻击者操纵措辞**：verbalized confidence 或 guard score 被格式诱导。
- **分布漂移沿用旧阈值**：线上风险失控。

## 何时不用单一置信度门槛

不同错误代价、用户群和任务类型需要不同决策规则。对高影响动作，应结合证据、工具验证、权限和人工复核，不以一个模型置信度授权。没有可接受 correctness label 时，也不应把 judge 分数硬解释为真实概率。

## 验证与报告卡

1. 常数概率、完美预测和高置信全错构造可手算基线。
2. 同时报 NLL、Brier、ECE 图和 risk–coverage。
3. 对关键 slice 给 paired/cluster 置信区间。
4. 在协议和时间变化后重新校准。
5. 对 verbalized confidence 做 prompt/语言/顺序扰动。
6. 对 semantic clustering 做人工一致性抽检。

```text
event and confidence definition
correctness evaluator and label quality
data/model/protocol revisions
coverage and invalid/missing statuses
NLL/Brier/ECE specification
risk–coverage operating points
semantic sampling/clustering budget
calibrator split and shift slices
confidence intervals and known blind spots
```

统计区间见[统计推断](statistical-inference.md)，事实 claim 的 support/unknown 见[幻觉与事实性](hallucination.md)，最小实现见[评测工具](../practice/evaluation-tooling.md)。
