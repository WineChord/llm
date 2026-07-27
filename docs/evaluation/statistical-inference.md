# 统计推断

评测分数是有限样本上的估计。统计推断要回答：比较对象是什么、哪些样本真正独立、差异有多大、区间有多宽，以及同时查看很多指标后结论还剩多少可信度。

## Estimand 与独立单位

先定义目标差异

$$
\Delta
=
\mathbb E_{z\sim P_{\text{target}}}
\left[m_B(z)-m_A(z)\right].
$$

若 A/B 在同一 item 上运行，逐项差值

$$
d_i=m_B(z_i)-m_A(z_i)
$$

保留了难度配对，通常比比较两个独立均值更有效。配对要求 item、输入、协议、预算和 evaluator 完全一致；缺少一侧输出的样本不能静默删除。

真正独立单位可能是题族、文档、repository、用户或环境，而不是每行。若一个 repository 有 100 个 issue，逐 issue bootstrap 会低估共同代码库带来的相关性。

## 分母与缺失

为每个系统保留：

```text
observed valid score
invalid output
timeout
environment / infrastructure failure
missing judge or verifier
not run
```

比较可同时报告：

1. **end-to-end effect**：所有分配样本，按预注册规则给失败状态赋结果；
2. **conditional effect**：双方均有效的可比子集；
3. **coverage difference**：有效执行比例之差。

只报 complete-case effect 隐含“缺失与表现无关”；在难题更容易 timeout 时，这一假设明显不成立。

## Paired bootstrap

对 $n$ 个配对 item：

1. 从索引 $\{1,\ldots,n\}$ 有放回抽取 $n$ 个；
2. 在同一抽样索引上同时取 A/B；
3. 计算 $\hat\Delta^*$；
4. 重复 $B$ 次；
5. 用 bootstrap 分布分位数构造区间。

点估计为

$$
\hat\Delta=\frac{1}{n}\sum_{i=1}^n d_i.
$$

二元指标也应对 item-level 差值重采样，而不是分别 bootstrap 两个准确率。随机种子、重采样次数、区间方法和双侧/单侧目标都需记录。

## Cluster bootstrap

若 item 属于 cluster $g$，先有放回抽 cluster，再带入该 cluster 的全部样本。估计单位取决于目标：

- 若目标总体按请求加权，cluster 被抽到后保留其原始样本数；
- 若目标希望每个 repository/用户等权，先求 cluster mean 再平均。

两者都合理，但 estimand 不同。cluster 数而不是行数决定区间的主要信息量；只有少数 cluster 时，普通 percentile bootstrap 也可能不稳定，应报告 cluster 数并做敏感性分析。

### 分层 cluster

若语言、领域或难度有固定目标权重，可在每个 stratum 内抽 cluster，再按预注册 $w_s$ 聚合：

$$
\hat\Delta_w
=\sum_s w_s\hat\Delta_s.
$$

这样既保留 slice 权重，又不打破 cluster 相关性。

## Effect 与置信区间

报告

```text
baseline score
candidate score
absolute effect
relative effect when meaningful
confidence interval
practical margin
```

相对提升在 baseline 接近零时会夸大，应始终保留绝对差。置信区间表示采样与协议下的估计不确定性，不覆盖 benchmark 污染、judge 偏差和目标总体错配等系统误差。

非劣检验预先给定允许回归 $\delta_0$。若 $\Delta=B-A$，候选非劣通常要求区间下界高于 $-\delta_0$；不能在结果出来后改变 margin。

## 功效与样本量

对配对差值方差 $\sigma_d^2$、希望检测的最小实际差异 $\delta_{\min}$，粗略样本量为

$$
n
\approx
\frac{
\left(z_{1-\alpha/2}+z_{1-\beta}\right)^2
\sigma_d^2
}{
\delta_{\min}^2
}.
$$

$1-\beta$ 是目标 power。二元配对指标的 $\sigma_d$ 主要由 A/B 不一致样本决定；应使用 pilot 的逐项差值或 simulation，而不是独立比例公式。cluster 数据要用设计效应或直接模拟 cluster bootstrap，不能把行数代入。

低 power 的“无显著差异”不等于等价。若样本不足，应报告区间仍允许多大提升或回归。

## Multiple comparison

同时查看多个 benchmark、slice、prompt 和 checkpoint，会提高至少一个偶然胜出的概率。先划分：

- 一个或少数 primary hypothesis；
- 预注册 secondary metrics；
- exploratory slices。

常用控制方式：

- family-wise error：Bonferroni 或 Holm；
- false discovery rate：适合较多探索假设；
- hierarchical testing：先总指标，再进入预定义 slice；
- 独立 holdout：选择后在未查看数据上确认。

若模型由同一批 benchmark 选出，最终报告的置信区间没有自动校正选择过程。最可靠做法是保留未参与选择的冻结测试。

## 重复生成与嵌套结构

一个 prompt 有多次 stochastic trial 时，trial 嵌套在 prompt 内。应先定义目标：

- 单次随机运行成功率；
- 给定 $k$ 次预算至少一次成功；
- 连续 $k$ 次都成功；
- prompt 间 macro success。

不能把所有 trial 展平成独立 item。Agent 环境还可能有 environment seed 和 user simulator 两层随机性，应按层级重采样或用明确的 mixed design。

## 实现契约

```text
item, cluster, stratum and trial IDs
paired run and protocol digests
primary/secondary/exploratory labels
denominator and missing-value policy
point estimator and slice weights
bootstrap seed/count/CI method
power assumptions and practical margin
multiplicity family and correction
```

原始逐样本分数和状态应可重算全部表格；只保存 bootstrap 区间无法审计分母。

## 正确性与失效

- **独立 bootstrap A/B**：丢失配对，区间无谓变宽。
- **逐行 bootstrap cluster 数据**：区间过窄。
- **只有几个 cluster 却报高精度**：有效样本量被夸大。
- **缺失样本 complete-case 删除**：难度相关 missingness 产生偏差。
- **只报 $p$ 值**：effect 和实际意义缺失。
- **置信区间跨零写成相同**：低 power 不等于等价。
- **试了很多 prompt 只报最好一个**：选择过程未校正。
- **同一测试集反复发布区间**：长期适应耗尽独立性。
- **bootstrap 覆盖系统误差的错觉**：污染和 judge bias 不在随机区间内。

## 何时可用简单区间

样本确实独立、A/B 完全配对、指标固定且没有多重选择时，item-level paired bootstrap 足够清晰。小样本、高度聚类或极低事件率需要更谨慎的 exact、permutation、Bayesian 或领域专用分析；不应以复杂方法掩盖数据不足。

## 报告卡

```text
estimand and target population
statistical / cluster / trial units
paired coverage and missing statuses
baseline/candidate/effect/CI
bootstrap or test specification
sample and cluster counts
power and minimum practical effect
primary family and multiplicity handling
slice weights and exploratory labels
systematic uncertainties outside the interval
```

指标与分母见[指标与评测设计](metrics.md)，paired/cluster 实现见[评测工具](../practice/evaluation-tooling.md)。

## Reference {#reference}

- [Bootstrap Methods: Another Look at the Jackknife](https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full)
- [Bootstrap-Based Improvements for Inference with Clustered Errors](https://direct.mit.edu/rest/article/90/3/414/57731/Bootstrap-Based-Improvements-for-Inference-with)
- [A Simple Sequentially Rejective Multiple Test Procedure](https://www.jstor.org/stable/4615733)
- [Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing](https://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_hochberg1995.pdf)
