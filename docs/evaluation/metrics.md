# 指标与评测设计

指标是对决策问题的压缩。设计顺序应是先定义目标总体和统计单位，再定义成功、失败、分母和聚合；从现成榜单反推目标，容易把易测量的代理量当成真正任务。

## 从决策到 estimand

同一个“准确率”可以回答完全不同的问题：

- 在冻结 benchmark 上选择 checkpoint；
- 估计真实请求分布中的成功率；
- 确认新版本不低于回归门槛；
- 找出某语言或难度 slice 的最差表现；
- 比较质量改善是否值得额外时延和成本。

一个 estimand 至少写明

$$
\theta
=
\mathbb E_{z\sim P_{\text{target}}}
\left[m(Y,z)\right],
$$

其中 $P_{\text{target}}$ 是目标总体，$Y$ 是系统输出，$m$ 是评分函数。benchmark 的经验均值只有在抽样、权重和协议能代表 $P_{\text{target}}$ 时，才是该目标的估计。

## 统计单位

| 单位 | 适合 | 常见相关性 |
| --- | --- | --- |
| Token | loss / PPL | 同文档 token 强相关 |
| Item | 独立题目 | 同题族、作者或模板 |
| Prompt | 多候选/偏好 | 同 prompt 回答 |
| Repository | 代码修复 | 同仓库 issue |
| Episode | Agent 任务 | 同环境、用户和工具 |
| User / session | 线上行为 | 同一用户多请求 |
| Time window | 事故与漂移 | 节假日、版本发布 |

有一百万 token 不等于一百万独立样本。置信区间的重采样层级应匹配最上层独立单位，详见[统计推断](statistical-inference.md)。

## 分母与缺失值

对 $N$ 个分配任务，先固定状态集合：

$$
N
=N_{\text{success}}+N_{\text{wrong}}+N_{\text{refusal}}
+N_{\text{invalid}}+N_{\text{timeout}}
+N_{\text{infra}}+N_{\text{missing}}.
$$

端到端成功率使用 $N$ 作为分母。能力口径可以在有效执行子集上计算，但必须同时报告 coverage：

$$
\operatorname{coverage}
=
\frac{N-N_{\text{infra}}-N_{\text{missing}}}{N}.
$$

无效解析、超时和拒答是否计错，取决于任务接口；不能在看到结果后选择对模型有利的口径。若 judge 缺失与回答难度相关，删除缺失样本会产生选择偏差。

## 指标家族

### 分类与抽取

- accuracy 适合类别互斥且代价接近；
- precision/recall/F1 适合稀有正类和不对称错误；
- macro average 先按类别或 slice 平均，强调尾部；
- micro average 汇总所有样本，易被大类主导。

任何 F1 都应说明正类、averaging 和空类处理。

### 生成与可执行任务

- exact match 对 parser 和规范化敏感；
- executable success 更接近代码、数学或状态任务的语义；
- pass@$k$ 衡量多次采样至少一次成功；
- pass$^k$ 衡量多次运行都成功的稳定性；
- pairwise win rate 依赖对手分布和 judge；
- atomic factual support 衡量长回答中可验证主张。

详见[语言模型评测协议](language-model-evaluation.md)、[Agent 与工具评测](agent-tool-evaluation.md)和[幻觉与事实性](hallucination.md)。

### 校准与选择性预测

NLL、Brier 和 ECE 衡量概率与经验正确率的关系；risk–coverage 衡量允许 abstain 时的错误–覆盖权衡。只报告回答后的准确率，会奖励无限拒答。具体公式见[校准与不确定性](calibration-uncertainty.md)。

### 时延与成本

质量比较要在相同资源约束下进行，至少记录：

```text
input/output tokens
number of samples and retries
tool / retrieval calls
wall-clock and queue time
accelerator or API cost
judge / verifier cost
```

可画 Pareto frontier，而不是把质量、时延和成本随意加权成一个不可解释分数。

## 聚合

### Macro、micro 与预定义权重

有 slice $g$、样本数 $n_g$、均值 $\bar m_g$：

$$
\hat\theta_{\text{micro}}
=\frac{\sum_g n_g\bar m_g}{\sum_g n_g},
\qquad
\hat\theta_{\text{macro}}
=\frac{1}{G}\sum_g\bar m_g.
$$

若目标总体有已知权重 $w_g$，使用

$$
\hat\theta_w=\sum_gw_g\bar m_g,
\qquad \sum_gw_g=1.
$$

权重必须在比较前冻结。看到某模型在哪些 slice 更强后再改权重，是另一种选择偏差。

### 严格成功与分项

多约束任务可同时报告：

- strict：所有必要条件都满足；
- component：每类约束成功率；
- severity-weighted：按预先定义的错误严重度；
- end-to-end：包括 parser、tool 和环境失败。

只看 strict 会隐藏接近完成的能力，只看分项又可能掩盖“没有一次完整成功”。

## Effect 与回归门槛

比较 A/B 时，主结果应是

$$
\hat\Delta=\hat\theta_B-\hat\theta_A
$$

及其置信区间，而不只是各自分数或 $p$ 值。回归 gate 可预先定义：

```text
primary metric non-inferiority margin
critical-slice hard floors
latency / cost budgets
safety and unauthorized-action zero/near-zero constraints
minimum coverage
```

非劣界 $\delta_{\min}$ 是产品或研究决策，不应由实验噪声反推。

## 数据与协议

- 开发集用于选择 prompt、阈值和路由，不再是无偏测试。
- 测试集按时间、题族、仓库、用户或环境分组冻结。
- 对开放生成保存原始输出，以便 judge 更新后桥接重评。
- harness、checkpoint、data 和 judge 都绑定 commit/revision。
- 动态 benchmark 记录执行日期和不可变 snapshot。
- 污染、跨语种释义和工具可访问性单独审计。

具体注册字段见[Benchmark 注册表](benchmark-registry.md)，污染边界见[评测污染](contamination.md)。

## 正确性与失效

- **平均分掩盖尾部**：高资源语言或简单题主导 micro average。
- **相关样本当 iid**：区间过窄，微小差异显得稳定。
- **无效样本静默排除**：coverage 下降却分数上升。
- **多次采样不报预算**：pass@$k$ 与单次 greedy 直接比较。
- **只报显著性**：样本很大时无意义差异也显著。
- **只报一个综合分**：能力、成本和安全 trade-off 无法判断。
- **测试集参与选权重**：最终区间不再覆盖预先定义的决策。

## 何时使用简单指标

接口完全确定、错误代价近似相同、样本独立且 parser 稳定时，accuracy 或 mean loss 是清晰起点。即使如此，也应保存状态分母、逐样本输出和版本信息；复杂统计应由真实相关结构和决策风险驱动，而不是为了显得完整。

## 报告卡

```text
decision and target population
estimand, statistical unit and slice weights
success / failure taxonomy
primary and secondary metrics
denominator and missing-value policy
model/data/harness/judge revisions
sample, cluster and trial counts
effect, confidence interval and practical margin
power and multiple-comparison policy
quality / latency / cost frontier
known coverage and contamination limits
```

计算 paired/cluster 区间见[统计推断](statistical-inference.md)，最小实现见[评测工具](../practice/evaluation-tooling.md)。

## Reference {#reference}

- [HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
- [MLPerf Inference Documentation](https://docs.mlcommons.org/inference/)
