# 语言模型评测协议

同一个 checkpoint 可以因模板、打分方式、few-shot、解码、parser 和 harness 实现得到不同结果。可比较的基本单位不是“模型名”，而是模型、数据、协议、执行代码和评分器的不可变组合。

## 可复现对象

一次运行至少冻结：

```text
model repository, checkpoint revision and weights digest
tokenizer / normalizer / chat template revisions
dataset repository, revision, split and asset digests
harness repository commit and task-definition digest
few-shot item IDs, order and seed
scoring / decoding / stop / parser configuration
tools, retrieval indexes and external-access policy
judge / verifier model, prompt, threshold and revision
precision, hardware, dependencies and execution date
```

[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 提供了广泛使用的公开实现与[任务定义指南](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)。框架名字仍不够：结果必须绑定 task YAML/代码 commit，因为模板、答案抽取和默认参数会变化。

## 统计对象与状态

逐样本记录保留原始输出和状态：

```text
correct
wrong
abstain or refusal
invalid format
context overflow
timeout
verifier / judge failure
infrastructure error
```

能力口径可在有效评分样本上计算，端到端口径则把 invalid、timeout 和 infra failure 作为系统结果。两者都要给出分母和 coverage；不得只保留解析成功的输出。

题目可能按主题、题族、文档、仓库或用户聚类。统计区间不能默认把每行当独立样本，详见[统计推断](statistical-inference.md)。

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

- tokenizer、文本编码与 normalizer；
- BOS/EOS、文档拼接与跨文档 attention；
- context window、滑动 stride 和重叠 token 去重计分；
- padding、文档边界与 ignore mask；
- 自然对数或其他底数；
- macro domain 与 micro token 聚合。

不同 tokenizer 的 token 单位不同，PPL 不宜直接横向比较。可补充 bits-per-byte：

$$
\operatorname{BPB}
=\frac{-\sum_{t\in\mathcal T}\log_2p_\theta(x_t\mid x_{<t})}
{N_{\text{bytes}}},
$$

但字节编码、normalization 和文档边界仍需一致。[PALOMA](https://arxiv.org/abs/2312.10523) 展示了按众多领域分层评估语言模型 fit，并强调 decontamination、训练顺序、词表与评测格式等控制。

## 多项选择

### 标签概率

prompt 末尾要求输出 `A/B/C/D`，比较标签 token 或 token 序列概率。标签 tokenization、前导空格和候选顺序都会影响分数。若标签需要多个 token，不能只取第一个 token。

### 选项文本条件概率

对候选 $c_i$：

$$
s_i^{\text{sum}}
=\sum_{t=1}^{|c_i|}
\log p(c_{i,t}\mid x,c_{i,<t}),
\qquad
s_i^{\text{mean}}
=\frac{s_i^{\text{sum}}}{|c_i|}.
$$

sum 与 mean 有不同长度偏置，均不是无条件正确。协议应预先固定，并报告选项长度和 calibration variant，而不能看到结果后选择。

### 自由生成后解析

让模型生成解释和答案，再用 parser 抽取，混合了知识、指令遵循、解码和格式能力。至少报告：

- raw correctness；
- parse success；
- conditional correctness among parsed；
- invalid/ambiguous 示例；
- parser revision 与规范化规则。

[MMLU](https://arxiv.org/abs/2009.03300) 等静态基准在不同 harness 中可能因上述选择产生差异；结果比较应采用相同 task definition，或做 bridge run 同时运行两个协议。

## Few-shot 与 in-context learning

示例来源、顺序、分隔符和答案格式都会改变结果。协议应：

1. 冻结示例 item IDs 与顺序；
2. 防止测试题、同题族和近重复进入示例；
3. 记录按类别平衡或随机抽样方式；
4. 用多个预注册顺序评估顺序方差；
5. 检查截断是否删除早期示例；
6. 比较 zero-shot 和相同模板基线；
7. 不把最佳顺序当无偏结果。

若为每个模型单独调 few-shot，比较包含不同的适配预算；这可以是产品优化，但不再是同协议模型比较。

## 生成与可执行任务

数学、代码和结构化任务优先使用 executable verifier。[HumanEval](https://arxiv.org/abs/2107.03374) 提出了代码生成中的 pass@$k$ 估计。每题生成 $n$ 个候选、其中 $c$ 个通过时：

$$
\operatorname{pass@}k
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}},
\qquad n-c\ge k.
$$

若 $n-c<k$，该题估计为 $1$。必须报告 $n$、$k$、temperature、top-$p$、stop、总 token、超时和 verifier。pass@10 与单次 greedy 不共享推理预算。

Agent 任务还关心连续多次都成功的 pass$^k$，含义与“至少一次成功”相反，见[Agent 与工具评测](agent-tool-evaluation.md)。

开放生成没有单一真值，可组合：

- reference-based metric；
- 人类 pairwise 或 rubric；
- LLM judge；
- atomic factual support；
- 指令、风格、长度和安全切片。

judge 协议见[生成式评测与 LLM Judge](generative-judges.md)，事实性见[幻觉与事实性](hallucination.md)。

## 工具、检索与外部访问

模型可访问检索或工具时，必须冻结：

```text
tool schema and implementation revision
retrieval corpus/index snapshot
network and hidden-test access
timeout/retry/cache policy
credentials and permission scope
environment initial state
```

工具增强结果不应与 closed-book 模型分数混为一列。若检索能读到答案、公开 solution 或 hidden tests，问题属于污染和协议泄漏，见[评测污染](contamination.md)。

## 多模态输入

图片、音频和视频还需冻结 asset digest、decoder、resize/crop、帧采样、音频采样率和模态 token 预算。仅保存原始文件名不足以重建模型实际输入，详见[多模态评测](multimodal-evaluation.md)。

## Judge 与 Guard

LLM judge 受位置、长度、格式、家族偏好和注入影响。pairwise 结果至少执行候选顺序交换，并用人工双盲子集校准。guard 评测则同时报告危险输入识别、危险输出拦截和无害任务过度拒绝；只报 accuracy 会隐藏阈值 trade-off。

具体协议见[生成式评测与 LLM Judge](generative-judges.md)和[安全评测](safety-evaluation.md)。

## 污染与版本时间

污染检查覆盖：

- 题干、选项、答案和解释；
- exact、near duplicate、释义与跨语种变体；
- 公开 solution、commit、harness 与 hidden tests；
- teacher、judge、检索和工具能否访问答案；
- 反复用于选 prompt、阈值和 checkpoint 的测试样本。

动态 benchmark 和实时网页必须记录执行时间与 snapshot；今天的“未公开题”不会永久保持未公开。完整方法见[评测污染](contamination.md)。

## 正确性与失效

- **只固定模型名**：checkpoint、量化、adapter 或模板不同。
- **harness 用 floating branch**：历史 task definition 无法恢复。
- **局部 parser 成功样本作分母**：端到端格式能力被排除。
- **不同 tokenizer 直接比较 PPL**：token 单位不一致。
- **sum/mean 看到结果后切换**：多选协议被调参。
- **few-shot 取最佳顺序**：顺序选择偏差。
- **pass@$k$ 不报总预算**：更多采样被误写成模型能力。
- **工具可访问答案**：closed-book 与 tool-augmented 目标混淆。
- **judge 升级直接覆盖旧分数**：时间序列断裂。
- **动态数据不锁 revision**：同名 benchmark 不再是同一集合。

## 何时使用简化协议

固定内部回归集、确定性解码和可执行 verifier 可以使用更精简的卡片，但 model/data/harness commit、逐样本状态和分母仍不能省略。探索阶段可快速迭代 prompt；一旦结果用于外部比较或发布，就必须冻结协议并在独立测试集重跑。

## 评测卡

```text
decision and target population
model/checkpoint/weights digest
tokenizer/template/system prompt/adapters
dataset revision/split/time window/asset digests
harness commit/task-definition digest/dependencies
few-shot IDs/order/scoring/decoding/parser
tools/retrieval/network/permissions/budgets
judge/verifier/guard versions and thresholds
statistical unit/denominator/missing-value policy
sample/cluster/trial counts and slice weights
effect/confidence interval/multiple comparisons
contamination audit/execution date/known limits
```

指标选择见[指标与评测设计](metrics.md)，区间与功效见[统计推断](statistical-inference.md)，最小实现见[评测工具](../practice/evaluation-tooling.md)。

## Reference {#reference}

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [lm-evaluation-harness task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)
- [PALOMA: A Benchmark for Evaluating Language Model Fit](https://arxiv.org/abs/2312.10523)
- [MMLU](https://arxiv.org/abs/2009.03300)
- [Evaluating Large Language Models Trained on Code / HumanEval](https://arxiv.org/abs/2107.03374)
