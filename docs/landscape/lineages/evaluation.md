# 从困惑度到运行中的评测

在封闭、单任务监督设置中，模型评测可以相对简单：给定固定测试集，报告困惑度或任务准确率。基础模型把大量任务压进同一个 checkpoint 后，问题变了。研究者不再只问“这个模型在一个任务上好不好”，而要同时追问它覆盖哪些能力、用什么提示、是否见过测试数据、输出怎样解析、成本多高、失败是否可恢复。评测的历史因此不是 benchmark 数量增长，而是测量对象逐层扩大的历史。

## 当一个数字还够用

自回归语言模型天然给出 token 级负对数似然：

$$
\operatorname{PPL}
=\exp\left(-\frac{1}{N}\sum_{t=1}^{N}\log p_\theta(x_t\mid x_{<t})\right).
$$

它适合比较同一 tokenizer、同一数据与同一归一化口径下的语言建模，却不能直接回答指令跟随、事实性或工具使用。tokenizer 一变，$N$ 与每个 token 的难度都变；领域分布一变，同一模型的 PPL 也会显著移动。困惑度从未“失效”，只是它测量的对象比后来希望知道的更窄。

## 从单任务到能力面板

[MMLU](https://arxiv.org/abs/2009.03300)用 57 个学科任务追问广泛知识与问题求解；[BIG-bench](https://arxiv.org/abs/2206.04615)通过大规模协作扩展任务空间，并观察不同任务随规模变化的形态；[HELM](https://arxiv.org/abs/2211.09110)则把焦点从“更多任务”转向“场景 × 指标 × 适配策略”的透明协议。

这里发生的关键变化，是分数不再脱离 harness：

$$
\hat m
=M(D,\theta,\text{prompt},\text{decoder},\text{parser},\text{budget}).
$$

同一个 checkpoint 只要更换 few-shot 示例、答案抽取规则或最大生成长度，就可能得到不同结果。[HELM 到 Chatbot Arena](../works/helm-arena.md)详细解释了为何评测对象从静态模型逐渐变成“模型加协议”。

## 开放式输出逼出新的裁判

多轮对话、写作和代码很难用 exact match 评分。[MT-Bench 与 LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)研究用强模型比较开放式答案，同时明确观察到位置、冗长度和自我偏好等偏差；[Chatbot Arena](https://arxiv.org/abs/2403.04132)把真实用户的成对偏好汇聚成排名。

这不是“自动裁判替代人工”的终点。它把误差从答案解析移到裁判模型与采样人群：

- judge agreement 高，不代表对每个领域、语言和安全边界都有效；
- pairwise 胜率依赖对手池与流量分布，榜单位置不是模型的固有常数；
- 人类偏好能衡量体验，却不自动验证事实、代码执行或环境终态。

因此，开放式评测需要把 rubric、judge 版本、顺序随机化、重复采样和人工校准一并冻结。具体统计模型见[生成式裁判](../../evaluation/generative-judges.md)和[统计推断](../../evaluation/statistical-inference.md)。

## 从 checkpoint 走向系统终态

RAG 与 Agent 把 corpus、工具和环境带入输出路径后，只评文本会漏掉真正的错误。一个回答可能引用了不存在的证据；一个 Agent 可能写出正确计划却没有完成操作。此时测量单位从样本输出变成 trajectory：

$$
\hat p_{\text{success}}
=\frac{\sum_i \mathbf 1[\operatorname{terminal}(s_i)\in G]}
{\sum_i \mathbf 1[\text{episode }i\text{ started}]},
$$

分母必须保留超时、工具异常、权限拒绝和预算耗尽。只统计“有最终文本的回合”会系统性抬高成功率。端到端契约见[Agent 与工具评测](../../evaluation/agent-tool-evaluation.md)和[生产可靠性](../../evaluation/production-reliability.md)。

## 静态榜单为什么会老化

公开 benchmark 一旦成为训练目标，就会同时推动真实进步和适应性过拟合。训练数据可能直接包含题目，合成数据也可能通过 teacher 间接带入答案；同一题库上的反复调参则产生更隐蔽的开发集泄漏。更可靠的方案不是放弃开放评测，而是组合：

1. 可复现的公开回归集；
2. 保密或滚动更新的 holdout；
3. 语义等价扰动和污染诊断；
4. 面向真实流量的 shadow/canary；
5. 对高风险失败的人工与对抗审查。

污染边界见[评测污染](../../evaluation/contamination.md)，安全与动态探测见[安全评测](../../evaluation/safety-evaluation.md)。

## 一条没有终点的脉络

评测的发展不断重复同一件事：模型获得新的自由度，协议就必须记录新的条件。从 tokenizer、prompt 到工具权限和环境版本，每个未冻结变量都可能伪装成能力进步。读一个新结果时，与其先看榜单名次，不如先问：

- estimand 是 checkpoint 能力、系统成功率，还是某类用户偏好；
- 样本、提示、解析、judge、预算和失败分母是否可重放；
- 置信区间是否覆盖随机解码、题目差异和 judge 波动；
- 结论是否越过了数据与场景支持的范围。

完整执行流程见[语言模型评测协议](../../evaluation/language-model-evaluation.md)，各 benchmark 的任务与风险见[Benchmark Registry](../../evaluation/benchmark-registry.md)。

## Reference {#reference}

- [MMLU](https://arxiv.org/abs/2009.03300)
- [BIG-bench](https://arxiv.org/abs/2206.04615)
- [Holistic Evaluation of Language Models / HELM](https://arxiv.org/abs/2211.09110)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [Chatbot Arena](https://arxiv.org/abs/2403.04132)
