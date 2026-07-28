# HELM、MT-Bench 与 Arena

当基础模型开始同时处理许多任务，传统论文中的“我们在这些数据集上报告分数”逐渐失去共同坐标。不同模型使用不同 prompt、few-shot 示例、解码参数和指标，表格看似可比，实际协议并不相同。HELM 尝试把这种隐含差异显式化；Chatbot Arena 则把开放式对话中难以预先写出的偏好问题交给真实成对比较。两者标志着评测从固定题库走向完整协议与运行中人群。

## HELM：先把测量矩阵写出来

[HELM](https://arxiv.org/abs/2211.09110) 不只增加 benchmark，而是定义 scenario、adaptation 与 metric 的组合。它强调覆盖率与多指标：准确率之外，还观察校准、鲁棒性、公平、偏差、毒性和效率，并公开 prompt 与 completion 以便审计。

这项工作的思想价值在于承认评测资源有限。完整任务空间无法穷尽，因此应先写出 taxonomy，再说明本轮覆盖了哪里、遗漏了哪里。一个稠密但边界清楚的矩阵，比每个模型各选最有利的任务更能支持横向比较。

评测记录至少应能重放：

```text
checkpoint + tokenizer
dataset revision + split
prompt / few-shot sampling
decoder + seed + budget
parser / metric
runtime + failure handling
```

模型名称只是其中一个字段。

## 从标准答案到生成式裁判

开放式对话没有唯一 reference。[MT-Bench 与 LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) 研究让强模型按 rubric 对回答评分或做 pairwise 比较，并系统讨论位置偏差、冗长度偏差与自我增强偏差。

成对偏好可用 Bradley–Terry 模型表示。模型 $i$ 相对 $j$ 的胜率为

$$
P(i\succ j)=\sigma(r_i-r_j).
$$

```python
import torch
def bt_loss(rating, winner, loser):
    return -torch.nn.functional.logsigmoid(rating[winner] - rating[loser]).mean()
rating = torch.zeros(3, requires_grad=True)
winner = torch.tensor([0, 0, 2, 2])
loser = torch.tensor([1, 1, 1, 0])
loss = bt_loss(rating, winner, loser)
loss.backward()
assert loss.ndim == 0 and torch.isclose(rating.grad.sum(), torch.tensor(0.0))
assert rating.grad[1] > 0
```

所有 rating 同时加一个常数不会改变胜率，因此模型只有相对位置；实现时需要固定参考点或加零均值约束。对手池、采样权重与平局规则也会改变估计。

## Chatbot Arena：把用户问题带进协议

[Chatbot Arena](https://arxiv.org/abs/2403.04132) 让用户在匿名模型之间做成对选择，再用统计模型聚合排名。它补足静态题库难以覆盖的真实交互分布，也引入新的条件：

- 谁会访问平台、提出什么语言和主题的问题；
- 哪些模型被配对、流量怎样分配；
- 版本何时切换，旧票能否与新 checkpoint 合并；
- 风格偏好与事实正确怎样分离。

Arena 分数因此是特定时期、平台人群和对手图上的估计，不是模型脱离环境后的绝对属性。排名接近时应看区间和 pairwise support，而不是只看序号。

## HELM 与 Arena 不是替代关系

HELM 倾向可复现、密集和多指标的标准协议；Arena 倾向开放式、动态和真实偏好。前者可能跟不上新交互形态，后者难以精确定位机制和保证人群稳定。可靠评测把二者与可执行 verifier、领域专家和生产终态结合：

| 问题 | 更合适的证据 |
| --- | --- |
| 数学答案是否正确 | 程序或标准答案 |
| 开放式回答是否更有帮助 | 盲化成对人类偏好 |
| 某类安全失败是否出现 | 对抗场景与人工审查 |
| Agent 是否完成任务 | 环境终态与副作用日志 |
| 模型是否跨场景稳健 | 预注册的多场景、多指标协议 |

## 后来的评测为何越来越像系统工程

RAG、工具和 Agent 让评测对象包含外部服务，推理时搜索让预算成为能力的一部分，模型裁判又引入另一个 checkpoint。每增加一个组件，都需要版本、超时和失败分母。评测不再是训练结束后的表格，而是贯穿数据、模型和部署的测量系统。

统计细节见[统计推断](../../evaluation/statistical-inference.md)与[生成式裁判](../../evaluation/generative-judges.md)，污染和动态 holdout 见[评测污染](../../evaluation/contamination.md)，完整历史位置见[从困惑度到运行中的评测](../lineages/evaluation.md)。

## Reference {#reference}

- [HELM 论文](https://arxiv.org/abs/2211.09110)与 [stanford-crfm/helm](https://github.com/stanford-crfm/helm)；
- [MT-Bench / LLM-as-a-Judge 论文](https://arxiv.org/abs/2306.05685)与 [FastChat LLM judge implementation](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)；
- [Chatbot Arena 论文](https://arxiv.org/abs/2403.04132)与承载其早期公开实现的 [FastChat](https://github.com/lm-sys/FastChat)；当前项目入口见 [LMArena](https://lmarena.ai/)。
