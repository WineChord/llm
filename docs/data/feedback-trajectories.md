# 偏好、过程与轨迹数据

后训练数据不只包含“输入—答案”。偏好对、过程标签、模型评分与环境轨迹表达不同监督语义，必须使用不同的数据契约和切分方式。

## 四种基本对象

| 对象 | 最小字段 | 能回答的问题 |
| --- | --- | --- |
| 示范 | prompt、response、provenance | 应该怎样回答 |
| 偏好对 | prompt、chosen、rejected、criterion | 两个回答哪个更符合某一标准 |
| 过程样本 | step sequence、step labels、outcome | 哪一步可能改善或失败 |
| 环境轨迹 | observations、actions、tool results、terminal state | 多步行动怎样改变外部状态 |

把这些对象全部压成纯文本会丢失比较关系、动作边界、终止原因和验证器版本。

## 偏好并非绝对标签

偏好记录应说明评价维度：事实性、帮助性、风格、安全、长度、代码是否通过或整体选择。若同一 chosen 同时被解释为“更正确”“更简洁”“更安全”，模型无法知道信号来自哪一项。

Bradley–Terry 模型常写为

$$
P(y_a\succ y_b\mid x)
=\sigma\left(r(x,y_a)-r(x,y_b)\right).
$$

它假设一个标量奖励足以解释成对选择。真实偏好可能非传递、依赖标注者或包含多目标冲突，因此数据中应保留标注者群体、准则、分歧和置信度，而不只保存多数票。

## 负样本怎样构造

负样本可以来自：

- 同一策略的低分采样；
- 人工编辑出的单一错误；
- 旧 checkpoint 或较弱模型；
- 正确答案的格式、事实或安全扰动；
- verifier 找到的失败轨迹。

过于容易的负样本让模型只学会表面风格；只用同一生成器又会把模型身份当作捷径。更稳妥的集合应覆盖局部错误、整体错误和“两个都可接受但一方更优”的细粒度比较。

## 过程监督

将解题或行动拆成步骤 $s_1,\ldots,s_T$ 后，可以为每步记录正确性、进展或风险。这里至少区分：

1. **局部有效**：本步本身是否成立；
2. **全局可达**：执行后是否仍能到达正确终态；
3. **效率**：是否制造无谓成本；
4. **权限**：动作是否被允许。

一个数学步骤局部正确，仍可能把搜索带入死路；一个工具调用返回成功，也可能没有满足业务终态。过程标签不能只由语言流畅度决定。

## 轨迹状态

多步数据需要记录：

```text
task and initial state
environment and tool-schema versions
policy and tokenizer versions
observation/action boundaries
raw tool status and normalized result
terminated / truncated / infrastructure_error
reward components and verifier version
cost and replay metadata
```

`terminated` 表示环境到达定义终态；`truncated` 表示预算、超时或外部中断。把二者都写成“失败”会污染价值估计与恢复训练。更严格的 token/action 契约见[轨迹与策略契约](../agentic-rl/trajectory-contract.md)。

## 切分与污染

随机切分单条记录往往泄漏：

- 同一 prompt 的多个候选被分到训练与测试；
- 同一仓库 issue、题目变体或模板跨集合；
- teacher 生成的近重复轨迹跨集合；
- verifier 或隐藏测试通过工具被模型读取。

应按任务族、来源对象、时间和生成模板成组切分，再做文本与语义近重复检查。对真实环境还要冻结依赖和初始状态，否则版本漂移会伪装成能力变化。

## 质量筛选

最终奖励不是唯一质量指标。可以为轨迹建立多维评分：

$$
q(\tau)=f(
\text{validity},
\text{success},
\text{replayability},
\text{diversity},
\text{efficiency},
\text{policy compliance}).
$$

失败轨迹也有价值：它们可训练错误识别、恢复和偏好比较。筛选的目标不是只保留最高分输出，而是保留能区分策略、可重放且覆盖失效面的学习信号。

## 数据卡

后训练数据卡至少报告：

- 任务分布、语言、长度和生成来源；
- 模型生成与人工编写的比例；
- 标注准则、分歧与质量抽检；
- chosen/rejected 的生成策略与去重；
- 过程标签或 verifier 的准确性边界；
- 环境、权限和可重放性；
- train/dev/test 的分组规则与污染审计；
- 许可、隐私、删除和保留期限。

目标函数见[奖励建模与偏好优化](../training/reward-preference.md)，在线环境数据见[Agentic RL 数据与环境](../agentic-rl/data-environments.md)。

## Reference {#reference}

- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [RLDS: an Ecosystem to Generate, Share and Use Datasets in Reinforcement Learning](https://arxiv.org/abs/2111.02767)
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)
