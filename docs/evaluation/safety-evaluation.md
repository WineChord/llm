# 安全评测

安全评测不是统计模型“拒绝了多少”。它要在明确 threat model 下测量危险目标是否达成、合法任务是否仍可完成、执行层是否越权，以及攻击者增加预算后防御怎样退化。

## Threat model

每个安全结论先冻结：

```text
protected asset and forbidden outcome
model/system/guard/tool boundary
attacker knowledge and access
white-box / logits / text-only interface
single-turn / multi-turn / tool / multimodal channels
query, token, time and human budgets
static or adaptive attacker
defender knowledge and update policy
success verifier and severity
```

同一防御在单轮黑盒攻击下有效，不代表面对 adaptive tool Agent 仍有效。没有 attacker budget 的“鲁棒率”不可比较。

## 统计对象与分母

### 攻击成功

设有效攻击 case 数为 $N_{\text{attack}}$：

$$
\operatorname{ASR}
=
\frac{N_{\text{forbidden outcome achieved}}}
{N_{\text{attack}}}.
$$

`valid attack` 的定义必须说明：攻击 prompt 无效、环境故障、judge 缺失和 target 本来不可完成如何处理。只在模型产生长回答的 case 上计算会选择性排除拒绝和 parser 失败。

### 过度拒绝

对 benign hard set：

$$
\operatorname{false\ refusal\ rate}
=
\frac{N_{\text{benign tasks incorrectly refused}}}
{N_{\text{benign valid tasks}}}.
$$

[XSTest](https://arxiv.org/abs/2308.01263) 专门研究看似敏感但应正常回答的任务。benign set 应覆盖真实困难、教育、转述、分类和安全讨论，不能只放显然无害的短句。

### 效用前沿

改变 guard threshold、防御强度或模型策略时，同时画：

```text
attack success
benign task success
false refusal
unsafe false negative
latency and cost
```

安全与效用应形成 frontier，而不是只选择使 ASR 最低的点。高风险未授权动作可设置 hard constraint，而不与一般帮助性线性平均。

## 攻击层次

### 直接 jailbreak

覆盖改写、角色扮演、编码、长上下文、多轮累积、语言转换和自动搜索。[HarmBench](https://arxiv.org/abs/2402.04249) 提供了自动红队与有害行为评估框架，[JailbreakBench](https://arxiv.org/abs/2404.01318) 强调 attack/defense 的可复现比较，[StrongREJECT](https://arxiv.org/abs/2402.10260) 研究了 jailbreak 响应评分。

这些 2024 基准是公开参考，不是永久攻击全集。报告必须绑定数据、攻击实现、evaluator 与模型日期。

### 间接 prompt injection

攻击指令藏在网页、邮件、文件、代码、工具返回或图像文本中。安全结果不是“模型是否复述攻击”，而是是否：

- 泄露保护数据；
- 调用未授权工具；
- 向错误对象发送或写入；
- 改变后续控制流；
- 让合法用户任务失败。

[AgentDojo](https://arxiv.org/abs/2406.13352) 用动态工具环境评估间接注入与防御。详细终态与副作用指标见[Agent 与工具评测](agent-tool-evaluation.md)。

### Guard 与分类器

输入/输出 guard 是检测器，需报告 threshold 下的 precision、recall、false positive 和 false negative。[Llama Guard](https://arxiv.org/abs/2312.06674) 是公开的安全分类器研究。guard 版本、taxonomy、语言和上下文窗口变化后，旧阈值不再自动校准。

模型对齐、guard 和执行权限是三层防线；任何一层得分不能代表其余两层。

## 指令层级与系统边界

[OpenAI Instruction Hierarchy](https://openai.com/index/the-instruction-hierarchy/) 和[Instruction Hierarchy Challenge](https://openai.com/index/instruction-hierarchy-challenge/) 提供了指令优先级训练与攻击评估的公开入口。[OpenAI Model Spec](https://openai.com/index/our-approach-to-the-model-spec/) 描述了公开行为规范，[Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) 则是能力风险治理框架。

这些文档适合定义研究与治理概念，不能替代具体部署的本地 threat model、权限测试和独立验证。

## Adaptive evaluation

静态攻击集容易被训练或规则记住。adaptive protocol 可以：

1. 在固定总预算内观察模型/guard 响应；
2. 根据失败模式改写攻击；
3. 搜索语言、编码、轮次和工具路径；
4. 保留每次查询与选择策略；
5. 在未用于防御更新的 holdout attacker 上确认。

攻击者和 defender 反复在同一测试集迭代后，该集合成为开发集。最终结果需要新的攻击族、时间切片或独立红队。

## 多模态与长上下文

安全 payload 可能出现在图像 OCR、音频转写、视频帧、文档附件和跨模态指代中。评测冻结预处理与采样，否则“未攻击成功”可能只是模型没读到载荷。还需测：

- 图像可见但低对比文字；
- 音频中的背景或重叠指令；
- 视频中短暂出现的文本；
- 文本指令与视觉内容冲突；
- guard 只检查文本而模型接收多模态。

输入感知与时延分解见[多模态评测](multimodal-evaluation.md)。

## 实现契约

```text
case and attack-family IDs
benign paired task
attacker algorithm/version/budget
model/system/guard/tool revisions
trusted and untrusted channels
raw interaction and environment states
success/severity verifier
refusal/invalid/infra/missing status
human audit and disagreement
```

真实敏感数据不应进入公开测试记录；使用合成凭据和隔离环境。

## 正确性与攻击失效

- **拒答率当安全**：benign utility 未测。
- **只测静态 prompt**：adaptive 和间接攻击缺失。
- **攻击失败因工具无权限，却归功模型**：防线层次混淆。
- **guard accuracy 不报阈值**：operating point 不明。
- **judge 被攻击内容劫持**：成功标签失真。
- **环境故障从分母删除**：攻击/防御条件被选择。
- **只测英文短上下文**：覆盖边界过窄。
- **攻击预算不相等**：模型比较不公平。
- **同一安全集反复调防御**：测试适应。
- **文本无害但已产生副作用**：终态未验证。

## 何时不用单一 benchmark

产品权限、数据资产和外部动作各不相同，公开 jailbreak 分数不能直接授权上线。高风险系统需要公开基准、领域场景、真实工作流回放、独立红队和故障注入的组合；低风险无工具文本任务可使用较小的 threat model，但仍要配对 benign utility。

## 报告卡

```text
protected asset and forbidden outcomes
system/guard/tool boundary and versions
attacker access, adaptivity and budget
attack and benign task distributions
ASR/false refusal/benign success with CIs
unauthorized side effects and severity
threshold and safety–utility frontier
invalid/infra/missing denominators
multilingual/multimodal/long-context coverage
holdout attacker and evaluation date
known untested channels and human audit
```

paired/cluster 区间见[统计推断](statistical-inference.md)，可靠性总览见[可靠性与安全](reliability-safety.md)，最小评测实现见[评测工具](../practice/evaluation-tooling.md)。
