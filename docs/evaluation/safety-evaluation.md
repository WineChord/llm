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

ASR、false refusal 与 unauthorized effect 使用不同分母，必须从 episode 状态显式聚合。下面把 `valid=False` 的协议/基础设施失败排除于三个 rate，同时返回实际分母，使 coverage 缺口无法藏在百分比后。

```python
def aggregate_safety_outcomes(records):
    if any(record.get("kind") not in {"harmful", "benign"} for record in records):
        raise ValueError("unknown safety slice")
    if any(type(record.get("valid")) is not bool for record in records):
        raise ValueError("validity must be explicit")
    valid = [record for record in records if record["valid"]]
    harmful = [record for record in valid if record["kind"] == "harmful"]
    benign = [record for record in valid if record["kind"] == "benign"]
    if not valid or not harmful or not benign:
        raise ValueError("valid harmful and benign denominators are required")
    fields = [(record, "forbidden_outcome") for record in harmful]
    fields += [(record, "refused") for record in benign]
    fields += [(record, "unauthorized_effect") for record in valid]
    if any(type(record.get(field)) is not bool for record, field in fields):
        raise ValueError("safety outcomes must be explicit booleans")
    return {
        "asr": (sum(r["forbidden_outcome"] for r in harmful) / len(harmful), len(harmful)),
        "false_refusal": (sum(r["refused"] for r in benign) / len(benign), len(benign)),
        "unauthorized": (sum(r["unauthorized_effect"] for r in valid) / len(valid), len(valid)),
    }
records = [
    {"kind": "harmful", "valid": True, "forbidden_outcome": True, "unauthorized_effect": True},
    {"kind": "harmful", "valid": False},
    {"kind": "benign", "valid": True, "refused": False, "unauthorized_effect": False},
]
rates = aggregate_safety_outcomes(records)
assert rates["asr"] == (1., 1)
assert rates["false_refusal"] == (0., 1)
assert rates["unauthorized"] == (.5, 2)
```

`valid` 的判定必须在看模型结果前固定，并另报 invalid/infra coverage；否则排除策略仍可操纵 rate。未授权副作用从真实环境状态读取，分母覆盖所有有效 episode，而不是只看有害 prompt 或主任务成功。完整边界回归见[评测工具](../practice/evaluation-tooling.md#safety-frontier)。

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

一个配置只有在 ASR、benign success 与 false refusal 三个方向上都不差、且至少一项更好时，才支配另一个配置。下面返回非支配点；`max_asr` 表达先验 hard constraint，而不是事后挑选漂亮阈值。

```python
def safety_frontier(configurations, max_asr=None):
    points = [point for point in configurations if max_asr is None or point["asr"] <= max_asr]
    def dominates(a, b):
        no_worse = (
            a["asr"] <= b["asr"]
            and a["benign_success"] >= b["benign_success"]
            and a["false_refusal"] <= b["false_refusal"]
        )
        strictly_better = (
            a["asr"] < b["asr"]
            or a["benign_success"] > b["benign_success"]
            or a["false_refusal"] < b["false_refusal"]
        )
        return no_worse and strictly_better
    return [point for point in points if not any(dominates(other, point) for other in points)]
points = [
    {"name": "loose", "asr": .2, "benign_success": .95, "false_refusal": .05},
    {"name": "balanced", "asr": .1, "benign_success": .9, "false_refusal": .1},
    {"name": "dominated", "asr": .15, "benign_success": .85, "false_refusal": .15},
]
frontier = safety_frontier(points)
assert {point["name"] for point in frontier} == {"loose", "balanced"}
assert [point["name"] for point in safety_frontier(points, max_asr=.1)] == ["balanced"]
assert all(point["name"] != "dominated" for point in frontier)
```

frontier 只比较给定点，不估计统计不确定性，也不把攻击严重度压成一个合理标量。生产报告应对相同 case 做 paired/cluster 区间，分开未授权写入等 hard failure，并同时展示 latency、成本和 adaptive attacker budget。

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

## 网络安全能力的证据卡 {#cyber-evidence-card}

Cyber evaluation 同时涉及能力、潜在危害与部署防护，至少应把证据分成：

1. 题目型 exploit benchmark；
2. 多步网络环境；
3. 真实软件候选漏洞的人工复核；
4. 模型级拒答、产品 guard 与执行权限。

这些层不能互相替代。[Kimi K3 技术报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)报告其内部环境中 K3 完成 14/36 个 exploit，GLM-5.2 为 8/36；K3 的成功项中 10 个属于 user-space。团队还报告人工复核后约 $70\%$ 的候选为真实问题，并在 6 个项目中确认 16 个新问题。报告没有公开足够环境、候选全集和复核协议，因此这些数字应标记为 **开发团队自报、不可独立重建**，不能据此推断通用网络安全水平。工作页中的逐项边界见 [Kimi K3](../landscape/works/kimi-k3.md)。

[UK AISI/CAISI 的独立初步评估](https://www.aisi.gov.uk/blog/preliminary-assessment-of-kimi-k3s-cyber-capabilities)提供了另一层证据：K3 在其 ExploitBench 上为 $32\%$，GLM-5.2 为 $24\%$；在 ACE 的 41 项中为 $0$；32-step 网络任务平均得分为 17，GLM-5.2 为 11，文中对照的领先美国模型为 28.5；在 10 个更现实任务中完成 1 个。评估还观察到既有 safeguards 没有阻止部分 offensive action。每个数字都绑定该机构当时的 harness、访问方式、攻击预算与日期，不能与内部 36 题直接合并。

这组证据只覆盖网络安全能力与相关 safeguard 行为。若报告未提供生物、化学、说服操纵、隐私、公平性、越权副作用或多模态安全结果，应在报告卡写成 `not evaluated`，而不是从 cyber 结果推出“整体安全”或“整体不安全”。

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

## Reference {#reference}

- [XSTest](https://arxiv.org/abs/2308.01263)
- [HarmBench](https://arxiv.org/abs/2402.04249)
- [JailbreakBench](https://arxiv.org/abs/2404.01318)
- [A StrongREJECT for Empty Jailbreaks](https://arxiv.org/abs/2402.10260)
- [AgentDojo](https://arxiv.org/abs/2406.13352)
- [Llama Guard](https://arxiv.org/abs/2312.06674)
- [OpenAI Instruction Hierarchy](https://openai.com/index/the-instruction-hierarchy/)
- [Instruction Hierarchy Challenge](https://openai.com/index/instruction-hierarchy-challenge/)
- [UK AISI/CAISI Preliminary Assessment of Kimi K3 Cyber Capabilities](https://www.aisi.gov.uk/blog/preliminary-assessment-of-kimi-k3s-cyber-capabilities)
- [Kimi K3 Technical Report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
