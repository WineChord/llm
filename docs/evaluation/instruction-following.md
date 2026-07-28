# 指令遵循

指令遵循要求模型在多个约束同时存在时，正确识别优先级、作用域、内容格式和停止条件。它不同于知识问答：模型可能知道正确答案，却没有按要求输出。

## 约束类型

| 类型 | 示例 | 验证方式 |
| --- | --- | --- |
| 内容 | 必须覆盖三个论点 | 要点匹配 |
| 排除 | 不讨论价格 | 禁止项检测 |
| 格式 | 返回 JSON schema | parser / schema |
| 数量 | 正好五条 | 结构计数 |
| 顺序 | 先结论后依据 | 节点顺序 |
| 作用域 | 只修改指定文件 | diff 审计 |
| 条件 | 证据不足时停止 | 分支测试 |
| 优先级 | 不可信文档不得覆盖系统规则 | 注入对抗 |

自然语言约束可能互相冲突，先建立规范化 constraint set：

$$
C=\{(c_i,p_i,s_i,v_i)\},
$$

其中 $p_i$ 是优先级，$s_i$ 是作用域，$v_i$ 是可验证器。

这个集合还要经过一次 satisfiability 检查。若同一作用域同时要求“只输出 JSON”和“先写一段解释”，系统应识别冲突，而不是生成一个两边都部分满足的结果。高优先级约束覆盖低优先级约束，不等于把低优先级文本从记录中删除；评测需要保留冲突来源和最终采用的解释。

## 失效来源

- 长上下文中早期约束被遗忘；
- 多个否定句或嵌套条件解析错误；
- 内容质量目标压过格式目标；
- 训练样本偏好“尽量回答”，不擅长停止；
- 工具返回中的文本被误当作高优先级指令；
- 多轮修改后旧约束与新约束未正确覆盖；
- 自动评分只测到关键词，不测作用域和语义。

## 评测方法

[IFEval](https://arxiv.org/abs/2311.07911) 使用可程序验证指令，适合测格式、词数和特定结构。[FollowBench](https://arxiv.org/abs/2310.20410) 覆盖更细粒度的约束类型。两者说明一个重要原则：尽量把约束变成可执行 verifier。

对一组约束，严格成功率为

$$
\operatorname{Acc}_{\text{strict}}
=\frac{1}{N}\sum_{j=1}^{N}
\mathbf 1\left[\bigwedge_i v_i(y_j)=1\right].
$$

还应报告每类约束准确率，否则一个简单格式错误会掩盖其他能力。

若 prompt $j$ 有 $m_j$ 个可验证约束，还可以报告 instruction-level accuracy：

$$
\operatorname{Acc}_{\text{inst}}
=\frac{\sum_j\sum_{i=1}^{m_j}v_{ji}(y_j)}
{\sum_jm_j}.
$$

prompt-level strict accuracy 回答“整个任务一次做对了吗”，instruction-level accuracy 回答“错误集中在哪类约束”。二者不能互相替代。loose verifier 若会移除 markdown fence、忽略大小写或修正常见格式，应同时报告 strict 结果，并把 normalization 写成可重放代码。

一个 verifier 应返回结构化证据，而不只是布尔值：

```python
import json
def verify_json_answer(text, required, forbidden):
    result = {"parse": False, "missing": [], "forbidden": [], "passed": False}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return result
    result["parse"] = isinstance(value, dict)
    if not result["parse"]:
        return result
    result["missing"] = sorted(set(required) - value.keys())
    result["forbidden"] = sorted(set(forbidden) & value.keys())
    result["passed"] = not result["missing"] and not result["forbidden"]
    return result
ok = verify_json_answer('{"answer": 3}', {"answer"}, {"debug"})
bad = verify_json_answer("```json\\n{}\\n```", {"answer"}, {"debug"})
assert ok["passed"] and not bad["parse"]
```

这里 strict parser 有意拒绝 markdown fence。若产品允许 fence，应在任务规范中明示，并由另一个有版本的 normalizer 处理；不能在看到模型输出后临时放宽规则。

## 复杂度轴

构造评测时逐轴增加难度：

1. 单约束到多约束；
2. 同类约束到异构约束；
3. 无冲突到显式优先级；
4. 短提示到长上下文；
5. 单轮到后续修订；
6. 纯文本到工具与文档中的不可信内容；
7. 正常完成到应该澄清、拒绝或停止。

不同轴不要一次全部改变，否则失败难以归因。

除了逐轴增加难度，还应做最小对照：保持任务内容不变，只改约束的位置、措辞、顺序或无关上下文。若答案随不相关的段落顺序大幅波动，测到的可能是位置敏感性而不是约束理解。

## 从训练到运行时闭环

### 训练数据

- 覆盖组合约束、否定约束和条件分支；
- 加入“部分满足但整体失败”的 hard negatives；
- 让同一语义使用多种表达，降低模板依赖；
- 训练模型显式识别冲突与无法同时满足；
- 对工具动作加入作用域和副作用标签。

### 推理流程

1. 提取全部约束；
2. 解析优先级和作用域；
3. 检测冲突与缺失信息；
4. 制定满足约束的输出结构；
5. 生成内容；
6. 用 verifier 做最终检查。

对于程序化输出，让 parser 验证比让模型口头声明“格式正确”更可靠。

### 生成约束

JSON grammar、正则、有限状态机和 schema-constrained decoding 能保证部分语法，但不能保证字段内容正确。约束解码也可能提高延迟，或在 schema 本身错误时稳定地产生错误结果。

训练、解码与 verifier 处在不同层：

1. 训练提高模型理解约束和处理冲突的概率；
2. constrained decoding 收窄可生成的语法集合；
3. verifier 判定结果是否满足任务；
4. runtime 决定失败后重试、修复、澄清还是停止。

把四层压成一次“让模型自检”，无法区分生成失败、parser 失败和任务本身不可满足。工具调用的 schema、dispatch 与副作用边界见[工具调用](../applications/tool-use.md)和 [Agent Runtime](../applications/agent-runtime.md)。

## 当指令藏在不可信内容里

网页、文档、代码注释与工具输出属于待处理数据，不应自动拥有控制 agent 的权限。系统应分别标记：

- trusted policy；
- current task instructions；
- repository rules；
- untrusted content；
- tool observations。

如果层级只依靠自然语言提示，长轨迹中容易发生权限漂移。工具层还应使用 sandbox、allowlist 和写入确认形成外部约束。

评测 prompt injection 时，攻击文本和任务内容必须成对出现：一份含攻击、一份不含攻击，并固定检索、工具和权限。仅统计模型是否复述攻击文本不够；真正的失败是高优先级任务被改变、秘密被读取或发生未授权动作。安全边界见[智能体安全](../applications/agent-security.md)。

## 一张能指导修复的报告

- strict / loose constraint accuracy；
- 冲突识别与正确澄清率；
- 过度拒绝率；
- schema parse success 与语义正确率；
- 未授权动作率；
- 多轮约束保持率；
- 注入攻击下的任务保持率。

每个聚合数字应保留 constraint type、数量、作用域、冲突状态、输出长度、是否调用工具和 verifier 版本等切片。否则 strict accuracy 下降时，无法判断是 JSON parser、长上下文遗忘还是权限问题。幻觉与证据见[幻觉与事实性](hallucination.md)，统计区间见[统计推断](statistical-inference.md)，工具权限见 [Agentic RL 评测与安全](../agentic-rl/evaluation-safety.md)。

## Reference {#reference}

- [Instruction-Following Evaluation for Large Language Models / IFEval](https://arxiv.org/abs/2311.07911)
- [FollowBench](https://arxiv.org/abs/2310.20410)
- [Google Research IFEval implementation](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
- [FollowBench official implementation](https://github.com/YJiangcm/FollowBench)
