# 搜索与验证

搜索把一次生成改写为状态空间探索。设状态 $s$ 表示当前问题、已生成前缀、工具结果和环境信息，动作 $a$ 是下一步推理或操作：

$$
s_{t+1}=F(s_t,a_t),
\qquad
a_t\sim\pi_\theta(a\mid s_t).
$$

搜索算法决定扩展哪些状态，verifier 决定哪些状态值得保留。若状态定义、可执行反馈或评分器不可靠，扩大搜索只会更系统地放大错误。

## Outcome 与 process verifier

### Outcome verifier

只对最终答案评分：

$$
v_{\mathrm{out}}(x,y)\in\mathbb R.
$$

优点是标注和执行简单，适合有单元测试、规则或确定答案的任务；缺点是无法定位中间错误，稀疏反馈也难指导早期剪枝。

### Process verifier

对中间状态或步骤评分：

$$
v_{\mathrm{proc}}(x,s_t)
\approx
P(\text{该前缀仍可导向正确答案}\mid x,s_t).
$$

[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)系统比较了结果监督与过程监督。过程分数适合搜索，但“当前步骤看似合理”不等于最终可完成；局部 verifier 也可能偏好冗长、模板化轨迹。

## Beam search

维护宽度为 $K$ 的候选集合：

$$
\mathcal B_{t+1}
=
\operatorname{TopK}
\left\{
\operatorname{score}(s\oplus a):
s\in\mathcal B_t,
a\in A(s)
\right\}.
$$

常见组合分数：

$$
\operatorname{score}(s)
=
\lambda\log\pi_\theta(s\mid x)
+
(1-\lambda)v(s)
-
\mu\,\operatorname{cost}(s).
$$

只累积 log-probability 会偏向模型熟悉的轨迹；只看 verifier 会产生 Goodhart。长度归一、重复惩罚和 hard constraint 必须在搜索前定义。

## Tree of Thoughts

[Tree of Thoughts](https://arxiv.org/abs/2305.10601)把可读的中间“thought”作为搜索节点，并通过生成、评价、选择和回溯探索多条路径。核心不是固定一种 BFS/DFS，而是：

1. 状态足以描述剩余子问题；
2. action 粒度允许修正；
3. evaluator 能在最终答案前提供信息；
4. 搜索预算能被严格控制。

若 thought 只是任意长度自然语言，等价状态难以去重，评分也容易受文风影响。可执行 DSL、程序状态或结构化草稿通常更容易验证。

## Monte Carlo Tree Search

节点 $s$ 的 PUCT 选择可写为

$$
a^\star
=
\arg\max_a
\left[
Q(s,a)
+
c_{\mathrm{puct}}
P(s,a)
\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}
\right].
$$

一次迭代包括 selection、expansion、evaluation/rollout 与 backup。LLM 场景需要额外定义：

- action 是 token、步骤、工具调用还是完整候选；
- 相同语义但不同文本是否合并；
- terminal 条件；
- value 来自 verifier、执行器还是 rollout；
- 随机环境和工具副作用是否允许重放。

## 可执行验证

确定性 checker 通常比学习式 verifier 更可靠：

- 代码编译与单元测试；
- 数学表达式代入或符号化验证；
- JSON Schema、语法和类型检查；
- 数据库只读查询与约束；
- 模拟器中的合法动作和终态。

但 checker 只验证其规格覆盖的性质。测试通过不等于需求完整，数值样例通过不等于证明正确，沙箱成功也不等于真实环境安全。

## Verifier 校准

设 verifier 输出 $\hat p$，应检查

$$
P(Y=1\mid \hat p\in[b,b+\Delta])
\approx
\mathbb E[\hat p\mid \hat p\in[b,b+\Delta]].
$$

除 AUROC 外，还应报告：

- top-1 selection accuracy；
- 含正确候选时的选择准确率；
- false-positive 高分尾部；
- 按题型、长度、格式和生成器切片的校准；
- 分布外生成器或新任务上的退化。

Verifier 在同一生成器数据上训练和测试，容易只学习其错误风格。跨生成器、跨模型和时间切分更能检验泛化。

## 搜索预算

树搜索成本近似为

$$
C
=
\sum_{s\in\mathcal E}
\left(
C_{\mathrm{expand}}(s)
+
C_{\mathrm{verify}}(s)
+
C_{\mathrm{tool}}(s)
\right),
$$

其中 $\mathcal E$ 是实际扩展节点。需要同时限制：

- 最大深度、宽度和节点数；
- 总生成 token；
- verifier batch 与调用次数；
- 工具时间、费用和副作用；
- 单题 wall-clock 与全局并发。

“搜索深度”若每层 action 长度不同，不能直接代表计算量。

## 实现契约

1. 状态使用稳定 ID，父子关系可重建；
2. score 分解保存 policy、verifier、cost 各分量；
3. terminal、invalid、timeout、pruned 状态分开；
4. 去重基于规范化状态而非原始自然语言；
5. 工具调用具有幂等键或只读隔离；
6. 被剪枝分支释放 KV 与外部资源；
7. backup 规则与 value 方向一致；
8. 停止条件不读取隐藏真值；
9. 搜索日志可重放但不保存不必要的敏感输入。

## 失效模式

- **Verifier hacking**：轨迹优化了评分器表面特征。
- **搜索坍缩**：不同分支共享同一高概率错误。
- **局部最优**：过程分数剪掉暂时低分但可完成的路径。
- **状态爆炸**：自然语言轻微差异阻止去重。
- **循环**：反思和工具调用重复同一状态。
- **错误 backup**：min/max、折扣或终态符号写反。
- **工具副作用**：探索分支改变真实环境。
- **隐藏 oracle**：调参或停止使用了参考答案信息。
- **成本失真**：节点数相同但 token 与工具成本相差巨大。

## 验证矩阵

| 目标 | 对照 |
| --- | --- |
| 搜索是否有益 | 同 token 预算的单轨迹、多采样、beam、树搜索 |
| Verifier 是否有益 | 随机、policy score、oracle、学习式 verifier |
| 剪枝是否安全 | 保留率与正确路径被剪比例 |
| 去重是否正确 | 同义状态、等价程序状态、不同关键变量 |
| Backup 是否正确 | 小型确定树的手算结果 |
| 工具是否安全 | sandbox、超时、取消、重放与幂等 |
| 泛化 | 新生成器、新题型、分布外长度 |
| 成本 | tokens、节点、verifier、工具、平均与尾延迟 |

本节的边界与阅读顺序见[推理与推理时计算总览](index.md)。搜索轨迹怎样用于策略与奖励训练，见[搜索、过程奖励与验证](../agentic-rl/search-verification.md)；候选生成和自适应停止见[推理时计算](test-time-compute.md)；最小 beam、PUCT 与校准实验见[测试时计算手撕实现](../practice/test-time-compute.md)。

## Reference {#reference}

- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
