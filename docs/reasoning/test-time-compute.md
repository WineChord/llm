# 推理时计算

推理时计算（test-time compute）是在参数固定时，为当前问题增加采样、反思、验证、搜索或工具执行预算。目标不是最大化生成 token，而是在给定成本下最大化正确完成概率：

$$
\max_{\pi}
\mathbb E[\operatorname{utility}(y)]
\quad
\text{s.t.}\quad
\mathbb E[\operatorname{cost}(\pi,x)]\le B.
$$

## 单轨迹与多轨迹

[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)展示了显式中间步骤可以改善部分算术、符号和常识推理任务。它改变的是生成轨迹；轨迹更长本身不保证更正确。

增加计算有两种基本方向：

- **纵向**：让单条轨迹更长，允许修正、工具调用或分层求解；
- **横向**：生成多条不同轨迹，再聚合或选择。

纵向依赖模型能否利用额外步骤，横向依赖样本多样性和选择器。两者都可能在错误模式高度相关时迅速饱和。

## Self-consistency

[Self-Consistency](https://arxiv.org/abs/2203.11171)从多个随机推理轨迹提取最终答案并投票：

$$
a^\star
=
\arg\max_a
\sum_{i=1}^{N}
\mathbf 1[
\operatorname{extract}(y_i)=a
].
$$

关键不在多数表决本身，而在两个接口：

1. 采样应产生有意义的解法多样性；
2. `extract` 应把数学等价、格式差异和单位表达归一为同一答案。

若答案是自由文本，可先聚类语义等价候选，再按簇大小与可信度选择；直接比较原始字符串会严重低估一致性。

## Best-of-$N$

给定 verifier $v(x,y)$：

$$
y^\star
=
\arg\max_{1\le i\le N}
v(x,y_i).
$$

若单样本独立成功率为 $p$，理想 oracle 选择器的覆盖率为

$$
P(\text{至少一次成功})
=
1-(1-p)^N.
$$

这只是上界直觉。实际候选通常相关，且 verifier 会产生 false positive/false negative。真实 selected accuracy 由候选覆盖率与选择准确率共同决定。

## Pass@$k$ 与选择准确率

代码或可自动判定任务常报告 pass@$k$。若从 $n$ 个样本中有 $c$ 个正确，均匀无放回抽取 $k$ 个至少一个正确的估计为

$$
\operatorname{pass@}k
=
1-
\frac{\binom{n-c}{k}}{\binom nk}.
$$

pass@$k$ 衡量候选集合覆盖，不代表系统能从中选出正确答案。还需报告：

$$
\operatorname{selection\ accuracy}
=
P(y^\star\text{ correct}\mid
\mathcal Y\text{ contains a correct candidate}).
$$

## 自适应预算

固定给每个问题 $N$ 个样本会浪费容易样本，并低估困难样本。设第 $i$ 个问题分配预算 $B_i$，成功概率为 $P_i(B_i)$：

$$
\max_{\{B_i\}}
\sum_iP_i(B_i)
\quad
\text{s.t.}\quad
\sum_iB_i\le B.
$$

理想策略按边际收益

$$
\Delta_i(b)
=
P_i(b+\delta)-P_i(b)
$$

分配下一单位预算。实际系统可用以下信号近似困难度或剩余收益：

- 候选答案是否已经稳定一致；
- verifier margin 是否足够大；
- 轨迹间是否出现新策略而非措辞改写；
- 工具执行是否提供了可判定反馈；
- 已用 token、时间和错误次数；
- 基础模型对任务类型的历史校准。

## 停止规则

常见停止条件包括：

- 同一答案达到阈值票数；
- top-1 与 top-2 verifier score 的 margin 足够大；
- 连续若干扩展没有产生新答案或更高分；
- 可执行检查通过；
- 达到 token、时间、费用或工具调用上限；
- 风险策略要求人工确认。

停止规则必须在评测前固定。观察到正确答案后再决定停止会引入 oracle 信息。

## 生成多样性

横向扩展需要区分“文本多样”与“解法多样”。可控制：

- temperature 与 top-$p$；
- prompt 中要求不同分解或不同工具；
- 不同草稿模型或角色；
- 对已探索前缀施加去重；
- 在搜索树的不同状态继续展开。

温度过低会重复同一错误；过高会增加无效或不可解析候选。应以正确候选覆盖率和错误相关性选择采样参数，而不是最大化表面差异。

## 成本模型

对并行采样，token 成本近似

$$
C_{\text{token}}
=
\sum_{i=1}^{N}
\left(
T_{\text{prefill},i}
+
T_{\text{decode},i}
\right).
$$

共享 prefix cache 可以减少重复 prefill 计算，但不会减少所有 decode 成本。并行执行降低 wall-clock，不降低总 accelerator time；串行自适应可以早停，却增加尾延迟。

比较方案时至少报告：

- input、visible output 与隐藏 reasoning token；
- prefix cache 是否复用；
- 并行宽度与硬件数量；
- verifier/工具额外成本；
- 平均、P95 和 P99 最终答案延迟。

## 实现契约

1. 候选 ID、随机种子和 sampling 参数可追踪；
2. answer extractor 在生成前冻结版本；
3. 等价答案归一规则覆盖单位、浮点、集合和代码；
4. verifier 不读取参考答案或其他候选的真值；
5. budget 同时约束 token、时间、费用和调用次数；
6. 取消的分支停止生成并释放 cache；
7. 并发失败和超时不会被静默当作错误答案；
8. 评测按问题聚合，不能把同题多个样本当独立测试样本。

## 失效模式

- **候选相关**：$N$ 增大但只是重复同一错误。
- **Verifier Goodhart**：生成器学会提高评分而非正确性。
- **长度偏置**：长答案因形式完整获得更高分。
- **答案解析错误**：等价答案被拆分，错误字符串被误归一。
- **预算反分配**：难题耗尽预算，简单题仍固定过度采样。
- **延迟隐藏**：只报告并行 wall-clock，不报告总计算。
- **测试泄漏**：verifier 或 prompt 包含 benchmark 特征。
- **不单调**：更长轨迹产生自我干扰或错误修正。

## 验证矩阵

| 维度 | 对照 |
| --- | --- |
| 候选覆盖 | pass@1、pass@$k$、unique answer 数 |
| 选择能力 | oracle、随机选择、verifier、majority |
| 相关性 | 同种采样与多策略采样 |
| 预算 | 固定 $N$ 与自适应早停 |
| 成本 | 相同 token、时间、费用预算 |
| 稳健 | extractor 扰动、超时、部分工具失败 |
| 校准 | verifier score 分桶后的真实正确率 |
| 泛化 | 不同任务、难度与分布外切片 |

[Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)给出了按问题难度选择计算策略的实证框架；[DeepSeek-R1 深读](../landscape/works/deepseek-r1.md)把可验证奖励、在线 RL、多阶段数据与蒸馏的公开证据拆开。对开放和闭源系统，都应把可观察的预算—质量曲线与未披露训练机制分开。

本节的边界与阅读顺序见[推理与推理时计算总览](index.md)，搜索与 verifier 的计算图见[搜索与验证](search-verification.md)，底层采样见[解码](../inference/decoding.md)，紧凑实现与测试见[测试时计算手撕实现](../practice/test-time-compute.md)。
