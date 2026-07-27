# Verifier、过程奖励与 Reward Shaping

Verifier 判断候选或状态，reward shaping 决定怎样把判断变成时间序列上的学习信号。两者不能合并：一个准确的终局 verifier 可能仍然只提供稀疏 reward；一个密集过程分数也可能改变最优策略。

本页负责 verifier taxonomy、过程/终局信号的 shaping 语义及攻击面，不重复定义完整 RLVR 训练闭环。任务采样、reward → advantage、policy update 与在线分布演化见[RLVR](rlvr.md)。

## 四类信号角色

| 对象 | 输入 | 输出 | 典型用途 |
| --- | --- | --- | --- |
| Outcome verifier | 完整答案 / 终态 | success / score | RLVR、最终评测 |
| Process verifier | 中间步骤 / prefix | 局部正确性 | 搜索、过程监督 |
| Value model | 当前 history | 未来 return 期望 | critic、搜索排序 |
| Reward model | response / trajectory | 偏好效用代理 | RLHF、reranking |

同一个模型可以被训练成不同角色，但目标和校准不能共享名字。PRM 的 step score 不自动等于 $V^\pi$；reward model 的 response score 也不自动是环境 return。

## 稀疏 outcome reward

若只有终局 reward $R_T$：

$$
R_t=0\quad(t<T),\qquad R_T\in\{0,1\}.
$$

Monte Carlo estimator 无偏但方差高；critic 和 GAE 可以缩短有效路径，却依赖 value 泛化。增加过程 reward 是另一种选择，但它直接改变优化目标。

## Potential-based shaping

经典结果表明，若 shaping 项为

$$
F(s,a,s')
=\gamma\Phi(s')-\Phi(s),
$$

则在标准 MDP 条件下可保持最优策略不变。沿轨迹求和：

$$
\sum_{t=0}^{T-1}\gamma^tF_t
=-\Phi(s_0)+\gamma^T\Phi(s_T),
$$

中间项望远镜抵消。若终态 potential 固定，策略比较只差常数。

一般 learned progress score、步骤正确率或“思考质量”不满足这个形式。把它们直接逐步相加，可能奖励冗长、循环或局部最优。

## Process reward 的三种用法

### 作为搜索启发式

只影响候选扩展和选择，不直接进入训练 return。最终仍由 outcome verifier 决定。

### 作为训练 target

用 step label 训练 PRM/value，再由 policy optimization 间接使用。此时误差通过 critic/search 传播。

### 作为 dense reward

直接定义

$$
\widetilde R_t
=R_t^{\mathrm{outcome}}
+\alpha R_t^{\mathrm{process}}.
$$

这是最强干预：$\alpha$、step 数量和 score calibration 都会改变最优行为。

## Prefix-score 增量与累计 reward

若 verifier 对 prefix 给累计分 $q_t$，直接把每个 $q_t$ 相加会重复奖励早期进展。更自然的增量候选是

$$
R_t^{\mathrm{increment}}
=q_t-q_{t-1}.
$$

但只有当 $q_t$ 真能作为同一量的累计势函数时，这种 prefix-score increment 才有清楚语义。生成式 judge 在不同 prefix 上的 score 未必可比较，差分可能放大噪声。这里不用 “difference reward” 一词：该术语在 multi-agent RL 中通常指用全局 reward 与移除某个 agent 贡献后的 counterfactual reward 之差，概念不同。

## Credit granularity

过程标签可能对应：

- token；
- reasoning span；
- natural-language step；
- tool call；
- environment transition；
- turn。

训练 mask 必须与标注粒度一致。把 turn-level score 复制到每个 token 后求和，会让长 turn 获得更大总权重；先做 token mean 则让每个 turn 等权。两者应由 estimand 决定。

## Verifier 的攻击面

### Specification gaming

policy 满足评分器表面规则，不满足真实目标。

### Parser gaming

利用格式、Unicode、浮点、异常或截断路径取得错误分数。

### Test gaming

记忆 visible tests、读取隐藏资产、修改测试或针对有限覆盖 hard-code。

### Judge gaming

使用冗长、自信、引用外观或特定短语影响生成式评审。

### State gaming

Agent 直接篡改目标状态、reward 文件或环境时钟。

因此 verifier 需要最小权限、输入隔离、不可变版本、隐藏测试和 evidence log。

## Reward ensemble

多个 verifier 可做保守组合：

$$
R=
\begin{cases}
R_{\mathrm{task}},&\text{all hard constraints pass},\\
R_{\mathrm{fail}},&\text{otherwise}.
\end{cases}
$$

或学习加权和。硬门禁适合不可补偿约束；加权和适合可交易目标。ensemble 一致不代表正确，如果组件共享数据、模型或 parser，它们会共同失败。

## Outcome 与 process 的联合评测

构造矩阵：

| | Outcome 成功 | Outcome 失败 |
| --- | --- | --- |
| Process 高 | 理想路径或幸运成功 | 局部合理但整体失败 / verifier 漏洞 |
| Process 低 | 非典型正确路径 / false negative | 普通失败 |

按任务类型、长度和搜索预算审计四象限。只在已有 policy 的分布上评估 PRM，不足以预测被优化后的 reward hacking。

## 与搜索的闭环

搜索使用 verifier 后，会改变训练数据：

```text
policy proposes
  -> verifier scores
  -> search selects
  -> selected / rejected paths become data
  -> policy changes
  -> verifier sees a new distribution
```

因此 verifier 需要周期性 OOD 审计。若 verifier 与 policy 同源或共同训练，错误相关性更高。

## 最小实验

1. 对已知 potential 验证 shaping 前后最优策略一致。
2. 比较累计 score 与 prefix-score increment，检查重复计费。
3. 固定 outcome reward，只改变 process coefficient 做 sweep。
4. 按 step 数报告总 process reward，检查长度 hacking。
5. 用独立 verifier 复核高训练 reward 失败样本。
6. parser fuzz：空输入、超长、Unicode、NaN、重复答案与异常退出。
7. 禁止 verifier 访问 policy 私有状态或训练标签。
8. 保存每个 reward 分量与 evidence，不只保存总分。

[RLVR](rlvr.md)负责可验证反馈进入 rollout → reward → advantage → policy update 的端到端范式，[语言模型信用分配](credit-assignment.md)关注 reward 怎样传播，[搜索与验证](../reasoning/search-verification.md)关注推理时算法。

## Reference {#reference}

- Ng, Harada, and Russell, [Policy Invariance under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- Christiano et al., [Deep Reinforcement Learning from Human Preferences](https://proceedings.neurips.cc/paper/2017/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html)
- Lightman et al., [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)
- Gao et al., [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
- Wolpert, Wheeler, and Tumer, [General Principles of Learning-Based Multi-Agent Systems](https://arxiv.org/abs/cs/9905005)
