# 语言模型中的信用分配

信用分配问的是：一个迟到的结果，应当怎样改变更早的决策。语言模型把时间轴拉得很长——token 构成 response，response 构成 turn，turn 又构成 episode；reward 却可能只在最终测试通过时出现。

## 四层时间轴

设 episode 含 $K$ 个 turn，第 $k$ 个 turn 含 $L_k$ 个 action token：

$$
\tau=
\left(
a_{1,1:L_1},o_1,
a_{2,1:L_2},o_2,\ldots,
a_{K,1:L_K},o_K
\right).
$$

可以在四种粒度估计 advantage：

| 粒度 | 一个 advantage 覆盖 | 主要优点 | 主要风险 |
| --- | --- | --- | --- |
| Token | 一个采样 token | 最细 | critic 与 reward 极难精确 |
| Span / turn | 一段 reasoning 或 tool call | 与环境动作接近 | 边界需要定义 |
| Response | 单轮完整回答 | 适合单轮 RLHF | 多轮中把过程压平 |
| Episode | 完整任务 | 终态最可信 | 方差大、早期信用模糊 |

粒度越细，不代表 bias 越小。一个不可靠的 step reward 会把错误监督传播到更多位置。

## Monte Carlo：终局 reward 直接回传

若只有终局 reward $R(\tau)$，最简单 estimator 是

$$
\widehat A_t=R(\tau)-b(h_t).
$$

没有 critic 时，action-independent baseline 可取不含当前样本的 leave-one-out prompt group mean、与当前采样动作独立的 greedy response reward，或其他仅依赖 $h_t$ 的统计量。普通 batch/group mean 若包含当前 reward，就依赖当前 action，不能直接套用 baseline invariance；固定 group size 下，self-including mean 是 RLOO estimator 的固定缩放，而再除以随机 group std 还会改变不同 prompt 的权重。细节见[无 learned critic 的策略梯度](critic-free-baselines.md)。

将同一 $\widehat A$ 复制到每个 token，再做 token sum，得到合法的 sequence policy gradient。若改成 token mean，则每条 response 的梯度除以长度，优化目标已经加入长度重权，不是无害的数值稳定化。

## Reward-to-go 与 causality

当中间 reward 可用时，动作 $a_t$ 不应为此前 reward 负责：

$$
G_t=\sum_{k=t}^{T}\gamma^{k-t}R_k.
$$

reward-to-go 利用因果性降低方差。语言任务里应先定义 reward 落点：

- 工具调用成本属于发起动作；
- 测试结果属于产生代码的 turn 或终态；
- 格式错误可在 parser 决定时产生；
- 基础设施失败不应伪装成策略 reward。

## TD、GAE 与 critic

critic 用当前 history 估计未来回报：

$$
V_\phi(h_t)\approx
\mathbb E_\pi[G_t\mid h_t].
$$

TD residual 为

$$
\delta_t
=R_t+\gamma m_tV_\phi(h_{t+1})-V_\phi(h_t),
\qquad
m_t=1-d_t,
$$

其中 $d_t=1$ 表示该 transition 后真正 terminal。GAE 的实现递推为

$$
\widehat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=\delta_t+\gamma\lambda m_t
\widehat A_{t+1}^{\mathrm{GAE}(\gamma,\lambda)}.
$$

展开式中的每一项都带有沿途 mask 的乘积，因此 trace 不会跨越 episode terminal。time-limit truncation 是否令 $m_t=0$ 取决于环境契约：若后继状态仍有定义，通常应 bootstrap，而不是把 truncation 冒充 terminal。$\lambda\to1$ 接近 Monte Carlo，偏差较小而方差较大；$\lambda\to0$ 更依赖 critic。长序列中 $\gamma\lambda$ 的指数衰减会弱化远端信号，固定超参数未必适合跨度差异巨大的 response。

完整推导见[多步回报与 GAE](multistep-traces.md)，critic 的训练耦合见 [Actor–Critic](actor-critic.md)。

## Observation 不等于 action

工具返回的 observation 会改变下一动作条件，却不是 policy 采样结果。若轨迹为

$$
[a_0,o_0,a_1,o_1,\ldots],
$$

policy loss 只覆盖 $a_i$。value 可以在包含 $o_i$ 的 history 上预测，但不能因为 observation 有很多 token，就把折扣按 observation token 数机械延长。

[SAO](../landscape/works/sao-compactionrl.md#sao) 的 Skip-Observation GAE 直接在 action token 链之间递推；这是特定时间尺度约定。其他系统可能在 turn 级定义一步。两者都应明确 $\gamma$ 对应 token、action span 还是环境 turn。

## Outcome reward 与 process reward

### Outcome reward

只判断最终答案、测试或环境状态。优点是目标清楚、可复现；缺点是稀疏且难定位错误。

### Process reward

给中间步骤、子目标或工具使用反馈。它缩短信号路径，却引入新的代理目标：

- 合法形式不等于有效推理；
- verbose reasoning 可能获得更多正分；
- 局部正确步骤可能通向错误全局计划；
- learned PRM 可被策略主动寻找漏洞。

若 shaping 满足

$$
F(s,a,s')
=\gamma\Phi(s')-\Phi(s),
$$

在标准条件下可保持最优策略不变；一般 learned process score 没有这项保证。

## 从搜索中产生信用

搜索可以提供比终局单样本更丰富的比较：

- sibling 候选间相对价值；
- verifier 对中间状态的估计；
- 成功路径与失败分叉；
- tree backup 得到的 state value。

但搜索策略、扩展预算和 verifier 共同决定数据分布。把搜索选中的最佳轨迹做 SFT，是 imitation of search；用节点 value 更新 policy，则更接近 policy improvement。二者不能只按“用了 tree search”归为同一算法。

详见[搜索、过程奖励与验证](../agentic-rl/search-verification.md)。

## 上下文压缩后的跨段信用

一次长任务被摘要拆成 $\sigma_1,\ldots,\sigma_K$ 后，每段单独计算 local return 会让早期段看起来离终局过近。需要记录：

- 所属原始 trajectory；
- segment 顺序；
- 后续 action token 数或 turn 数；
- summary 是否由 policy 采样；
- reward 是否在各段重复，还是只保留一份。

[CompactionRL](../landscape/works/sao-compactionrl.md#compactionrl) 按后续优化 token 数对 local advantage 追加折扣。它恢复终局 reward 的 token-distance，但仍不是完整跨边界 GAE：后续每个 TD residual 并未全部回传。

## 层级信用

长任务可把高层 option $z_k$ 与低层 action 分开：

$$
z_k\sim\pi_H(\cdot\mid h_{t_k}),
\qquad
a_t\sim\pi_L(\cdot\mid h_t,z_k).
$$

高层 policy 对子目标选择和终止负责，低层 policy 对执行负责。层级分解缩短信号路径，却新增：

- option 边界是谁决定；
- 子目标完成由谁验证；
- 高低层 reward 是否一致；
- 失败恢复是否继续同一 option；
- 两层更新是否 on-policy。

在没有可靠子目标标注时，层级结构可能只把信用问题换了名字。

## 诊断

1. 用两三步轨迹手算 return、TD residual、GAE 与 bootstrap。
2. 分开 terminal、truncated、invalid 和 infrastructure error。
3. 将 observation 内容扰动；若 action mask 不变，policy target 数量应不变。
4. 按 token、turn、episode 分别汇总 advantage mean/std。
5. 检查早期 action 的 advantage 是否随 horizon 指数消失。
6. process reward 关闭后，用相同 rollout 预算重跑 outcome baseline。
7. 按 verifier 置信度和最终正确性构造四象限，审计高分错误轨迹。
8. 对压缩或分段轨迹检查 reward 是否重复计数。

对应的最小实现见[手撕强化学习](../practice/reinforcement-learning.md)和 [Packed trajectory GAE](../practice/llm-policy-optimization.md#gae)。

## Reference {#reference}

- Williams, [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696)
- Sutton et al., [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)
- Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- Lightman et al., [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)
- Ng, Harada, and Russell, [Policy Invariance under Reward Transformations](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- Hou et al., [Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508)
- Li et al., [CompactionRL: Reinforcement Learning with Context Compaction for Long-Horizon Agents](https://arxiv.org/abs/2607.05378)
