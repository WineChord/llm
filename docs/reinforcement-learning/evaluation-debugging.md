# 强化学习实验与诊断

强化学习曲线把数据分布、reward、策略、critic、采样系统和评测器压进一个数。训练 reward 上升可能来自真实能力，也可能来自长度漂移、verifier 漏洞、policy lag 变化或分母改变。诊断的目标是把这些机制重新拆开。

## 先定义实验单位

至少同时记录四类预算：

| 预算 | 说明 |
| --- | --- |
| Environment episodes | 完整任务或交互次数 |
| Rollout tokens | 采样 policy 实际生成量 |
| Train tokens | 进入反向传播的有效 action token |
| Wall-clock / compute | GPU-hours、并行度、失败重跑与 verifier 成本 |

只固定 optimizer step 不公平：不同方法的 group size、过滤率、response length 和 epoch 数可能完全不同。

## 数据流漏斗

从任务到梯度逐层记分母：

```text
sampled tasks
  -> started episodes
  -> completed / truncated / failed
  -> reward available
  -> valid trajectories
  -> accepted by freshness / ratio gates
  -> effective action tokens
  -> optimizer updates
```

每层都报告数量和原因。否则 reward 上升可能只是丢掉了更难或更长的轨迹。

## Policy 指标

### Ratio 与 KL

监控

$$
\rho_t
=\exp\left(
\log\pi_\theta(a_t\mid h_t)
-\log\mu(a_t\mid h_t)
\right)
$$

的中位数、分位数、极值和按长度/lag 的分层，而不只看 mean。同步 rollout 在更新前应满足 $\rho\approx1$；明显偏离通常意味着 token、模板、mask 或 behavior log-prob 错位。

KL 指标必须标明方向、采样分布和 reduction。全局 token mean 会掩盖少数 prompt 的大漂移。

### Entropy 与多样性

policy entropy 下降可能是学习，也可能是模式坍缩。配合观察：

- unique response / action pattern；
- 同 prompt reward 方差；
- 成功路径的结构多样性；
- 长度和终止方式；
- pass@$k$ 与 pass$^k$。

## Critic 指标

value loss 低不等于 advantage 可靠。至少检查：

### Explained variance

$$
\operatorname{EV}
=1-\frac{\operatorname{Var}(G-V)}
{\operatorname{Var}(G)}.
$$

当 return 几乎常数时，分母很小，EV 不稳定；应同时报告 return variance 和样本数。

### Calibration 与分层残差

按 horizon、任务难度、终止类型、reward 桶和 policy version 比较 $V$ 与 realized return。一个只拟合短任务的 critic 可能在全局 MSE 上很好，却系统性低估长任务。

### Actor–critic 速度

监控 critic update 次数、learning rate、gradient norm、target freshness 与 actor KL。actor 变化远快于 critic 时，advantage 会持续基于过时状态价值。

## Reward 指标

将 reward 拆成原始分量：

```text
task success
format / parser
process score
cost and length
safety or permission
KL / regularization
```

总 reward 上升时，逐项检查贡献。再建立 reward 与独立 outcome 的四象限：

| | 独立 outcome 成功 | 独立 outcome 失败 |
| --- | --- | --- |
| 高训练 reward | 目标一致 | reward hacking 候选 |
| 低训练 reward | reward false negative | 普通失败 |

最值得人工审计的是右上和左下，而不是随机看平均样本。

## 终止与故障

至少分开：

- `success`
- `task_failure`
- `invalid_action`
- `timeout`
- `truncated_budget`
- `environment_error`
- `verifier_error`
- `infrastructure_error`

把所有非成功都记成 reward $0$，会让系统可用性问题进入 policy gradient。环境和 verifier 故障率本身应成为系统 SLO，不应由模型“学习规避”。

## 组相对方法

RLOO、GRPO 等按 prompt 成组时，报告：

- group size 的实际分布；
- all-correct、all-wrong 与 mixed group 比例；
- reward std 为零的组；
- 因无信号而重采样的 token 成本；
- 每个 prompt 对梯度的最终权重；
- response length 与组优势的相关性。

动态过滤可以提高有效梯度比例，也会改变任务分布。比较方法时应保留过滤前后的题目难度与总采样预算。

## 异步训练

按 policy lag 分桶统计：

| 指标 | 作用 |
| --- | --- |
| weight version distance | 粗略陈旧度 |
| token log-ratio | 实际动作分布差异 |
| accepted / clipped / dropped fraction | 校正强度 |
| queue wait 与 rollout duration | staleness 来源 |
| learner idle / rollout idle | 资源失衡 |

版本差不能替代 probability ratio；ratio 也不能完全修正状态访问分布变化。过旧轨迹可能需要直接丢弃。

## 公平比较

一次算法比较至少固定或显式报告：

1. base/SFT checkpoint；
2. prompt 与环境数据；
3. verifier、reward 和 parser；
4. 生成 temperature、top-$p$、最大长度；
5. rollout token 与 train token；
6. group size、过滤和 replay；
7. optimizer update 与 epoch；
8. 超参数搜索预算；
9. 独立评测协议与重复次数；
10. 硬件、并行度和 wall-clock。

PPO 与 GRPO 的显存不同、RLOO 与 PPO 的 rollout 数不同、异步系统的利用率不同；只固定 step 或 batch size 会把资源差异误当成算法差异。

## 最小回归测试

### 数学

下面既有逐样本断言，也有只在期望上成立的性质；测试时不能混为一谈。

- 新旧 policy 相同时 ratio 为 $1$；
- 全零 advantage 时 policy gradient 为零；
- 在小离散策略上枚举全部 action，或用大量独立 Monte Carlo sample 估计 expectation：加入不依赖 action 的常数 baseline 后，期望 policy gradient 不变；有限 minibatch 的两次样本估计不要求逐项相等；
- terminal 不 bootstrap，truncated 按契约处理；
- 全同 group reward 产生零相对优势；
- 空 action mask 明确报错。

### 数据

- 保存 token ID 与重算 token ID 一致；
- prompt、observation、padding 不进入 action loss；
- reward、policy、tokenizer、environment 与 verifier version 完整；
- resume 前后 data cursor 和 policy version 连续。

### 端到端

- 随机 policy / SFT / rejection sampling 是受控基线；
- hidden verifier 不参与训练；
- 失败轨迹保留并分类型；
- 训练 reward 与最终 outcome 同时画曲线；
- 多 seed 或重复 rollout 给出不确定性。

可执行断言见[手撕强化学习](../practice/reinforcement-learning.md)、[训练目标](../practice/training-objectives.md)与[评测工具](../practice/evaluation-tooling.md)。多轮系统指标继续见 [Agentic RL 评测与安全](../agentic-rl/evaluation-safety.md)。

## Reference {#reference}

- Henderson et al., [Deep Reinforcement Learning that Matters](https://ojs.aaai.org/index.php/AAAI/article/view/11694)
- Agarwal et al., [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)
- Engstrom et al., [Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
