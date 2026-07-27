# RLVR：可验证奖励怎样改变语言模型训练

Reinforcement Learning with Verifiable Rewards 使用程序、形式系统或环境终态自动判断结果。它解决的是 reward 扩展性，不指定 policy optimizer：PPO、RLOO、GRPO 或其他策略梯度都可以优化可验证 reward。

本页负责 RLVR 的端到端训练范式：任务如何进入 rollout、原始验证结果怎样映射为 reward、怎样形成 advantage、何时构成在线闭环，以及怎样评测 policy improvement。Verifier 的详细 taxonomy、process shaping 与攻击面由[Verifier、过程奖励与 Reward Shaping](verifiers-reward-shaping.md)维护，避免两页重复给出不同定义。

## 端到端闭环

```text
task distribution
  -> evolving policy rollout
  -> verifier status / score / evidence
  -> versioned reward mapping
  -> return / advantage estimator
  -> policy-gradient update
  -> refreshed rollout distribution
```

一个 verifier $v(x,y,e)$ 接收任务 $x$、policy 输出 $y$ 与环境 $e$，先返回结构化观察：

$$
v(x,y,e)
\to
(\text{status},\text{score},\text{evidence}).
$$

它仍不是 advantage。训练配方必须显式定义版本化映射

$$
R=g(\text{status},\text{score},\text{cost},\text{infra state}),
$$

再经 critic/GAE、leave-one-out、group normalization 或其他 estimator 得到 $\widehat A$。把 `timeout`、`invalid` 与确定错误都压成同一个零分，或者直接把 verifier confidence 当作 advantage，都会隐藏训练目标。

## Reward mapping 与信号密度

最简单的 outcome mapping 是

$$
R(x,y)=\mathbb 1\{v(x,y)=\text{correct}\}.
$$

它不需要学习 reward model，却可能极度稀疏：成功率接近 $0$ 时几乎没有正样本，接近 $1$ 时组内相对方法缺少方差。课程与采样调度可以维持有信息量的成功分布，但最终评测仍要覆盖原始任务分布。

格式、过程、资源成本与终局正确性应保留为独立原始分量，再由明确规则聚合。过程分数是否能作为 dense reward、怎样避免重复计费，以及 potential-based shaping 何时保持最优策略，统一见[Verifier、过程奖励与 Reward Shaping](verifiers-reward-shaping.md)。

## 采样、搜索与 RL 的边界

### Rejection sampling

采样 $N$ 个候选，保留 verifier 通过者做 SFT。它是在成功条件下模仿当前 policy 的数据，不直接执行 policy gradient。

### Best-of-$N$

推理时生成候选并由 verifier 选最好者。它增加 inference compute，不更新参数。

### Search

beam、tree search 或 MCTS 使用 verifier/value 决定扩展。搜索结果可继续用于 SFT、preference 或 RL，但每种回流方式目标不同。

### Policy update 与在线闭环

固定 verified dataset 上的 SFT、DPO 或 policy gradient 都是离线更新；当前 policy 产生 rollout 但只用于一次性分析，也不构成在线训练。只有当不断演化的 policy 产生新 rollout，验证结果被用于更新后续 policy，并再次改变采样分布时，才形成这里所说的 online policy-improvement loop。

在 RLVR 中，标准做法是把 reward 转成 return/advantage 后执行 policy-gradient update。迭代 rejection sampling + SFT 也可以形成在线数据闭环并改善 policy，但其训练目标是 imitation，不应仅因使用 verifier 和新采样就改称 RL。

完整搜索机制见[推理时搜索](../reasoning/search-verification.md)，搜索数据怎样回到训练见[推理后训练](../training/reasoning-posttraining.md)。

## Group-relative RLVR

同题采样 $K$ 个答案并用二值 reward 时：

$$
\bar R=\frac1K\sum_iR_i.
$$

只有 mixed group 同时包含成功和失败时，中心化 reward 才产生非零信号。于是 group size、temperature、题目难度与 pass rate 共同决定有效梯度。

动态过滤全对/全错组可以提高 train-token 效率，却增加 rollout 成本并改变题目分布。公平报告应包含：

- 原始采样题数和 token；
- mixed group 比例；
- 被过滤原因；
- 进入训练的 token；
- 完整目标分布上的独立评测。

## Verifier 完整性与训练隔离

程序 reward 仍会遭遇 specification、parser、test、judge 与 environment state gaming。具体攻击分类和 shaping 风险见[Verifier、过程奖励与 Reward Shaping](verifiers-reward-shaping.md)；端到端 RLVR pipeline 额外负责隔离训练 verifier 与最终 evaluator、冻结 reward mapping 版本、区分基础设施失败，并保存能重放 reward 的 evidence。

## 从正确答案到可靠能力

RLVR 直接提高的是所选 verifier 下的期望 reward。要推断更广能力，需要独立检查：

- 未见题型与难度；
- 解题稳定性与 pass@$k$；
- 校准与放弃策略；
- 非目标能力回归；
- reasoning length 与成本；
- 对 verifier 变体的鲁棒性；
- 隐藏测试和污染。

训练 verifier 与最终 evaluator 同源时，分数上升可能只是协议适配。

## 历史位置

[Training Verifiers](https://arxiv.org/abs/2110.14168)训练 learned verifier，并在推理时从多份数学解答中选择答案；它是验证器训练与 test-time selection 的重要前史，但论文没有用 verifier reward 对解题 policy 做 end-to-end RLVR。[DeepSeekMath](https://arxiv.org/abs/2402.03300)把 group-relative policy optimization 用于数学推理；[DeepSeek-R1](https://arxiv.org/abs/2501.12948)进一步组织冷启动、可验证 reward、RL 与蒸馏。它们分别改变 verifier、optimizer 和训练流程，不应压缩成单一“GRPO 公式”。

## 验证清单

1. 保存 task split、rollout policy、采样配置与 exact outputs。
2. verifier 原始状态、reward mapping、聚合分数和 evidence 同时保存。
3. 明确 reward 怎样变成 return / advantage，以及使用哪个 denominator。
4. 区分 wrong、invalid、timeout 与 infrastructure error。
5. 按 pass-rate、长度与题类报告有效组和过滤比例。
6. rejection sampling、Best-of-$N$、SFT 与 RL 使用相同采样预算比较。
7. 新 policy 必须重新采样，在线改善不能只由训练 loss 推断。
8. 最终评测使用隔离数据和独立 verifier revision。

reward 与反馈的整体坐标见[反馈制度](feedback-regimes.md)，过程 reward 的边界见[Verifier 与 reward shaping](verifiers-reward-shaping.md)。

## Reference {#reference}

- Cobbe et al., [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
- Lightman et al., [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)
- Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- Guo et al., [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
