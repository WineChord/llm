# 奖励建模

Reward model 把人类偏好、规则或过程判断映射成可优化分数。它不是客观效用函数，而是对特定数据、标注协议和回答分布的统计外推；策略一旦针对它优化，原本微小的 shortcut 会被放大。

## Bradley–Terry 模型

给定 prompt $x$ 与回答 $y_w\succ y_l$，标量奖励 $r_\phi(x,y)$ 常使用

$$
P(y_w\succ y_l\mid x)
=
\sigma(r_w-r_l),
$$

$$
\mathcal L_{\text{BT}}
=-\log\sigma(r_w-r_l)
=\operatorname{softplus}(-(r_w-r_l)).
$$

只有差值可辨识。对同一 prompt 的所有回答加常数 $c(x)$：

$$
(r_w+c)-(r_l+c)=r_w-r_l,
$$

偏好概率不变。因此奖励绝对零点没有由 pairwise 数据确定，跨模型、跨 prompt 或跨 checkpoint 比较 raw reward 必须先建立校准协议。

尺度也不是无关紧要：若所有 reward 乘常数，pairwise 排序可能不变，但在线 RL 中 reward 与 KL、value loss 和 clipping 的相对强度会变化。

### Bradley–Terry loss {#bradley-terry-loss-reference}

`reward_a` 与 `reward_b` 是同一 prompt 下两个候选的标量分数，`preference` 是 $a$ 胜出的概率；硬标签取 $0/1$，平局或软标签可取中间值。函数输出逐 pair loss，保留维度以便外层按标注置信度加权。

```python
import torch
import torch.nn.functional as F

def bradley_terry_loss(reward_a, reward_b, preference):
    if not (reward_a.shape == reward_b.shape == preference.shape):
        raise ValueError("one aligned preference is required per reward pair")
    if not (reward_a.device == reward_b.device == preference.device):
        raise ValueError("rewards and preferences must share a device")
    if not torch.isfinite(reward_a).all() or not torch.isfinite(reward_b).all():
        raise ValueError("rewards must be finite")
    if not torch.isfinite(preference).all() or torch.any((preference < 0) | (preference > 1)):
        raise ValueError("preference probabilities must lie in [0, 1]")
    margin = reward_a - reward_b
    return F.binary_cross_entropy_with_logits(
        margin, preference.to(margin.dtype), reduction="none"
    )

ra = torch.tensor([2., -1., 0.])
rb = torch.tensor([0., 1., 0.])
q = torch.tensor([1., 0., 0.5])
loss = bradley_terry_loss(ra, rb, q)
assert torch.allclose(loss, bradley_terry_loss(ra + 7, rb + 7, q))
assert torch.allclose(loss, bradley_terry_loss(rb, ra, 1 - q))
assert loss[:2].max() < bradley_terry_loss(torch.zeros(2), torch.zeros(2), q[:2]).min()
try:
    bradley_terry_loss(ra, rb, torch.tensor([2., 0., .5]))
except ValueError:
    pass
else:
    raise AssertionError("out-of-range preference probabilities must be rejected")
```

加同一常数不改变输出，交换候选并翻转标签也不改变 loss；这两个断言正对应 pairwise 模型的可辨识边界。reference 不包含 reward head、padding mask、listwise 采样和跨 prompt 校准，生产路径还必须防止同一候选跨数据切分泄漏。

## 数据契约

每个 pair 或 list 至少记录：

```text
prompt and context source
candidate texts and generator versions
sampling / decoding configuration
presentation order and randomization
annotator or judge protocol
winner / tie / invalid / disagreement
reason codes and rubric dimensions
language, domain, length and safety slice
```

候选生成器身份、模板、长度和格式若与 label 强相关，reward model 会把它们当捷径。训练切分应按 prompt/题族和候选来源分组，而不是随机拆 pair。

## 标签语义

### 软标签与平局

若偏好概率或标注者比例为 $q\in[0,1]$，可使用

$$
\mathcal L
=
-q\log\sigma(\Delta r)
-(1-q)\log(1-\sigma(\Delta r)),
\qquad
\Delta r=r_a-r_b.
$$

将所有 tie 随机改成 win/loss 会制造不存在的强偏好。也可把 tie 单独建模，但训练与评测必须使用一致语义。

### Listwise 排序

当同一 prompt 有多个候选，独立 pair 会重复利用样本且忽略整体排序。可使用 listwise likelihood 或先构造有控制的 pair；无论哪种方式，都要避免把同一候选跨 train/test。

### 多维 rubric

正确性、帮助性、事实性、风格和安全可能冲突。把它们直接压成一个标量会隐藏 trade-off。可训练分项 head 或保留分项标签，再由策略层明确组合；总 reward 仍需能回溯到各分项。

## Shortcut 与分布外问题

常见 shortcut 包括：

- 长回答或更多标题；
- 固定礼貌语和拒答模板；
- 生成器身份、tokenizer 或格式；
- 引用数量而非引用支持；
- 代码块存在而非测试通过；
- 推理步骤更长而非结论更正确。

控制方法包括长度匹配 pair、顺序交换、格式扰动、只改变一个事实的 hard negative、匿名化生成器信息和可执行 verifier。[Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) 研究了代理 reward 被持续优化时与真实目标之间的偏离，说明 held-out pair accuracy 不能保证策略优化后的可靠性。

## Outcome 与 process reward

Outcome reward 只评估最终答案，标注便宜但信用分配稀疏。Process reward 对中间步骤评分，能定位错误，却需要定义步骤边界、前缀可行性和局部正确是否足以通向全局解。

[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) 比较了数学推理中的 outcome 与 process supervision。其结果属于特定任务和数据，不意味着所有开放式推理都适合步骤标量。对工具轨迹，还要区分模型动作、环境 observation、超时和终态。

## 训练与校准

除了 pairwise accuracy，还应报告：

- log loss 与 Brier-style 概率误差；
- margin 分布和 tie slice；
- 长度、语言、领域、生成器与安全切片；
- 顺序交换后的预测一致性；
- 新策略样本上的校准；
- reward 与可执行正确、人工判断的相关；
- ensemble 或重复训练的不确定性。

从 SFT 生成器切换到经过优化的 policy 后，回答分布会移动。reward model 必须在当前策略样本上重新评估；历史测试集不能证明在线区域仍可靠。

## 正确性与失效

- **把绝对 reward 当可辨识量**：pairwise 训练只约束差值。
- **训练/测试随机拆 pair**：同一 prompt 或候选泄漏。
- **无视 tie 与 disagreement**：人为制造强标签。
- **候选长度不受控**：模型学会长度捷径。
- **只报 accuracy**：严重错分和轻微错分被同等计数。
- **评测仍来自旧 policy**：策略优化后进入 reward model 的分布外区域。
- **infra error 记为低 reward**：环境故障被错归因模型行为。
- **过程步骤没有稳定边界**：PRM 标签与 token 或 action 错位。
- **reward 总和掩盖分项**：某一维提升抵消关键安全或正确性退化。

## 何时不需要 reward model

若任务有可靠、低成本的可执行 verifier，可直接使用结果奖励或 rejection sampling；若只有高质量离线 pair，DPO/IPO 类方法可能更简单；若目标能由 SFT 示范清晰表达，显式 reward model 可能增加不必要外推风险。

## 验证

1. 给所有候选 reward 加同一常数，BT loss 必须不变。
2. 交换候选顺序与标签，预测概率应互补。
3. 按 prompt/题族分组切分，并检查生成器和模板泄漏。
4. 构造长度相同、格式扰动和单事实错误的 counterfactual pair。
5. 在当前 policy、旧 policy 和外部生成器样本上分别测校准。
6. 对 reward 高分但 verifier/人工失败的样本做优先审计。
7. 进入在线 RL 前冻结 reward 版本、归一化、clip 和组合权重。

偏好数据结构见[偏好、过程与轨迹数据](../data/feedback-trajectories.md)，从 reward 到策略优化见[在线 RL](online-rl.md)，最小 BT 目标与校准工具见[训练目标实现](../practice/training-objectives.md)和[评测工具](../practice/evaluation-tooling.md)。

## Reference {#reference}

- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
