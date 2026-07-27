# 监督微调

Supervised Fine-Tuning（SFT）让基础模型在给定上下文后模仿目标输出。它既可以教授任务接口与对话格式，也会改变回答风格、拒答边界和能力分布；“只是在小数据上继续预训练”不足以描述这些变化。

## 目标函数

对 prompt $x$ 与 response $y=(y_1,\ldots,y_T)$，常见目标为

$$
\mathcal L_{\text{SFT}}
=-\sum_{t=1}^{T}
\log p_\theta(y_t\mid x,y_{<t}).
$$

若整段对话中只训练 assistant token，loss mask 应与消息边界严格一致。是否训练 system、user、tool result、role marker 与 EOS 是配方的一部分，详见[序列构造与打包](../data/sequence-construction.md)。

### Completion-only loss {#completion-only-loss-reference}

下面的输入已经完成 next-token 对齐：`labels[b,t]` 是 `logits[b,t]` 要预测的 token，`response_mask` 只在纳入监督的 assistant 位置为 $1$，`sample_weight` 为每条样本的权重。函数返回分子与加权 token 分母，分布式路径必须先分别归约二者。

```python
import torch
import torch.nn.functional as F

def completion_only_terms(logits, labels, response_mask, sample_weight):
    if (logits.ndim != 3 or logits.shape[:2] != labels.shape
            or labels.shape != response_mask.shape):
        raise ValueError("logits, labels and response mask must align on [batch, time]")
    if sample_weight.ndim != 1 or sample_weight.shape[0] != logits.shape[0]:
        raise ValueError("sample_weight must contain one value per sample")
    if not torch.isfinite(sample_weight).all() or (sample_weight < 0).any():
        raise ValueError("sample weights must be finite and non-negative")
    valid = response_mask.bool() & sample_weight[:, None].ne(0)
    weight = sample_weight[:, None].expand_as(labels)[valid].to(logits.dtype)
    if not valid.any():
        return logits[valid].sum(), weight.sum()
    nll = F.cross_entropy(logits[valid], labels[valid], reduction="none")
    return (nll * weight).sum(), weight.sum()

z = torch.tensor([[[float("nan")] * 3, [0.] * 3, [0.] * 3]], requires_grad=True)
labels = torch.tensor([[-100, 1, 2]])
mask = torch.tensor([[False, True, True]])
num, den = completion_only_terms(z, labels, mask, torch.tensor([2.]))
assert den.item() == 4 and torch.isfinite(num)
(num / den).backward()
assert torch.isfinite(z.grad).all() and z.grad[:, 0].abs().sum() == 0
empty_num, empty_den = completion_only_terms(
    z, torch.full_like(labels, -100), torch.zeros_like(mask), torch.tensor([2.])
)
assert empty_num == 0 and empty_den == 0
try:
    completion_only_terms(z.expand(2, -1, -1), labels.expand(2, -1),
                          mask.expand(2, -1), torch.ones(2, 1))
except ValueError:
    pass
else:
    raise AssertionError("rank-two sample weights must be rejected")
```

不变量是非 response token 在交叉熵之前被排除，对 loss 与梯度均无贡献；样本权重同时作用于分子和分母。局部 batch 没有有效目标时返回 `(0, 0)`，data-parallel 调用方归约后必须拒绝全局分母仍为零的更新。这里没有解析 role 或 chat template；这些边界必须由版本化的序列构造器产生。多种归一化与目标的组合测试见[训练目标实现](../practice/training-objectives.md)。

## 数据比数量更重要的地方

一条示范同时提供：

- 任务是什么；
- 输入和输出的结构；
- 允许使用的知识与工具；
- 推理或解释的可见形式；
- 答案长度、语气和停止方式；
- 面对不确定或冲突时怎样处理。

因此重复的模板噪声也会被稳定学习。高质量不是单一模型评分，而是相对于目标任务的正确性、覆盖、难度、边界和多样性。

## Chat、completion 与 tool data

### Completion

固定前缀后预测续写，边界最简单，适合领域文本和结构化生成。

### Multi-turn chat

需要统一 role、历史轮次、generation prompt 与截断策略。随机切断一轮对话可能留下没有问题的答案或没有答案的问题。

### Tool use

至少区分：

1. 模型选择工具；
2. 生成结构化参数；
3. 外部系统返回结果；
4. 模型根据结果继续或终止。

只训练成功调用会让模型不熟悉超时、权限不足和部分成功。工具 schema 版本应随样本保存。

### 可扩展的 Agent 序列接口

当一条样本同时包含对话、推理状态、工具定义、调用和返回值时，chat template 已经是一项协议设计，而不只是字符串拼接。一个可演进的接口至少要分开：

- 对整段会话长期生效的 global options；
- 在历史消息之后临时注入的 one-shot options；
- assistant 的思考、面向用户响应与工具调用 channel；
- 调用序号、工具名、参数类型和对应返回值；
- 未知字段的前向兼容与 schema revision。

[Kimi K3 技术报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)中的 XTML 是这种 typed markup 的一个实例：全局选项位于消息历史之前，一次性选项在历史之后；工具调用用 index 对齐结果，并允许显式类型参数。可迁移的重点是“结构和作用域由 parser 验证”，而不是复制某组标签。训练序列化器、rollout parser 与服务端必须共享同一 golden corpus，逐 token 对齐 role、channel、escaping、EOS 和 loss mask。更完整的接口演进见 [Kimi K3](../landscape/works/kimi-k3.md)。

内部思考是否保存、是否进入监督、是否在下一轮可见、是否对最终用户返回，是四个独立开关。任何实现都应按产品和安全边界显式定义，而不能因为序列里存在一个 reasoning channel 就默认公开或监督全部中间文本。

## Packing 与权重

短样本 packing 能提高有效 token 比，但不能让独立对话跨边界 attention。若每个 batch 先按 token 求均值，长回答贡献更多梯度；若先按样本求均值，短回答权重提高。两者都合理，但必须与数据混合目标一致。

对数据源 $i$，可以显式使用样本权重 $w_i$：

$$
\mathcal L
=\frac{\sum_i w_i\sum_t m_{i,t}\ell_{i,t}}
{\sum_i w_i\sum_t m_{i,t}}.
$$

分布式训练中，分子和有效 token 分母都要跨 rank 归约；简单平均各 rank 的局部均值会在长度不均时产生偏差。

## 全量微调与参数高效微调

全量微调更新全部参数，容量最大，也需要完整梯度和 optimizer state。LoRA 等参数高效方法冻结基座，只学习低秩增量，降低训练显存和多任务存储成本。选择时要看：

- 目标域与基座差异；
- 数据规模和任务多样性；
- 是否需要改变 embedding、norm 或输出头；
- 多 adapter 的装载与合并方式；
- 训练节省是否转化为服务简化。

具体方法见[参数高效训练与压缩](peft-compression.md)。

若目标部署使用低精度 expert 权重或 activation，量化契约还可能从 SFT 延续到在线 RL。这样做可减少“训练高精度、rollout 低精度”造成的策略分布偏差，但也把 scale、格式和 kernel 版本变成训练数据的一部分；K3 报告的 MXFP4 expert weight、MXFP8 expert input 路线是一项具体实例。通用 QAT 边界见[量化](../inference/quantization.md)，不能仅凭存储位宽推断质量或速度。

## 学习率与训练长度

SFT 数据通常远少于预训练语料，重复 epoch 很快。应同时监控：

- train 与 held-out token loss；
- 目标任务成功率；
- 通用能力和多语言回归；
- 输出长度、格式和拒答分布；
- 训练样本的逐字记忆；
- 数据源分项梯度或指标。

低 train loss 不代表行为更好。继续训练可能只强化固定措辞，降低多样性或覆盖基础模型已有能力。

## Completion-only mask 的验证

在正式训练前，抽取一条多轮样本，打印：

```text
token id
decoded token
role
segment id
label
loss enabled
position id
```

然后把目标 response 替换为明显不同的短字符串，确认只有相应位置 loss 改变。若框架自动应用 chat template 或 label shift，还要验证没有重复处理。

## 能力混合与遗忘

不同能力源可按采样权重混合：

$$
p(D)=\sum_i\alpha_iD_i.
$$

但真实影响还取决于序列长度、loss mask、难度与重复度。视觉问答、代码、长推理和普通对话的同样“样本数”不等价。若某一新能力训练后通用能力下降，可检查：

- 该源的有效 token 占比是否过高；
- 模板是否与原 checkpoint 不兼容；
- 学习率是否过大；
- 是否缺少通用 replay；
- 评测是否被输出格式变化误伤。

## 最小验收

1. 数据可追溯，模板和 tokenizer 固定版本。
2. loss mask、shift、packing 和分布式归一化有小样例测试。
3. 单 batch 可过拟合，resume 后曲线连续。
4. 目标能力、通用能力、安全与格式均有独立回归集。
5. 推理服务使用完全一致的模板、特殊 token 与 adapter。

后续偏好阶段见[奖励建模与偏好优化](reward-preference.md)，训练稳定性见[优化与稳定性](optimization.md)。

## Reference {#reference}

- [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)
- [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155)
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)
- [Kimi K3 Technical Report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
