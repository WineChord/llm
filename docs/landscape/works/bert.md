# BERT

[BERT](https://arxiv.org/abs/1810.04805) 于 2018 年 10 月首次公开，正式发表于 [NAACL 2019](https://aclanthology.org/N19-1423/)。它针对的矛盾不是“Transformer 不能编码文本”，而是当时强势的自回归预训练只能让每个位置使用单侧上下文；许多理解任务却希望一个 token 的表示同时结合左右证据。

作者公开了 [TensorFlow 代码与预训练模型](https://github.com/google-research/bert)。该仓库明确说明，公开的预训练数据代码能生成论文描述的数据，但并非论文所用内部 C++ 程序的完全相同实现，且原始程序还有额外复杂度。复现时应分别记录论文配方、公开脚本和实际生成的数据。

## 双向表示需要新的预测目标

若直接让双向 encoder 在每个位置预测当前 token，输入就已经泄露答案。BERT 随机选择约 15% token 构造破坏后的序列 $\tilde x$，只在选中集合 $\mathcal M$ 上计算 masked language modeling：

$$
\mathcal L_{\mathrm{MLM}}
=-\sum_{i\in\mathcal M}\log p_\theta(x_i\mid\tilde x).
$$

选中 token 中，80% 替换为 `[MASK]`，10% 替换为随机 token，10% 保持不变。后两种情况减轻预训练只见 `[MASK]` 的输入偏移，但没有消除 corruption mismatch。保持不变的 token 仍要计入损失，否则就改变了原始目标。

输入表示由 token、segment 和 position embedding 相加。原始工作还联合训练 next sentence prediction（NSP），区分第二段是否为真实后继：

$$
\mathcal L=\mathcal L_{\mathrm{MLM}}+\mathcal L_{\mathrm{NSP}}.
$$

NSP 是 BERT 原始配方的一部分，不应从历史描述中删除；它也不应被提升为所有双向预训练不可缺少的原则。

原始输入最多由两段组成，序列开头加入 `[CLS]`，段间与结尾使用 `[SEP]`，segment embedding 区分 A/B。NSP 的正例来自语料中的真实后继，负例从语料随机抽取；因此它同时包含篇章关系和数据构造信号。后续工作移除或替换 NSP 时，也往往改变 batch、数据量、masking 和训练步数，不能把结果归因给单一开关。

## 原始 masking 的最小实现

下面实现 15% 选择和 80/10/10 替换，并用 `ignore_index` 保证未选位置不进入 MLM loss：

```python
import torch
import torch.nn.functional as F
def corrupt_for_bert(tokens, special, vocab_size, mask_id, generator):
    """tokens/special:[B,T]; special=True positions are never selected."""
    if tokens.shape != special.shape or special.dtype != torch.bool:
        raise ValueError("special must be a boolean mask matching tokens")
    chosen = (torch.rand(tokens.shape, generator=generator) < 0.15) & ~special
    labels = tokens.clone()
    labels[~chosen] = -100
    draw = torch.rand(tokens.shape, generator=generator)
    masked = chosen & (draw < 0.8)
    randomed = chosen & (draw >= 0.8) & (draw < 0.9)
    corrupted = tokens.clone()
    corrupted[masked] = mask_id
    random_tokens = torch.randint(vocab_size, tokens.shape, generator=generator)
    corrupted[randomed] = random_tokens[randomed]
    return corrupted, labels, chosen, masked, randomed
g = torch.Generator().manual_seed(11)
vocab, mask_id = 97, 4
tokens = torch.randint(5, vocab, (128, 128), generator=g)
special = torch.zeros_like(tokens, dtype=torch.bool)
special[:, 0] = True
corrupted, labels, chosen, masked, randomed = corrupt_for_bert(
    tokens, special, vocab, mask_id, g
)
rate = chosen.float().mean().item()
assert 0.13 < rate < 0.17 and (labels[:, 0] == -100).all()
assert 0.76 < masked.sum().item() / chosen.sum().item() < 0.84
assert 0.07 < randomed.sum().item() / chosen.sum().item() < 0.13
assert torch.equal(corrupted[~chosen], tokens[~chosen])
logits = torch.randn(*tokens.shape, vocab, requires_grad=True)
loss = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1))
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
```

这段 reference 允许随机 token 偶然等于原 token，符合“从词表随机采样”的语义。真实数据管线还要处理 WordPiece、句段边界、short sequence、重复 corruption、全局随机性和样本分片；不能把一个 collator 当成完整预训练复现。

公开脚本与原始论文数据程序之间还有一个重要边界：原始 BERT 预先生成带 mask 的训练实例，并通过 `dupe_factor` 产生多份 corruption；现代 collator 常在每次读取时动态 masking。二者期望目标相近，但样本相关性、随机性与可重放方式不同，必须在复现实验中说明。

论文的长度课程也不是“从第一步就用 512 token”：大部分 step 使用较短序列以节约平方 attention 成本，后段再训练 512 长度。若只复制最大长度而忽略 step 分布，计算预算与位置覆盖都会改变。

## 一个 backbone 怎样服务不同任务

BERT 不为每个任务重新设计 backbone。句级任务读取 `[CLS]` 表示，token 级任务读取对应位置，问答则预测答案起止位置；全模型与一个轻量任务头共同微调。论文的贡献因此同时包括双向预训练目标和简单、统一的参数迁移接口。

这种接口也有边界：

- encoder 没有因果生成语义，不能直接替代自回归 decoder；
- 每个样本只有约 15% 位置获得 token 目标，监督密度较低；
- `[MASK]` 不出现在自然下游输入中；
- NSP 的收益依赖数据构造与比较配方，不能脱离实验条件概括；
- checkpoint 结果还依赖 WordPiece 词表、序列长度阶段和优化配方。

微调接口看似简单，评测却仍需固定：

- `[CLS]`、span start/end 或 token head 的初始化；
- max sequence length 与截断方向；
- 多 seed 的方差，尤其小数据任务；
- whole-model fine-tuning 还是冻结 backbone；
- task-specific preprocessing 与答案解析。

“同一个预训练模型加一个小 head”减少了架构分叉，不意味着下游协议已经统一。

## 历史结论与后续修正怎样共存

BERT 系统展示了深层双向 Transformer 表示可以通过大规模无标注预训练迁移到多类理解任务；它没有证明 MLM、NSP 或 encoder-only 在所有任务上最优。

后来的 RoBERTa 等工作通过更长训练、更大数据、动态 masking 和去除 NSP 重估配方，说明 BERT 的关键遗产应拆成“双向 corruption 预训练”和“一套具体 2018 recipe”。历史页保留后者，canonical 目标页则比较可迁移机制。三类预训练信息流见[预训练目标与信息流](../lineages/pretraining-objectives.md)。通用机制见 [Transformer](../../architecture/transformer.md) 与[语言建模](../../foundations/language-modeling.md)，mask 与损失实现约束见[概率、损失与梯度](../../foundations/probability-objectives.md)和[手撕：训练目标](../../practice/training-objectives.md)。

## Reference {#reference}

- [BERT 2018 preprint](https://arxiv.org/abs/1810.04805)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)
- [TensorFlow 代码与预训练模型](https://github.com/google-research/bert)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
