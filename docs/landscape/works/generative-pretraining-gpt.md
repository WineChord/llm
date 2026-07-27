# GPT-1：生成式预训练如何成为通用接口

2018 年的 GPT 技术报告并不是第一个神经语言模型，也不是第一个使用 Transformer 的工作。它真正固定下来的是一套后来极具扩展性的迁移接口：先在大量无标注文本上训练 causal Transformer，再用很少的任务结构改造和监督数据完成下游适配。[报告](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)与[官方实现](https://github.com/openai/finetune-transformer-lm)应作为同一个历史对象阅读；它是 technical report，不应被补写成不存在的会议论文。

## 前序矛盾：任务数据太少，模型接口太碎

早期 NLP 常为分类、文本蕴含、相似度和多项选择分别设计架构。预训练词向量能够迁移词级语义，却没有把长程上下文和整个模型一起迁移。GPT-1 选择一个简单目标：

$$
\mathcal L_{\mathrm{LM}}
=-\sum_t \log p_\theta(x_t\mid x_{<t}),
$$

用 decoder-only Transformer 训练通用参数，然后在监督任务上继续优化。causal 目标天然与生成一致，也意味着每个位置只能读取左侧上下文。

## Task-aware input transformation

不同任务被改写成同一种 token 序列，再在最后位置接线性分类头。例如：

```text
classification: <s> document <e>
entailment:     <s> premise <delim> hypothesis <e>
similarity:     两种顺序分别编码后相加
multiple choice:<s> context <delim> answer_i <e>
```

模型主体没有为每个任务重做结构；任务差异主要落在输入序列化和读出位置。这正是后来 prompt/interface 思想的早期形态，但 GPT-1 仍然更新全部参数，并不属于今天所说的纯 in-context learning。

## 监督目标仍保留语言建模

报告在下游训练中组合任务损失与辅助语言模型损失：

$$
\mathcal L
=\mathcal L_{\mathrm{task}}+\lambda\mathcal L_{\mathrm{LM}}.
$$

辅助目标试图减少小数据微调对预训练表示的破坏。下面的最小模型同时固定 causal mask、next-token shift 与最后有效 token 的任务读出：

```python
import torch
from torch import nn
import torch.nn.functional as F
class TinyGPT1(nn.Module):
    def __init__(self, vocab, d, classes):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        layer = nn.TransformerEncoderLayer(d, 2, 2 * d, batch_first=True, dropout=0)
        self.body = nn.TransformerEncoder(layer, 1)
        self.lm = nn.Linear(d, vocab, bias=False)
        self.task = nn.Linear(d, classes)
    def forward(self, tokens):
        n = tokens.size(1)
        mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
        h = self.body(self.embed(tokens), mask=mask)
        return self.lm(h[:, :-1]), self.task(h[:, -1])
torch.manual_seed(0)
model = TinyGPT1(vocab=17, d=8, classes=3)
tokens = torch.tensor([[1, 4, 7, 2], [1, 5, 6, 2]])
lm_logits, task_logits = model(tokens)
loss = F.cross_entropy(lm_logits.reshape(-1, 17), tokens[:, 1:].reshape(-1))
loss = loss + 0.5 * F.cross_entropy(task_logits, torch.tensor([0, 2]))
loss.backward()
assert lm_logits.shape == (2, 3, 17) and task_logits.shape == (2, 3)
assert model.embed.weight.grad is not None
```

这段代码没有位置 embedding、预训练语料或任务专用 pooling，只保留两个目标共享同一 causal body 的关键逻辑。完整 decoder block 见[从零实现 Transformer](../../practice/transformer-from-scratch.md)。

## 为什么 causal objective 适合继续扩展

causal LM 的连续性首先在于预训练与生成共享同一种 next-token factorization。GPT-1 的下游迁移仍使用任务特定的输入变换与分类、相似度等读出头，并没有让任意任务都直接共享语言模型输出头。

[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)才进一步强调把任务序列化成文本、通过自然语言条件完成零样本迁移；[GPT-3](https://arxiv.org/abs/2005.14165)又把这种接口推到大规模 few-shot in-context learning。它们延续的是 causal 目标，并逐步减少任务专用读出，不是说微调从此失去价值。

## 与 BERT、T5 的分叉

GPT 选择左到右可见性，适合生成，却不能让一个 token 在预训练时同时看见左右上下文。[BERT](bert.md)用 masked prediction 换取双向表示；[T5](t5.md)用 span corruption 与 encoder–decoder 把双向输入和自回归输出结合。三者不是简单版本升级，而是信息流和下游接口的不同选择，完整比较见[预训练目标谱系](../lineages/pretraining-objectives.md)。

## 论文证明到哪里

GPT-1 在当时一组语言理解任务上展示了生成式预训练与判别式微调的有效组合，也通过消融支持 auxiliary LM objective 和 Transformer 表示的价值。它没有证明：

- causal LM 对所有理解任务都优于双向目标；
- 输入序列化可以消除任务数据分布差异；
- 更大模型必然获得可靠推理或指令遵循；
- 官方仓库能重建原始数据、硬件和完整训练运行。

它留下的最重要遗产，是一个足够简单、能不断扩展的目标—接口组合。现代预训练、SFT 与对话 mask 的完整契约见[预训练](../../training/pretraining.md)和[监督微调](../../training/supervised-finetuning.md)。
