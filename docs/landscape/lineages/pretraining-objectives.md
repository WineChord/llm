# 预训练目标与信息流

GPT、BERT 与 T5 不应被排成一条简单的替代链。它们共享 Transformer 组件，却分别选择了不同的可见范围、预测目标和下游接口。目标函数规定模型训练时能读取什么、在哪里收到监督；架构只是实现这种信息流的载体。

## 三种选择不是同一个问题的排名

| 路线 | 主体架构 | 预训练可见性 | 被预测内容 | 自然接口 |
| --- | --- | --- | --- | --- |
| GPT-1 | decoder | 每个位置只看左侧 | 每个 next token | 续写与任务微调 |
| BERT | encoder | 被破坏序列上的双向上下文 | 选中的原始 token，另含 NSP | 表示抽取与任务微调 |
| T5 | encoder–decoder | encoder 双向，decoder 因果 | 被移除的连续 span | text-to-text |

这张表描述的是原始工作，不代表后来所有同名模型都保留相同目标。

## GPT：用因果预测承接无标注文本

[OpenAI 2018 technical report](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)先在大量无标注文本上优化自回归目标：

$$
\mathcal L_{\mathrm{CLM}}
=-\sum_t\log p_\theta(x_t\mid x_{<t}).
$$

统一的 causal interface 使预训练模型可以直接估计序列概率和生成文本。原始 GPT-1 随后通过 task-aware input transformations 与监督头适配任务，微调时还组合辅助语言建模目标；它不是后来“只写 prompt、完全不更新参数”的 few-shot 范式。作者公开的历史实现见 [finetune-transformer-lm](https://github.com/openai/finetune-transformer-lm)。

这套“生成式预训练 + 任务微调”接口及其与 GPT-2/3 的边界见 [GPT-1 深读](../works/generative-pretraining-gpt.md)。

代价来自左向信息流：位置 $t$ 的表示不能利用右侧词。对需要整句判别或 token 级理解的任务，只靠最后状态或任务专用变换并不自然。

## BERT：把双向条件化变成可训练目标

[BERT 2018 preprint](https://arxiv.org/abs/1810.04805)选择 Transformer encoder，并只在选中位置计算 masked language modeling：

$$
\mathcal L_{\mathrm{MLM}}
=-\sum_{i\in\mathcal M}\log p_\theta(x_i\mid\tilde x).
$$

$\tilde x$ 是破坏后的序列。原始配方选择约 15% token，其中 80% 替换为 `[MASK]`、10% 替换为随机 token、10% 保持不变；同时训练 next sentence prediction。这样每层表示都能结合左右上下文，却只有少数位置提供 token 监督，并引入预训练 corruption 与自然输入之间的差异。

该工作于 2018 年 10 月首次公开，正式版本发表于 [NAACL 2019](https://aclanthology.org/N19-1423/)。目标、数据构造和实现边界见 [BERT 深读](../works/bert.md)；作者的 [TensorFlow 仓库](https://github.com/google-research/bert)也明确说明，公开预训练数据代码不是论文所用内部 C++ 流程的完全相同实现。

## T5：把任务也统一成文本生成

[T5](https://arxiv.org/abs/1910.10683)于 2019 年首次公开，后发表于 [JMLR 2020](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf)。它不只是提出一个更大模型，而是在统一实验框架中比较架构、无监督目标、数据和迁移方式，并采用 text-to-text 接口。

span corruption 从输入移除连续片段，以 sentinel token 标记缺口；decoder 按顺序生成被移除内容：

$$
\mathcal L_{\mathrm{span}}
=-\sum_t\log p_\theta
(y_t\mid x_{\mathrm{corrupt}},y_{<t}).
$$

连续 span 比独立 token masking 更贴近 encoder–decoder 的条件生成接口，也减少 target 长度。统一文本接口降低了任务头碎片化，却保留 encoder 与 decoder 两套激活和 cross-attention 成本；它不是任何部署条件下的统一最优解。历史实现见 [text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer)，现代研究框架见 [T5X](https://github.com/google-research/t5x)。

原始 sentinel 构造、span 采样与 text-to-text 迁移边界见 [T5 深读](../works/t5.md)。

## 读目标时应冻结哪些条件

比较这些路线时，至少同时记录：

- tokenizer、corruption rate 与有效监督 token 数；
- attention mask 和 encoder/decoder 参数预算；
- 训练 token、序列长度与总计算；
- 预训练输入和下游输入是否同分布；
- 下游是冻结表示、全量微调、生成还是上下文学习。

否则，架构、目标、数据和计算的收益会被混成一个数字。通用概率口径见[语言建模](../../foundations/language-modeling.md)与[概率、损失与梯度](../../foundations/probability-objectives.md)，训练流程见[预训练](../../training/pretraining.md)，紧凑实现见[手撕：训练目标](../../practice/training-objectives.md)。
