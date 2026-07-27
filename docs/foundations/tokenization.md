# 分词与表示

Tokenizer 在字符与模型词表之间建立可逆或近似可逆的映射。它不仅是预处理：相同文本被切成多少 token，会直接影响上下文占用、训练采样权重、推理成本与跨语言公平性。

## 常见方法

### Byte Pair Encoding

BPE 反复合并高频相邻符号。词表越大，常见片段越短；但词表矩阵和 softmax 计算更大，低资源语言与特殊格式也可能被切得不均衡。子词 BPE 在神经机器翻译中的经典描述见 [Sennrich 等](https://arxiv.org/abs/1508.07909)。

### Unigram Language Model

Unigram 从较大的候选词表出发，迭代删除贡献较低的子词，通过概率模型选择切分。它天然支持多种候选切分，可用于 subword regularization。

### 字节与混合方案

字节级方案避免未知字符，但序列可能更长。工程系统常对空白、数字、代码和多字节字符加入专门规则；比较 tokenizer 时必须用真实语料测量 token/字符、token/词和长尾分布。

[SentencePiece](https://arxiv.org/abs/1808.06226) 提供与语言无关、可直接从原始句子训练的实现框架。

## Embedding 与输出层

输入 token ID 经矩阵 $E\in\mathbb{R}^{V\times d}$ 映射为向量。输出层把隐藏状态投影回 $V$ 维 logits。权重 tying 可让输入 embedding 与输出矩阵共享参数，但会约束二者表示。

## 评估清单

- 多语言、代码、数学、表格、URL 和 Unicode 是否出现异常膨胀。
- 特殊 token 是否有唯一、稳定且不会与正文冲突的编码。
- chat template 与 tokenizer 的 BOS、EOS、role token 是否一致。
- 训练与推理使用的 tokenizer 文件、normalizer 和版本是否完全相同。
- 词表扩展后，新增 embedding 如何初始化，旧检查点是否兼容。

Tokenizer 变化通常意味着数据统计、上下文长度和 checkpoint 接口一起变化，不能只替换一个词表文件。

模板、BOS/EOS、label shift 与 packing 的落地见[序列构造与打包](../data/sequence-construction.md)，token 单位怎样影响 PPL 见[语言模型评测协议](../evaluation/language-model-evaluation.md)。
