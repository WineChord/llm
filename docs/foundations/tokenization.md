# 分词与表示

Tokenizer 把文本变成有限词表上的整数序列。它不是可随意替换的预处理器：切分结果会改变训练样本的长度权重、有效上下文、推理成本、复制精度以及 checkpoint 的输入输出接口。

一个完整 tokenizer 通常包含四层：

$$
\text{raw text}
\xrightarrow{\text{normalize}}
\text{normalized text}
\xrightarrow{\text{pre-tokenize}}
\text{pieces}
\xrightarrow{\text{model}}
\text{token ids}
\xrightarrow{\text{template}}
\text{model sequence}.
$$

比较两个 tokenizer 时，应逐层说明差异；只比较词表大小不足以解释行为。

## 可逆性与边界

理想的普通文本路径满足

$$
\operatorname{decode}(\operatorname{encode}(x))=x.
$$

但 normalizer 可能主动折叠兼容字符、大小写或空白，因此实际契约可能只是

$$
\operatorname{decode}(\operatorname{encode}(x))
=
\operatorname{normalize}(x).
$$

这一区别对代码、数学公式、路径、URL 和结构化输出尤其重要。若模型必须逐字复制输入，应避免不可逆规范化，或保存原始字节旁路。

## Byte Pair Encoding

[子词 BPE](https://arxiv.org/abs/1508.07909) 从较小的基础符号集合开始，反复合并语料中高频的相邻符号对。设当前序列集合中的相邻对频次为

$$
c(a,b)=\sum_{x\in\mathcal D}\operatorname{count}_{x}(a,b),
$$

一次训练迭代选择

$$
(a^\star,b^\star)=\arg\max_{(a,b)}c(a,b)
$$

并把所有不重叠的 $(a^\star,b^\star)$ 合并为新符号。最终词表由基础符号、特殊 token 和依次生成的 merge token 构成。

### 编码不是重新统计频次

训练得到的是有顺序的 merge 表。编码新文本时，应始终应用当前可合并对中 rank 最小的一项，而不是重新选择新文本中出现次数最多的 pair。否则训练器和编码器定义了两个不同算法。

实现还要明确：

- 一轮中重叠 pair 如何处理，例如 `aaa` 中的 `aa`；
- merge 是否允许跨空白或预分词边界；
- 文本起始、词首或空白是否有显式标记；
- byte、Unicode code point 还是字符片段作为基础符号。

## Unigram Language Model

[Unigram 分词](https://arxiv.org/abs/1804.10959) 从较大的候选词表开始，为每个子词 $z$ 分配概率 $p(z)$。一个字符串 $x$ 的切分 $s=(z_1,\ldots,z_m)$ 满足

$$
p(s)=\prod_{i=1}^{m}p(z_i),
\qquad
s^\star=\arg\max_{s\in\mathcal S(x)}
\sum_{i=1}^{m}\log p(z_i).
$$

最优切分可以用 Viterbi 动态规划求解。训练时交替估计 token 概率并删除对似然贡献较小的候选，直到达到目标词表大小。

与确定性 BPE 相比，Unigram 自然保留多个候选切分。训练时按候选后验采样可形成 subword regularization，使模型不依赖唯一边界；推理和评测仍应固定采样配置。

## 字节回退

若基础词表覆盖全部字节，则任意输入都可编码而无需 `[UNK]`。UTF-8 文本首先变成字节序列：

$$
x\longrightarrow (b_1,\ldots,b_n),
\qquad b_i\in\{0,\ldots,255\}.
$$

Byte fallback 提供完整覆盖，但罕见字符可能展开为多个 token。它解决的是“能否表示”，不是“表示是否高效”。应在自然语言、代码、emoji、组合字符、控制字符和损坏 UTF-8 等切片上分别报告膨胀率。

[SentencePiece](https://arxiv.org/abs/1808.06226) 展示了直接从原始句子训练 BPE 或 Unigram、把空白纳入模型并保持语言无关接口的工程方式；其[官方实现](https://github.com/google/sentencepiece)也适合核对 normalizer、sampling 和序列化语义。

## 特殊 token 与模板

特殊 token 不只是词表中的几个整数。模型输入通常由模板构造：

$$
[BOS]\,
[ROLE_{\text{user}}]\,
x\,
[END]\,
[ROLE_{\text{assistant}}]\,
y\,
[EOS].
$$

必须固定：

- BOS、EOS、PAD、UNK 的 ID 与是否自动添加；
- system、user、assistant、tool 等角色边界；
- 特殊 token 在普通文本中出现时是字面文本还是控制符；
- generation prompt 是否追加 assistant 起始标记；
- 哪些位置参与 loss，哪些位置仅用于条件。

模板、tokenizer 与模型权重构成一个版本化接口。只替换其中任意一个，都可能造成停止失效、角色串位或训练—推理分布漂移。

## Embedding 与输出层

词表大小为 $V$、隐藏维为 $d$ 时，输入 embedding 为

$$
E\in\mathbb R^{V\times d},
\qquad h_t=E[x_t].
$$

输出层产生

$$
z_t=W_{\text{out}}h_t+b,
\qquad W_{\text{out}}\in\mathbb R^{V\times d}.
$$

Weight tying 令 $W_{\text{out}}=E$，减少约 $Vd$ 个参数并共享输入输出几何，但也约束两种角色使用同一表示。扩展词表时，需要同步处理 embedding、输出层、配置中的词表大小以及 tied weight；新行的初始化不能依赖未更新的旧 checkpoint 形状。

## 复杂度与统计口径

Tokenizer 的效率不应只报告平均 token 数。对样本集合 $\mathcal D$，至少统计

$$
r_{\text{char}}
=
\frac{\sum_{x\in\mathcal D}|\operatorname{encode}(x)|}
{\sum_{x\in\mathcal D}|x|},
$$

并给出分位数、模态和领域切片。还应记录：

- tokens/UTF-8 byte、tokens/字符和 tokens/词；
- 长尾样本的 P95、P99 膨胀；
- 不同语言、代码、数学、表格、URL 与日志；
- 特殊 token 和模板本身的固定开销；
- 相同字符预算下各领域获得的训练 token 权重。

Perplexity 依赖 tokenization，两个词表上的 token-level PPL 不能直接横向比较。可在共同的 byte/character 口径或下游任务上补充对照。

## 实现契约

一个可验证实现应显式保存：

1. normalizer 和版本；
2. pre-tokenizer 规则；
3. 基础符号集合与 byte fallback；
4. token 到 ID 的双向表；
5. BPE merge rank 或 Unigram token 概率；
6. 特殊 token 及其添加策略；
7. chat template；
8. 序列化格式的校验值。

训练和服务加载后，应比较这些对象的内容而不只比较文件名。完整的最小训练器与编码器练习见[Tokenizer 手撕实现](../practice/tokenizers.md)。

## 常见失效

- **规范化漂移**：训练时使用 NFKC，服务时保留原文，导致 ID 序列不同。
- **空白漂移**：前导空格、换行和 tab 被不同预处理器折叠。
- **Merge 优先级错误**：编码时按局部频率而不是训练 rank 合并。
- **特殊 token 碰撞**：控制标记能被普通文本伪造，或被拆成普通子词。
- **截断后破坏模板**：EOS、tool result 或图像占位符被截掉。
- **字节膨胀**：覆盖完整但序列长度在长尾语言或二进制片段上失控。
- **词表扩展不完整**：输入 embedding 已扩展，输出头或 tied weight 未同步。
- **不可比指标**：把不同 tokenizer 的 token-level loss/PPL 直接排序。

## 验证矩阵

| 验证层 | 必测项 |
| --- | --- |
| 单元 | merge rank、Viterbi 路径、特殊 token、空字符串 |
| 可逆 | 多语言、组合字符、emoji、空白、代码、任意 byte |
| 一致 | 训练器、离线编码、在线服务输出完全相同 |
| 统计 | 平均与 P95/P99 膨胀、各领域 token 占比 |
| 模板 | 单轮、多轮、tool call、截断、generation prompt |
| Checkpoint | embedding/output shape、ID 范围、版本与 hash |

Tokenizer 生成的序列如何参与 packing、label shift 与 loss mask，见[序列构造与打包](../data/sequence-construction.md)；token 单位怎样影响评测，见[语言模型评测协议](../evaluation/language-model-evaluation.md)。
