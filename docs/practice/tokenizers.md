# 手撕：Tokenizer

Tokenizer 是模型输入协议。训练合并表、编码优先级、Unicode 规范化、特殊 token 与 byte fallback 任一不一致，都会让同一 checkpoint 接收不同 token 序列。

## Byte-level BPE 训练

下面把 UTF-8 byte 作为初始符号，反复合并语料中频率最高的相邻 pair。`bytes` token 天然可拼回原输入：

```python
from collections import Counter
import unicodedata

def merge_pair(tokens, pair):
    """tokens:list[bytes] -> merge all non-overlapping occurrences."""
    out, i = [], 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out

def train_byte_bpe(texts, merge_steps):
    """texts:list[str] -> merge-rank dict[(bytes,bytes),int]."""
    sequences = [[bytes([value]) for value in text.encode("utf-8")] for text in texts]
    ranks = {}
    for rank in range(merge_steps):
        counts = Counter(
            pair
            for sequence in sequences
            for pair in zip(sequence, sequence[1:])
        )
        if not counts:
            break
        pair = min(counts, key=lambda item: (-counts[item], item))
        ranks[pair] = rank
        sequences = [merge_pair(sequence, pair) for sequence in sequences]
    return ranks
```

频率相同时显式按 byte pair 排序，保证相同语料与步数可复现。真实训练还应按词或 pre-tokenizer boundary 统计权重，处理大语料 streaming 与最小频率。

## BPE 编码与解码

编码时，当前相邻 pair 中优先合并训练 rank 最小者：

```python
def encode_byte_bpe(text, ranks):
    """text:str -> list[bytes] under a fixed merge table."""
    tokens = [bytes([value]) for value in text.encode("utf-8")]
    while len(tokens) > 1:
        candidates = [
            (ranks[pair], pair)
            for pair in zip(tokens, tokens[1:])
            if pair in ranks
        ]
        if not candidates:
            break
        _, pair = min(candidates)
        tokens = merge_pair(tokens, pair)
    return tokens

def decode_byte_bpe(tokens):
    return b"".join(tokens).decode("utf-8", errors="strict")
```

```python
corpus = ["banana", "bandana", "你好，世界", "hello  world"]
ranks = train_byte_bpe(corpus, merge_steps=20)
for text in corpus + ["未知🙂", "\n\t"]:
    assert decode_byte_bpe(encode_byte_bpe(text, ranks)) == text
assert len(encode_byte_bpe("banana", ranks)) < len("banana".encode())
```

这段实现是 $O(T^2)$ reference。生产 encoder 会使用 merge rank、链表或堆减少重复扫描，但输出必须相同。

## Unicode 规范化

视觉相同字符串可以有不同 code point：

```python
def normalize_text(text, form=None):
    """form is None, NFC, NFD, NFKC or NFKD."""
    if form is None:
        return text
    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError("unknown Unicode normalization")
    return unicodedata.normalize(form, text)
```

```python
composed = "é"
decomposed = "e\u0301"
assert composed != decomposed
assert normalize_text(composed, "NFC") == normalize_text(decomposed, "NFC")
```

规范化会改变某些兼容字符，不能在部署时临时开启。训练与推理必须固定同一规则，并保存 tokenizer artifact 哈希。

## Unigram 的 Viterbi 编码

Unigram tokenizer 从候选 token 概率模型中选择总负对数概率最小的分词：

$$
z^*=\arg\min_{z:\operatorname{concat}(z)=x}
\sum_{u\in z}-\log p(u).
$$

```python
def unigram_viterbi(text, token_cost, unknown_cost=20.0):
    """text:str, token_cost:dict[str,float] -> minimum-cost token list."""
    n = len(text)
    cost, back = [float("inf")] * (n + 1), [None] * (n + 1)
    cost[0] = 0.0
    max_len = max(map(len, token_cost), default=1)
    for end in range(1, n + 1):
        for start in range(max(0, end - max_len), end):
            token = text[start:end]
            if token in token_cost and cost[start] + token_cost[token] < cost[end]:
                cost[end], back[end] = cost[start] + token_cost[token], (start, token)
        if cost[end - 1] + unknown_cost < cost[end]:
            cost[end], back[end] = cost[end - 1] + unknown_cost, (end - 1, text[end - 1:end])
    output, cursor = [], n
    while cursor:
        if back[cursor] is None:
            raise ValueError("input cannot be segmented")
        cursor, token = back[cursor]
        output.append(token)
    return output[::-1]
```

```python
cost = {"北": 3.0, "北京": 1.0, "大学": 1.0, "大": 2.0, "学": 2.0}
assert unigram_viterbi("北京大学", cost) == ["北京", "大学"]
```

完整 Unigram 训练还会用 EM 估计 token 概率并剪枝候选词表；这里仅固定编码的动态规划。

## Special token

特殊 token 应在普通文本与控制通道之间有明确契约：

- 是否允许用户原文出现相同字符串；
- added token 在 pre-tokenization 前还是后匹配；
- BOS/EOS 是否由模板或 tokenizer 自动插入；
- padding ID 是否与 EOS 共享；
- tool、image、audio token 是否占连续区间；
- 词表扩展后 embedding 与 output head 怎样初始化。

编码 API 最好分别接受文本片段和已授权的控制 token，而不是先拼成一个字符串再解析。

## 验证

- round-trip：随机 Unicode、空白、控制字符与 emoji；
- determinism：相同 artifact 在不同进程输出相同 IDs；
- merge rank：频率 tie 与重叠 pair；
- special token：普通文本不能意外注入控制 token；
- compatibility：训练、推理、数据生成与评测使用同一哈希；
- length：按语言、代码和 byte fallback slice 报 token 膨胀。

subword BPE 在 NMT 中的经典应用见 [Sennrich et al.](https://arxiv.org/abs/1508.07909)，byte-level 变体可从 [GPT-2 技术报告](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)回溯，Unigram 与可逆预处理见 [SentencePiece](https://arxiv.org/abs/1808.06226)。模型侧影响见[分词与表示](../foundations/tokenization.md)。
