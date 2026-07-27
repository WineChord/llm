# 手撕：Decoder-only Transformer

本页把 embedding、RoPE、GQA、pre-norm block、语言建模损失与增量 KV cache 连接成一个最小可运行模型。实现没有训练器和配置框架，但保留训练—解码一致性所需的关键状态。

## Attention

输入 $x\in\mathbb R^{B\times T\times H}$，query head 数为 $A$，KV head 数为 $A_{\mathrm{kv}}$，head dimension 为 $D=H/A$。

```python
import math
import torch
from torch import nn
import torch.nn.functional as F

def apply_rope(x, start=0, base=10000.0):
    """x:[B,A,T,D] -> rotated x at absolute positions start..start+T-1."""
    d, t = x.size(-1), x.size(-2)
    if d % 2:
        raise ValueError("head dimension must be even")
    inv = base ** (-torch.arange(0, d, 2, device=x.device).float() / d)
    pos = torch.arange(start, start + t, device=x.device).float()
    angle = pos[:, None] * inv[None, :]
    c = angle.cos().to(x.dtype)[None, None]
    s = angle.sin().to(x.dtype)[None, None]
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)
```

```python
assert apply_rope(torch.randn(1, 2, 3, 8).bfloat16()).dtype == torch.bfloat16
```

cache 保存已旋转的 K 与未变换的 V。增量 token 从 `past` 开始应用位置编码：

```python
class CausalGQA(nn.Module):
    def __init__(self, hidden, n_heads, n_kv_heads):
        super().__init__()
        if hidden % n_heads or n_heads % n_kv_heads:
            raise ValueError("incompatible head counts")
        self.nh, self.nkv, self.d = n_heads, n_kv_heads, hidden // n_heads
        self.q = nn.Linear(hidden, n_heads * self.d, bias=False)
        self.k = nn.Linear(hidden, n_kv_heads * self.d, bias=False)
        self.v = nn.Linear(hidden, n_kv_heads * self.d, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, cache=None):
        b, tq, _ = x.shape
        past = 0 if cache is None else cache[0].size(-2)
        q = self.q(x).view(b, tq, self.nh, self.d).transpose(1, 2)
        k = self.k(x).view(b, tq, self.nkv, self.d).transpose(1, 2)
        v = self.v(x).view(b, tq, self.nkv, self.d).transpose(1, 2)
        q, k = apply_rope(q, past), apply_rope(k, past)
        if cache is not None:
            k, v = torch.cat((cache[0], k), -2), torch.cat((cache[1], v), -2)
        group = self.nh // self.nkv
        kr = k.repeat_interleave(group, 1)
        vr = v.repeat_interleave(group, 1)
        score = q @ kr.transpose(-2, -1) / math.sqrt(self.d)
        qpos = torch.arange(past, past + tq, device=x.device)
        kpos = torch.arange(k.size(-2), device=x.device)
        score.masked_fill_(kpos[None, :] > qpos[:, None], float("-inf"))
        y = (score.softmax(-1) @ vr).transpose(1, 2).reshape(b, tq, -1)
        return self.o(y), (k, v)
```

`repeat_interleave` 仅是 reference。生产 GQA kernel 会直接按 query head 映射 KV head，避免复制 cache。

## RMSNorm、SwiGLU 与 Block

pre-norm block：

$$
y=x+\operatorname{Attn}(\operatorname{RMSNorm}(x)),\qquad
z=y+\operatorname{MLP}(\operatorname{RMSNorm}(y)).
$$

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        y = xf * (xf.square().mean(-1, keepdim=True) + self.eps).rsqrt()
        return y.to(x.dtype) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate = nn.Linear(hidden, intermediate, bias=False)
        self.up = nn.Linear(hidden, intermediate, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class DecoderBlock(nn.Module):
    def __init__(self, hidden, intermediate, n_heads, n_kv_heads):
        super().__init__()
        self.n1, self.n2 = RMSNorm(hidden), RMSNorm(hidden)
        self.attn = CausalGQA(hidden, n_heads, n_kv_heads)
        self.mlp = SwiGLU(hidden, intermediate)

    def forward(self, x, cache=None):
        a, cache = self.attn(self.n1(x), cache)
        x = x + a
        return x + self.mlp(self.n2(x)), cache
```

这里省略 dropout。若训练配方包含 dropout，增量 cache 对照应在 `eval()` 下进行，并明确 RNG 与 activation checkpoint 的一致性。

## Language model

embedding 与输出层共享权重，减少参数并固定词表空间：

```python
class TinyLM(nn.Module):
    def __init__(self, vocab, hidden, intermediate, layers, heads, kv_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList([
            DecoderBlock(hidden, intermediate, heads, kv_heads)
            for _ in range(layers)
        ])
        self.norm = RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens, cache=None):
        x = self.embed(tokens)
        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError("cache must contain one entry per decoder block")
        old = [None] * len(self.blocks) if cache is None else cache
        new = []
        for block, layer_cache in zip(self.blocks, old):
            x, layer_cache = block(x, layer_cache)
            new.append(layer_cache)
        return self.lm_head(self.norm(x)), new

def next_token_loss(logits, tokens, ignore_index=-100):
    """logits:[B,T,V], tokens:[B,T] -> next-token scalar mean."""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        tokens[:, 1:].reshape(-1),
        ignore_index=ignore_index,
    )
```

如果 batch 包含 padding 或拼接文档，不能只靠 `ignore_index`：attention mask、segment boundary 与 loss mask 都要与[序列构造](../data/sequence-construction.md)一致。

## 增量解码

首次调用处理完整 prompt，后续只输入最新 token。cache 长度就是下一 token 的绝对位置：

```python
@torch.no_grad()
def generate(model, prompt, max_new_tokens, temperature=1.0, top_k=None, generator=None):
    """prompt:[B,T] -> [B,T+max_new_tokens]."""
    out, cache, step = prompt, None, prompt
    for _ in range(max_new_tokens):
        logits, cache = model(step, cache)
        z = logits[:, -1].float() / temperature
        if top_k is not None:
            cutoff = z.topk(min(top_k, z.size(-1))).values[:, -1, None]
            z = z.masked_fill(z < cutoff, float("-inf"))
        token = torch.multinomial(z.softmax(-1), 1, generator=generator)
        out, step = torch.cat((out, token), 1), token
    return out
```

`temperature` 必须大于零；greedy decoding 应作为单独分支使用 `argmax`，而不是除以零。真实服务还要处理 per-request RNG、stop sequence、grammar state、取消和动态 batch。

## 一致性测试

对同一 token 序列，完整 forward 与逐 token cache forward 的 logits 应接近：

```python
torch.manual_seed(7)
model = TinyLM(vocab=31, hidden=32, intermediate=64, layers=2, heads=4, kv_heads=2)
model.eval()
tokens = torch.randint(0, 31, (2, 6))
full, _ = model(tokens)
cache, parts = None, []
for i in range(tokens.size(1)):
    logits, cache = model(tokens[:, i:i + 1], cache)
    parts.append(logits)
stepwise = torch.cat(parts, 1)
torch.testing.assert_close(full, stepwise, atol=1e-5, rtol=1e-5)
assert all(k.size(-2) == tokens.size(1) for k, _ in cache)
try:
    model(tokens[:, :1], cache=[])
except ValueError:
    pass
else:
    raise AssertionError("invalid cache length was accepted")
```

还应覆盖：

- prompt 长度为 1；
- batch 内不同长度时的显式 padding/cache 元数据；
- MHA、MQA 与 GQA；
- cache dtype 与模型计算 dtype 不同；
- RoPE extension 后的位置；
- full forward 与 chunked prefill；
- 训练权重加载、weight tying 和词表扩展。

结构推导见 [Transformer](../architecture/transformer.md)与 [Decoder Block](../architecture/decoder-block.md)，cache 的物理管理见[手撕推理引擎](inference-engine.md)。

## Reference {#reference}

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
