# 最小实现

最小实现用于固定数学与 shape 语义，再与框架、融合 kernel 和分布式路径比较。它追求可审计，不追求生产性能。

## Stable softmax 与交叉熵

```python
import torch

def stable_log_softmax(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    z = x - m
    return z - z.exp().sum(dim=dim, keepdim=True).log()

def cross_entropy(logits, labels):
    logp = stable_log_softmax(logits, dim=-1)
    return -logp.gather(-1, labels[..., None]).squeeze(-1)
```

验证点：

```python
x = torch.tensor([[1000.0, 1001.0, 999.0]], dtype=torch.float64)
y = torch.tensor([1])
loss = cross_entropy(x, y)
ref = torch.nn.functional.cross_entropy(x, y, reduction="none")
torch.testing.assert_close(loss, ref)
```

梯度应满足 $p-y$。可在 FP64 小张量上与 autograd 和有限差分比较。

## RMSNorm

```python
def rms_norm(x, weight, eps=1e-6):
    scale = (x.float().square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    return (x.float() * scale).to(x.dtype) * weight
```

归约用 FP32，再转回输入 dtype。若生产 kernel 使用不同累加精度或把 weight 融入其他算子，先在固定输入上比较，再做端到端 logits 回归。

## GQA

下面的 reference 显式展开 K/V head，适合核对 head 映射：

```python
import math

def gqa(q, k, v, causal=True):
    b, hq, tq, d = q.shape
    _, hkv, tk, _ = k.shape
    if hq % hkv:
        raise ValueError("query heads must be divisible by KV heads")
    group = hq // hkv
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    score = q @ k.transpose(-2, -1) / math.sqrt(d)
    if causal:
        qi = torch.arange(tq, device=q.device) + tk - tq
        kj = torch.arange(tk, device=q.device)
        mask = kj[None, :] > qi[:, None]
        score = score.masked_fill(mask, float("-inf"))
    return score.softmax(dim=-1) @ v
```

生产实现不应真的复制 K/V；它应在 kernel 内映射 query head 到 KV head。这里保留低效展开，目的是让语义清楚。

## Online softmax 合并

对两个 score block，可以合并最大值、分母和加权 value：

```python
def merge_softmax(m1, l1, o1, m2, l2, o2):
    m = torch.maximum(m1, m2)
    a = torch.exp(m1 - m)
    b = torch.exp(m2 - m)
    l = a * l1 + b * l2
    o = (a * l1 * o1 + b * l2 * o2) / l
    return m, l, o
```

在小矩阵上把 score 沿 key 维切成不同 block，合并结果应与一次完整 softmax 接近。还要覆盖全 mask row、极大 logit 和非整除 block。

## 分页 block table

逻辑 token 位置 $t$ 到物理 slot 的最小映射：

```python
def physical_slot(block_table, block_size, token_pos):
    logical_block, offset = divmod(token_pos, block_size)
    physical_block = block_table[logical_block]
    return physical_block * block_size + offset
```

真实运行时还需处理 block 未分配、copy-on-write、引用计数、不同层 cache base 与设备分片。这个函数只固定“逻辑序列不要求物理连续”的核心语义。

## 组相对优势

```python
def group_advantage(reward, group, eps=1e-6):
    advantage = torch.empty_like(reward)
    for g in group.unique():
        idx = group == g
        r = reward[idx]
        advantage[idx] = (r - r.mean()) / (r.std(unbiased=False) + eps)
    return advantage
```

同组只有一个样本或奖励全相等时，优势接近零。生产实现还要明确是否按 prompt 分组、是否过滤无信号组，以及 sequence advantage 怎样广播到 action token。

## 验证层级

每个 reference 都按同一顺序升级：

1. 手算或解析公式；
2. FP64 小张量；
3. 框架 reference；
4. 自定义高性能 kernel；
5. mixed precision；
6. backward；
7. 分布式 shape；
8. 真实模型质量与性能。

若某层失败，回到第一个分叉点，不要通过放宽最终 logits 容差掩盖语义错误。数学目标见[概率、损失与梯度](../foundations/probability-objectives.md)，性能路径见[Kernel 与性能](../systems/kernels-performance.md)。
