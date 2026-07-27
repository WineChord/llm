# 手撕：递推与记忆模型

状态空间、线性注意力与测试时记忆都把历史压进显式状态。reference 的重点是比较 recurrent、chunked 与矩阵形式，并检查 reset、精度和固定容量失效。

## Selective scan

对逐元素递推：

$$
h_t=a_t\odot h_{t-1}+b_t\odot x_t,\qquad
y_t=c_t\odot h_t.
$$

```python
import torch

def selective_scan(x, a, b, c, initial=None):
    """x/a/b/c:[B,T,H], initial:[B,H] -> y:[B,T,H], final:[B,H]."""
    if not (x.shape == a.shape == b.shape == c.shape):
        raise ValueError("scan tensors must share shape")
    state = torch.zeros_like(x[:, 0]) if initial is None else initial
    output = []
    for step in range(x.size(1)):
        state = a[:, step] * state + b[:, step] * x[:, step]
        output.append(c[:, step] * state)
    return torch.stack(output, dim=1), state
```

这不是完整 Mamba block；它只固定输入依赖系数下的递推语义。[Mamba](https://arxiv.org/abs/2312.00752)还包含离散化、卷积与硬件感知 scan。

## 显式矩阵对照

第 $j$ 个输出对第 $i$ 个输入的权重：

$$
M_{j,i}=c_jb_i\prod_{k=i+1}^{j}a_k,\qquad j\ge i.
$$

```python
def scalar_scan_matrix(a, b, c):
    """a/b/c:[T] -> lower-triangular mixing matrix [T,T]."""
    t = a.numel()
    matrix = torch.zeros(t, t, dtype=a.dtype, device=a.device)
    for j in range(t):
        carry = torch.ones((), dtype=a.dtype, device=a.device)
        for i in range(j, -1, -1):
            matrix[j, i] = c[j] * b[i] * carry
            carry = carry * a[i]
    return matrix
```

```python
torch.manual_seed(4)
x = torch.randn(1, 6, 1, dtype=torch.float64)
a = torch.sigmoid(torch.randn_like(x))
b, c = torch.randn_like(x), torch.randn_like(x)
y, _ = selective_scan(x, a, b, c)
matrix = scalar_scan_matrix(a[0, :, 0], b[0, :, 0], c[0, :, 0])
torch.testing.assert_close(y[0, :, 0], matrix @ x[0, :, 0])
```

该对照能发现 scan 方向、离散化和 chunk boundary 的 off-by-one。

## Chunked scan

chunk 之间只传最终 state：

```python
def chunked_scan(x, a, b, c, chunk_size):
    """Same semantics as selective_scan with explicit chunk boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    state, output = None, []
    for start in range(0, x.size(1), chunk_size):
        stop = start + chunk_size
        y, state = selective_scan(
            x[:, start:stop], a[:, start:stop], b[:, start:stop], c[:, start:stop], state
        )
        output.append(y)
    return torch.cat(output, dim=1), state
```

```python
full, final = selective_scan(x, a, b, c)
chunked, chunk_final = chunked_scan(x, a, b, c, chunk_size=4)
torch.testing.assert_close(full, chunked)
torch.testing.assert_close(final, chunk_final)
```

低精度下，不同 scan 树会有舍入差异；应分别检查短序列 reference 与真实长度稳定性。

## Delta-rule fast weight {#delta-rule-fast-weight}

状态 $S_t\in\mathbb R^{D_k\times D_v}$ 保存 key 到 value 的线性映射。先读预测，再用误差写入：

$$
\hat v_t=k_t^\top S_{t-1},\qquad
S_t=S_{t-1}
+\beta_t k_t(v_t-\hat v_t)^\top.
$$

```python
def delta_memory(keys, values, beta, initial=None):
    """keys:[B,T,Dk], values:[B,T,Dv], beta:[B,T,1] -> reads, final state."""
    b, _, dk = keys.shape
    dv = values.size(-1)
    state = values.new_zeros(b, dk, dv) if initial is None else initial
    reads = []
    for step in range(keys.size(1)):
        key = keys[:, step]
        prediction = torch.einsum("bd,bdv->bv", key, state)
        error = values[:, step] - prediction
        state = state + beta[:, step, :, None] * key[:, :, None] * error[:, None, :]
        reads.append(prediction)
    return torch.stack(reads, 1), state
```

key 的范数、$\beta$ 与 state dtype 决定稳定性。固定维度 state 在大量相似 key 下会发生干扰，不能等同于精确 KV cache。

## Transformer-XL segment memory {#transformer-xl-segment-memory}

段级记忆保存过去 hidden state，但切断跨段梯度：

```python
def update_segment_memory(memory, hidden, limit):
    """memory:[B,M,H]|None, hidden:[B,T,H] -> detached [B,min(M+T,limit),H]."""
    if limit < 0:
        raise ValueError("memory limit must be non-negative")
    history = hidden if memory is None else torch.cat((memory, hidden), dim=1)
    return history[:, -limit:].detach() if limit else history[:, :0].detach()
```

```python
h = torch.randn(2, 5, 7, requires_grad=True)
memory = update_segment_memory(None, h, 3)
assert memory.shape == (2, 3, 7) and not memory.requires_grad
next_memory = update_segment_memory(memory, torch.randn(2, 4, 7), 3)
assert next_memory.shape == memory.shape
```

训练时 reset 的边界必须与部署请求隔离一致；否则模型可能依赖部署时不存在的跨样本状态。

## kNN probability memory

给 query 找 top-$k$ key，用距离温度形成邻居权重并按 token ID 汇总：

```python
def knn_distribution(query, keys, token_ids, vocab_size, k, temperature=1.0):
    """query:[D], keys:[N,D], token_ids:[N] -> distribution:[V]."""
    if query.ndim != 1 or keys.ndim != 2 or query.numel() != keys.size(1):
        raise ValueError("query and memory key shapes disagree")
    if not 1 <= k <= keys.size(0) or vocab_size <= 0 or temperature <= 0:
        raise ValueError("invalid memory")
    if keys.size(0) != token_ids.numel():
        raise ValueError("every key needs one token ID")
    if token_ids.numel() and (token_ids.min() < 0 or token_ids.max() >= vocab_size):
        raise ValueError("memory token ID is outside the vocabulary")
    distance = (keys.float() - query.float()).square().sum(-1)
    value, index = distance.topk(min(k, keys.size(0)), largest=False)
    weight = (-value / temperature).softmax(0)
    probability = torch.zeros(vocab_size, device=keys.device)
    probability.scatter_add_(0, token_ids[index], weight)
    return probability

def interpolate_distribution(model_prob, memory_prob, weight):
    if not 0 <= weight <= 1:
        raise ValueError("interpolation weight must be in [0,1]")
    return weight * model_prob + (1 - weight) * memory_prob
```

memory key、tokenizer、model version 和权限域必须兼容。旧记忆或投毒检索会直接改变输出概率。

## Associative-recall 压力测试

```python
def associative_recall_batch(batch, pairs, vocab, generator=None):
    """Return key/value stream and query keys for exact-copy evaluation."""
    if pairs <= 0 or vocab < 2 * pairs:
        raise ValueError("vocab must support distinct keys and values")
    stream, query, answer = [], [], []
    for _ in range(batch):
        permutation = torch.randperm(vocab, generator=generator)
        keys, values = permutation[:pairs], permutation[pairs:2 * pairs]
        target = int(torch.randint(pairs, (), generator=generator))
        stream.append(torch.stack((keys, values), dim=1).flatten())
        query.append(keys[target])
        answer.append(values[target])
    return torch.stack(stream), torch.stack(query), torch.stack(answer)
```

扫描 pair 数、查询距离、key 冲突和 state dimension，才能观察固定状态的容量边界。[Zoology](https://arxiv.org/abs/2312.04927)系统研究了长程 recall 任务中的架构差异。

## 验证边界

- recurrent、chunked 与显式矩阵在 FP64 对齐；
- state 在 request、batch 与文档边界正确 reset；
- state dtype 和序列长度的误差曲线；
- copy、associative recall、聚合与自然语言任务分别评测；
- 理论 $O(T)$ 与真实短/长 shape wall-clock 分别报告；
- hybrid attention 的局部精确路径与递推路径分别消融。

架构推导见[状态空间与线性注意力](../architecture/state-space-linear-attention.md)与[记忆架构](../architecture/memory-architectures.md)。

## Reference {#reference}

- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
- [Generalization through Memorization: Nearest Neighbor Language Models](https://arxiv.org/abs/1911.00172)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Zoology](https://arxiv.org/abs/2312.04927)
- [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620)
