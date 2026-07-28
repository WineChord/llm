# 解码

模型给出下一个 token 的条件分布，解码算法把这些局部分布变成完整序列。它会改变多样性、搜索成本与输出分布，但不会修复缺失知识、错误上下文或不可靠奖励。

## Logit 处理顺序

设原始 logits 为 $z$，实际采样前可能依次应用：

1. repetition/frequency/presence penalty；
2. 禁止 token 与结构 grammar；
3. temperature；
4. top-$k$、top-$p$ 或 min-$p$ 截断；
5. 重新归一化；
6. 采样与停止判断。

顺序改变结果。例如先 top-$p$ 再 temperature 与先 temperature 再 top-$p$ 的候选集合可能不同。服务必须把具体顺序和版本作为解码契约。

## Greedy

每步选择

$$
y_t=\arg\max_i p(i\mid x,y_{<t}).
$$

greedy 可复现且便宜，但局部最大不保证整条序列概率最大。早期一个略高概率 token 可能把后续带入较差分支。

## Beam search

beam search 保留 $B$ 个累计分数最高的前缀。序列 log-prob 随长度单调减小，因此常使用长度归一化：

$$
s(y)=\frac{\log p(y\mid x)}{L(y)^\alpha}.
$$

beam 适合机器翻译、语音和受约束生成等“答案分布相对集中”的任务；开放式对话中，大 beam 可能产生通用、重复和缺乏多样性的输出。实现还要处理：

- finished beam 是否继续参与 top-$B$；
- EOS 与最小/最大长度；
- KV Cache 的分支共享与 copy-on-write；
- 多个 beam 的重新排序；
- length penalty 与 early stopping；
- batch 内不同请求的 beam width。

## 随机采样

### Temperature

$$
p_i(\tau)=
\frac{\exp(z_i/\tau)}
{\sum_j\exp(z_j/\tau)}.
$$

$\tau\to0$ 接近 greedy；较大温度使分布更平。温度不能用零直接相除，通常将零温度映射为 greedy 分支。

### Top-$k$

只保留概率最大的 $k$ 个 token。固定 $k$ 不适应分布尖锐程度：某一步可能只需两个候选，另一步可能有大量合理候选。

### Top-$p$

按概率降序选择累计概率至少为 $p$ 的最小集合：

$$
\mathcal V_p
=\min\left\{
\mathcal V':\sum_{i\in\mathcal V'}p_i\ge p
\right\}.
$$

它让候选数随分布变化，但排序和截断有额外开销。截断后必须重新归一化。

## 结构化与受约束解码

JSON schema、正则或 grammar 可将非法 token 置为 $-\infty$，保证输出属于形式语言。约束状态机必须随每个请求、beam 和 speculative branch 更新。它只能保证语法，不保证字段值、引用、工具参数或业务动作正确。

若 grammar 在某状态屏蔽全部 token，应返回明确错误，而不是产生 NaN 或静默解除约束。

增量 decode 的关键正确性不变量是：同一 token prefix 上，prefill/step 路径的下一 token logits 必须与完整重算一致。下面把模型接口抽象成 `prefill`、`step` 和 `full`，并用一个累加状态的玩具核验证 harness；真实 Transformer 的 state 则是逐层 KV、position 与 processor 状态。

```python
import torch
def assert_incremental_consistency(prefill, step, full, prompt, continuation):
    prefix = list(prompt)
    state, logits = prefill(prefix)
    torch.testing.assert_close(logits, full(prefix))
    for token in continuation:
        prefix.append(token)
        state, logits = step(state, token)
        torch.testing.assert_close(logits, full(prefix))
    return logits
embedding = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
projection = torch.tensor([[1., -1., 0.], [0., 1., -1.]])
full = lambda ids: embedding[ids].sum(0) @ projection
prefill = lambda ids: (embedding[ids].sum(0), full(ids))
step = lambda state, token: (
    state + embedding[token],
    (state + embedding[token]) @ projection,
)
last = assert_incremental_consistency(prefill, step, full, [0, 1], [2, 0])
torch.testing.assert_close(last, full([0, 1, 2, 0]))
assert last.shape == (3,)
assert torch.isfinite(last).all()
```

这个 harness 不把近似量化误差自动视为正确；生产测试应为 dtype/布局设显式容差，并逐位置比较 logits 或概率。遇到 beam 重排、speculative rejection、字符串 stop 或 grammar 状态时，还要确认 KV rollback、RNG counter 与增量 detokenizer 一起恢复。

## 停止语义

停止条件可能是：

- EOS token；
- token ID 序列；
- 解码后文本字符串；
- grammar 接受态；
- 最大 token、时间或成本；
- 外部取消。

字符串 stop 可能跨 token 边界，流式输出前需要保留最长 stop 前缀的缓冲，否则客户端可能先看到本应截断的字符。Unicode 与 byte-level tokenizer 还要求增量解码器保留未完成字节。

## Speculative decoding

draft policy $q$ 连续提出若干 token，target policy $p$ 一次并行验证。对 draft token $x$，一种精确接受规则为

$$
a(x)=\min\left(1,\frac{p(x)}{q(x)}\right).
$$

若拒绝，则从残差分布

$$
p'(x)\propto\max(0,p(x)-q(x))
$$

采样。这样在假设满足时保持 target 分布不变。[Speculative Decoding](https://arxiv.org/abs/2211.17192) 的收益近似取决于接受长度与 draft/verify 成本，而不是只看 draft 模型更小。

需要验证：

- draft 与 target tokenizer/词表是否兼容；
- logit processors、grammar 与随机数是否一致；
- target 验证 batch 是否能有效并行；
- 拒绝后 KV 与 sampler state 是否正确回滚；
- 高并发时 draft 是否占用本可服务 target 的资源。

## 可复现性

固定 seed 不必然得到跨硬件、跨并行度的相同输出。并行 reduction、top-$k$ tie、异步请求顺序与不同 kernel 都可能改变边界概率。若需要严格重放，应同时固定：

```text
model and tokenizer
prompt bytes and template
logit processor order
sampling parameters
RNG algorithm, seed and per-request counter
kernel/library versions
parallel layout
stop and detokenization rules
```

KV 分支与分页见 [KV Cache](kv-cache.md)，请求级状态见[推理运行时](runtime.md)，加速选择见[加速与量化](acceleration.md)。

连续批处理、增量状态与 decode 一致性测试见[推理引擎手撕实现](../practice/inference-engine.md)。

## Reference {#reference}

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
