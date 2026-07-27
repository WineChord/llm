# 手撕：推理时计算

推理时计算用更多采样、搜索或验证预算换取更高成功率。实现重点不是“让输出更长”，而是答案归一化、候选多样性、verifier 偏差、预算和提前停止。

## Self-consistency

对 $N$ 条推理轨迹提取最终答案并投票：

$$
a^*=\arg\max_a\sum_{i=1}^{N}
\mathbf 1[\operatorname{extract}(y_i)=a].
$$

```python
from collections import Counter
import math
import re
import unicodedata

def normalize_answer(answer):
    """Conservative normalization for exact-answer voting."""
    text = unicodedata.normalize("NFKC", answer).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def self_consistency(samples, extract):
    """samples:list[trajectory], extract(sample)->answer."""
    if not samples:
        raise ValueError("voting needs samples")
    answers = [normalize_answer(extract(sample)) for sample in samples]
    counts = Counter(answers)
    winner, votes = min(counts.items(), key=lambda item: (-item[1], item[0]))
    return {"answer": winner, "votes": votes, "total": len(samples), "histogram": counts}
```

```python
result = self_consistency(
    ["work... 42", "other...４２ ", "mistake...41"],
    lambda text: text.rsplit("...", 1)[-1],
)
assert result["answer"] == "42" and result["votes"] == 2
```

等价数学表达式、单位和自由文本不能只靠字符串规则合并，需要任务专用 canonicalizer 或 verifier。[Self-Consistency](https://arxiv.org/abs/2203.11171)展示了采样投票路线。

## Best-of-$N$

```python
def best_of_n(candidates, verifier):
    """verifier(candidate)->finite scalar; stable tie by original order."""
    if not candidates:
        raise ValueError("selection needs candidates")
    scored = [(float(verifier(candidate)), index, candidate)
              for index, candidate in enumerate(candidates)]
    if any(not math.isfinite(score) for score, _, _ in scored):
        raise ValueError("verifier returned non-finite score")
    score, index, candidate = max(scored, key=lambda item: (item[0], -item[1]))
    return {"candidate": candidate, "score": score, "index": index}
```

verifier 可能偏好长度、格式或自身模型家族。Best-of-$N$ 应同时报告候选生成成本、验证成本、oracle pass@$N$ 与 verifier 实际选择率。

## Verifier-guided beam

以下搜索只保存状态、累计分数和终止标志；`expand` 与 `score` 由任务定义：

```python
def verifier_beam(initial, expand, score, is_terminal, width, depth):
    """Return terminal and frontier states ordered by verifier score."""
    if width <= 0 or depth <= 0:
        raise ValueError("width and depth must be positive")
    beam, terminal = [initial], []
    for _ in range(depth):
        candidates = []
        for state in beam:
            if is_terminal(state):
                terminal.append(state)
            else:
                candidates.extend(expand(state))
        if not candidates:
            break
        unique = {}
        for state in candidates:
            key = repr(state)
            if key not in unique or score(state) > score(unique[key]):
                unique[key] = state
        beam = sorted(unique.values(), key=score, reverse=True)[:width]
    terminal.extend(state for state in beam if is_terminal(state))
    return sorted(terminal or beam, key=score, reverse=True)
```

`repr` 只是 reference 的去重键；生产实现应使用规范化状态哈希。真实副作用环境不能盲目复制搜索分支。

## PUCT selection

MCTS 中常用：

$$
a^*=\arg\max_a
\left[
Q(s,a)+c_{\mathrm{puct}}P(s,a)
\frac{\sqrt{1+\sum_bN(s,b)}}{1+N(s,a)}
\right].
$$

```python
def puct_select(q_value, prior, visits, c_puct=1.5):
    """q/prior/visits equally sized lists -> selected action index."""
    if not q_value or not (len(q_value) == len(prior) == len(visits)):
        raise ValueError("PUCT arrays must align")
    total = sum(visits)
    value = [
        q + c_puct * p * math.sqrt(total + 1) / (1 + n)
        for q, p, n in zip(q_value, prior, visits)
    ]
    return max(range(len(value)), key=lambda index: (value[index], -index))
```

这里的 $+1$ 让根节点在零访问时仍按 prior 区分动作；也可使用其他显式初始化约定。Q 的尺度、先验温度、扩展数和 terminal reward 会共同改变行为。只有 verifier 可靠且状态可复制时，树搜索才值得额外复杂度。

## 边际收益分配

给每个问题预估第 $j$ 个额外样本的边际收益 $\Delta_i(j)$，贪心分配离散预算：

```python
def allocate_budget(marginal_gain, total):
    """marginal_gain:list[list[float]] -> samples allocated per item."""
    if total < 0:
        raise ValueError("budget cannot be negative")
    allocation = [0] * len(marginal_gain)
    for _ in range(total):
        choices = [
            curve[allocation[i]] if allocation[i] < len(curve) else float("-inf")
            for i, curve in enumerate(marginal_gain)
        ]
        item = max(range(len(choices)), key=lambda i: (choices[i], -i))
        if choices[item] <= 0:
            break
        allocation[item] += 1
    return allocation
```

```python
gain = [[0.4, 0.1], [0.8, 0.3, 0.05], [0.2]]
assert allocate_budget(gain, total=4) == [1, 2, 1]
```

这是假定边际收益已知且可加的 oracle baseline。实际系统需要由 prompt 特征、早期样本分歧或 verifier uncertainty 估计难度，并防止难题永久吞掉预算。

## Consensus early stop

若当前领先答案即使剩余预算全投给第二名也不会被反超，可以安全停止离散多数投票：

```python
def majority_decided(counts, remaining):
    """counts: mapping answer->votes; tie-break is not considered decided."""
    values = sorted(counts.values(), reverse=True)
    first = values[0] if values else 0
    second = values[1] if len(values) > 1 else 0
    return first > second + remaining
```

统计置信提前停止需要额外分布假设；不要把同一模型、同一 prompt 的高度相关样本当独立 Bernoulli。

## 预算化评测

每个结果至少绑定：

- 生成样本数、平均/最大 token；
- search nodes、depth 与 branching；
- verifier 次数、版本和阈值；
- wall time、TTFT、费用与并发；
- oracle coverage、实际选择率与最终正确率；
- easy/hard slice 的预算分布；
- 失败候选、解析失败和 verifier unknown。

固定 $N$ 比较模型时仍可能计算量不同；公平比较应同时给固定 token、固定时间和固定费用等视角。

## 失效测试

- 所有样本相同，增加 $N$ 无收益；
- 等价答案被 canonicalizer 分裂；
- verifier 偏好更长但错误的候选；
- 高分候选包含评测器提示注入；
- 搜索状态去重碰撞；
- easy prompt 被过度分配；
- 提前停止在 tie 或相关样本下失效；
- 测试数据或 hidden verifier 泄漏进训练。

机制与证据见[推理时计算](../reasoning/test-time-compute.md)、[验证与搜索](../reasoning/search-verification.md)和[推理训练](../training/reasoning-posttraining.md)。
