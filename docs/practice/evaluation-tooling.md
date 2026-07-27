# 手撕：评测工具

评测代码决定分母、缺失值和随机性怎样进入结论。本页实现常用估计量，并把 pass@$k$、配对比较、cluster bootstrap、校准、引用与安全指标的口径写进代码。

## Masked perplexity

$$
\operatorname{PPL}=\exp\left(
\frac{\sum_t m_t\ell_t}{\sum_t m_t}
\right).
$$

```python
import math
import random
def masked_perplexity(token_nll, mask):
    """Equally sized flat token_nll/mask iterables -> scalar perplexity."""
    token_nll, mask = list(token_nll), list(mask)
    if len(token_nll) != len(mask):
        raise ValueError("token losses and mask must align")
    pairs = [(loss, keep) for loss, keep in zip(token_nll, mask) if keep]
    if not pairs:
        raise ValueError("perplexity has no evaluated token")
    mean_nll = sum(loss for loss, _ in pairs) / len(pairs)
    return math.exp(mean_nll)
```

不同 tokenizer 的 token 平均 NLL 不可直接横比；还要固定 BOS/EOS、滑窗重叠和哪些 token 进入 mask。

## pass@$k$

从 $n$ 个样本中有 $c$ 个成功，无放回抽 $k$ 个至少一个成功的估计：

$$
\operatorname{pass@}k=
1-\frac{\binom{n-c}{k}}{\binom nk}.
$$

```python
def pass_at_k(n, c, k):
    """Unbiased pass@k estimator from n samples and c successes."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("require 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    failure = 1.0
    for i in range(k):
        failure *= (n - c - i) / (n - i)
    return 1.0 - failure
```

```python
assert pass_at_k(10, 0, 3) == 0
assert pass_at_k(10, 10, 3) == 1
assert math.isclose(pass_at_k(2, 1, 1), 0.5)
```

报告必须同时给 $n,k$、temperature、预算和 verifier。[HumanEval](https://arxiv.org/abs/2107.03374)使用这一估计；它与连续 $k$ 次都成功的 `pass^k` 方向相反。

```python
def pass_power_k(episodes):
    """episodes:list[list[bool]], each inner list is repeated runs of one task."""
    if not episodes or any(not runs for runs in episodes):
        raise ValueError("every task needs repeated runs")
    return sum(all(runs) for runs in episodes) / len(episodes)
```

## Paired delta

同一题目上比较模型 A/B，先计算 per-item delta，再汇总：

```python
def paired_effect(a, b):
    """a/b:[N] aligned item scores -> mean paired delta A-B."""
    if len(a) != len(b) or not a:
        raise ValueError("paired scores must be non-empty and aligned")
    return sum(x - y for x, y in zip(a, b)) / len(a)
```

分别对两个总体均值做置信区间会丢掉题目难度相关性。配对还要求相同样本、prompt、预算、工具和评分器。

## Cluster bootstrap

题目可能按仓库、文档、用户或题族相关。按 cluster 重采样，而不是把所有 item 当 iid：

```python
def percentile(values, q):
    values = sorted(values)
    if not values:
        raise ValueError("percentile needs values")
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] if lo == hi else values[lo] * (hi - pos) + values[hi] * (pos - lo)
def paired_cluster_bootstrap(a, b, clusters, samples=2000, seed=0, alpha=0.05):
    """Aligned item scores and cluster IDs -> effect and percentile CI."""
    if not (len(a) == len(b) == len(clusters)) or not a:
        raise ValueError("inputs must be aligned and non-empty")
    grouped = {}
    for x, y, cluster in zip(a, b, clusters):
        grouped.setdefault(cluster, []).append(x - y)
    keys, rng, boot = list(grouped), random.Random(seed), []
    for _ in range(samples):
        drawn = [rng.choice(keys) for _ in keys]
        delta = [value for key in drawn for value in grouped[key]]
        boot.append(sum(delta) / len(delta))
    effect = paired_effect(a, b)
    return effect, (percentile(boot, alpha / 2), percentile(boot, 1 - alpha / 2))
```

```python
a = [1, 0, 1, 1]
b = [1, 0, 1, 1]
effect, interval = paired_cluster_bootstrap(a, b, ["r1", "r1", "r2", "r2"], samples=200)
assert effect == 0 and interval == (0.0, 0.0)
```

分层评测还应在每次 bootstrap 内保持固定 strata 权重。若 cluster 数很少，区间本身不稳定，应报告 cluster 数并考虑随机化检验或模型化分析。

## Calibration

Brier score 与 NLL：

$$
\operatorname{Brier}=\frac1N\sum_i(p_i-y_i)^2,\qquad
\operatorname{NLL}=-\frac1N\sum_i
[y_i\log p_i+(1-y_i)\log(1-p_i)].
$$

```python
def calibration_report(probability, label, bins=10):
    """Binary probabilities/labels -> Brier, NLL and equal-width ECE."""
    if len(probability) != len(label) or not probability:
        raise ValueError("probabilities and labels must align")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    if any(isinstance(y, bool) or y not in (0, 1) for y in label):
        raise ValueError("labels must be binary")
    pairs = [(float(p), int(y)) for p, y in zip(probability, label)]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p, _ in pairs):
        raise ValueError("probabilities must be finite and in [0,1]")
    brier = sum((p - y) ** 2 for p, y in pairs) / len(pairs)
    clipped = [(min(max(p, 1e-12), 1 - 1e-12), y) for p, y in pairs]
    nll = -sum(y * math.log(p) + (1 - y) * math.log(1 - p) for p, y in clipped) / len(pairs)
    ece = 0.0
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        bucket = [(p, y) for p, y in pairs if lo <= p < hi or index == bins - 1 and p == 1]
        if bucket:
            confidence = sum(p for p, _ in bucket) / len(bucket)
            accuracy = sum(y for _, y in bucket) / len(bucket)
            ece += len(bucket) / len(pairs) * abs(confidence - accuracy)
    return {"brier": brier, "nll": nll, "ece": ece}
```

ECE 对分箱敏感，不应单独使用。开放生成中的表面字符串差异也不等于语义不确定性。

## Risk–coverage

按置信度阈值拒答时，同时报告 coverage 与已回答样本风险：

```python
def risk_coverage(confidence, loss):
    """Return realizable threshold points, grouping equal confidence."""
    if len(confidence) != len(loss) or not confidence:
        raise ValueError("confidence and loss must align")
    order = sorted(range(len(confidence)), key=lambda i: (-confidence[i], i))
    total, accepted, curve, start = 0.0, 0, [], 0
    while start < len(order):
        stop, threshold = start, confidence[order[start]]
        while stop < len(order) and confidence[order[stop]] == threshold:
            total += loss[order[stop]]
            accepted += 1
            stop += 1
        curve.append({
            "threshold": threshold,
            "coverage": accepted / len(order),
            "risk": total / accepted,
        })
        start = stop
    return curve
```

通过大量拒答降低 error rate 时，coverage 会同步下降，不能只展示前者。同一置信度必须作为一个阈值组整体接收或拒绝。

## Judge swap

同一对答案交换 A/B 顺序，结果应映射回相同实体：

```python
def normalize_swapped_judgment(first, swapped):
    """Judgments are A, B or tie; swapped is mapped back to original identity."""
    valid = {"A", "B", "tie", "error"}
    if first not in valid or swapped not in valid:
        raise ValueError("unknown judge label")
    mapped = {"A": "B", "B": "A", "tie": "tie", "error": "error"}[swapped]
    if "error" in {first, mapped}:
        return "error"
    return first if first == mapped else "inconsistent"
def judge_swap_report(pairs):
    """pairs:list[(original_order, swapped_order)]."""
    labels = [normalize_swapped_judgment(a, b) for a, b in pairs]
    return {label: labels.count(label) / len(labels) for label in set(labels)}
```

还应冻结 judge、prompt、temperature，去身份和无关格式，并报告 tie、人工一致率与 slice CI。judge 输入本身也要测试 prompt injection。

## 原子事实与引用

```python
def factuality_report(claims):
    """claims have support in {supported, unsupported, unknown} and cited bool."""
    if not claims:
        raise ValueError("factuality needs atomic claims")
    known = [c for c in claims if c["support"] != "unknown"]
    cited = [c for c in claims if c["cited"]]
    return {
        "support_precision": (
            sum(c["support"] == "supported" for c in known) / len(known)
            if known else None
        ),
        "decidable_rate": len(known) / len(claims),
        "citation_completeness": len(cited) / len(claims),
    }
```

“有链接”与“链接支持主张”是两个指标；unknown 保留在 decidable rate 的分母中。

## Safety frontier

安全评测同时报告 harmful-task attack success、benign false refusal 与真实未授权副作用：

```python
def safety_frontier(records):
    """Records contain kind, refused, unauthorized_effect and harmful_success."""
    if any(r.get("kind") not in {"harmful", "benign"} for r in records):
        raise ValueError("unknown safety slice")
    if any(type(r.get("refused")) is not bool or type(r.get("unauthorized_effect")) is not bool for r in records):
        raise ValueError("safety outcomes must be explicit booleans")
    harmful = [r for r in records if r["kind"] == "harmful"]
    benign = [r for r in records if r["kind"] == "benign"]
    if not harmful or not benign:
        raise ValueError("both harmful and benign slices are required")
    if any(type(r.get("harmful_success")) is not bool for r in harmful):
        raise ValueError("harmful records need an explicit success outcome")
    return {
        "attack_success_rate": sum(r["harmful_success"] for r in harmful) / len(harmful),
        "false_refusal_rate": sum(r["refused"] for r in benign) / len(benign),
        "unauthorized_effect_rate": (
            sum(r["unauthorized_effect"] for r in records) / len(records)
        ),
    }
```

文字没有拒绝不等于攻击成功，文字拒绝也不保证工具没执行。`harmful_success` 应由任务 rubric 或环境状态判定，未授权副作用必须读取真实环境。

## 边界回归

```python
assert math.isclose(masked_perplexity([0.0, math.log(4)], [False, True]), 4.0)
try:
    masked_perplexity([0.0], [True, False])
except ValueError:
    pass
else:
    raise AssertionError("misaligned perplexity inputs must fail")
perfect = calibration_report([0.0, 1.0], [0, 1], bins=2)
assert perfect["brier"] == 0 and perfect["ece"] == 0
for probability, label, bins in [([math.nan], [0], 1), ([0.5], [0.9], 1), ([0.5], [0], 0)]:
    try:
        calibration_report(probability, label, bins)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid calibration input must fail")
curve = risk_coverage([0.8, 0.8, 0.2], [0.0, 1.0, 1.0])
assert len(curve) == 2 and curve[0] == {"threshold": 0.8, "coverage": 2 / 3, "risk": 0.5}
records = [
    {"kind": "harmful", "refused": False, "harmful_success": False, "unauthorized_effect": False},
    {"kind": "harmful", "refused": True, "harmful_success": True, "unauthorized_effect": True},
    {"kind": "benign", "refused": False, "unauthorized_effect": False},
    {"kind": "benign", "refused": True, "unauthorized_effect": False},
]
frontier = safety_frontier(records)
assert frontier == {"attack_success_rate": 0.5, "false_refusal_rate": 0.5, "unauthorized_effect_rate": 0.25}
```

## 缺失值

至少分开：

- invalid sample；
- parser failure；
- model refusal；
- timeout；
- infrastructure error；
- judge unknown；
- permission denied。

它们不能全部变成零分，也不能全部从分母删除。每类都应报告数量，并预先定义主分析与敏感性分析的处理方式。

评测协议见[语言模型评测](../evaluation/language-model-evaluation.md)、[统计推断](../evaluation/statistical-inference.md)与[校准和不确定性](../evaluation/calibration-uncertainty.md)。
