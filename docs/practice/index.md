# 实验方法

最有价值的最小实验能隔离一个机制，并让质量、性能与资源同时可测。

## 实验卡

每次实验记录：

- 问题与可证伪假设；
- 基线、唯一主要变量与保持不变的条件；
- 代码、配置、数据、模型和 tokenizer 版本；
- 硬件、驱动、框架、精度与随机种子；
- 指标、样本数、重复次数和停止规则；
- 原始日志、输出、失败样本与结论边界。

## 一个最小注意力实现

以下实现用于核对 shape、mask 和缩放语义，不替代优化 kernel：

```python
import math
import torch

def attention(q, k, v, causal=True):
    scale = 1 / math.sqrt(q.size(-1))
    score = q @ k.transpose(-2, -1) * scale
    if causal:
        n, m = q.size(-2), k.size(-2)
        mask = torch.ones(n, m, dtype=torch.bool, device=q.device).triu(m - n + 1)
        score = score.masked_fill(mask, float("-inf"))
    prob = score.softmax(dim=-1)
    return prob @ v
```

先在小张量上与手算比较，再与框架 reference、融合 kernel 和增量 KV 路径比较。

## 性能测试

1. 固定硬件电源与时钟策略。
2. 预热 kernel、allocator、图和缓存。
3. 明确是否包含 tokenizer、网络和后处理。
4. 固定长度分布、batch、并发与采样参数。
5. 同时报告吞吐、P50/P95/P99、显存和输出质量。
6. 保存 profiler trace，并解释关键路径而非只贴 utilization。

## 复现层级

“能运行”只证明接口可用；“指标接近”还需要相同数据与协议；“机制成立”则需要对照和消融。复现结论应明确属于哪一层。

softmax、RMSNorm、GQA、online softmax、分页映射与组相对优势的可审计 reference 见[最小实现](minimal-implementations.md)，跨层故障定位见[调试手册](debugging.md)。
