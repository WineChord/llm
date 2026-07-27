# Flamingo、BLIP-2 与 LLaVA：复用预训练组件怎样架桥

当强视觉 encoder 与强语言模型分别出现后，一个诱人的问题是：能否复用两端，只训练必要的连接或适配部分？2022—2023 年的 Flamingo、BLIP-2 与 LLaVA 给出了三种影响深远但经常被混为一谈的答案。它们共同使用预训练组件，却针对交错上下文、模态对齐与视觉指令跟随选择了不同的数据、冻结策略和计算路径。

## Flamingo：让视觉条件反复进入语言层

[Flamingo](https://arxiv.org/abs/2204.14198)先把视觉特征送进 Perceiver Resampler，用固定数量 latent queries 吸收可变数量的图像或视频 token；随后在冻结语言模型的若干层之间插入 gated cross-attention。门控从接近零开始，使新分支在训练初期不会突然破坏原语言模型。

这套设计服务于交错图文与 few-shot prompting：视觉条件不是只在输入端投影一次，而能在语言深层反复被读取。代价是桥接计算贯穿多层，KV、mask 与 media position 都需要单独管理。

## BLIP-2：先让查询学会“看什么”

[BLIP-2](https://arxiv.org/abs/2301.12597)冻结图像 encoder 和 LLM，中间放置 Q-Former。learned queries 通过 cross-attention 从视觉特征中提取有限数量的信息。训练分两阶段：先建立视觉—语言表示关系，再把 query 输出接到 LLM 做生成。

这里的 bottleneck 是有意设计的。固定数量 query 控制送入 LLM 的视觉 token 数，却可能压掉 OCR、小目标和密集空间细节。query 数量不是越少越好；它同时决定成本上限与信息容量。

## LLaVA：简单 projector 与指令数据

[LLaVA](https://arxiv.org/abs/2304.08485)使用视觉 encoder、线性 projector 与语言模型，把视觉特征映射为语言 embedding 序列。它的两阶段边界很重要：feature alignment 阶段冻结视觉 encoder 与 LLM，只训练 projector；视觉指令微调阶段继续冻结视觉 encoder，但同时更新 projector 与 LLM。[LLaVA 1.5](https://arxiv.org/abs/2310.03744)进一步显示，MLP projector、更高视觉分辨率和更合适的数据混合可以显著加强这一简单基线。

这条路线的启发不应被简化成“复杂桥接无用”。LLaVA 的目标是可访问的视觉对话模型；Flamingo 的交错 few-shot 场景和 BLIP-2 的冻结两端预训练约束并不相同。架构结论必须与数据、冻结策略和任务一起读。

## 一个最小 cross-attention bridge

下面的实现保留三个关键对象：learned queries、视觉 cross-attention 和固定长度输出。它更接近 BLIP-2/Perceiver bridge 的共同骨架，不复刻任一完整模型。

```python
import torch
from torch import nn
class VisualBridge(nn.Module):
    def __init__(self, d, q, heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(q, d) / d**0.5)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
    def forward(self, visual, pad=None):
        b = visual.size(0)
        query = self.query.expand(b, -1, -1)
        out, weight = self.attn(query, visual, visual, key_padding_mask=pad)
        return self.norm(query + out), weight
torch.manual_seed(0)
bridge = VisualBridge(d=16, q=4, heads=4)
visual = torch.randn(2, 9, 16)
tokens, weight = bridge(visual)
assert tokens.shape == (2, 4, 16)
assert weight.shape == (2, 4, 9)
assert torch.allclose(weight.sum(-1), torch.ones(2, 4), atol=1e-6)
```

真实系统还需决定视觉位置、多个媒体片段的 causal mask、projector 到 LLM embedding 维度的映射，以及 bridge 输出是否参与语言 loss。上面的断言只固定注意力归一化和 shape，不代表它保留了所有视觉信息。

## 三种桥怎样比较

| 问题 | Flamingo | BLIP-2 | LLaVA |
| --- | --- | --- | --- |
| 主要目标 | 交错图文 few-shot | 冻结两端的高效预训练 | 视觉指令跟随 |
| 视觉压缩 | Perceiver latent | Q-Former query | patch token 经 projector |
| 注入位置 | 多层 gated cross-attention | query 接入 LLM | 输入 embedding 序列 |
| 训练重点 | 大规模交错语料 | 两阶段桥接目标 | 对齐与指令数据 |
| 典型风险 | 深层注入成本与 mask | query bottleneck | 高分辨率 token 成本 |

真正的比较应固定视觉 encoder、LLM、分辨率、数据和训练预算。只比较 trainable parameter 数量，会遗漏冻结组件的前置训练成本。

## 从桥接走向统一模型

这些工作在各自的数据、冻结策略和任务中展示了“复用强单模态组件”这条路线的可行性与竞争力，也暴露了桥接的长期张力：视觉信息必须适配语言 token 接口，而语言模型未必保留空间结构。后来的多分辨率切片、二维位置编码、视觉 token pruning 和原生多模态训练都在缓解这一问题。

这段历史的前一站是[CLIP](clip.md)，后一站是[统一理解与生成](../../multimodal/unified-understanding-generation.md)。训练计算图、冻结策略与数据阶段见[融合与训练](../../multimodal/architecture-training.md)，完整谱系见[从“看懂”到“生成”](../lineages/multimodal-generation.md)。
