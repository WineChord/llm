# CLIP：把自然语言变成视觉接口

在 CLIP 之前，用文本监督视觉并不是新想法；真正改变后续路线的是规模、目标和接口同时对齐。固定分类标签被替换为自然语言描述，训练目标被写成 batch 内图文匹配，推理时的分类器则由文本 prompt 临时生成。[原论文](https://arxiv.org/abs/2103.00020)与[官方实现](https://github.com/openai/CLIP)共同展示了这一闭环。

## 它接住了什么问题

监督分类把“猫”“X 光片异常”或“卫星图里的港口”都压进预先定义的类别表。表外概念必须重新标注和训练。自然语言监督提供了更开放的概念空间，但早期实验的数据规模不足，难以与强监督视觉模型竞争。CLIP 使用约 4 亿图文对，把这个问题从小数据的表示实验推到大规模预训练。

关键不是模型第一次“理解文本”，而是文本 encoder 在推理时仍然参与计算。类别名和描述经过编码后成为零样本分类权重，任务接口由语言定义。

## 对称对比目标

设归一化图像表示 $u_i\in\mathbb R^d$、文本表示 $v_j\in\mathbb R^d$，温度为 $\tau$：

$$
s_{ij}=\frac{u_i^\top v_j}{\tau}.
$$

一个 batch 内，第 $i$ 个图像只与第 $i$ 个文本配对。损失同时做 image-to-text 与 text-to-image 分类：

$$
\mathcal L_{\text{CLIP}}
=\frac12\left[
\frac1B\sum_i-\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}}
+
\frac1B\sum_j-\log\frac{e^{s_{jj}}}{\sum_i e^{s_{ij}}}
\right].
$$

对称性很重要：只做一个方向，另一侧多个样本之间的竞争关系没有被同样约束。batch size 也不是普通实现细节；batch 内其他配对构成 negatives，更大的全局 batch 改变了目标本身。

```python
import torch
import torch.nn.functional as F
def clip_loss(image, text, logit_scale):
    image = F.normalize(image, dim=-1)
    text = F.normalize(text, dim=-1)
    logits = logit_scale.exp() * image @ text.T
    target = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2
torch.manual_seed(0)
image = torch.randn(4, 8, requires_grad=True)
text = torch.randn(4, 8, requires_grad=True)
scale = torch.tensor(1.0, requires_grad=True)
loss = clip_loss(image, text, scale)
loss.backward()
assert loss.ndim == 0 and image.grad.shape == image.shape
```

这段 reference 没有实现分布式 all-gather。真实训练若每个 rank 只把本地样本当 negatives，目标等价于更小 batch；若 gather 后错误地切断需要的梯度，又会改变 encoder 更新。

## 零样本分类不是“直接写类别名”

对类别 $c$，先把若干模板 $t_k(c)$ 编码并归一化聚合：

$$
w_c=\operatorname{normalize}\left(\frac1K\sum_k
\operatorname{normalize}(f_T(t_k(c)))\right).
$$

图像特征与所有 $w_c$ 比较得到分类 logits。prompt engineering 在这里不是聊天技巧，而是把类别映射到预训练时更常见的语言分布。模板集、类别同义词和语言都会改变结果，应作为评测协议的一部分。

## 它没有解决什么

CLIP 的目标奖励全局语义匹配，不要求精确计数、局部定位或关系推理。数据来自互联网，也会继承长尾不足、文字捷径与社会偏差。零样本迁移强并不意味着对分布外细粒度任务可靠；线性 probe、few-shot 与 fully supervised 结果也不能混为一类。

后来 [ALIGN](https://arxiv.org/abs/2102.05918)继续探索噪声图文数据的规模，[SigLIP](https://arxiv.org/abs/2303.15343)把 batch softmax 改为独立 sigmoid pair loss，缓解全局 negative 归一化的部分约束。这些变化都应回到“正负样本怎样定义、跨设备怎样归一化”来理解。

## 为什么它成为多模态基础件

CLIP 同时给出了视觉表示、文本表示和二者可比较的坐标系。后来的生成模型用它做文本条件或质量信号，视觉语言模型把其视觉 encoder 接到 LLM，开放词表检测和分割也借用相同思想。它的影响并非一种固定架构，而是一种接口：语言可以在推理时描述视觉任务。

接下来读[冻结模型之间怎样架桥](visual-language-bridges.md)，会看到相似度空间怎样进一步变成可生成、可对话的视觉上下文；完整机制和训练陷阱见[视觉语言模型](../../multimodal/vision-language.md)与[多模态手撕实现](../../practice/multimodal.md)。

## Reference {#reference}

- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [openai/CLIP](https://github.com/openai/CLIP)
- [ALIGN](https://arxiv.org/abs/2102.05918)
- [SigLIP](https://arxiv.org/abs/2303.15343)
