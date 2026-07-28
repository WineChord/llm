# 手撕：多模态原语

多模态实现的核心是 token 预算、坐标、mask、模态归一化与采样过程。以下代码覆盖视觉 patch、对比学习、resampler、grounding、统一序列、VQ、diffusion、flow、音频 RVQ 与视频 tubelet。

这页负责把跨模态原语放在一起做组合验证；机制为什么成立、在系统中放在哪里，则分别由[信号与 Token 化](../multimodal/foundations/signals-tokenization.md)、[对齐与融合](../multimodal/foundations/alignment-fusion.md)、[Diffusion 与 Score](../multimodal/image-generation/diffusion-score.md)、[音频生成](../multimodal/audio/generation-streaming.md)、[视频生成](../multimodal/video/generation.md)、[世界模型](../world-models/dynamics-planning.md)与[具身策略](../embodied/state-action-policies.md)展开。

## ViT patchify

图像 $x\in\mathbb R^{B\times C\times H\times W}$，patch 边长 $P$，token 数：

$$
N=\frac HP\frac WP.
$$

```python
import torch
from torch import nn
import torch.nn.functional as F
def patchify(image, patch):
    """image:[B,C,H,W] -> patches:[B,(H/P)(W/P),C*P*P]."""
    b, c, h, w = image.shape
    if h % patch or w % patch:
        raise ValueError("image size must be divisible by patch size")
    x = image.view(b, c, h // patch, patch, w // patch, patch)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(b, -1, c * patch * patch)
def unpatchify(tokens, channels, height, width, patch):
    """Inverse of patchify for a known image shape."""
    b = tokens.size(0)
    x = tokens.view(b, height // patch, width // patch, channels, patch, patch)
    return x.permute(0, 3, 1, 4, 2, 5).reshape(b, channels, height, width)
```

```python
image = torch.arange(2 * 3 * 8 * 12).view(2, 3, 8, 12)
tokens = patchify(image, 4)
torch.testing.assert_close(unpatchify(tokens, 3, 8, 12, 4), image)
assert tokens.shape == (2, 6, 48)
```

动态分辨率还需保存 tile 顺序、原图尺寸、resize/crop 变换和二维位置；只拼 patch token 会丢失几何来源。

## CLIP 与 SigLIP {#clip-siglip}

CLIP 对 batch 内图文做对称分类：

```python
def clip_loss(image_embedding, text_embedding, logit_scale):
    """embeddings:[B,D] -> symmetric image-text contrastive loss."""
    image = F.normalize(image_embedding.float(), dim=-1)
    text = F.normalize(text_embedding.float(), dim=-1)
    logits = logit_scale.exp().clamp(max=100) * image @ text.T
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
```

[CLIP](https://arxiv.org/abs/2103.00020)的 in-batch negatives 会把重复 caption 或同义图像当假负例。[SigLIP](https://arxiv.org/abs/2303.15343)对全部图文 pair 使用独立 sigmoid loss。令 $y_{ij}=1$ 表示匹配 pair，否则 $y_{ij}=-1$：

$$
\mathcal L_{\mathrm{SigLIP}}
=
\frac{1}{B}
\sum_{i=1}^{B}
\sum_{j=1}^{B}
\operatorname{softplus}(-y_{ij}z_{ij}).
$$

```python
def siglip_loss(image_embedding, text_embedding, logit_scale, bias=0.0):
    """embeddings:[B,D] -> all-pair sigmoid loss normalized by batch size."""
    image = F.normalize(image_embedding.float(), dim=-1)
    text = F.normalize(text_embedding.float(), dim=-1)
    logits = logit_scale.exp() * image @ text.T + bias
    target = torch.eye(logits.size(0), device=logits.device).mul(2).sub(1)
    return F.softplus(-target * logits).sum() / logits.size(0)
```

```python
image = torch.eye(2)
text = torch.eye(2)
scale = torch.tensor(0.0)
logits = image @ text.T
target = torch.eye(2).mul(2).sub(1)
expected = F.softplus(-target * logits).sum() / 2
torch.testing.assert_close(siglip_loss(image, text, scale), expected)
```

分母是 image batch size $B$，不是 pair 数 $B^2$；因此每个样本增加更多 negatives 时，loss 规模也会增长。分布式实现必须明确是 local pairs 还是 all-gather 后的 global pairs，并保持同一归一口径。SigLIP 改变了对比目标，不自动解决细粒度 grounding。

## Fixed-query resampler {#fixed-query-resampler}

固定 $M$ 个可学习 query 把可变视觉 token $Z\in\mathbb R^{B\times N\times D}$ 压为 $M$ 个：

```python
class QueryResampler(nn.Module):
    def __init__(self, width, queries, heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(queries, width) / width ** 0.5)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm_q, self.norm_kv = nn.LayerNorm(width), nn.LayerNorm(width)
    def forward(self, visual, padding_mask=None):
        """visual:[B,N,D] -> [B,M,D]."""
        if padding_mask is not None:
            if padding_mask.shape != visual.shape[:2] or padding_mask.dtype != torch.bool:
                raise ValueError("padding_mask must be bool [batch, visual_tokens]")
            if padding_mask.all(dim=-1).any():
                raise ValueError("every row needs at least one visible visual token")
        query = self.query[None].expand(visual.size(0), -1, -1)
        output, _ = self.attn(
            self.norm_q(query), self.norm_kv(visual), self.norm_kv(visual),
            key_padding_mask=padding_mask, need_weights=False,
        )
        return query + output
```

```python
torch.manual_seed(0)
resampler = QueryResampler(width=8, queries=3, heads=2)
visual = torch.randn(2, 5, 8)
padding = torch.tensor([[False] * 4 + [True]] * 2)
output = resampler(visual, padding)
changed = visual.clone()
changed[:, -1] = 1_000
torch.testing.assert_close(resampler(changed, padding), output)
assert output.shape == (2, 3, 8) and torch.isfinite(output).all()
for bad in (torch.ones(2, 5), torch.ones(2, 5, dtype=torch.bool)):
    try:
        resampler(visual, bad)
    except ValueError:
        continue
    raise AssertionError("mask must be boolean and leave one visible token per row")
```

固定 $M$ 带来稳定语言侧 token 成本，也可能丢失密集小字和大量目标。应扫描分辨率、query 数与任务性能。

## 坐标离散与几何变换 {#coordinate-geometry}

```python
def quantize_box(box, width, height, bins):
    """box=(x1,y1,x2,y2) in pixels -> integer coordinates in [0,bins-1]."""
    if min(width, height) <= 0 or bins < 2:
        raise ValueError("invalid canvas or bins")
    scale = torch.tensor([width, height, width, height], dtype=torch.float64)
    value = torch.as_tensor(box, dtype=torch.float64) / scale
    return (value.clamp(0, 1) * (bins - 1)).round().long()
def dequantize_box(box, width, height, bins):
    scale = torch.tensor([width, height, width, height], dtype=torch.float64)
    return torch.as_tensor(box).double() / (bins - 1) * scale
def transform_box(box, crop_xy, resize_scale):
    """Apply crop translation then independent x/y resize scales."""
    x0, y0 = crop_xy
    sx, sy = resize_scale
    x1, y1, x2, y2 = box
    return ((x1 - x0) * sx, (y1 - y0) * sy, (x2 - x0) * sx, (y2 - y0) * sy)
```

量化误差约为一个 bin 的像素尺度。crop、pad、tile 和 resize 后必须同步变换坐标；文档和 GUI grounding 见[文档、图表与 GUI](../multimodal/document-gui-grounding.md)。

## Context–target attention mask

统一序列中，context token 彼此可见，target token 可看全部 context 与过去 target，但 context 不看 target：

```python
def context_target_mask(role):
    """role:[T], 0=context, 1=target -> allowed attention:[T,T]."""
    role = torch.as_tensor(role)
    if not torch.all((role == 0) | (role == 1)):
        raise ValueError("role must be context or target")
    query, key = role[:, None], role[None, :]
    causal = torch.arange(role.numel())[:, None] >= torch.arange(role.numel())[None, :]
    return (query == 0) & (key == 0) | (query == 1) & ((key == 0) | causal)
```

```python
mask = context_target_mask([0, 0, 1, 1])
assert mask[0].tolist() == [True, True, False, False]
assert mask[3].tolist() == [True, True, True, True]
```

具体模型若把图像 token 也作为待生成 target，mask 与采样 schedule 必须同步改变。

## 按模态归一的 loss

```python
def modality_loss(token_loss, modality, weights):
    """token_loss/modality:[B,T], weights:dict[modality_id,float]."""
    total = token_loss.new_zeros(())
    present = 0
    for kind, weight in weights.items():
        mask = modality == kind
        if mask.any():
            total = total + weight * token_loss[mask].mean()
            present += 1
    if not present:
        raise ValueError("batch has no configured modality")
    return total
```

先在各模态有效 token 内归一，再用显式 $\lambda_m$ 加权，避免图像 token 数量仅因更多而支配梯度。

## Vector quantization

对 encoder latent $z_e$ 找最近 codebook：

$$
k^*=\arg\min_k\lVert z_e-e_k\rVert_2^2,\qquad z_q=e_{k^*}.
$$

```python
def vector_quantize(latent, codebook):
    """latent:[...,D], codebook:[K,D] -> straight-through quantized, indices."""
    flat = latent.reshape(-1, latent.size(-1)).float()
    code = codebook.float()
    distance = flat.square().sum(1, keepdim=True) + code.square().sum(1) - 2 * flat @ code.T
    index = distance.argmin(-1)
    quantized = codebook[index].view_as(latent)
    straight_through = latent + (quantized - latent).detach()
    return straight_through, index.view(latent.shape[:-1]), quantized
```

commitment、codebook update 与重建 loss 仍需显式实现。监控 code usage、perplexity 与 dead code，不能只看重建 loss。[VQ-VAE](https://arxiv.org/abs/1711.00937)给出完整目标。

## DDPM forward 与 CFG

$$
x_t=\sqrt{\bar\alpha_t}x_0
+\sqrt{1-\bar\alpha_t}\epsilon.
$$

```python
def q_sample(x0, alpha_bar, noise=None):
    """x0:[B,...], alpha_bar:[B] -> noisy sample and exact noise."""
    noise = torch.randn_like(x0) if noise is None else noise
    shape = (x0.size(0),) + (1,) * (x0.ndim - 1)
    alpha = alpha_bar.to(x0).view(shape)
    return alpha.sqrt() * x0 + (1 - alpha).sqrt() * noise, noise
def classifier_free_guidance(unconditional, conditional, scale):
    if unconditional.shape != conditional.shape:
        raise ValueError("CFG predictions must align")
    return unconditional + scale * (conditional - unconditional)
```

`scale=0` 返回 unconditional，`scale=1` 返回 conditional。$\epsilon$、$x_0$ 与 $v$ 参数化不可混用；noise schedule 与 sampler 也必须配套。[DDPM](https://arxiv.org/abs/2006.11239)给出前向与反向过程。

## Flow Euler sampler

Flow matching 学习速度场 $v_\theta(x,t)$，数值积分：

```python
def euler_flow(velocity, initial, steps, start=0.0, end=1.0):
    """velocity(x,t)->dx/dt; integrate from start to end."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    x, dt = initial, (end - start) / steps
    for index in range(steps):
        time = start + index * dt
        x = x + dt * velocity(x, time)
    return x
```

```python
x0 = torch.tensor([2.0])
result = euler_flow(lambda x, t: torch.ones_like(x) * 3, x0, steps=8)
torch.testing.assert_close(result, torch.tensor([5.0]))
```

时间方向与训练 target velocity 写反会得到完全错误的采样器。[Flow Matching](https://arxiv.org/abs/2210.02747)给出相应连续路径框架。

## Residual Vector Quantization

多个 codebook 依次量化残差：

$$
r_0=z,\quad
k_m=\arg\min_k\lVert r_{m-1}-e_{m,k}\rVert^2,\quad
r_m=r_{m-1}-e_{m,k_m}.
$$

```python
def residual_vector_quantize(latent, codebooks):
    """latent:[...,D], codebooks:[M,K,D] -> reconstruction, indices:[...,M]."""
    residual = latent.float()
    reconstruction = torch.zeros_like(residual)
    indices = []
    for codebook in codebooks.float():
        flat = residual.reshape(-1, residual.size(-1))
        distance = flat.square().sum(1, keepdim=True) + codebook.square().sum(1) - 2 * flat @ codebook.T
        index = distance.argmin(-1)
        quantized = codebook[index].view_as(residual)
        reconstruction = reconstruction + quantized
        residual = residual - quantized
        indices.append(index.view(latent.shape[:-1]))
    return reconstruction.to(latent.dtype), torch.stack(indices, dim=-1)
```

音频模型还需定义多 codebook delayed pattern、帧率、streaming state 与丢包恢复。[EnCodec](https://arxiv.org/abs/2210.13438)是 RVQ 音频 codec 的重要实例。

## Video tubelet

视频 $[B,C,T,H,W]$ 按 $(P_t,P_h,P_w)$ 分块：

```python
def tubelet_patchify(video, tubelet):
    """video:[B,C,T,H,W] -> [B,(T/Pt)(H/Ph)(W/Pw),C*Pt*Ph*Pw]."""
    b, c, t, h, w = video.shape
    pt, ph, pw = tubelet
    if t % pt or h % ph or w % pw:
        raise ValueError("video shape must be divisible by tubelet")
    x = video.view(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
    return x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(
        b, -1, c * pt * ph * pw
    )
```

```python
video = torch.randn(1, 3, 8, 16, 12)
tubelets = tubelet_patchify(video, (2, 4, 3))
assert tubelets.shape == (1, 4 * 4 * 4, 3 * 2 * 4 * 3)
```

稀疏采样会漏掉短事件；视频评测应扫描帧率、时间位置和 rollout horizon，而不只报告总 token 数。

## 验证矩阵

- patch/tubelet round-trip 与非整除错误；
- dynamic resize/crop 后坐标一致；
- duplicate positives 对对比 loss 的影响；
- resampler query 数与小目标召回；
- mask 不泄漏 target；
- 模态 token 数改变时 loss 权重不漂移；
- VQ code usage、dead code 和重建上限；
- diffusion 的 $t=0/1$、参数化与 schedule；
- flow 时间方向和 step-size 收敛；
- RVQ stage 数、音频实时系数与 streaming 连续性。

概念页见[多模态融合](../multimodal/architecture-training.md)、[理解与生成统一](../multimodal/unified-understanding-generation.md)、[图像生成](../multimodal/generative-modeling.md)、[音频](../multimodal/audio-language-models.md)与[视频](../multimodal/video-world-models.md)。

## Reference {#reference}

- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [SigLIP](https://arxiv.org/abs/2303.15343)
- [VQ-VAE](https://arxiv.org/abs/1711.00937)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [EnCodec](https://arxiv.org/abs/2210.13438)
