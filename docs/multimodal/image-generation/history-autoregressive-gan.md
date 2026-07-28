# 图像生成的第一条主线：似然、对抗学习与视觉 Token

今天的图像生成常被概括为 autoregressive、diffusion 或 flow。若从历史因果而非模型名称回看，真正反复出现的是三个问题：

1. <strong>怎样定义“像真实图像”</strong>：逐像素概率、判别器反馈，还是感知空间里的距离；
2. <strong>怎样把二维图像变成可计算的生成过程</strong>：按像素展开、压成离散 token，或一次并行修补一组未知位置；
3. <strong>怎样在质量、覆盖度与采样成本之间取舍</strong>。

这条线索从显式似然出发，经由 GAN 把“似然难写”改成“真假可判”，又借 VQ tokenizer 回到序列建模。后来 diffusion 与 flow 并没有让它失效：它们仍在使用 autoencoder、Transformer、adversarial perceptual loss 与离散条件接口。理解早期路线，才能看清今天系统里每个组件究竟继承了什么。

## 先区分三种概率观

设真实数据分布为 $p_{\mathrm{data}}(x)$，模型分布为 $p_\theta(x)$。

### 显式似然

模型直接给出可计算的 $p_\theta(x)$ 或其严格下界，以最大化

$$
\mathbb E_{x\sim p_{\mathrm{data}}}\log p_\theta(x).
$$

自回归模型属于这一类；normalizing flow 还要求可逆变换，使 change of variables 可精确计算。优点是训练目标清楚、可做 likelihood 比较，缺点是一个方便计算的分解未必符合图像的二维结构，而且高 likelihood 不必然等于人眼偏好的锐利样本。

### 隐式分布

生成器把简单噪声映射为图像：

$$
z\sim p(z),\qquad x=G_\theta(z).
$$

它可以采样，却未必能对任意图像求密度。[GAN](https://arxiv.org/abs/1406.2661) 用判别器学习一个可训练的分布差异，绕开显式像素似然。

### 变分下界

[VAE](https://arxiv.org/abs/1312.6114) 引入潜变量 $z$ 与近似后验 $q_\phi(z\mid x)$：

$$
\log p_\theta(x)
\ge
\mathbb E_{q_\phi(z\mid x)}\log p_\theta(x\mid z)
-
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\|p(z)\right).
$$

它仍有概率解释，但优化的是下界。后来的 VQ tokenizer、latent diffusion 和 representation autoencoder 都沿用了“先学习表示，再在表示上建模”的分层思想；区别在于 latent 是连续、离散还是高维语义特征。

## 自回归：把图像变成一句很长的话

[PixelRNN](https://arxiv.org/abs/1601.06759) 把二维像素按固定顺序展开：

$$
p(x)=\prod_{i=1}^{H}\prod_{j=1}^{W}
p(x_{i,j}\mid x_{<i,<j}).
$$

实际还需分解 RGB channel。这个目标没有近似：训练时所有位置可用 teacher forcing 并行计算 loss，采样时却必须等待前一个位置。于是出现一个关键不对称：

$$
\text{训练并行度高},\qquad
\text{生成关键路径长度}=HWC.
$$

[Image Transformer](https://arxiv.org/abs/1802.05751) 说明 attention 可以代替 recurrent/convolutional 条件器，但没有消除序列长度和扫描顺序。Raster order 给出了合法概率分解，却把本来相邻的上下像素隔开一整行，也人为规定“左上角永远先发生”。

这类模型的优势并不是“像语言模型所以先进”，而是：

- 每个 token 都有标准 categorical likelihood；
- prefix、inpainting、交错图文可落入统一序列协议；
- sampling policy 可以显式加入 top-$k$、temperature 与约束。

代价则是 exposure bias、误差逐步累积，以及 token 数增长带来的线性串行延迟。

## GAN：不再逐像素解释，而是学习怎样挑错

GAN 的原始极小极大目标为

$$
\min_G\max_D
\mathbb E_{x\sim p_{\mathrm{data}}}\log D(x)
+
\mathbb E_{z\sim p(z)}\log(1-D(G(z))).
$$

对固定 $G$，最优判别器为

$$
D^\star(x)
=
\frac{p_{\mathrm{data}}(x)}
{p_{\mathrm{data}}(x)+p_G(x)}.
$$

代回后，生成器最小化与 Jensen–Shannon divergence 相关的目标。这里的历史转折是：模型不再被要求逐像素说明“为什么这张图有高概率”，只需产生能骗过一个不断学习的 critic 的样本。人眼敏感的边缘、纹理和全局统计可以通过判别器进入训练，因此早期 GAN 样本通常比像素独立似然模型锐利。

但理论平衡不等于训练轨迹稳定。当 $D$ 很容易区分两种分布时，原始生成器目标会饱和。实践常改用 non-saturating loss：

$$
L_G
=
-\mathbb E_z\log D(G(z)).
$$

[DCGAN](https://arxiv.org/abs/1511.06434) 给出卷积架构经验，[BigGAN](https://arxiv.org/abs/1809.11096) 展示规模、类别条件和正则化可以显著提升高分辨率生成，[StyleGAN](https://arxiv.org/abs/1812.04948) 则把 latent 的不同层级注入 synthesis network，让粗结构与细节更可控。它们共同说明：GAN 的有效性来自目标、归一化、架构和数据规模的耦合，不能只复制一条 loss。

### WGAN 为什么改变了训练语言

[WGAN](https://arxiv.org/abs/1701.07875) 从“真假概率”转向 1-Lipschitz critic：

$$
W_1(p_{\mathrm{data}},p_G)
=
\sup_{\|f\|_L\le1}
\mathbb E_{p_{\mathrm{data}}}f(x)
-
\mathbb E_{p_G}f(x).
$$

当真实与生成分布支撑集几乎不重叠时，Wasserstein-1 仍能提供随样本移动而连续变化的信号。[WGAN-GP](https://arxiv.org/abs/1704.00028) 用插值点的梯度惩罚近似 Lipschitz 约束：

$$
L_{\mathrm{GP}}
=
\lambda
\mathbb E_{\hat x}
\left(\|\nabla_{\hat x}D(\hat x)\|_2-1\right)^2.
$$

它不是“让所有梯度越小越好”，而是把特定采样路径附近的范数推向 $1$。若忘记对插值样本开启梯度、把 batch 维也并入 norm，或让 fake 没有 detach，代码能运行却不再对应该目标。

## 视觉 tokenizer 让序列重新变短

逐像素自回归太慢，连续 VAE 又常在强压缩下模糊。[VQ-VAE](https://arxiv.org/abs/1711.00937) 把 encoder 输出映射到有限码本：

$$
k^\star
=
\arg\min_k \|z_e-e_k\|_2^2,\qquad
z_q=e_{k^\star}.
$$

图像先被压成 $h\times w$ 个 code，再建模

$$
p(k_{1:N}\mid c)
=
\prod_{n=1}^{N}p(k_n\mid k_{<n},c),
\qquad N=hw.
$$

若空间下采样率为 $f$，序列长度约从 $HW$ 降为 $HW/f^2$。这不是免费压缩：prior 再强，也恢复不了 tokenizer 已丢掉的小字、细线或面孔细节。

[VQGAN](https://arxiv.org/abs/2012.09841) 用 perceptual 与 adversarial loss 提高重建的感知质量，并以 Transformer 建模离散 code；[DALL·E](https://arxiv.org/abs/2102.12092) 把文本与图像 token 放入统一自回归序列；[Parti](https://arxiv.org/abs/2206.10789) 进一步验证大规模文本到图像的序列建模。这里形成了后来统一多模态模型仍在使用的接口：

$$
\text{continuous media}
\xrightarrow{\text{tokenizer}}
\text{finite vocabulary}
\xrightarrow{\text{sequence model}}
\text{tokens}.
$$

## 不一定要从左上角生成到右下角

自回归只要求一个因果偏序，并不要求 raster scan。[MaskGIT](https://arxiv.org/abs/2202.04200) 从全 mask 开始，并行预测未知 token，每轮保留高置信位置、重新 mask 其余位置。其迭代状态可以写成

$$
\mathcal M_{s+1}
=
\operatorname{SelectLowConfidence}
\left(p_\theta(z_i\mid z_{\bar{\mathcal M}_s},c),\,r_{s+1}\right),
$$

其中 $\mathcal M_s$ 是第 $s$ 轮未知位置，$r_s$ 是剩余 mask 比例。生成步数从 $N$ 降为固定迭代轮数，但它不再给出同一个标准 left-to-right likelihood；置信度校准与 remask schedule 成为算法的一部分。

[Muse](https://arxiv.org/abs/2301.00704) 沿这一路线扩展文本条件 masked generation。[VAR](https://arxiv.org/abs/2404.02905) 把“下一 token”改写为“下一尺度”，先生成低分辨率布局，再补更高分辨率细节。[MAR](https://arxiv.org/abs/2406.11838) 则表明自回归顺序并不必绑定离散码本，可以在连续表示上使用 diffusion loss。它们共同揭示一个更一般的设计空间：

> 自回归规定的是信息何时可见；token 的类型、分组方式和每一步内部采用的生成目标仍可独立选择。

## 两个最容易写错的核心

下面代码只实现两个语义原子：二维 raster causal mask，以及 non-saturating GAN loss 与 WGAN-GP。约定 attention mask 中 `True` 表示<strong>禁止读取</strong>；critic 输出 shape 为 `[batch]` 或 `[batch, 1]`。

```python
import torch
import torch.nn.functional as F
def raster_causal_mask(height, width, device=None):
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    n = height * width
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), 1)
def nonsaturating_gan_losses(real_logits, fake_logits):
    if real_logits.shape != fake_logits.shape:
        raise ValueError("real and fake batches must align")
    d = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
    g = F.softplus(-fake_logits).mean()
    return d, g
def gradient_penalty(critic, real, fake):
    if real.shape != fake.shape or real.ndim < 2:
        raise ValueError("real and fake must share [batch, ...]")
    shape = [real.size(0)] + [1] * (real.ndim - 1)
    alpha = torch.rand(shape, device=real.device)
    mixed = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    score = critic(mixed).reshape(real.size(0), -1).mean(1)
    grad, = torch.autograd.grad(score.sum(), mixed, create_graph=True)
    return (grad.flatten(1).norm(2, dim=1) - 1).square().mean()
mask = raster_causal_mask(2, 2)
assert mask.shape == (4, 4) and not mask.diag().any()
assert mask[0, 1] and not mask[3, 0]
real, fake = torch.tensor([2., 1.]), torch.tensor([-1., -2.])
d_loss, g_loss = nonsaturating_gan_losses(real, fake)
assert d_loss > 0 and g_loss > 0
linear = torch.nn.Linear(3, 1, bias=False)
gp = gradient_penalty(linear, torch.randn(4, 3), torch.randn(4, 3))
assert gp.ndim == 0 and torch.isfinite(gp)
```

`mask[i,j]=True` 表示第 $i$ 个 query 不可看未来 key $j$；如果所用 attention API 采用相反语义，必须在接口处转换。GAN 训练还要分别执行 discriminator 与 generator optimizer step：更新 $D$ 时 fake 应从生成图 detach，更新 $G$ 时冻结 $D$ 参数但不能切断 $D(G(z))$ 对 $G$ 的输入梯度。

## 实现契约

| 组件 | 必须固定的契约 |
| --- | --- |
| 像素 AR | channel 分解、扫描顺序、离散化、padding、BOS/EOS |
| Token AR | tokenizer 版本、码本大小、空间 layout、特殊 token、temperature |
| Masked generation | mask ratio 分布、迭代轮数、置信度、remask schedule、最终兜底 |
| GAN | loss 变体、更新比、正则化间隔、EMA、real/fake 预处理 |
| WGAN-GP | critic 无 sigmoid、插值分布、gradient norm 维度、$\lambda$ |
| 条件生成 | 条件 dropout、文本编码器版本、空条件、CFG 或 truncation |

最关键的边界是：<strong>tokenizer、prior 与 decoder 是一个版本化整体</strong>。只替换其中一部分，即使 tensor shape 相同，也可能因为 latent scale、code id 或归一化不同而静默失效。

## 失败模式：锐利不等于覆盖

### Mode collapse

生成器集中到少数能骗过判别器的模式。样本可能漂亮，但 prompt、姿态和背景多样性不足。应做 fixed-noise trajectory、nearest-neighbor、类条件覆盖与 pairwise diversity，而不是只挑最好看的网格。

### Mode dropping 与似然目标的另一面

Maximum likelihood 倾向覆盖数据中的多种模式，却可能把不确定性平均成模糊；adversarial loss 倾向感知锐利，却可能遗漏低频模式。这不是两条绝对定律，而是目标施加的压力不同。VQGAN 等混合目标正是在两者间设权重。

### 顺序与 exposure bias

自回归训练看到真实 prefix，推理看到自己的 prefix。一个早期结构错误会改变后续所有条件。随机扫描、block order、masked refinement 和重打分只能改变误差传播方式，不能自动消除分布偏移。

### Tokenizer ceiling

若 reconstruction 已经丢文字、手指或细粒度纹理，prior 的 FID 再好也不能说明生成模型学会了这些能力。必须先报告 tokenizer-only reconstruction，再报告 prior sample。

## 评测：把“像”“全”“听话”分开

| 问题 | 代表测量 | 常见误读 |
| --- | --- | --- |
| 样本是否接近真实分布 | FID/KID | 单一 embedding 不能覆盖文字与计数 |
| 是否覆盖多样性 | precision/recall、类覆盖、pair diversity | 好看的 best-of-$N$ 会隐藏 mode dropping |
| 是否遵循条件 | compositional benchmark、人评、VQA-based score | 自动问答器自身可能有偏差 |
| tokenizer 是否保真 | PSNR/SSIM/LPIPS、OCR、identity | 感知锐利可能牺牲逐像素保真 |
| 代价如何 | 顺序步数、吞吐、首样本延迟、峰值显存 | 参数量不能代表串行关键路径 |

公平比较必须固定分辨率、数据、prompt、采样数、reranking、truncation/temperature 与是否使用外部 caption rewrite。GAN 的一次前向、AR 的 $N$ 次解码和 masked 模型的若干轮 refinement，应同时报告端到端延迟与硬件，而不是只比较 FLOPs。

对视觉表示与压缩的下一层分析见 [Autoencoder 与视觉 Tokenizer](autoencoders-tokenizers.md)；diffusion 的概率路径见 [Diffusion 与 Score](diffusion-score.md)；控制、编辑与组合评测见[可控生成、编辑与评测](control-editing-evaluation.md)。

离散视觉 token 与生成原语的组合测试见[多模态手撕实现](../../practice/multimodal.md)。

## Reference {#reference}

- [Kingma and Welling, Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Goodfellow et al., Generative Adversarial Nets](https://arxiv.org/abs/1406.2661)
- [Radford et al., Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- [van den Oord et al., Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759)
- [Arjovsky et al., Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [Gulrajani et al., Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- [van den Oord et al., Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
- [Parmar et al., Image Transformer](https://arxiv.org/abs/1802.05751)
- [Brock et al., Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)
- [Karras et al., A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)
- [Esser et al., Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841)
- [Ramesh et al., Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092)
- [Chang et al., MaskGIT: Masked Generative Image Transformer](https://arxiv.org/abs/2202.04200)
- [Yu et al., Scaling Autoregressive Models for Content-Rich Text-to-Image Generation](https://arxiv.org/abs/2206.10789)
- [Chang et al., Muse: Text-To-Image Generation via Masked Generative Transformers](https://arxiv.org/abs/2301.00704)
- [Tian et al., Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905)
- [Li et al., Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838)
