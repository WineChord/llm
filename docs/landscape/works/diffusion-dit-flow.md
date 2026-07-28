# 从 DDPM 到 DiT 与 Flow：生成模型如何换一种提问

自回归模型把生成写成“下一个符号是什么”，GAN 把它写成生成器与判别器的博弈。扩散模型的关键转折，是把复杂的一步生成拆成许多局部去噪问题：如果每个噪声尺度上都能判断怎样回到数据分布，采样就可以从高斯噪声逐步走向样本。后来 Latent Diffusion、DiT 与 Flow Matching 的演进，都在重新安排表示空间、去噪网络和概率路径，而不是简单替换 backbone。

## DDPM：学习逆转一个已知破坏过程

[DDPM](https://arxiv.org/abs/2006.11239) 定义前向马尔可夫链：

$$
q(x_t\mid x_{t-1})
=\mathcal N(\sqrt{1-\beta_t}x_{t-1},\beta_t I).
$$

令 $\alpha_t=1-\beta_t$、$\bar\alpha_t=\prod_{s=1}^t\alpha_s$，可直接从干净样本采任意时刻：

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

常用训练目标让网络预测这次加入的噪声：

$$
\mathcal L_\epsilon
=\mathbb E_{x_0,t,\epsilon}
\left\|\epsilon-\epsilon_\theta(x_t,t,c)\right\|_2^2.
$$

```python
import torch
def q_sample(x0, alpha_bar, t, noise):
    a = alpha_bar[t].view(-1, *([1] * (x0.ndim - 1)))
    return a.sqrt() * x0 + (1 - a).sqrt() * noise
torch.manual_seed(0)
x0 = torch.randn(3, 4)
noise = torch.randn_like(x0)
alpha_bar = torch.tensor([1.0, 0.81, 0.49, 0.16])
t = torch.tensor([0, 1, 3])
xt = q_sample(x0, alpha_bar, t, noise)
assert torch.allclose(xt[0], x0[0])
assert torch.allclose(xt[1], 0.9 * x0[1] + alpha_bar[1].sub(1).neg().sqrt() * noise[1])
```

这个 reference 固定了 closed-form 加噪。训练中最容易错的并非公式本身，而是 timestep dtype、broadcast 轴、$\epsilon/x_0/v$ 参数化之间的转换，以及 loss 对空间与 batch 的归一化。

## Latent Diffusion：先决定在哪里生成

像素空间的长宽直接决定计算量。[Latent Diffusion](https://arxiv.org/abs/2112.10752) 先训练 autoencoder：

$$
z=E(x),\qquad \hat x=D(z),
$$

再对 $z$ 做扩散。压缩比越大，denoiser 越便宜，但 autoencoder 丢掉的细节无法由扩散过程凭空恢复。于是生成质量被拆成两个问题：表示是否保真，latent prior 是否建模准确。

文本条件通常通过 cross-attention 注入。classifier-free guidance 在推理时组合有条件和无条件预测：

$$
\hat\epsilon
=\epsilon_\theta(x_t,t,\varnothing)
+w\left[
\epsilon_\theta(x_t,t,c)-\epsilon_\theta(x_t,t,\varnothing)
\right].
$$

$w$ 增大通常加强条件一致性，也可能牺牲多样性、饱和颜色或放大伪影。它是采样器的一部分，不能只记录训练 checkpoint。

## DiT：Transformer 成为去噪 backbone

[DiT](https://arxiv.org/abs/2212.09748) 把 latent 切成 patch token，用 Transformer 替代 U-Net，并通过 timestep 与类别条件调制 block。论文在 ImageNet latent diffusion 和所测试的 DiT 家族中观察到 FID 随模型计算量增加而改善的趋势，但这并没有把扩散变成自回归，也不能脱离数据、训练与采样配方外推：

- token 轴表示空间 patch，不是已生成前缀；
- attention 通常是双向的，没有 causal mask；
- 每个采样步都重新处理整张 latent；
- 训练仍然回归噪声、速度或数据，而不是 next token。

因此，DiT 的系统成本要乘采样步数。单次 forward 的 FLOPs 降低，不保证端到端生成延迟降低。

## Flow Matching：直接学习速度场

[Flow Matching](https://arxiv.org/abs/2210.02747) 选择从噪声分布到数据分布的概率路径 $p_t$，训练向量场 $v_\theta(x,t)$ 匹配条件速度 $u_t$：

$$
\mathcal L_{\text{FM}}
=\mathbb E_{t,x_t}\|v_\theta(x_t,t)-u_t(x_t)\|_2^2.
$$

若约定 $x_0$ 来自噪声分布、$x_1$ 来自数据分布，线性耦合 $x_t=(1-t)x_0+tx_1$ 的条件速度为 $u_t=x_1-x_0$；交换端点会同时反转时间方向和速度符号。推理时求解 ODE：

$$
\frac{dx_t}{dt}=v_\theta(x_t,t).
$$

Flow Matching 把“噪声 schedule”推广为“概率路径与速度场”，但采样误差仍取决于路径曲率、solver 和步长。[Rectified Flow](https://arxiv.org/abs/2209.03003) 进一步研究怎样让路径更直，以更少数值步逼近。

## 这条线真正改变了什么

从 DDPM 到 DiT 与 Flow，三个选择逐渐被解耦：

1. **表示**：像素、连续 latent 还是离散 token；
2. **模型**：U-Net、Transformer 或混合网络；
3. **路径**：离散扩散、概率流 ODE 或其他插值。

新工作若只宣布换用 Transformer，却没有说明表示压缩、参数化、采样器和等算力比较，就很难知道收益来自哪里。更完整的采样公式、离散表示和验证矩阵见[多模态生成模型](../../multimodal/generative-modeling.md)，理解与生成怎样合流见[多模态技术谱系](../lineages/multimodal-generation.md)。

## Reference {#reference}

- [DDPM 论文](https://arxiv.org/abs/2006.11239)与 [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion)；
- [Latent Diffusion 论文](https://arxiv.org/abs/2112.10752)与 [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)；
- [DiT 论文](https://arxiv.org/abs/2212.09748)与 [facebookresearch/DiT](https://github.com/facebookresearch/DiT)；
- [Flow Matching 论文](https://arxiv.org/abs/2210.02747)；后续的 [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) 提供了通用 [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching)，它不是 2022 原论文实验代码。
