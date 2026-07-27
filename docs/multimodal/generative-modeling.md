# 图像生成：离散 Token、Diffusion 与 Flow

图像生成模型需要把简单先验分布或离散序列变成高维像素。主要路线分别建模：

- 离散视觉 token 的条件概率；
- 加噪过程的反向去噪；
- 噪声到数据之间的连续向量场。

三者的训练目标、采样路径和失败模式不同，不能只按最终样例比较。

[DDPM、DiT 与 Flow Matching](../landscape/works/diffusion-dit-flow.md)把去噪目标、Transformer backbone 和连续向量场放在同一组符号下比较；它们与视觉语言理解路线的汇合位置见[多模态理解与生成](../landscape/lineages/multimodal-generation.md)。

## VQ 表示

Encoder 产生连续 latent：

$$
z_e=E(x).
$$

对每个位置选择最近码本向量：

$$
k^\star
=
\arg\min_j\|z_e-e_j\|_2^2,
\qquad
z_q=e_{k^\star}.
$$

[VQ-VAE](https://arxiv.org/abs/1711.00937)的典型目标包含重建、码本和 commitment：

$$
L
=
L_{\mathrm{recon}}
+
\|\operatorname{sg}(z_e)-e\|_2^2
+
\beta
\|z_e-\operatorname{sg}(e)\|_2^2.
$$

Straight-through 路径可写为

$$
z_{\mathrm{st}}
=
z_e+\operatorname{sg}(z_q-z_e),
$$

前向使用 $z_q$，反向把 decoder 梯度传给 encoder。

[VQGAN](https://arxiv.org/abs/2012.09841)加入感知与对抗目标，提高重建的感知质量。重建更锐利不一定意味着码本更适合语义建模；tokenizer 需同时评估 reconstruction、code usage 与下游生成。

## 离散自回归与 masked generation

量化后可建模

$$
p(z\mid c)
=
\prod_{t=1}^{N}
p(z_t\mid z_{<t},c).
$$

优点是与语言模型、交错图文和受约束解码统一；缺点是生成步数随视觉 token 数增长，且 raster 顺序对二维局部性不友好。

[MaskGIT](https://arxiv.org/abs/2202.04200)并行预测被 mask 的图像 token，并迭代保留高置信位置。迭代 masked generation 需要明确每步 remask schedule、置信度和最终未填位置处理。

## Diffusion 前向过程

[DDPM](https://arxiv.org/abs/2006.11239)定义逐步加噪。记

$$
\alpha_t=1-\beta_t,
\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s,
$$

则可以直接采样任意时刻：

$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

噪声预测目标：

$$
L_\epsilon
=
\mathbb E_{x_0,t,\epsilon}
\left[
w(t)
\|\epsilon-\epsilon_\theta(x_t,t,c)\|_2^2
\right].
$$

模型也可预测 $x_0$、score 或 velocity。Prediction type、noise schedule 与 sampler 必须成套匹配。

## DDIM 与采样

[DDIM](https://arxiv.org/abs/2010.02502)构造与 DDPM 共享训练边缘分布的非马尔可夫采样路径，可减少采样步数并控制随机性。少步不自动保持质量；时间步子集、参数化和 discretization 都会改变误差。

## Latent diffusion

[Latent Diffusion](https://arxiv.org/abs/2112.10752)先用 autoencoder 压缩图像：

$$
z=E(x),
\qquad
\hat x=D(z),
$$

再在 latent $z$ 上扩散。空间压缩降低计算，但 decoder 重建误差构成上限。实现必须固定 autoencoder 版本、latent scale、通道布局和像素归一化。

[DiT](https://arxiv.org/abs/2212.09748)使用 Transformer 处理 latent patches，并通过时间和条件调制 block。其“Transformer”是去噪 backbone，不等于自回归语言模型。

## Conditioning

常见条件接口包括：

- cross-attention；
- AdaLN/FiLM 尺度与偏置；
- 条件 token 拼接；
- control encoder；
- classifier-free guidance。

Classifier-free guidance：

$$
\hat\epsilon
=
\epsilon_{\varnothing}
+
w(\epsilon_c-\epsilon_{\varnothing}).
$$

$w$ 提高通常增强条件遵循，但可能降低多样性、产生过饱和和伪影。训练时条件 dropout、无条件分支定义与 batch 拼接顺序必须和推理一致。

## Flow matching

连续流满足

$$
\frac{dx_t}{dt}=v_\theta(x_t,t,c).
$$

[Flow Matching](https://arxiv.org/abs/2210.02747)直接回归选定概率路径的条件向量场。对简单线性插值：

$$
x_t=(1-t)x_0+tx_1,
\qquad
u_t=x_1-x_0.
$$

目标：

$$
L_{\mathrm{FM}}
=
\mathbb E
\|v_\theta(x_t,t,c)-u_t\|_2^2.
$$

[Rectified Flow](https://arxiv.org/abs/2209.03003)研究把耦合路径变直以减少积分误差。训练不需要完整解 ODE，推理仍需 solver：

$$
x_{t+\Delta t}
=
x_t+\Delta t\,v_\theta(x_t,t,c)
$$

是最简单 Euler 步。时间方向、边界分布和步长写反会生成完全错误的过程。

## 少步与一致性

[Consistency Models](https://arxiv.org/abs/2303.01469)学习同一概率流轨迹上不同点映射到一致输出，可支持一步或少步生成。应分别报告：

- 是否从预训练 diffusion 蒸馏；
- 一步、少步和多步质量；
- 采样成本与训练额外成本；
- 新分布和强 guidance 下的稳定性。

## Shape 与实现契约

图像 latent 常写为

$$
z\in\mathbb R^{B\times C\times H'\times W'}.
$$

DiT patchify 后：

$$
z_{\mathrm{seq}}
\in
\mathbb R^{B\times N\times d},
\qquad
N=\frac{H'}P\frac{W'}P.
$$

实现应固定：

1. 像素范围与 channel order；
2. autoencoder latent scale；
3. $\beta_t,\alpha_t,\bar\alpha_t$ 的索引约定；
4. prediction type；
5. 时间 $t$ 的方向和离散网格；
6. conditional/unconditional batch 对齐；
7. solver、步数和随机性；
8. VQ codebook、特殊 token 与扫描顺序。

## 失效模式

- **Codebook collapse**：少数 code 占据大部分位置。
- **Dead codes**：部分 embedding 永远不被选择。
- **重建上限**：生成错误来自 tokenizer/autoencoder 而非 prior。
- **参数化错配**：模型预测 $v$，sampler 按 $\epsilon$ 解释。
- **Schedule 错位**：训练与推理时间索引不一致。
- **CFG 过强**：条件遵循提高但多样性和自然度下降。
- **Flow 方向错误**：从数据积分到噪声而非反向。
- **Solver 截断**：少步误差集中在细节或构图。
- **评测混淆**：只看感知质量，不测文字、计数和属性绑定。

## 验证矩阵

| 层级 | 测试 |
| --- | --- |
| VQ | 最近邻、STE 梯度、code usage、原图重建 |
| Forward noise | $t=0$、末端分布、经验均值与方差 |
| Prediction | $\epsilon/x_0/v$ 相互转换 |
| CFG | $w=0,1$ 与 batch 对齐 |
| Flow | 边界、方向、Euler 步与解析向量场 |
| Sampling | seed、步数、solver、guidance 网格 |
| 语义 | 文字、数量、空间、属性绑定与身份 |
| 系统 | 采样步数、延迟、峰值显存与吞吐 |

理解与生成怎样共享主干见[理解与生成统一](unified-understanding-generation.md)，音视频的离散表示见[音频与语音](audio-language-models.md)和[视频与世界模型](video-world-models.md)，最小 VQ、加噪、CFG 与 flow sampler 见[多模态手撕实现](../practice/multimodal.md)。
