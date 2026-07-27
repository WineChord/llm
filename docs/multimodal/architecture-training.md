# 多模态融合、位置与训练

多模态架构首先要解决 shape 对齐：感知 encoder 输出的 token 数和维度通常与语言主干不同。设

$$
Z_m=E_m(x_m)\in\mathbb R^{B\times N_m\times d_m},
\qquad
H_m=P_m(Z_m)\in\mathbb R^{B\times N'_m\times d}.
$$

$P_m$ 同时决定信息瓶颈、token budget 和 checkpoint 兼容性。

## Projector

最简单的接口是线性层：

$$
H_m=Z_mW+b,
\qquad
W\in\mathbb R^{d_m\times d}.
$$

MLP projector 可以增加非线性：

$$
H_m=W_2\sigma(W_1Z_m).
$$

[LLaVA](https://arxiv.org/abs/2304.08485)展示了预训练视觉 encoder、projector 与语言模型对齐的一条简洁路线。Projector 不减少 $N_m$；高分辨率输入的成本会原样传给 LLM。

## Querying 与 resampling

固定 $N_q$ 个可学习 query：

$$
Q\in\mathbb R^{B\times N_q\times d},
\qquad
H_m'
=
\operatorname{Attn}(Q,K=Z_m,V=Z_m).
$$

无论原始 token 数 $N_m$ 多大，语言主干只接收 $N_q$ 个结果。[BLIP-2](https://arxiv.org/abs/2301.12597)的 Q-Former 和 [Perceiver IO](https://arxiv.org/abs/2107.14795)的 latent querying 提供了代表性设计。

固定压缩率有明确风险：全局语义可能保留，小文字、密集目标和稀有区域更容易被丢弃。应测性能随 $N_q$ 的曲线，而不是只报告一个配置。

## Cross-attention

在语言层中加入对模态特征的 cross-attention：

$$
H_{\text{text}}'
=
H_{\text{text}}
+
g\odot
\operatorname{Attn}
\left(
Q=H_{\text{text}},
K=Z_m,
V=Z_m
\right).
$$

[Flamingo](https://arxiv.org/abs/2204.14198)使用 gated cross-attention 处理交错图文。Cross-attention 让文本 token 按需读取媒体，但改变主干结构，也增加每层媒体 K/V、checkpoint 转换和服务 runtime 的复杂度。

## Early fusion

另一条路线把媒体 token 投影到共同维度后，与文本一起进入 self-attention：

$$
H_0
=
[H_{\text{text}}^{(1)};
H_{\text{image}};
H_{\text{text}}^{(2)};\ldots].
$$

优点是任意位置可直接交互；代价是所有模态共享序列长度和二次 attention。还必须定义：

- 每种模态的 type embedding；
- causal、bidirectional 或 block mask；
- 多图、多段音频与视频的边界 token；
- position IDs 是否重置；
- 哪些媒体 token 参与语言 loss。

## 对齐目标

[CLIP](https://arxiv.org/abs/2103.00020)对归一化图文表示使用批内对比：

$$
s_{ij}=\frac{u_i^\top v_j}{\tau}.
$$

对图到文、文到图分别做交叉熵：

$$
L_{\mathrm{CLIP}}
=
\frac12
\left(
L_{\mathrm{i2t}}+L_{\mathrm{t2i}}
\right).
$$

[SigLIP](https://arxiv.org/abs/2303.15343)改为 pairwise sigmoid loss：

$$
L
=
\frac1{B^2}
\sum_{i,j}
\log
\left(
1+\exp[-y_{ij}(s_{ij}+b)]
\right).
$$

全局对比学习建立语义对齐，但不自动提供字符、区域和关系级监督。重复 caption、同类图像和弱文本会形成 false negative 或错误对应。

## 分辨率与 patch

对图像大小 $H\times W$、patch $P_h\times P_w$：

$$
N_v
=
\frac{H}{P_h}\frac{W}{P_w}.
$$

分辨率翻倍时，patch 数约增长四倍。常见策略包括：

- 固定 resize；
- 保留宽高比并 padding；
- 全局缩略图加局部 tiles；
- 动态分辨率或 dynamic tiling；
- token merge、pooling 或 resampler。

动态切片必须保存 tile 顺序、原图 offset、缩放比例和二维坐标。否则模型虽然获得高清 crop，却无法恢复跨 tile 空间关系。

## 多维位置

图像 token 使用 $(h,w)$，视频使用 $(t,h,w)$。若采用多维旋转位置：

$$
\operatorname{RoPE}(q;t,h,w)
=
[R_tq^{(t)};R_hq^{(h)};R_wq^{(w)}].
$$

[Qwen2-VL](https://arxiv.org/abs/2409.12191)提供了 M-RoPE 与动态视觉 token 的公开设计。实现中应固定每个轴的通道切分、文本 token 的坐标推进和多媒体边界。

## 训练阶段

一种常见但非唯一的顺序是：

1. 感知 encoder/tokenizer 单模态预训练；
2. 冻结主干，训练 projector/resampler 对齐；
3. 解冻部分或全部组件做多模态预训练；
4. 指令、grounding、OCR、工具和多轮数据；
5. 偏好、安全与生产分布适配。

每阶段需记录：

- 冻结参数与 trainable parameter 数；
- 数据混合、采样权重和分辨率；
- 模态 token、文本 token 与 packing；
- optimizer、学习率和各组件 schedule；
- loss mask 与模态 loss 归一；
- encoder、tokenizer 和 chat template 版本。

## Loss 平衡

若文本与媒体 token 数量差异很大，直接按全部 token 平均会让占比高的模态支配梯度。可按模态有效 token 归一：

$$
L
=
\lambda_t
\frac{\sum_im_i^{(t)}\ell_i^{(t)}}{\sum_im_i^{(t)}}
+
\lambda_m
\frac{\sum_jm_j^{(m)}\ell_j^{(m)}}{\sum_jm_j^{(m)}}.
$$

$\lambda$ 控制的是容量分配，不只是数值尺度。应监控每个目标的梯度范数、训练曲线和遗忘，而不只监控总 loss。

## 实现契约

1. 每个模态张量的 layout、dtype 和有效长度明确；
2. projector 输出与 LLM hidden size、norm 约定一致；
3. media placeholder 数量与实际注入 token 完全一致；
4. position IDs、attention mask 和 loss mask 独立构造；
5. resize、crop、tile 与 grounding 坐标使用同一变换；
6. 多图、视频和音频在 batch packing 后仍保持边界；
7. frozen module 处于正确的 train/eval 状态；
8. checkpoint 保存 encoder、projector、tokenizer 和模板版本。

## 失效模式

- **模态遗漏**：移除媒体后答案几乎不变。
- **语言先验**：常见答案模式掩盖感知失败。
- **压缩瓶颈**：固定 query 无法保留小目标与文字。
- **占位错位**：placeholder 与视觉 token 数不同。
- **坐标漂移**：crop 后 bbox 仍使用原图尺度。
- **目标竞争**：媒体 loss 增长造成文本能力退化。
- **冻结错误**：名义冻结但 BatchNorm/dropout 状态仍变化。
- **模板漂移**：训练与服务的媒体边界 token 不一致。
- **权限混淆**：图像内文本被当成高权限指令。

## 验证

| 维度 | 测试 |
| --- | --- |
| Shape | 多尺寸、多图、空模态、最大 token |
| 对齐 | 正确配对、错配、重复 caption、hard negative |
| 证据依赖 | 遮挡、替换、移除媒体与反事实图像 |
| 空间 | tile 顺序、crop、bbox、二维位置 |
| Loss | 各模态梯度、冻结组件、训练阶段切换 |
| 系统 | token 数、prefill、峰值显存、吞吐 |
| 鲁棒 | 压缩、旋转、噪声、模态缺失、媒体内攻击 |

可执行的 patchify、CLIP/SigLIP loss、resampler、模态 mask 与位置练习见[多模态手撕实现](../practice/multimodal.md)。文档和坐标见[文档、图表、GUI 与 Grounding](document-gui-grounding.md)，统一生成目标见[理解与生成统一](unified-understanding-generation.md)。
