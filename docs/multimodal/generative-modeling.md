# 多模态生成模型

图像、音频与视频生成既可以被离散化为 token 做自回归预测，也可以在连续空间中学习去噪或向量场。三条路线优化的对象、采样路径和系统瓶颈不同。

## 自回归离散 token

encoder 与量化器把连续信号映射为码本索引：

$$
z=q(E(x)),
\qquad z_i\in\{1,\ldots,K\}.
$$

模型按顺序建模

$$
p(z\mid c)=\prod_t p(z_t\mid z_{<t},c).
$$

优点是与语言建模、交错模态和 grammar 统一；缺点是序列长、生成串行，且 tokenizer 的重建误差形成质量上限。二维图像还需要确定 raster、分层或其他 token 顺序。

## Diffusion

前向过程逐步向数据加噪，常写为

$$
x_t=\alpha_tx_0+\sigma_t\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

模型可以预测噪声、原始样本、velocity 或 score。以噪声预测为例：

$$
\mathcal L_{\epsilon}
=\mathbb E_{x_0,\epsilon,t}
\left[
w(t)
\lVert
\epsilon-\epsilon_\theta(x_t,t,c)
\rVert_2^2
\right].
$$

采样从噪声出发，数值积分反向过程。步数、scheduler、guidance 和 latent decoder 都影响质量与速度；训练 loss 不能单独预测感知质量。

## Flow Matching

Continuous Normalizing Flow 用 ODE 描述样本随时间移动：

$$
\frac{dx_t}{dt}=v_\theta(x_t,t,c).
$$

[Flow Matching](https://arxiv.org/abs/2210.02747)选择一族连接噪声与数据的条件概率路径，并直接回归对应向量场：

$$
\mathcal L_{\text{FM}}
=\mathbb E_{t,x_t}
\left[
\lVert
v_\theta(x_t,t,c)-u_t(x_t\mid x_1)
\rVert_2^2
\right].
$$

训练无需沿 ODE 完整模拟路径；推理仍需 ODE solver。路径选择、时间采样和 solver 步长共同决定效率，不能把“flow”简单解释为固定少步生成。

## Conditioning

文本条件可通过：

- cross-attention；
- AdaLN/FiLM 式尺度与偏置；
- 拼接 token；
- 独立 control encoder；
- classifier-free guidance。

classifier-free guidance 常用条件与无条件预测组合：

$$
\hat\epsilon
=\epsilon_\varnothing
+w(\epsilon_c-\epsilon_\varnothing).
$$

增大 $w$ 往往提高条件遵循，却可能降低多样性、饱和颜色或放大伪影。文本 encoder、negative prompt 和训练时条件 dropout 都属于配方。

## Latent 与 pixel space

在压缩 latent 中生成可降低空间尺寸和算力，但 autoencoder 会丢高频细节，并可能对文字、脸部或细线产生结构性误差。在 pixel/token space 直接生成避免这一瓶颈，却需要更大模型或更长序列。

视频还增加时间压缩。若独立生成每帧，局部质量可以很高但时间一致性差；联合时空 latent 则显著增加内存与训练数据要求。

## 理解与生成的共享

视觉理解偏好语义不变性，生成偏好可逆的局部细节。可采用：

| 方案 | 共享部分 | 风险 |
| --- | --- | --- |
| 完全独立 | 仅由 LLM/tool 协调 | 能力割裂、误差跨模块 |
| 共享主干、独立 encoder/decoder | 高层推理状态 | 路由与表示对齐 |
| 统一离散 token | tokenizer 与自回归主干 | 模态竞争、长序列 |
| 连续生成头接语言主干 | 条件与语义表示 | 接口瓶颈与训练目标冲突 |

“一个模型完成全部模态”必须落实到哪些参数和目标真正共享。

## 评测

### 图像

- 感知质量与多样性；
- 文本、计数、空间关系和属性绑定；
- identity/character consistency；
- 局部编辑是否保持非目标区域；
- 采样步数、延迟和显存。

### 视频

- 运动与物理一致性；
- 时间连续、镜头切换和长时身份；
- 文本条件在全时段是否保持；
- 帧率、分辨率、时长与生成成本。

### 音频

- 内容正确性、说话人、韵律和噪声；
- 流式首包与实时系数；
- 语音理解和生成之间的一致性。

自动感知指标应与人工 pairwise、条件 verifier 和失效切片共同使用。

## 实现检查

- 时间参数、noise schedule 与训练/采样定义一致；
- latent scale 与 autoencoder 版本一致；
- prediction type 与 scheduler 匹配；
- guidance 的 conditional/unconditional batch 对齐；
- ODE/SDE solver 的方向、步长和边界正确；
- mixed precision 下 norm、attention 和 decoder 无溢出；
- 固定 seed 的可复现范围明确。

统一表示的讨论见[原生多模态与生成](native-generation.md)，视觉接口见[视觉语言模型](vision-language.md)。
