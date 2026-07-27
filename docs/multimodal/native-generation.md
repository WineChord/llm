# 原生多模态与生成

“原生多模态”没有统一定义。判断一个系统是否真正联合建模多个模态，应检查训练目标、表示空间和参数更新方式，而不是只看产品是否接受图片。

## 三种能力边界

| 形态 | 计算路径 | 优点 | 局限 |
| --- | --- | --- | --- |
| 工具旁路 | LLM 调用 OCR、检测、ASR 或生成模型 | 模块可替换、结果可审计 | 误差跨模块传播，无法端到端学习 |
| 适配器融合 | encoder + projector/resampler + LLM | 复用成熟主干，训练成本低 | 表示瓶颈，理解与生成常分离 |
| 联合建模 | 多模态 token 或共享 Transformer | 可学习跨模态生成与推理 | 序列更长、目标竞争、训练更难 |

这些形态可以组合。原生视觉理解模型仍可能调用 OCR；统一 token 模型也可以保留独立 diffusion decoder。

## 表示的两条路线

### 连续特征

视觉 encoder 输出

$$
Z_v=E_v(I)\in\mathbb{R}^{n_v\times d_v},
$$

再经 projector $P$ 映射到语言维度：

$$
H_v=P(Z_v)\in\mathbb{R}^{n_v\times d}.
$$

连续特征适合理解任务，保留语义空间的平滑结构；但如果要生成像素，还需要额外 decoder 或 diffusion 模型。

### 离散 token

视觉 tokenizer 将图像映射为离散码

$$
z=q(E(I)),\qquad z_i\in\{1,\ldots,K\},
$$

模型在文本与图像 token 上使用统一自回归目标：

$$
\mathcal L=-\sum_t\log p_\theta(x_t\mid x_{<t}).
$$

统一目标简洁，却要求视觉 tokenizer 同时兼顾重建质量和语义可预测性。码本坍塌、长序列和模态频率失衡都会影响训练。

## 理解与生成为何会冲突

理解需要抽象且不变的语义表示：颜色轻微变化不应改变“这是一只猫”。生成需要保留局部纹理和空间细节。强行使用同一表示可能导致：

- 语义特征不足以重建高频细节；
- 像素级特征给语言推理带来冗余；
- 文本 token 频率压倒图像 token；
- 不同损失的梯度方向相互干扰。

[Janus](https://arxiv.org/abs/2410.13848)采用解耦视觉编码路径、共享自回归主干的思路，说明“统一模型”不必等于“统一所有表示”。

## 自回归、扩散与混合生成

### 自回归图像 token

$$
p(z)=\prod_{t=1}^{n}p(z_t\mid z_{<t},c).
$$

它与语言模型接口一致，适合交错图文和长序列推理；缺点是生成步数随 token 数增长。

### 扩散生成

前向过程逐步加噪，模型学习噪声或 score：

$$
\mathcal L_{\text{diff}}=
\mathbb E_{x,\epsilon,t}
\left[\lVert\epsilon-\epsilon_\theta(x_t,t,c)\rVert_2^2\right].
$$

扩散对连续高维信号有效，但与离散语言序列需要额外条件接口。

### 混合系统

LLM 负责意图、布局、文字和迭代策略，图像 decoder 负责像素生成。评价时应分别测试语义遵循、空间关系、文字渲染、身份一致性和编辑局部性。

## 训练配方

联合系统通常包含：

1. 单模态 encoder/tokenizer 预训练；
2. 图文对齐与 caption 数据；
3. 交错文档和多图上下文；
4. 理解、生成、编辑与 grounding 联合训练；
5. 多模态指令与偏好训练；
6. agent 环境中的视觉反馈。

联合目标可写作

$$
\mathcal L=
\lambda_{\text{text}}\mathcal L_{\text{text}}+
\lambda_{\text{understand}}\mathcal L_{\text{understand}}+
\lambda_{\text{generate}}\mathcal L_{\text{generate}}.
$$

$\lambda$ 不仅控制数值尺度，也控制容量分配；固定权重可能在不同训练阶段造成模态遗忘。

## 评测清单

- **感知**：OCR、小目标、图表、空间关系、视频时序；
- **推理**：答案是否真正依赖图像，而非语言先验；
- **生成**：构图、属性绑定、文字、审美与安全；
- **编辑**：非目标区域是否保持不变；
- **一致性**：多图人物、场景和风格能否连续；
- **鲁棒性**：裁切、旋转、压缩和图像内提示攻击；
- **系统**：视觉 token、prefill、显存和首 token 延迟。

架构接口见[多模态融合与训练](architecture-training.md)，家族案例见[Kimi](kimi.md)与[DeepSeek](deepseek.md)。
