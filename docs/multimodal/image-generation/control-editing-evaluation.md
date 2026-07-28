# 可控生成、编辑与评测：让条件真正改变正确的部分

文本到图像只给出“想要什么”，可控生成还要回答：

- 哪些结构必须保持；
- 哪些区域允许变化；
- 条件之间冲突时谁优先；
- 输出变化是否真的由条件引起；
- 更好看的结果是否只是更强 guidance 或 best-of-$N$ 的产物。

因此，可控生成不是向模型多塞一个 embedding。它更像一个条件化逆问题：从不完整、含噪甚至互相矛盾的约束中，寻找位于自然图像分布上的解。

## 先把控制拆成四类

设输出图像为 $x$，文本为 $c_t$，空间结构为 $c_s$，参考外观或身份为 $c_r$，源图为 $x_{\mathrm{src}}$，编辑区域为 $m$。目标可抽象为

$$
p(x\mid c_t,c_s,c_r,x_{\mathrm{src}},m).
$$

这些条件承担不同角色：

| 条件 | 约束对象 | 典型形式 |
| --- | --- | --- |
| 语义 | 物体、动作、风格、关系 | 文本 token |
| 几何 | 边缘、深度、姿态、分割、layout | 与图像对齐的 dense map |
| 外观/身份 | 人物、物体、配色、材质 | reference image embedding |
| 编辑边界 | 哪些内容变、哪些保持 | source latent、mask、轨迹 |

若把四者都压成一句 prompt，系统就无法知道“红色”应绑定哪个对象，也无法保证背景保持。可控架构的核心是让条件拥有合适的<strong>粒度、位置与强度接口</strong>。

<div markdown="block">
<figure class="paper-figure paper-figure--wide" id="dit-figure-03" data-paper-source="dit" data-paper-asset="dit-figure-03" markdown="1">
[![DiT 将加噪 latent 切成 patch token，并比较 adaLN-Zero、cross-attention 与条件 token 拼接三种条件注入方式](../../assets/papers/dit/figure-03-architecture-conditioning.png){ width="2150" height="883" loading="lazy" decoding="async" }](../../assets/papers/dit/figure-03-architecture-conditioning.png)
<figcaption><strong>Figure 3 说明“加入条件”至少包含位置、参数路径和初始化三项选择。</strong>adaLN 把条件变成通道调制，cross-attention 建立独立键值记忆，in-context 方案则让条件直接占用序列位置；后续加入边缘、深度、身份或 mask 时，必须继续说明它们复用哪条路径，以及多条件冲突怎样被训练和评测。<span class="paper-figure__source">图源：<a href="https://arxiv.org/pdf/2212.09748v2#page=3">Scalable Diffusion Models with Transformers, Figure 3, p. 3</a>；Copyright © 2023 William Peebles and Saining Xie，<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>。</span></figcaption>
</figure>
</div>

## Guidance：最轻量的控制是改变 score

Classifier guidance 利用

$$
\nabla_x\log p(x\mid c)
=
\nabla_x\log p(x)
+
\nabla_x\log p(c\mid x),
$$

沿 classifier 认为更符合条件的方向修改 score。[Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) 把条件与无条件预测放到同一网络：

$$
\hat y
=
y_\varnothing
+
w(y_c-y_\varnothing),
$$

其中 $y$ 可以是 $\epsilon$、$v$ 或其他与 sampler 匹配的 target。

CFG 的语义是对条件方向做线性外推，不是“条件权重的概率”。$w>1$ 往往提高 prompt adherence，却把状态推到训练分布外。多个条件若各自相对无条件分支线性相加：

$$
\hat y
=
y_\varnothing
+
\sum_i w_i(y_{c_i}-y_\varnothing),
$$

还隐含假设它们的交互可加。人物身份、姿态与风格常有强交互，独立增大 scale 可能相互破坏。可靠系统需要联合条件分支或显式优先级，而不只是更多 slider。

## 图像编辑从“加多少噪声”开始

[SDEdit](https://arxiv.org/abs/2108.01073) 先把输入图像扰动到中间噪声层级，再反向生成：

$$
x_t
=
\alpha_tx_{\mathrm{src}}
+
\sigma_t\epsilon.
$$

$t$ 小，模型只能做局部修复，source fidelity 高；$t$ 大，系统获得更多重绘自由，语义变化更大。这个单一参数已经揭示编辑的基本 Pareto：

$$
\text{source preservation}
\longleftrightarrow
\text{edit strength}.
$$

所谓 image-to-image `strength` 必须映射到明确的 noise/sigma 区间。不同 scheduler 下同一个数值未必对应相同 signal-to-noise ratio。

对于 mask 编辑，理想边界是

$$
x_{\mathrm{out}}\odot(1-m)
\approx
x_{\mathrm{src}}\odot(1-m),
$$

同时在 $m$ 内满足指令。若只在最终像素混合，边界可能不连贯；若在每个 denoising step 把非编辑区域替换为 source 在同一噪声层级的 latent，能更强地保持上下文，但要求 source noise 与当前 sample 使用一致 schedule 和 noise realization。

## 文字条件编辑：改 token 还是改 attention

[Prompt-to-Prompt](https://arxiv.org/abs/2208.01626) 观察到 text-to-image diffusion 的 cross-attention map 携带词到空间的对应关系。对原 prompt 与编辑 prompt，在若干层/时刻复用或替换 attention，可在改变词语时保留构图。

若

$$
A=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt d}\right),
$$

交换 prompt token embedding 会同时改变 $K,V$；固定旧 $A$ 则保留“哪个空间位置看哪个词”的路由，但允许新 $V$ 提供新语义。这个解释比“复制 attention 就能编辑”更准确，也说明它的边界：

- tokenizer 若把词拆分不同，token alignment 先要解决；
- 词序和句法大改时，旧 attention 未必仍有意义；
- attention map 不是唯一因果通道，self-attention 与 residual 也保存结构；
- 不同 layer/timestep 控制的空间尺度不同。

## 个性化：embedding、权重与 reference adapter

### Textual Inversion

[Textual Inversion](https://arxiv.org/abs/2208.01618) 冻结生成模型，只为少量参考图学习新 token embedding $v_\star$。优化近似为

$$
v_\star
=
\arg\min_v
\mathbb E_{x,t,\epsilon}
\|\epsilon-\epsilon_\theta(x_t,t,\operatorname{text}(v))\|^2.
$$

存储小、组合方便，但一个 embedding 的容量有限，容易在身份细节和可编辑性之间取舍。

### DreamBooth

[DreamBooth](https://arxiv.org/abs/2208.12242) 用罕见 identifier 与 class noun 微调模型，并用 class prior preservation 缓解语言漂移。容量更强，也更容易过拟合少量视角、把参考背景绑进主体，或改变基础模型对整个类别的分布。

### IP-Adapter

[IP-Adapter](https://arxiv.org/abs/2308.06721) 把图像条件与文本 cross-attention 解耦。简化表示为

$$
\operatorname{Attn}_{\mathrm{total}}
=
\operatorname{Attn}(Q,K_t,V_t)
+
\lambda
\operatorname{Attn}(Q,K_i,V_i).
$$

独立 projection 允许调整 reference 强度，又不能自动区分“保持身份”还是“复制姿态、背景与光照”。训练数据、crop 与 reference encoder 决定 adapter 学到的相似性。

个性化评测必须同时看 identity similarity 与 prompt editability。只复刻训练图不是成功，只遵循 prompt 却丢身份也不是。

## 空间控制：把 dense condition 接入冻结主干

[ControlNet](https://arxiv.org/abs/2302.05543) 复制预训练 diffusion backbone 的部分编码路径，并通过零初始化卷积把控制残差接回冻结主干。抽象地：

$$
h_{\mathrm{out}}
=
F_{\mathrm{frozen}}(h)
+
Z_2\!\left(
F_{\mathrm{ctrl}}(h+Z_1(c_s))
\right),
$$

其中 $Z_1,Z_2$ 初始输出为零。训练开始时系统近似原模型，控制分支再逐步学会偏移，这降低了小数据直接破坏 backbone 的风险。

[T2I-Adapter](https://arxiv.org/abs/2302.08453) 以更轻的 adapter 提取条件特征并注入冻结生成模型。两者的共同思想不是某个具体卷积层，而是：

> 保留大模型已学到的图像 prior，让新模块只学习“怎样沿条件方向修正中间特征”。

Dense map 仍需明确坐标契约。Canny、depth、pose 与 segmentation 的值域和含义不同；resize 时对 label map 使用双线性插值会制造不存在的类别，对 depth 归一化按图单独处理又会丢绝对尺度。控制强度还可随 timestep 调度：

$$
\lambda(t)
=
\begin{cases}
\lambda_{\mathrm{early}}, & \text{布局阶段}\\
\lambda_{\mathrm{late}}, & \text{纹理阶段}.
\end{cases}
$$

早期更影响全局结构，后期更影响细节，但具体分界由 schedule 与模型决定，不应硬编码成通用常数。

## InstructPix2Pix：用合成监督学“怎么改”

[InstructPix2Pix](https://arxiv.org/abs/2211.09800) 构造“源图—编辑指令—目标图”数据，让模型同时条件于 source image 与 instruction。其双条件 guidance 可写成多个分支差分；关键并不是某一组固定系数，而是训练与推理必须拥有一致的条件 dropout 组合。

合成编辑对能扩大覆盖，却会继承 teacher 的偏差：

- 指令说改颜色，target 可能连背景也改变；
- source/target 几何未严格对齐；
- 很少出现“无需编辑”或冲突指令；
- instruction 文风窄，真实用户表达迁移不稳。

因此编辑模型要用 counterfactual 数据检查：换掉 instruction、遮掉 source、打乱 source-target pair，观察输出究竟依赖哪一项。

## 一个最小的多条件与 mask 契约

下面代码实现两个不可约操作：多个条件相对同一 unconditional prediction 的线性组合，以及每个 denoising step 对非编辑区域的 source-preserving blend。约定 prediction tensor 为 `[batch, channel, height, width]`，mask 为 `[batch, 1, height, width]`，`1` 表示允许编辑。

```python
import torch
def compose_guidance(unconditional, conditions, scales):
    if len(conditions) != len(scales) or not conditions:
        raise ValueError("conditions and scales must be non-empty and aligned")
    guided = unconditional.clone()
    for conditional, scale in zip(conditions, scales):
        if conditional.shape != unconditional.shape:
            raise ValueError("all guidance branches must align")
        guided = guided + scale * (conditional - unconditional)
    return guided
def preserve_outside_mask(proposal, source_at_same_time, edit_mask):
    if proposal.shape != source_at_same_time.shape or proposal.ndim != 4:
        raise ValueError("latents must align as [B,C,H,W]")
    expected = (proposal.size(0), 1, proposal.size(2), proposal.size(3))
    if edit_mask.shape != expected:
        raise ValueError("edit_mask must be [B,1,H,W]")
    if torch.any((edit_mask < 0) | (edit_mask > 1)):
        raise ValueError("edit_mask values must be in [0,1]")
    return edit_mask * proposal + (1 - edit_mask) * source_at_same_time
u = torch.zeros(1, 2, 2, 2)
text, pose = torch.ones_like(u), torch.full_like(u, 2.)
torch.testing.assert_close(compose_guidance(u, [text, pose], [1., .5]),
                           torch.full_like(u, 2.))
mask = torch.tensor([[[[1., 0.], [0., 1.]]]])
source, proposal = torch.zeros_like(u), torch.ones_like(u)
edited = preserve_outside_mask(proposal, source, mask)
torch.testing.assert_close(edited[:, :, 0, 0], torch.ones(1, 2))
torch.testing.assert_close(edited[:, :, 0, 1], torch.zeros(1, 2))
```

`source_at_same_time` 不是干净 source latent，而是按当前 timestep、同一 schedule 得到的对应 noisy latent。多条件线性组合没有显式交互项；若 joint branch $y_{c_1,c_2}$ 可得，应用消融决定是否需要 inclusion–exclusion 或独立 learned fusion。

## 评测先问：控制了什么，代价是什么

### 分布质量

[FID](https://arxiv.org/abs/1706.08500) 比较 Inception 特征的 Gaussian 均值与协方差：

$$
\operatorname{FID}
=
\|\mu_r-\mu_g\|_2^2
+
\operatorname{Tr}\!\left(
\Sigma_r+\Sigma_g
-2(\Sigma_r\Sigma_g)^{1/2}
\right).
$$

它依赖特征网络、sample 数与预处理，小样本估计有偏。[KID](https://arxiv.org/abs/1801.01401) 使用 kernel MMD 的无偏估计，但同样继承 embedding 的盲区。二者都不直接判断 prompt 是否满足。

### 覆盖与真实性应分开

[Improved Precision and Recall](https://arxiv.org/abs/1904.06991) 在 feature manifold 中分别估计 sample quality 与 mode coverage。生成器可以 precision 高而 recall 低；精选样图通常只展示前者。

### 文本遵循不是一个 cosine

[CLIPScore](https://arxiv.org/abs/2104.08718) 用图文 embedding 相似度提供 reference-free 信号，但对计数、否定、空间关系和文字拼写并不可靠。[GenEval](https://arxiv.org/abs/2310.11513) 把对象、数量、颜色、位置等拆成组合任务，仍依赖 detector 的能力边界。最好把评测写成能力矩阵，而不是用一个总分替代所有语义。

### 编辑需要双目标

编辑评测至少同时测：

$$
\text{instruction success}
\quad\text{与}\quad
\text{non-target preservation}.
$$

后者要按 mask 外像素、perceptual feature、identity、layout 与背景对象切片。若输出大幅重绘后恰好满足指令，单看 CLIP alignment 会错误奖励它。

## 公平比较协议

| 变量 | 为什么必须披露 |
| --- | --- |
| Prompt | 是否重写、扩写、翻译、加入负面词 |
| 输入 | resize/crop、mask feather、depth/pose detector 版本 |
| Sampling | seed 数、步数、solver、CFG、control scale/schedule |
| Selection | best-of-$N$、reranker、人选、失败重试 |
| External model | captioner、detector、face encoder、safety filter |
| Cost | 每个候选的端到端延迟、总 NFE、峰值显存 |
| Data | benchmark 泄漏、主体是否出现在训练/个性化样本 |

推荐三层报告：

1. <strong>固定协议自动评测</strong>：可复现的大规模切片；
2. <strong>盲法人评</strong>：质量、遵循、保持分别提问；
3. <strong>失败案例簿</strong>：冲突条件、小目标、文字、多人身份、遮挡和极端 mask。

## 快速变化的产品边界

截至 2026-07-28，[OpenAI 对 GPT‑4o 图像生成的公开介绍](https://openai.com/index/introducing-4o-image-generation/)强调原生图像生成、文字渲染与多轮编辑；[Google DeepMind 的 Imagen 页面](https://deepmind.google/models/imagen/)展示其公开产品能力与安全说明。这些属于<strong>产品披露</strong>，可用于描述公开接口和演示边界，不足以反推出未发布的训练数据、loss、tokenizer 或 sampler 架构。

对快速更新系统，评测记录必须包含产品/模型版本、日期、区域、API 参数和输入输出原件。网页演示或作者精选样例不是固定 benchmark；版本更新后的结果也不能追溯覆盖旧结论。

## 常见失效模式

- <strong>条件抢占</strong>：reference image 复制背景，压过文本编辑。
- <strong>空间漂移</strong>：pose/depth 看似满足，物体身份或左右关系改变。
- <strong>Mask 泄漏</strong>：编辑区外颜色、光照或纹理被全局 attention 改写。
- <strong>身份—可编辑性冲突</strong>：identity scale 高时姿态与风格难改变。
- <strong>多主体绑定错误</strong>：属性落到错误人物或物体上。
- <strong>控制图捷径</strong>：模型学到 detector 风格而非几何语义。
- <strong>评测器共谋</strong>：生成模型利用 CLIP/detector 的偏差提高自动分。
- <strong>挑图偏差</strong>：只展示多个 seed 中最符合预期的结果。

排查顺序应从接口开始：先单独关闭每个条件，做 scale sweep，再测试 pairwise interaction；随后固定噪声比较 source/control/prompt 的 counterfactual。直接看最终成图很难知道是哪条通道真正生效。

Diffusion prediction 与 CFG 的数学基础见 [Diffusion 与 Score](diffusion-score.md)；latent、DiT 与 flow sampler 的接口见 [Latent Diffusion、DiT 与 Flow](latent-dit-flow.md)；个性化所依赖的表示上限见 [Autoencoder 与视觉 Tokenizer](autoencoders-tokenizers.md)。

条件 mask、CFG 与编辑区域不变量的组合测试见[多模态手撕实现](../../practice/multimodal.md)。

## Reference {#reference}

- [Meng et al., SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073)
- [Ho and Salimans, Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [Hertz et al., Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626)
- [Gal et al., An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/abs/2208.01618)
- [Ruiz et al., DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242)
- [Brooks et al., InstructPix2Pix: Learning to Follow Image Editing Instructions](https://arxiv.org/abs/2211.09800)
- [Zhang et al., Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)
- [Mou et al., T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability for Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.08453)
- [Ye et al., IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://arxiv.org/abs/2308.06721)
- [Heusel et al., GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
- [Bińkowski et al., Demystifying MMD GANs](https://arxiv.org/abs/1801.01401)
- [Kynkäänniemi et al., Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991)
- [Hessel et al., CLIPScore: A Reference-free Evaluation Metric for Image Captioning](https://arxiv.org/abs/2104.08718)
- [Ghosh et al., GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment](https://arxiv.org/abs/2310.11513)
- [OpenAI, Introducing 4o Image Generation](https://openai.com/index/introducing-4o-image-generation/)
- [Google DeepMind, Imagen](https://deepmind.google/models/imagen/)
