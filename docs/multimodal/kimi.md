# Kimi 多模态与 Agent 演化

Kimi 家族展示了一条从长上下文语言模型、稀疏专家、视觉输入到长时 agent 的连续路线。分析时应把模型能力、训练方法和产品 harness 分开；本页公开信息核验截至 2026-07-27。

## 家族节点

| 节点 | 公开对象 | 可复用观察 |
| --- | --- | --- |
| Kimi K2 | [技术报告与权重](https://github.com/MoonshotAI/Kimi-K2) | 大规模 MoE、MuonClip 与 agentic data |
| Kimi K2.5 | [技术报告](https://arxiv.org/abs/2602.02276) | 视觉输入、thinking 与 agent swarm 能力共同训练 |
| Kimi K3 | [官方技术博客](https://www.kimi.com/blog/kimi-k3) | 更大稀疏模型、原生视觉、长上下文与长时 coding |
| Kimi-Audio | [官方仓库](https://github.com/MoonshotAI/Kimi-Audio) | 音频理解、生成与对话是独立模态路线 |

版本号不自动代表统一架构：具体参数、训练阶段、开放权重和 API 能力应逐版本核对。

## K2：MoE 与优化器

MoE 路由仍可写成

$$
y=\sum_{i\in \operatorname{TopK}(g(x))}p_iE_i(x),
$$

但大规模训练的关键不只是 top-$k$。专家容量、跨节点 all-to-all、路由稳定性和 optimizer state 共同决定可训练性。K2 公开材料强调 Muon 系优化方法与大规模稀疏训练的结合，阅读时应追踪：

- 矩阵参数与向量参数是否使用不同更新规则；
- 正交化或谱约束如何近似；
- learning rate、weight decay 和 clipping 怎样与规模联动；
- optimizer 的额外计算是否被通信或主干计算掩盖。

优化器名称不能替代完整配方，见[优化与稳定性](../training/optimization.md)。

## K2.5：能力组合不等于模块拼接

视觉、thinking、工具使用和并行 agent 若分别训练，容易出现相互覆盖：视觉微调削弱代码，工具轨迹使输出格式固化，长推理又增加延迟。联合训练的核心问题是数据混合与能力路由：

$$
p(D)=
\alpha D_{\text{text}}+
\beta D_{\text{vision}}+
\gamma D_{\text{reasoning}}+
\delta D_{\text{agent}}.
$$

系数不仅是样本比例，还受到序列长度、loss mask 和采样难度影响。公开 benchmark 只能显示结果，不能反推出精确混合配方。

## K3：结构、规模与长时任务

官方资料将 K3 描述为带原生视觉和长上下文的大规模稀疏模型，并引入 Kimi Delta Attention 与 Attention Residuals。对这类新架构，适合按三层阅读：

1. **数学对象**：状态如何更新，跨 token 或跨层信息如何压缩；
2. **系统实现**：训练 kernel、缓存、并行与通信是否匹配；
3. **任务证据**：收益来自 checkpoint，还是长时任务 harness 与额外推理预算。

长时 coding 结果尤其不能只归因于模型。上下文压缩、任务检查点、工具权限、失败恢复和缓存都会进入系统质量，见[Coding Agent](../applications/coding-agents.md)。

## Agent swarm 的判断框架

并行 agent 不是自动增益。若把任务拆成 $m$ 个子任务，总时间近似

$$
T\approx \max_i T_i+T_{\text{coord}}+T_{\text{merge}}+T_{\text{verify}}.
$$

只有当子任务依赖弱、合并成本低且验证可分解时，$\max_i T_i$ 的并行收益才可能覆盖协调开销。模型自报“多个 agent”不能证明它们拥有独立状态或真实并行。

评测应记录：

- 并行度、模型调用数和总 token；
- 共享上下文与写冲突策略；
- 合并者是否重新验证所有结果；
- 单 agent 等预算基线；
- 成功率与尾延迟，而非最佳展示。

## 多模态边界

“原生视觉”至少应核对：

- 视觉数据是否进入主训练，而非上线时外接描述器；
- 图像 token 与文本 token 如何融合；
- 高分辨率、多图、视频和 OCR 的预算；
- 视觉输出是否由同一模型生成；
- 图像内指令与系统指令的权限关系。

通用机制见[原生多模态与生成](native-generation.md)，模型谱系记录方法见[模型谱系](../landscape/index.md)。
