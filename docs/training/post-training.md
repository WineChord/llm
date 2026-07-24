# 后训练与偏好学习

后训练把基础模型的分布能力转化为可交互行为。它不能凭空补回预训练缺失的知识，也不能只靠一个奖励分数定义所有可靠性。

## Supervised Fine-Tuning

SFT 对示范答案做条件语言建模。关键变量包括数据质量、任务混合、chat template、只对 response 计算 loss 与否、长度截断以及拒答样本比例。高质量小数据可能优于低质量大数据，但“质量”必须由目标任务和人工审查定义。

## Reward Modeling 与 RLHF

偏好数据给出 \(y_w\succ y_l\)。Bradley–Terry 风格奖励模型常优化：

\[
\mathcal{L}_{RM}
=-\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))
\]

策略优化通常在提高奖励的同时，用 KL 约束限制偏离参考策略。[InstructGPT](https://arxiv.org/abs/2203.02155) 给出了 SFT、reward model 与 PPO 的代表性流程。

风险包括：reward hacking、标注者偏差、长度偏好、风格偏好、分布外奖励失真，以及优化后多样性下降。

## Direct Preference Optimization

[DPO](https://arxiv.org/abs/2305.18290) 把带 KL 正则的偏好优化改写为对策略与参考策略 log-ratio 的分类目标，避免单独训练显式奖励模型和在线 PPO rollout。它实现更简单，但仍依赖参考策略、偏好数据和温度超参数，且不自动解决标签噪声或分布外泛化。

## 可验证奖励与推理训练

数学、代码和结构化任务可用单元测试、答案检查器或形式验证器提供结果奖励。过程监督可对中间步骤打分，但标注成本更高，且评分器可能被格式投机。

[DeepSeekMath](https://arxiv.org/abs/2402.03300) 描述了 Group Relative Policy Optimization（GRPO）的一种实现，用同组样本的相对奖励构造优势而不训练独立 value model。方法名称不能替代目标推导：应明确采样组、baseline、KL、裁剪和奖励组成。

## RLAIF 与原则约束

AI feedback 可扩展标注，但会继承评审模型偏差。[Constitutional AI](https://arxiv.org/abs/2212.08073) 展示了用显式原则指导自我批评和偏好训练的路线。原则文本、评审模型与最终策略仍需独立评测。

## 评测

同时测量有用性、事实性、指令遵循、拒答边界、校准、风格、长度、工具行为和能力回归。自动 judge 只能作为一个测量通道，必须用人工抽检、对抗样本和可执行验证器校准。
