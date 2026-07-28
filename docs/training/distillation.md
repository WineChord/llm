# 知识蒸馏

知识蒸馏让 student 学习 teacher 的概率分布、生成行为或中间判断。它既可压缩模型，也可把难以直接标注的行为迁移到另一模型；若 student 没有变小，蒸馏仍可能用于改变数据效率或部署接口，而不属于模型压缩。

## 问题

普通 SFT 只看到一个目标 token 或一条示范。teacher 分布还包含“哪些替代 token 也合理”“错误选项有多不合理”等信息，但 token-level 蒸馏要求 teacher 与 student 在同一个词表和 prefix 上比较。生成式任务还面临 exposure gap：训练时总在 teacher/真实前缀上学习，不代表 student 偏离后仍会被纠正。

## Token-level 目标

teacher 与 student 的温度分布为

$$
q_T(v\mid h)
=\operatorname{softmax}\left(\frac{z^{\text{teacher}}(h)}{T}\right)_v,
$$

$$
p_T(v\mid h)
=\operatorname{softmax}\left(\frac{z^{\text{student}}(h)}{T}\right)_v.
$$

对有效位置 mask $m_t$，soft distillation loss 为

$$
\mathcal L_{\text{KD}}
=
\frac{T^2}{\sum_t m_t}
\sum_t m_t
D_{\mathrm{KL}}
\left(q_T(\cdot\mid h_t)\,\|\,p_T(\cdot\mid h_t)\right).
$$

$T>1$ 会软化概率；前面的 $T^2$ 用于抵消 softmax 导数随温度缩小的梯度尺度。它不是可随意省略的装饰。常与 hard-label loss 混合：

$$
\mathcal L
=(1-\lambda)\mathcal L_{\text{hard}}
+\lambda\mathcal L_{\text{KD}}.
$$

$\lambda$、$T$、KL 方向和 token 归一化共同定义目标。只写“使用 KD”不足以复现。

### 最小语义实现 {#masked-temperature-kl}

`masked_kd` 接收对齐的 teacher/student logits `[B,T,V]` 和有效 token mask，计算 $T^2D_{\mathrm{KL}}(q_T\Vert p_T)$。teacher 在函数内停止梯度；布尔选择发生在 softmax 之前，所以无效位置的 NaN 不会先污染 KL 再被乘以零。

```python
import torch
import torch.nn.functional as F

def masked_kd(student, teacher, mask, temperature=2.):
    assert student.shape == teacher.shape and student.shape[:-1] == mask.shape
    assert temperature > 0
    valid = mask.bool()
    if not valid.any():
        raise ValueError("token-mean KL needs at least one valid target")
    selected_student = student[valid]
    selected_teacher = teacher.detach()[valid]
    log_student = F.log_softmax(selected_student / temperature, dim=-1)
    log_teacher = F.log_softmax(selected_teacher / temperature, dim=-1)
    teacher_probability = log_teacher.exp()
    return temperature ** 2 * (
        teacher_probability * (log_teacher - log_student)
    ).sum(-1).mean()

student = torch.tensor([[[2., 0.], [float("nan")] * 2]], requires_grad=True)
teacher = student.detach().clone()
mask = torch.tensor([[True, False]])
loss = masked_kd(student, teacher, mask)
assert loss.abs() < 1e-7
loss.backward()
assert torch.isfinite(student.grad).all()
assert student.grad[0, 1].abs().sum() == 0
rejected = False
try:
    masked_kd(student, teacher, torch.zeros_like(mask))
except ValueError:
    rejected = True
assert rejected
```

它输出当前张量的 token-mean soft target，因而空有效集合直接拒绝。hard-label 混合也必须先选有效 label，不能让 masked `-100` 进入普通 gather；不同词表映射、缓存 top-$k$ 残余质量与跨 rank 分母则要在外层明确定义。完整 hard/soft 组合见[训练目标：Knowledge distillation](../practice/training-objectives.md#knowledge-distillation)。

## 数据与计算契约

Token-level KD 需要：

```text
teacher and student exact revisions
shared vocabulary or an explicit token mapping
identical prefix tokens and position semantics
teacher/student chat templates
temperature, KL direction and hard/soft weight
response / padding / action mask
teacher-logit dtype, top-k truncation and storage format
```

teacher 必须停止梯度。若只保存 top-$k$ logits，还需记录剩余概率质量如何处理；把未保存 token 直接设为零会改变 KL。

teacher 和 student tokenizer 不同，或模板产生不同 prefix 时，不应强行按位置对齐 logits。可改用 sequence-level 蒸馏、共享字符/字节空间的辅助目标，或先构造明确映射。

## 蒸馏机制

### Teacher-forced token KD

teacher 与 student 都在同一真实或 teacher 序列前缀上计算分布。它稳定、可离线缓存 logits，却没有覆盖 student 自己犯错后到达的 prefix。

### Sequence distillation

teacher 生成完整回答，student 用 SFT 学习筛选后的序列。它不要求逐 token 词表相同，也能利用工具和 verifier 过滤；代价是 teacher 的分布被单条或少量样本压缩，概率信息与多样性可能丢失。

### On-policy distillation

student 先生成 prefix，再由 teacher 对这些 prefix 给出目标。[Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649) 研究了这类 student-generated output 上的蒸馏，以缓解 teacher-forcing 与部署分布的差距。

它更接近 student 实际状态，也更昂贵且随训练非平稳。生成样本必须记录 student policy version；teacher 查询失败、截断和拒答不能静默变成普通标签。

#### 多 teacher 的 on-policy token reward {#mopd-token-reward}

若不同领域或推理预算由不同 teacher 擅长，可先按样本条件 $(d,e)$ 选择 teacher，再让 student 在自己的 prefix 上生成。对实际动作 $y_t$，一种稠密奖励是

$$
r_t^{\mathrm{OPD}}
=\operatorname{clip}\!\left(
\operatorname{sg}\!\left[
\log \pi_{\mathrm{teacher}}^{d,e}(y_t\mid x,y_{<t})
-\log \pi_{\mathrm{student}}(y_t\mid x,y_{<t})
\right],
-R_{\max},R_{\max}
\right).
$$

`sg` 表示 reward construction 不反向穿过 teacher 或 student log-prob；真正的梯度由后续 policy objective 产生。下面只实现 action-token reward 与裁剪，不替代完整 RL loss：

```python
import torch

def mopd_token_reward(student_logp, teacher_logp, action_mask, rmax):
    if student_logp.shape != teacher_logp.shape or student_logp.shape != action_mask.shape:
        raise ValueError("student, teacher and action mask must align")
    if rmax <= 0 or not torch.isfinite(torch.tensor(rmax)):
        raise ValueError("rmax must be finite and positive")
    mask = action_mask.bool()
    if not mask.any():
        raise ValueError("at least one action token is required")
    with torch.no_grad():
        reward = (teacher_logp - student_logp).clamp(-rmax, rmax)
        return torch.where(mask, reward, torch.zeros_like(reward))

student = torch.tensor([[-2., -1., -4.]], requires_grad=True)
teacher = torch.tensor([[-1., -3., -2.]], requires_grad=True)
mask = torch.tensor([[True, False, True]])
reward = mopd_token_reward(student, teacher, mask, .5)
torch.testing.assert_close(reward, torch.tensor([[.5, 0., .5]]))
assert not reward.requires_grad
```

[Kimi K3 技术报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)把这种 Multi-Teacher On-Policy Distillation（MOPD）用于三类领域与 low/high/max 三档 effort，共九个 teacher，并把 token reward 接入 RL。其 $R_{\max}$、teacher routing 细节和完整优化目标未公开，不能从报告补造；报告还指出 top-$k$ logit 蒸馏在该设置中没有显示清晰增益，这不是对其他模型和预算的普遍否定。机制在整条后训练流水线中的位置见 [Kimi K3](../landscape/works/kimi-k3.md)。

#### DeepSeek-V4：全词表、多教师 OPD {#deepseek-v4-full-vocabulary-opd}

[DeepSeek-V4](../landscape/works/deepseek-v4.md#on-policy-distillation) 同样先训练多个 domain specialist，再由学生从自身 policy 采样；区别在于它不把 sampled-token log-ratio 当作 RL advantage，而在每个已访问前缀显式计算

$$
\mathcal L_{\mathrm{OPD}}
=\sum_i w_i
D_{\mathrm{KL}}\!\left(
\pi_\theta(\cdot\mid s_t)
\middle\|
\pi_{E_i}(\cdot\mid s_t)
\right).
$$

全词表目标以更高计算和通信换取更低方差。为了避免同时物化十余个 $T\times |V|$ teacher logits，V4 只缓存 teacher last hidden state，按 teacher 对样本排序，再逐个装入输出 head、重建 logits 并流式累计 KL；teacher 主体权重存于集中式存储并按需做 ZeRO-like sharding。

因此“是否 on-policy”“KL 方向”“全词表还是 sampled-token estimator”“一个还是多个 teacher”是四个独立开关。完整推导、几何混合解释与可执行实现见 [On-Policy Distillation 深读](../landscape/works/on-policy-distillation.md)。

### 推理与过程蒸馏

可将 teacher 或搜索产生的答案、步骤、验证结果转成：

- 最终答案 SFT；
- chosen/rejected pair；
- step-level process labels；
- student rollout 上的 teacher correction。

[DeepSeek-R1 深读](../landscape/works/deepseek-r1.md)还原了把推理行为蒸馏到较小模型的实验及其与在线 RL 的边界。该结果属于具体 teacher、数据和模型族；长推理文本本身不是正确性证明，必须用独立 verifier 和新题评测。搜索到训练的闭环见[推理后训练](reasoning-posttraining.md)。

### 剪枝后的恢复

student 也可以是结构化剪枝后的模型。[Minitron](https://arxiv.org/abs/2407.14679) 结合 pruning 与 distillation 恢复质量。此时应把“结构删除带来的损失”和“蒸馏恢复的收益”分别测量。

## 正确性与失效

- **漏掉 $T^2$**：改变不同温度下 soft loss 的梯度尺度。
- **mask 不一致**：teacher 在 prompt 或 padding 上贡献梯度，student 学到错误目标。
- **teacher/student prefix 不同**：位置对齐的 KL 不再比较同一条件分布。
- **teacher logits 未 detach**：无意中更新 teacher 或浪费图内存。
- **词表不同时硬对齐 token ID**：同一 ID 不代表同一符号。
- **只蒸馏 greedy 序列**：teacher 的替代解与不确定性消失。
- **student 只见 teacher prefix**：部署偏离后的状态未训练。
- **teacher 与评测同源污染**：student 看似迁移能力，实际复制答案。
- **温度和 top-$k$ 缓存未记录**：离线 logits 无法解释或重建。

## 何时不应蒸馏

teacher 在目标域不可靠、teacher 查询成本超过直接标注、两者词表和接口差异过大、数据使用条件不允许派生，或目标是引入 teacher 本身没有的新知识时，不应把蒸馏作为默认方案。若只需教授少量格式，高质量 SFT 往往更透明。

## 验证

1. teacher 与 student logits 完全相同时，soft KL 应接近零。
2. 改变 padding、batch 切分和 mask 后，有效 token 梯度应保持一致。
3. 分别报告 hard loss、soft loss、teacher entropy 与 student–teacher KL。
4. 比较 teacher-forced、sequence 与 on-policy 数据在相同 teacher 查询预算下的收益。
5. 在 teacher 未见的新来源、时间和题族上评测，检查答案污染。
6. 同时报 student baseline、teacher 上限、蒸馏后能力、校准、长度和安全回归。
7. 对 tokenizer、模板、logit 截断和 teacher 版本做恢复测试。

目标函数的最小实现与不变量见[训练目标实现](../practice/training-objectives.md)，合成样本的谱系与筛选见[合成数据](../data/synthetic-data.md)。

## GLM-5 的跨阶段 OPD {#glm-opd}

GLM-5 把 On-Policy Distillation 放在多阶段后训练的最后：学生从当前策略产生前缀，SFT、Reasoning RL 与 General RL 的 final checkpoints 作为教师，在对应训练 prompt 上提供 token 级信号。报告没有把 Agentic RL checkpoint 明确列入这份 teacher 清单。它用 stop-gradient 的 teacher/student log-ratio 替换 advantage，并采用 group size 1、batch size 1024；教师 logits 由推理引擎提供。

它与 outcome RL 的区别不是“有没有 rollout”，而是同一条学生轨迹上得到逐 token 的 teacher signal，而不是只有稀疏终局奖励。报告的式 (2) 显式使用 sampled token 上的 teacher / student log-ratio；它不能直接等同于每个位置都枚举全词表的 KL。GLM-5.2 又把这一接口扩展到十余个教师并行服务。损失方向、单样本估计与全词表 KL 不应混称，完整推导见 [On-Policy Distillation](../landscape/works/on-policy-distillation.md#reverse-kl)，系统调度见 [slime 与异步 Agentic RL](../landscape/works/slime-async-agentic-rl.md#glm-52)。

## Reference {#reference}

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Generalized Knowledge Distillation for Auto-Regressive Sequence Models](https://arxiv.org/abs/2306.13649)
- [Minitron](https://arxiv.org/abs/2407.14679)
- [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [Kimi K3 Technical Report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
- [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348)
- [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763)
- [GLM-5.2 发布说明](https://z.ai/blog/glm-5.2)
