# 轨迹与策略契约

强化学习训练能否成立，取决于“这条轨迹究竟由哪个策略、在什么环境、对哪些动作采样而来”。只保存 prompt、response 和 reward，无法可靠重算 probability ratio、advantage 或环境结果。

## 三层表示

一条语言 agent 轨迹可分为：

1. **环境层**：observation、structured action、tool result 与状态转移；
2. **消息层**：system/user/assistant/tool message；
3. **token 层**：input ID、position、log-prob 与 action mask。

环境层适合审计业务语义，token 层适合训练。两层之间必须有可重放的序列化规则。

## Token 契约

对每个 token $x_t$，至少知道：

```text
token id
message and action-span id
position id
attention/segment metadata
is_observation
is_action
old log-probability if sampled
sampling configuration
```

策略损失只应覆盖策略选择的 action token。system、user、tool result 与 padding 是条件，不是动作：

$$
\mathcal L_{\text{policy}}
=-\sum_tm_t
\log\pi_\theta(x_t\mid x_{<t})\hat A_t,
$$

其中 $m_t=1$ 仅表示可归因于策略的 token。

## old log-probability

PPO 或 off-policy 校正需要 behavior policy 的概率：

$$
r_t(\theta)
=\exp\left(
\log\pi_\theta(x_t\mid h_t)
-\log\pi_{\text{old}}(x_t\mid h_t)
\right).
$$

最稳妥的是在 rollout 时保存实际采样 token 的 `old_logprob`。以后用“同名 checkpoint”重算仍可能不同，因为 tokenizer、模板、量化、kernel、logit processor 或随机策略已变化。

保存的 log-prob 必须对应经过 temperature、mask 和截断后的哪个分布。若训练使用原始 policy probability、采样却来自 top-$p$ 截断分布，importance ratio 的语义需要明确，不能混用。

## Policy version

每条 episode 绑定不可变版本：

```text
weights digest
tokenizer and chat-template digest
adapter set
inference precision and quantization
sampling implementation version
tool schema
environment and verifier version
```

“latest”不是版本。若一个 episode 中途热更新权重，应按 span 保存 behavior version；否则整条轨迹不来自单一 policy。

## Action 粒度

### Token-level

每个 token 使用独立 ratio 和 advantage。粒度细，但 sequence-level reward 被复制到很多 token，长度会改变总梯度。

### Span-level

将一条消息或工具调用视为 action：

$$
\log\pi(a_k\mid h_k)
=\sum_{t\in a_k}\log\pi(x_t\mid x_{<t}).
$$

它更接近环境转移语义，却可能让长 span 的 ratio 极端。使用平均 log-ratio 会改变目标，不是数值等价替换。

### Episode-level

整条轨迹共享回报，简单但信用最粗。适合强 outcome verifier 与短轨迹；长时任务通常需要 turn-level、critic 或层级分解。

## Observation 插入

工具结果会进入下一次上下文，但不是策略生成。需要记录：

- 原始状态码与规范化状态；
- stdout/stderr 或结构化 payload 的截断；
- 不可信文本边界；
- 重试、超时和幂等语义；
- 结果进入 prompt 的精确模板。

若训练时只保存渲染文本，无法判断某段内容是模型 action 还是环境 observation，mask 容易错误。

## 终止与奖励

建议将终态分开：

```text
success
task_failure
policy_stop
budget_truncation
user_cancel
environment_error
verifier_error
```

奖励也保存分量：

$$
R=
w_sR_{\text{success}}
+w_pR_{\text{process}}
-w_cC_{\text{cost}}
-w_vP_{\text{violation}}.
$$

只保存加权总分会让以后无法修改权重、审计奖励突增或区分能力与基础设施问题。

## Advantage 对齐

若 reward 在 tool-call 结束后产生，需要定义它赋给：

- 生成工具名的 token；
- 参数 JSON 的 token；
- 整个 action span；
- 从上一个 observation 到本次 transition 的所有 token；
- 更早的规划 action。

这种对齐不是数据清洗细节，而是 credit assignment。过程奖励见[搜索与验证](search-verification.md)，数学估计量见[数学与算法](math-algorithms.md)。

## Padding 与 batch

不同轨迹打包时，应独立维护：

- episode/segment boundary；
- attention mask；
- action mask；
- advantage/value；
- old/reference log-prob；
- terminal/bootstrap mask。

padding token 的 advantage 设为零还不够；它必须从 loss 分母、KL、entropy 和统计指标中排除。

### 轨迹语义校验 {#trajectory-mask-validator-reference}

输入 `tokens` 保留 token 顺序，每项声明来源 `kind`、policy mask 与 rollout 时保存的 `old_logprob`；`terminated` 表示环境真正终止，`truncated` 表示时间或预算截断。输出给出 action token 数和 value bootstrap mask。

```python
import math
def validate_trajectory(tokens, terminated, truncated):
    if terminated and truncated:
        raise ValueError("termination and truncation are distinct outcomes")
    action_tokens = 0
    for index, token in enumerate(tokens):
        kind = token["kind"]
        if kind not in {"action", "observation"}:
            raise ValueError(f"unknown token kind at {index}")
        is_action = kind == "action"
        if token["mask"] != int(is_action):
            raise ValueError(f"policy mask disagrees with token kind at {index}")
        old_logprob = token.get("old_logprob")
        if is_action and (old_logprob is None or not math.isfinite(old_logprob)):
            raise ValueError(f"missing behavior probability at {index}")
        if not is_action and old_logprob is not None:
            raise ValueError(f"observation carries behavior probability at {index}")
        action_tokens += int(is_action)
    return {"action_tokens": action_tokens, "bootstrap_mask": 0 if terminated else 1}
trace = [{"kind": "observation", "mask": 0, "old_logprob": None},
         {"kind": "action", "mask": 1, "old_logprob": -0.7}]
assert validate_trajectory(trace, terminated=True, truncated=False) == {
    "action_tokens": 1, "bootstrap_mask": 0
}
assert validate_trajectory(trace, terminated=False, truncated=True)["bootstrap_mask"] == 1
```

不变量是 observation 永不进入 policy loss，action 必须能追溯 behavior probability，termination 禁止 bootstrap，而 time-limit / budget truncation 保留 bootstrap。生产数据还需验证 episode / span ID、最终 observation、版本摘要与 padding 分母；action mask、ratio 和 bootstrap 的组合实验见[手撕：强化学习](../practice/reinforcement-learning.md)。

## 异步消费

rollout 进入 buffer 后，learner 已可能更新多次。每批至少统计：

$$
\Delta v=v_{\text{learner}}-v_{\text{behavior}},
$$

以及 current/behavior KL、ratio 分位数和被裁剪 token 比例。版本差只是 lag 的代理；相同版本间不同解码设置也可能产生不同 behavior distribution。

轨迹过旧时可以丢弃、降权或使用截断 importance correction。不能只改元数据宣称它来自当前策略。

## 可重放测试

1. 用保存的 token 重建完整消息与环境 action。
2. 用固定 checkpoint 对每个 action token 重算 log-prob，并与保存值在容差内比较。
3. 从环境快照重放 tool transition。
4. 重算 reward components 与终态。
5. 对 batch packing 后的 mask、advantage 和分母做逐 token 可视化。
6. 模拟 tokenizer、schema 或 verifier 版本不匹配，确认系统明确拒绝。

数据来源与切分见[数据与环境](data-environments.md)，异步系统见[训练系统](training-systems.md)。

## Reference {#reference}

- [RLDS: an Ecosystem to Generate, Share and Use Datasets in Reinforcement Learning](https://arxiv.org/abs/2111.02767)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
- [Gymnasium: Handling Time Limits](https://gymnasium.farama.org/v0.26.3/tutorials/handling_time_limits/)
