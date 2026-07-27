# 训练系统

Agentic RL 把推理服务、交互环境和分布式训练绑成一个系统。GPU 利用率只是局部指标；真正目标是在可控 policy lag 下持续产生有效、可验证的轨迹。

## 数据流

```text
task queue
  -> rollout scheduler
  -> inference workers
  -> environment workers
  -> verifier
  -> trajectory buffer
  -> learner
  -> checkpoint registry
  -> inference refresh
```

每条轨迹应携带 policy、tokenizer、prompt template、tool schema、environment 和 verifier 版本。缺少任何一项都可能使奖励无法解释。

## Rollout 与训练的资源冲突

Rollout 偏好大量解码实例和高显存 KV cache；训练偏好大 batch、反向计算和高带宽通信。常见部署方式：

| 方式 | 优点 | 风险 |
| --- | --- | --- |
| colocated | 复用 GPU，减少空闲 | 权重切换、显存碎片与抖动 |
| disaggregated | 推理与训练独立扩缩 | 参数同步、网络与调度复杂 |
| time-shared | 阶段清晰，易实现 | GPU 等待，周期长 |
| elastic pool | 适配动态任务长度 | 资源管理和容错最复杂 |

端到端吞吐更接近

$$
\text{throughput}=
\frac{\text{valid training tokens}}
{\max(T_{\text{rollout}},T_{\text{verify}},T_{\text{learn}},T_{\text{sync}})}
$$

而不是单个 kernel 的峰值。

## 长度不均衡

agent 轨迹长度重尾：简单任务几步结束，困难任务可能运行很久。固定 batch 会产生 straggler。可采用：

- 按估计长度或环境类型分桶；
- continuous batching；
- 超时与阶段性检查点；
- 动态补充短 episode；
- 将验证与环境 I/O 异步化。

不能简单截断所有长轨迹，因为尾部可能包含最终奖励和恢复行为。需要区分预算终止、环境故障和策略主动终止。

## Policy lag

设轨迹由版本 $v_b$ 生成，learner 当前版本为 $v_l$，则 lag 可按更新步、KL 或 wall-clock 表示。只记录“最新模型”不足以审计。

控制手段包括：

- 限制 buffer 最大年龄；
- 每次 learner 更新后逐步刷新 inference worker；
- 对过旧轨迹降低权重或丢弃；
- 监控 behavior/current policy 的 token-level KL；
- 在 checkpoint registry 中使用不可变版本。

参数热更新必须保证一个 episode 内模型版本是否允许变化；若允许，轨迹概率就不再来自单一 policy。

[SAO](../landscape/works/sao-compactionrl.md#sao)进一步把 prompt 内的多 rollout 等待视为异步 barrier：单条轨迹完成后即可进入训练队列，但 learner 仍须保存真实 behavior log-probability、限制 policy lag，并审计 DIS 丢弃了哪些 token。

## Reward 与 Verifier 服务

可验证奖励经常包含编译、测试、浏览器或远程 API，吞吐与 GPU 推理不同。Verifier 集群需要：

- sandbox 与资源配额；
- 依赖缓存和镜像版本；
- 确定性 seed 与时间控制；
- 测试隔离，防止轨迹篡改；
- 对 timeout、infra error、wrong answer 分别编码；
- 重试的幂等性与成本记录。

把基础设施错误记为零奖励会训练模型回避特定任务，而不是提高能力。

## 检查点

除了模型权重，还应保存：

- optimizer、scheduler 与随机数状态；
- rollout/learner 全局步；
- tokenizer、模板和 tool schema；
- reward/verifier 版本；
- 已消费任务与 replay buffer 游标；
- 数据混合与 curriculum 状态；
- 分布式拓扑和关键依赖版本。

恢复后先做小规模 canary：固定任务、固定 seed、比较 logits/奖励/工具协议，再恢复全量生产。

## 观测指标

### 学习

- reward 分布与各分量；
- KL、entropy、clip fraction、优势均值与方差；
- 训练/验证任务成功率；
- 长度、终止类型和工具错误率。

### 系统

- rollout tokens/s、有效轨迹/s；
- TTFT、decode throughput、KV cache 命中；
- 环境 reset/step/verifier 延迟；
- learner MFU、通信和 checkpoint 时间；
- policy lag、队列年龄与失败重试。

只看平均值会隐藏长尾；至少同时报告 p50、p95、p99。

## 开源系统入口

[veRL](https://github.com/volcengine/verl)提供面向大语言模型 RL 的训练与 rollout 组织，可用于理解 actor、critic、reference、reward 和推理引擎如何分离。阅读框架时，优先追踪数据对象与版本边界，而非只记命令行参数。

behavior version、old log-prob 与异步 batch 字段见[轨迹与策略契约](trajectory-contract.md)，分布式基础见[集合通信与状态分片](../systems/collectives-sharding.md)、[模型并行](../systems/model-parallelism.md)和[检查点与容错](../systems/checkpointing.md)。

## Reference {#reference}

- [veRL](https://github.com/volcengine/verl)
- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System](https://arxiv.org/abs/2505.24298)
- [THUDM/slime](https://github.com/THUDM/slime)
