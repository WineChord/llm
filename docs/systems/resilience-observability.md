# 系统韧性与可观测性

大模型系统的可靠性目标不是“进程还能运行”，而是持续产生可验证的训练进度或满足 SLO 的推理结果。一个 rank 卡住、网络降速、GPU 产生不可纠正错误、checkpoint 部分完成，都可能让进程表面存活却不再产生有效 goodput。

韧性设计应从故障模型、状态边界和恢复目标开始，再决定 heartbeat、重试、弹性或冗余。

## 先定义服务目标

训练系统可使用有效 token goodput：

$$
G_{\mathrm{train}}
=\frac{\text{通过数值与数据一致性检查的 token}}
{\text{wall time}}.
$$

推理系统则以满足全部 TTFT、TPOT、端到端延迟和质量约束的请求计数：

$$
G_{\mathrm{serve}}
=\frac{\text{满足 SLO 的完成请求}}
{\text{wall time}}.
$$

可用性和正确性不能相互替代。自动重试可能提高完成率，却也可能造成重复训练 step、重复计费、重复流式 token 或不一致的工具副作用。

恢复目标至少包括：

- RPO：允许丢失多少已完成进度；
- RTO：故障后多久恢复有效工作；
- 数据一致性：是否允许重样本、跳样本或近似 resume；
- 服务语义：请求是 at-most-once、at-least-once 还是幂等重放；
- 降级边界：允许牺牲哪些质量、长度或吞吐。

## 故障分类

| 层次 | 典型故障 | 需要的证据 |
| --- | --- | --- |
| GPU | ECC、Xid、温度、降频、kernel hang | 设备健康、错误事件、时钟与功耗 |
| 进程 | crash、OOM、死锁、异常退出 | exit code、stack、allocator 与 trace |
| Collective | rank mismatch、超时、链路降速 | 各 rank 调用序列、count、dtype、网络计数 |
| 主机 | 内存、NUMA、磁盘、重启 | OS 与资源 telemetry |
| 网络 | packet loss、拥塞、NIC / switch 故障 | link、RDMA、重传、端到端带宽 |
| 存储 | 超时、部分写入、损坏 | manifest、checksum、提交标记 |
| 数据 | 坏样本、游标漂移、重复或跳样本 | sample ID、token cursor、版本 |
| 模型 | NaN、梯度爆炸、质量漂移 | loss、gradient、scale、固定探针 |
| 服务控制面 | 错路由、stale version、容量失配 | request / model / cache schema |

故障检测要区分 fail-stop、slowdown、Byzantine-like corruption 和控制面不一致。只对 crash 做重启不能处理静默数值损坏或 straggler。

## Collective 的正确性先于超时

[NCCL collective 语义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)要求各 rank 的 collective 调用匹配 count、datatype 与顺序。违反契约可能表现为 hang、crash 或静默数据损坏。

因此 timeout 只能说明“在时间窗内没有完成”，不能单独证明网络故障。排查顺序应包括：

1. 每个 rank 是否进入同一个逻辑 step；
2. process group 与 collective 顺序是否一致；
3. tensor count、dtype、device 和 stream 是否匹配；
4. 是否存在提前异常或 OOM 的 rank；
5. 最后再判断链路、拥塞和硬件。

watchdog 若在一个 rank 仍使用通信 buffer 时直接重启其他 rank，可能扩大故障。需要先终止或隔离同一一致性组，再进行恢复。

## Straggler 是可靠性问题

同步训练由最慢 rank 决定 step time。可定义尾部比：

$$
\rho_{\mathrm{tail}}
=\frac{T_{p99,\mathrm{rank}}}
{\operatorname{median}_r T_r}.
$$

还应按 phase 分解 compute、collective、input 和 checkpoint。一个 rank 的 step 慢可能来自：

- GPU 降频或错误恢复；
- NUMA / CPU 数据线程；
- 特定 NIC、交换链路或拥塞；
- allocator / page fault；
- 数据 shape 偏斜；
- expert 或 sequence 负载不平衡；
- 异步 checkpoint staging 抢占 CPU / 网络。

只看平均 GPU utilization 会把这些原因混在一起。[NVIDIA DCGM profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html) 与[诊断工具](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html)可提供设备级证据，但仍需与模型 phase 和分布式 trace 对齐。

## 状态机与故障域

把训练作业表示为

$$
\text{initializing}
\rightarrow
\text{running}
\rightarrow
\text{quiescing}
\rightarrow
\text{checkpointing}
\rightarrow
\text{recovering}.
$$

每个状态应定义允许的外部动作和持久化边界。恢复不能简单地“从最近目录读取”，而应读取最近一个经过原子提交、checksum 和兼容性检查的 checkpoint。

故障域决定一致性组：

- 一个 DP replica 失败是否需要重启整个 world；
- pipeline 某 stage 能否由备用副本接管；
- serving worker 失败是否只丢失本地 KV；
- router 和 KV store 是否持有可恢复的全局状态；
- 网络分区时哪一侧允许继续接受写入。

弹性缩放改变 world size 时，还会改变 shard、batch 和数据游标语义，必须与[检查点与容错](checkpointing.md)的跨拓扑恢复共同设计。

## 重试、弹性与冗余

[TorchElastic agent](https://docs.pytorch.org/docs/stable/elastic/agent.html) 提供 worker group 的失败检测和重启语义；它不会自动证明数据游标、optimizer 或自定义外部状态的一致性。

[TorchFT](https://github.com/pytorch/torchft) 探索训练中的容错与副本协调；[NVIDIA Resiliency Extension](https://github.com/NVIDIA/nvidia-resiliency-ext) 提供 fault detection、straggler detection 与进程管理等能力。采用这些机制前仍需回答：

- 哪些状态被复制；
- 复制在何时达到一致；
- 故障切换会丢失多少 step；
- 外部数据和 checkpoint 如何去重；
- 网络带宽和显存开销是否进入容量模型。

冗余并不总是最优。若 checkpoint 恢复很快、故障率低，备用 GPU 的持续成本可能大于收益；若模型 step 很长或集群 MTBF 随规模快速下降，局部恢复与冗余更有价值。

## 可观测性数据模型

指标应围绕状态与因果关系组织，而不是堆积 dashboard。

### 训练

- global step、有效 token、数据游标；
- step、forward、backward、optimizer、collective 分位；
- MFU / HFU 与 HBM / network throughput；
- 各 rank tail ratio；
- loss、gradient norm、overflow、scale；
- checkpoint staging、写入、提交和恢复时间；
- restart、lost work、重复或跳过样本；
- GPU Xid、ECC、温度、功耗与时钟。

### 推理

- queue time、TTFT、TPOT、E2E 和 goodput；
- active / waiting / preempted / cancelled 请求；
- KV pool、fragmentation、prefix hit 与 eviction；
- prefill、decode、transfer 与 sampler 时间；
- model / adapter / cache schema version；
- OOM、timeout、stream disconnect、retry 与重复输出；
- compile、graph miss 和 fallback 比例。

高基数字段如 request ID、rank、model version 用于 trace 或结构化 log；不要把 prompt、生成文本或 token 内容默认放入 telemetry。指标 label 无界增长会反过来破坏观测系统。

## Trace 与关联

一次训练 step 或推理请求应具有稳定关联 ID，使以下事件可拼接：

$$
\text{queue}
\rightarrow
\text{compute}
\rightarrow
\text{collective/transfer}
\rightarrow
\text{commit/stream}.
$$

时钟不同步会破坏跨节点 trace 顺序；需要记录单调时钟 duration，并对 wall-clock 做同步质量监测。采样 trace 时应保留错误、尾延迟和版本切换事件，不能只随机保留平均请求。

## 健康检查

- **liveness**：进程或服务循环仍能推进；
- **readiness**：当前实例能够安全接收新工作；
- **progress**：有效 step、token 或请求确实增长；
- **correctness probe**：固定输入仍满足数值或协议不变量。

只检测端口或 heartbeat 不能证明 collective、GPU 或模型可用。反过来，短时没有新请求也不是服务失败。健康检查应结合当前状态和工作负载。

## 正确性契约

- 同一训练一致性组只认可一个 active generation / epoch。
- restart count、checkpoint ID、data cursor 和 model version 单调。
- committed checkpoint 不可在后台修改。
- 请求重放必须有 idempotency key 或明确的重复语义。
- streaming token 的序号单调，finish 只发生一次。
- readiness 在模型、adapter、KV schema 未加载完成时保持 false。
- 旧 worker drain 后不得接收新版本请求。
- 观测事件携带版本和状态，但默认不携带敏感内容。
- alert 必须引用可操作证据，并区分症状与根因。

## 自动恢复何时会伤害系统

- collective mismatch 尚未修复，重试只会重复 hang；
- checkpoint 未原子提交，自动回退可能读到部分状态；
- OOM 来自确定性 shape，原配置重启必然再次失败；
- NaN 来自数据或模型错误，跳过并继续会掩盖质量损坏；
- 非幂等外部副作用可能被重复；
- 网络分区中双方都继续写入，产生 split brain；
- stale model / KV schema 被重新加入流量；
- retry storm 占满控制面、存储或网络。

自动化应有速率上限、熔断、隔离和人工升级边界。无法证明安全重放时，应停止而不是追求表面可用。

## 故障注入与验证

验证矩阵至少覆盖：

1. 单 worker 正常退出、强制退出和无响应；
2. collective 前后某 rank 失败；
3. NIC 降速、丢包、网络分区和节点内链路退化；
4. GPU OOM、Xid、降频和长 kernel；
5. 数据读取超时、坏样本和游标不一致；
6. checkpoint shard 缺失、损坏、重复和 partial manifest；
7. 恢复时改变 DP / TP / PP / EP 拓扑；
8. serving worker 带活跃 KV 退出；
9. model、adapter 或 cache schema 灰度切换；
10. telemetry 后端不可用，确认业务不会被观测链路拖垮。

每次实验记录 detection time、quiesce time、lost work、restore time、重复工作和最终数值 / SLO。故障注入的通过标准不是“自动重启成功”，而是状态不变量仍成立、错误可定位且恢复成本符合目标。

## 何时采用更简单的方案

单机短作业、可快速重跑的实验不一定需要复杂弹性框架；小规模服务也可能用进程隔离、健康检查和无状态重启即可。复杂恢复机制只有在以下条件下才值得：

- 故障频率和 lost-work 成本已量化；
- 状态边界清楚且可测试；
- 恢复时间确实优于全量重启；
- 引入的副本、网络和控制面成本可接受；
- 团队能维护故障注入和升级兼容矩阵。

可靠性的成熟度应以恢复演练和长期 goodput 证明，而不是以配置项数量衡量。collective、全局归一化与 checkpoint 提交的不变量可结合[手撕：分布式与容错](../practice/distributed-systems.md)逐项验证。

## Reference {#reference}

- [NCCL collective 语义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
- [NVIDIA DCGM profiling](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html)
- [NVIDIA DCGM diagnostics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html)
- [TorchElastic agent](https://docs.pytorch.org/docs/stable/elastic/agent.html)
- [TorchFT](https://github.com/pytorch/torchft)
- [NVIDIA Resiliency Extension](https://github.com/NVIDIA/nvidia-resiliency-ext)
