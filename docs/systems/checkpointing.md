# 检查点与容错

大模型训练的 checkpoint 不是一个权重文件，而是恢复计算轨迹所需的分布式状态快照。系统需要同时回答：

1. 哪些状态决定下一步计算？
2. 多个 shard 怎样形成一个不可分割的提交？
3. 保存时怎样避免阻塞训练或复制正在变化的 buffer？
4. world size 与并行拓扑改变后怎样恢复同一个逻辑 tensor？

只恢复模型权重属于 warm start，不等价于严格 resume。

## 恢复状态

严格 resume 通常需要：

- 模型参数与 buffer；
- optimizer state、参数分组和更新步数；
- learning-rate scheduler；
- gradient scaler、amax history 与低精度 scale；
- global step、已见 token 数、样本与 packing 游标；
- CPU、GPU、dropout 与随机舍入 RNG；
- data loader、shuffle 与动态采样状态；
- DP、TP、PP、CP、EP 的 mesh 与 tensor placement；
- 模型、tokenizer、代码、容器、依赖和数据版本。

状态可按恢复强度分层：

| 模式 | 状态 | 可承诺的语义 |
| --- | --- | --- |
| 权重加载 | 模型参数 | 新优化轨迹的 warm start |
| 训练恢复 | 权重、optimizer、scheduler、scaler | 优化状态连续 |
| 严格恢复 | 再加数据游标、RNG、拓扑和版本 | 在误差契约内延续同一轨迹 |

分布式 reduction 和低精度 kernel 可能使恢复后不具备 bitwise 一致性；应提前定义允许的 loss、梯度与收敛误差，而不是事后降低标准。

## 逻辑 tensor 与物理 shard

checkpoint 应优先描述逻辑 tensor：

$$
T=
\operatorname{assemble}
\left(
\{T_r,\operatorname{offset}_r,\operatorname{shape}_r\}_{r=1}^{p}
\right).
$$

每个 shard 记录 global shape、global offset、local shape、dtype 和 placement。旧 topology 恢复到新 topology 时，流程是

$$
\text{old shards}
\rightarrow
\text{logical tensor ranges}
\rightarrow
\text{new shards}.
$$

若格式只用文件名中的 local rank 表示身份，就难以在 TP / PP / EP size 改变后安全 reshard。global parameter ID、expert ID 和 pipeline layer range 必须独立于当前 rank 编号。

[PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/2.8/distributed.checkpoint.html)提供分布式 state dict、planner 与跨 rank 保存 / 读取接口。具体格式仍需由调用方补齐数据游标、版本、外部状态和提交协议。

## Manifest

一个 snapshot manifest 至少包含：

- 不可变 snapshot ID；
- 创建时间、global step、token 与 data cursor；
- 模型、tokenizer、代码、数据和 schema version；
- 原 topology 与 process mesh；
- 每个逻辑 tensor 的 shape、dtype、placement；
- 每个 shard 的对象名、offset、长度与 checksum；
- optimizer、scheduler、scaler 和 RNG 的存在性；
- parent snapshot 或增量依赖；
- writer 数量和格式版本；
- committed 状态。

manifest 是一致性边界，不只是文件清单。恢复端必须先验证 schema，再验证全部必需 shard 和 checksum，最后才分配或加载模型状态。

## 原子提交协议

对象存储或分布式文件系统通常不能让一组文件天然原子。可采用两阶段协议：

1. 为本次保存生成唯一 snapshot ID。
2. 各 rank 写入临时命名空间，只写不可变 shard。
3. writer 完成后返回 size、checksum 与逻辑 range。
4. coordinator 收齐并验证所有必需 shard。
5. 写入完整 immutable manifest。
6. 最后原子创建很小的 committed marker，或原子更新 latest 指针。

恢复端只枚举 committed snapshot：

$$
\text{readable}
\iff
\text{marker exists}
\land
\text{manifest valid}
\land
\forall i,\ \operatorname{checksum}(S_i)=c_i.
$$

marker 必须最后写入。仅依赖目录存在、文件数量或修改时间，会把部分写入误认为可恢复状态。`latest` 只是可重建索引，不应成为唯一真相。

失败保存产生的临时对象可以延迟回收，但回收器不得删除仍被某个 committed 增量快照引用的 parent 或 shard。

## 同步与异步保存

同步保存使所有 rank 在一致点停下，语义简单但暂停时间长。异步保存通常分两阶段：

1. 在一致 step 边界把本地状态复制为不可变 staging snapshot；
2. 后台将 staging 数据写入持久存储。

设本 rank checkpoint 大小为 $M_{\mathrm{local}}$，异步路径至少需要约

$$
M_{\mathrm{staging}}
\gtrsim M_{\mathrm{local}}
$$

的额外 CPU 或其他稳定存储空间，除非采用更细粒度流水。保存中的 buffer 不能被下一训练 step 修改或 allocator 复用。

[PyTorch 异步 checkpoint recipe](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html)明确把 state 搬到 CPU staging，再由后台 future 完成写入，并建议限制同时只有一个异步保存。若无界累积多个 future，CPU 内存、PCIe、网络和存储队列都可能失控。

异步并不意味着零开销：

- device-to-host copy 竞争 PCIe / 内存带宽；
- state snapshot 会短暂增加显存或 CPU 内存；
- 后台压缩、checksum 和写入消耗 CPU；
- 共享网络上的 checkpoint 会干扰 collective；
- 下一次保存可能等待上一次完成。

需要在 trace 中分别记录 snapshot pause 与 background drain。

## 一致保存点

checkpoint 边界应明确位于：

$$
\text{optimizer step complete}
\rightarrow
\text{state snapshot}
\rightarrow
\text{next data advance}.
$$

若在 gradient accumulation 中间保存，还需保存已累积 gradient、microbatch index 和相应 RNG。否则恢复后无法重建同一更新。

各 rank 的 global step 相同并不足够。还要检查：

- optimizer 是否全部完成；
- scheduler 是否已经推进；
- data cursor 是当前 batch 还是下一 batch；
- scaler overflow 分支是否完成；
- pipeline 是否已 drain；
- 异步 collective 与 kernel 是否完成使用相关 buffer。

一致点可通过 barrier 或协议实现，但 barrier 本身不能修复不同 rank 已处于不同逻辑 step 的错误。

## 跨拓扑恢复

从旧网格

$$
(p_{\mathrm{DP}},p_{\mathrm{TP}},p_{\mathrm{PP}},
p_{\mathrm{CP}},p_{\mathrm{EP}})
$$

恢复到新网格时，需要分别处理：

- replicated state 是否只读一个副本；
- TP shard 怎样按全局维度重切；
- PP layer range 怎样重新分配；
- optimizer state 是否与参数使用相同 placement；
- expert global ID 怎样映射到新 EP rank；
- data-parallel batch 与数据游标怎样保持 token 语义；
- RNG 是按 global sample 还是按 rank 派生。

跨拓扑恢复不应要求某个 rank 聚合完整模型，否则大模型会在 CPU 或单卡产生不可接受的峰值。更好的 planner 直接计算旧 shard 与新 shard range 的交集并流式传输。[ByteCheckpoint](https://arxiv.org/abs/2407.20143)研究了大规模训练中高效、弹性的分布式 checkpoint，可作为 2024 年的系统设计实例，而非通用格式标准。

## 保存间隔

设一次 checkpoint 的不可隐藏耗时为 $C$，系统平均故障间隔为 $M$，保存间隔为 $\tau$。忽略恢复时间和高阶项时，单位时间浪费近似为

$$
f(\tau)
\approx
\frac{C}{\tau}
+\frac{\tau}{2M}.
$$

第一项是保存开销，第二项是假设故障在区间内均匀发生时的平均 lost work。求极值得到 Young 一阶近似：

$$
\tau_{\mathrm{Young}}\approx\sqrt{2CM}.
$$

[Daly 的高阶分析](https://doi.org/10.1016/j.future.2004.11.016)在 Poisson 单组件故障、长作业等假设下，对“两个 checkpoint 之间的有效计算时间”给出更精细的近似：

$$
\tau_{\mathrm{Daly}}
\approx
\sqrt{2CM}
\left(
1+\frac{1}{3}\sqrt{\frac{C}{2M}}
+\frac{1}{9}\frac{C}{2M}
\right)-C,
\qquad C<2M.
$$

恢复时间会增加总 wall time，但在该高阶最优间隔近似中不直接出现。实际选择仍需加入异步保存的隐藏比例、集群规模相关 MTBF、存储限流和业务 RPO；故障相关、重尾分布、检测延迟不可忽略或 $C$ 接近 $M$ 时，应使用完整模型或实测优化，不能机械套用 Young / Daly。

## 增量与分层 checkpoint

完整快照简单但写入量大；增量快照只保存变化块，却引入依赖链。恢复成本可写为

$$
T_{\mathrm{restore}}
=T_{\mathrm{base}}
+\sum_{i=1}^{d}T_{\mathrm{delta},i},
$$

其中 $d$ 是增量链深度。链越长，任一 parent 丢失或损坏的影响越大。

分层策略可以组合：

- 高频本地 / 节点级快照；
- 低频远端持久快照；
- optimizer state 与权重不同周期；
- 训练恢复格式与部署权重导出格式分离。

只有远端持久层能覆盖整机、机架或站点故障。本地快照不应被统计为相同故障域下的完整 RPO 保证。

## 正确性契约

- snapshot ID 全局唯一且提交后不可变。
- 每个逻辑 tensor 的 ranges 不重叠、不缺失，除非 schema 明确允许复制。
- manifest 写入前，全部必需 shard 已持久化并校验。
- committed marker 只在 manifest 完整后出现。
- 恢复端永不读取未提交 snapshot。
- staging buffer 在后台完成前不可修改或释放。
- 同时进行的异步保存数量有界。
- data cursor、global token、scheduler step 与 optimizer step 关系明确。
- RNG、scaler、amax 与低精度 scale 属于恢复状态。
- 格式升级有显式 schema version 和转换器。
- strict resume、approximate resume 与 warm start 在接口和日志中区分。

ZeRO/FSDP shard 与 reshard 的基础语义见[集合通信与状态分片](collectives-sharding.md)。

## 失效模式

- **半写目录被发现**：缺少最终 committed marker。
- **不同 rank 保存不同 step**：只有局部 step，没有全局一致点。
- **异步 buffer 被覆盖**：快照不是不可变副本。
- **只校验文件存在**：静默截断或 bit rot 未被发现。
- **rank 文件名即身份**：拓扑改变后无法定位逻辑 range。
- **只恢复模型与 optimizer**：数据、scheduler 或 RNG 漂移。
- **增量 parent 被清理**：manifest 仍在但依赖链断裂。
- **保存导致训练降速**：PCIe、CPU、网络或存储争用未进入关键路径模型。
- **重试写入同名对象**：不同 attempt 互相覆盖，manifest 混合。

对于短实验或可从头重跑的小模型，复杂增量和跨拓扑格式可能得不偿失；一个有 checksum 的同步完整快照通常更可靠。

## 故障注入

至少覆盖：

1. 某 rank 在 shard 写入前、写入中、manifest 前和 marker 前退出；
2. shard 截断、checksum 错误、重复 range 和缺失 range；
3. coordinator 失败并由新 coordinator 重试；
4. 对象存储出现超时、重排可见性或重复响应；
5. 异步保存尚未完成时触发下一次保存；
6. 保存过程中训练进程 OOM 或被抢占；
7. 从不同 DP / TP / PP / EP 网格恢复；
8. schema、模型、tokenizer 或 optimizer 版本不兼容；
9. data cursor 恢复后出现重样本或跳样本；
10. 本地快照可用但远端持久层不可用。

每个故障点都应证明：未提交快照不会被读取，最近一次已提交快照仍可恢复，临时对象可安全回收。

## 恢复验证

1. 在独立进程和干净环境中读取，而不是仅由 writer 自检。
2. 核对 manifest、shard checksum、global shape、dtype 与 range 覆盖。
3. 比较恢复前后的模型参数、optimizer step、scheduler 和 scaler。
4. 对固定 batch 比较 loss、logits、梯度统计和下一次更新。
5. 比较样本 ID、token cursor 与 RNG 派生序列。
6. 在至少一个不同 topology 上完成 reshard 与下一 step。
7. 记录 snapshot pause、后台 drain、存储吞吐、RTO 和 lost work。
8. 长期演练 partial write、worker failure 和存储故障。

恢复排查可结合[调试手册](../practice/debugging.md)，manifest 原子提交与恢复校验的紧凑 reference 见[手撕：分布式与容错](../practice/distributed-systems.md)。通过一次读取测试只证明一个样本可读；可靠 checkpoint 需要持续的故障注入、跨版本迁移和独立恢复证据。
