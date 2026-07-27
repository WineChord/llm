# 检查点与容错

大模型训练的 checkpoint 不是单个权重文件，而是恢复计算轨迹所需的分布式状态集合。

## 完整状态

- 模型参数与 buffer；
- optimizer state 和参数分组；
- learning-rate scheduler；
- gradient scaler 与混合精度状态；
- global step、已见 token 数与样本游标；
- Python、NumPy、CPU/GPU 随机数状态；
- data loader、shuffle、packing 与动态采样状态；
- 并行拓扑、分片元数据和模型配置；
- 代码、容器、依赖与数据版本。

只恢复权重属于 warm start，不等价于严格 resume。

## 分布式格式

每个 rank 独立保存本地 shard 能减少聚合内存，但文件数量、元数据一致性和拓扑变化更复杂。好的格式应支持：

- 原子完成标记，避免读取半写 checkpoint；
- checksum 与 tensor shape/dtype 校验；
- 并行度变化时 reshard；
- 异步保存且不复用仍在计算的 buffer；
- 旧版本迁移和最小可读工具。

## 故障模型

硬件故障只是其中一类。还需处理网络抖动、单 rank 卡死、数据损坏、存储超时、collective mismatch、OOM、NaN、主机重启与作业抢占。

## 恢复验证

1. 在小规模作业中定期注入中断。
2. 比较中断前后的数据样本 ID 和 learning rate。
3. 对固定 batch 比较恢复前后的 loss 与梯度统计。
4. 验证 checkpoint 能在独立环境读取和转换。
5. 记录恢复时间目标与允许丢失的 step 数。

容错不是“失败后能重新提交”，而是能证明没有静默跳样本、重样本、错 scheduler 或读到不一致 shard。

ZeRO/FSDP shard 与 reshard 语义见[集合通信与状态分片](collectives-sharding.md)，数值与数据游标的恢复排查见[调试手册](../practice/debugging.md)。
