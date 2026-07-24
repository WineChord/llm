# 调试手册

大模型问题往往跨越数据、数学、精度、分布式和服务层。最快的方法是建立最小参考，并逐层增加复杂度。

## Loss 异常

1. 打印 token、label、mask 与每样本 loss。
2. 验证 shift、padding、packed boundary 和 ignore index。
3. 固定单 batch，关闭 dropout，使用高精度单卡。
4. 比较恢复 checkpoint 前后的 optimizer、scheduler 与数据游标。
5. 再打开混合精度、gradient accumulation 和并行。

NaN 出现后降低学习率可能暂时隐藏问题；先定位第一个非有限张量及其上游。

## 分布式 Hang

- 确认所有 rank 以相同顺序进入 collective。
- 对比 tensor shape、dtype、process group 和条件分支。
- 检查 dataloader 是否让某些 rank 提前耗尽。
- 区分网络超时、kernel hang、OOM 后未退出和 Python 死锁。
- 用更小 world size 和同步执行建立可复现点。

## OOM

先记录峰值与各状态理论账本，再区分：

- 静态权重/optimizer 放不下；
- activation 随 batch 或长度增长；
- KV Cache 或临时 workspace；
- allocator 碎片；
- 某个 rank 因长度或专家路由不均衡超载；
- 内存泄漏或未释放计算图。

## 训练与推理不一致

检查 tokenizer、chat template、BOS/EOS、position IDs、norm、精度、rope scaling、attention mask、权重 tying、量化 scale 和 adapter。对同一前缀逐层比较 hidden state 与 logits，找到第一个分叉位置。

## 服务尾延迟

把端到端 trace 分为排队、tokenizer、prefill、decode、通信、采样与网络发送。检查长 prompt、GC、prefix cache miss、KV 抢占、batch 形状变化和过载队列。平均 tokens/s 正常时，P99 仍可能因少数长请求失控。

## 最小原则

保留一个慢但可信的 reference path。优化路径只有在固定输入上同时通过数值、任务质量和性能验证后才可替代参考。
