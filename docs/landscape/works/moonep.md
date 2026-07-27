# MoonEP：用动态冗余 Expert 固定每个 Rank 的负载

Mixture of Experts（MoE）把每个 token 只送给少量 expert，因而能在扩大总参数量的同时控制激活计算量。但稀疏激活没有自动解决系统负载：router 可以在这一层偏向 expert 7，下一层又突然偏向 expert 203；expert 的归属 rank 是静态的，token 分布却随 batch、层和训练进度不断变化。

[MoonEP 官方仓库](https://github.com/MoonshotAI/MoonEP)给出了一条很直接的系统路线：不改变 router 选中的 global expert，而是在当前 step 动态规划少量 expert 副本，把同一 expert 的 token 分给不同执行 rank，使每个 rank 接收完全相同数量的 routed token。它解决的不是“平均看来还算均匀”，而是同步系统中最慢 rank 决定关键路径的问题。

## 为什么动态路由特别难

设一个 expert-parallel group 有 $R$ 个 rank，每个输入 rank 有 $S$ 个 token，每个 token 选择 top-$K$ expert，总 routed record 数为

$$
N_{\mathrm{route}}=RSK.
$$

若共有 $E$ 个 routed expert，expert $e$ 的本 step token 数为 $T_e$，则

$$
\sum_{e=1}^{E}T_e=RSK.
$$

传统 expert parallelism 常把 expert 固定放在 home rank。若每个 rank 持有 $E/R$ 个 expert，rank $r$ 的计算负载是

$$
L_r=\sum_{e\in\mathcal E_r}T_e.
$$

即使长期平均 $\mathbb E[L_r]$ 接近 $SK$，某一步的 $\max_r L_r$ 仍可能很大。同步训练要等最慢 rank 完成 dispatch、expert GEMM 和 combine；在线 decode 的小 token batch 又更容易让个别 expert 形成 skinny GEMM。更麻烦的是，$L_r$ 每步变化会让通信 split、临时张量和 grouped GEMM shape 同时变化，产生 host 同步、allocator 碎片和 graph 复用困难。

因此“加一个 load-balancing loss”并不等于系统问题消失。辅助目标会改变模型学习，而系统仍需为极端 batch 留容量；capacity factor、padding 或 token drop 又分别浪费计算或改变模型语义。router 与 MoE 层的数学边界见[稀疏 MoE](../../architecture/moe.md)，dispatch、placement 与 grouped GEMM 的系统契约见[MoE 系统](../../systems/moe-systems.md)。

## Perfect balance 的目标

MoonEP 把规划结果写成非负整数分配 $x_{e,r}$：expert $e$ 的多少条 routed record 由 rank $r$ 执行。它需要同时满足

$$
\sum_{r=1}^{R}x_{e,r}=T_e
\qquad \forall e,
$$

以及

$$
\sum_{e=1}^{E}x_{e,r}=SK
\qquad \forall r.
$$

第一条保证每个 expert 的 token 守恒；第二条让每个 rank 恰好执行 $SK$ 条真实 routed record。若 $r$ 不是 expert $e$ 的 home rank 且 $x_{e,r}>0$，该 rank 就需要访问或预取 $e$ 的权重副本。

这里的 redundant expert 是 **同一组权重的执行副本**，不是 router 多出一个新选项：

- token 的 top-$K$ global expert ID 不变；
- route weight 与 combine 位置不变；
- 副本参数必须与 home expert 同版本；
- 训练产生的副本梯度最终归约回 home expert。

所以 perfect rank balance 保持模型函数不变，却不保证所有硬件时间都完全相同。不同 expert 的 GEMM shape、节点拓扑、预取距离、padding 和 kernel occupancy 仍可能产生尾部；它固定的是最重要、也最容易审计的 token-count 与 buffer-shape 不变量。

## 从 Router 输出到 Online Plan

规划不能离线做好，因为 $T_e$ 只有当前 router forward 后才知道。一次可审计的数据流是：

```text
local router top-k
  -> aggregate per-expert counts across EP ranks
  -> build expert-to-destination allocation
  -> materialize token offsets and cu_seqlens
  -> prefetch required remote expert weights
  -> dispatch directly into expert-grouped slots
  -> grouped expert GEMM
  -> combine to original token order
```

官方实现把 planning 放在 GPU 上，并将 plan 同时交给 prefetch、dispatch、combine 和 backward。这样 backward 可以复用 forward 的 token placement，而不是根据一个可能不同的顺序重新规划。[Kimi K3 技术报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)还在附录中分析了冗余 expert 的上界：对 $E$ 个 expert、$R$ 个 rank，其构造把每个 rank 所需冗余 expert 数限制在 $E/R$ 量级。这个结论解释了为什么训练时可以预留有界的副本槽；具体 GPU planner 的线程组织、原子操作顺序和端到端收益仍应以当前实现与目标硬件实测为准。

### 最小分配模型 {#exact-rank-plan}

下面的 reference 只固定两条核心语义：每个 expert 的 token 数守恒、每个 rank 的真实负载相等。它先尽量使用 home rank，再把溢出分配到剩余容量；它没有复现官方 GPU planner，也不声称最小化副本数量。

```python
def exact_rank_plan(expert_counts, ranks):
    experts = len(expert_counts)
    if ranks <= 0 or experts == 0 or experts % ranks:
        raise ValueError("experts must divide evenly across positive ranks")
    if any(type(count) is not int or count < 0 for count in expert_counts):
        raise ValueError("expert counts must be non-negative integers")
    total = sum(expert_counts)
    if total % ranks:
        raise ValueError("total routed tokens must divide evenly across ranks")
    capacity, local = total // ranks, experts // ranks
    allocation = [[0] * experts for _ in range(ranks)]
    spill = []
    for expert, count in enumerate(expert_counts):
        home = expert // local
        used = sum(allocation[home])
        keep = min(count, capacity - used)
        allocation[home][expert] = keep
        if count > keep:
            spill.append([expert, count - keep])
    for item in spill:
        expert, remaining = item
        for rank in range(ranks):
            room = capacity - sum(allocation[rank])
            take = min(remaining, room)
            allocation[rank][expert] += take
            remaining -= take
        if remaining:
            raise AssertionError("global rank capacity is inconsistent")
    assert all(sum(row) == capacity for row in allocation)
    assert all(sum(row[e] for row in allocation) == count
               for e, count in enumerate(expert_counts))
    return allocation

plan = exact_rank_plan([9, 1, 2, 0, 3, 5, 4, 8], ranks=4)
assert [sum(row) for row in plan] == [8, 8, 8, 8]
assert [sum(row[e] for row in plan) for e in range(8)] == [9, 1, 2, 0, 3, 5, 4, 8]
assert plan[1][0] > 0
```

最后一个断言展示了动态副本的本质：expert 0 的 home rank 容量不足时，一部分 expert 0 token 在其他 rank 执行，但它们仍计入 expert 0。生产 planner 还需优化 remote expert 数、迁移距离、prefetch slot 和并行 kernel 开销；不能把这个计数 reference 当成调度算法的性能实现。

## Zero-copy 与静态 Buffer

完美平衡的另一个价值，是把“这一层究竟要多少真实 token slot”固定为每 rank $SK$。MoonEP 的物理 buffer 还可以包含按 VM group 对齐的 padding，因此公开 API 中常见的是固定 `NvS` 容量；逻辑有效 token 仍由 `cu_seqlens` 定义。

官方仓库披露的权重布局为每个 expert projection 建立连续 symmetric-memory 范围：

```text
[0, E)       home expert rows
[E, E + B)   local prefetch slots
```

grouped GEMM 只读取一份连续的 `[E+B,H,H']` 视图，`cu_seqlens` 指出本 step 哪些 expert row 拥有 token。dispatch 可以把 token 直接写入远端最终的 expert-grouped 位置，并把通信 buffer view 交给 GEMM；combine 再从同一布局恢复 token-major 顺序。这样避免“通信 buffer → 用户 buffer”的额外边界 copy，也避免每层为动态 shape 重新分配。

Zero-copy 会把生命周期约束变得更严格：返回 view 与通信 buffer alias，下一次 dispatch/combine 可能覆盖它；若 autograd 需要长期保存 activation，就必须切换到拥有独立存储的路径。静态 shape 也不是“无视 padding”，而是让 padding、有效区间和未定义尾部都由 plan 明确描述。

## 训练与推理不是同一配置

训练需要保证副本权重参与 forward/backward 后，梯度只归并一次。MoonEP 的公开接口为 prefetch-slot gradient 使用独立 reduce buffer，再由 expert home rank 读取并累加，避免框架原有 data-parallel reduction 把临时副本误当成独立参数。

官方仓库给出的配置边界是：

- **训练**：prefetch slot 数 $B=E/R$，使 planner 选中的 remote expert 都可在本地副本执行；副本梯度归约回 owner；
- **推理**：没有梯度，允许 $B<E/R$；超出 prefetch slot 的 remote expert 可通过 symmetric mapping 直接读取 home 权重，以较少显存换取可能更慢的访问。

这一区别不能简化成“推理把 $B$ 调小即可”。在线服务还受 continuous batching、请求取消、graph bucket、模型热更新和尾延迟约束；训练则更关注 backward plan 重放、gradient visibility 与固定 step time。两者都需把 plan、权重 revision、stream event 和 buffer owner 作为一个事务。

## 与 Kimi K3 的关系

[Kimi K3](kimi-k3.md)采用 896 个 routed expert、每个 token 激活 16 个 routed expert，并用 Quantile Balancing 改善 router 负载；MoonEP 位于它的系统侧，处理即使经过模型级平衡后仍会出现的 step-level rank skew。两层作用不同：

```text
router balancing
  -> 改善 expert 选择分布，但可能改变学习动力学
MoonEP planning
  -> 保持 expert 选择不变，重新安排执行位置
```

报告把 MoonEP、专家 GEMM、并行布局和内存策略放在同一训练系统中。这说明通信库的收益必须放回端到端关键路径：planner、权重预取、dispatch、GEMM、combine 和 gradient reduction 都要计时，不能只比较裸 all-to-all。

## 正确性与验证

最小验证矩阵应覆盖：

1. 每个 token 保持原 top-$K$ expert ID、route weight 与 combine 顺序；
2. 每个 expert 的全局 token 数在规划前后完全一致；
3. 每个 rank 的真实 routed token 数恰为 $SK$；
4. 空 expert、单一热点 expert、全部均匀与极端偏斜；
5. dispatch/combine 与单 rank reference 的输出和梯度对齐；
6. 副本梯度只回到正确 home expert，且没有重复 reduction；
7. zero-copy view 的覆盖、释放和 stream event 生命周期；
8. padding、`cu_seqlens`、静态 capacity 与 grouped GEMM 读取区间一致；
9. checkpoint 或 EP size 改变后，global expert ID 与 owner 映射可恢复；
10. planning、prefetch、通信、GEMM 和 backward 分项计时。

collective 与通信次序见[集合通信与状态分片](../../systems/collectives-sharding.md)，更完整的 token permutation、并行线性层和 checkpoint reference 见[分布式系统手撕实现](../../practice/distributed-systems.md)。

## 证据边界

MoonEP 当前公开入口包含实现、测试、benchmark 脚本和集成契约，但不是一篇给出全部算法证明、硬件覆盖与多集群结果的完整论文。官方仓库的 benchmark 绑定其披露的设备、EP 配置和路由生成方式；K3 报告中的系统结论又绑定 K3 的模型形状与训练栈。由此可以确认 perfect balance、online planning、静态 buffer、zero-copy、prefetch 和 gradient reduction 的机制，不能据此宣称：

- 任意网络拓扑和模型形状都有同样加速；
- token 数严格相等就意味着 wall-clock 完全相等；
- inference 的小 $B$ 配置适合训练；
- 冗余 expert 不增加任何显存、带宽或同步成本；
- 动态副本可以替代 router 的质量与稳定性治理。

## Reference {#reference}

- [MoonshotAI/MoonEP Expert Parallelism Library](https://github.com/MoonshotAI/MoonEP)
- [Kimi K3 Technical Report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)
- [DeepSeek-AI/DeepEP Communication Library](https://github.com/deepseek-ai/DeepEP)
