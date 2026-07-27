# 数据与环境

Agentic RL 的数据不是静态问答对，而是带环境版本、状态变化、工具结果和终止原因的轨迹。若这些信息不可重放，训练集就无法审计。

## 轨迹模式

```text
episode_id
task_spec
environment_image / commit / seed
policy_version
observation_0
action_0
tool_result_0
...
terminal_state
reward_components
verifier_version
cost
```

工具返回的完整原始内容可能很大，也可能含敏感信息。训练存储应保留可重建的结构化字段、必要哈希与脱敏摘要，而不是无边界复制日志。

## 任务来源

| 来源 | 优点 | 风险 |
| --- | --- | --- |
| 人工编写 | 目标清晰、可控 | 规模小、风格单一 |
| 真实历史任务 | 分布真实 | 隐私、许可与不可重复状态 |
| 模板生成 | 覆盖系统化 | 容易被模板特征泄漏 |
| 模型自生成 | 扩展快 | 错误累积、难度虚假 |
| 程序化环境 | 自动验证 | 任务空间可能过窄 |
| curriculum | 可控制难度 | 难度指标不一定等于能力 |

合成数据应使用独立 verifier，并保留生成模型与筛选规则，否则 student 可能只学习 teacher 的格式偏差。

## 环境的四类状态

1. **静态状态**：代码仓库、文档、数据库快照；
2. **可变状态**：文件编辑、进程、浏览器页面、游戏局面；
3. **外部状态**：API、网络服务、时间和第三方数据；
4. **权限状态**：可读、可写、需确认与禁止动作。

只有静态快照通常可完全重放。外部服务应使用版本化模拟器、录制回放或明确的在线评测窗口。

## 工具契约

动作空间最好是结构化 schema，而不是从自然语言中猜测命令：

```json
{
  "tool": "read_file",
  "arguments": {
    "path": "src/model.py",
    "start_line": 1,
    "end_line": 120
  }
}
```

训练时需统一：

- 工具名、参数类型与错误码；
- stdout/stderr 截断规则；
- 超时、重试与幂等语义；
- 路径、网络和写权限；
- 结果是否进入下一轮上下文。

工具 schema 变更会造成 environment drift，应与模型和轨迹共同版本化。

## Verifier

强 verifier 直接判断目标状态，如单元测试、形式证明、棋局结果或数据库约束。弱 verifier 依赖模型评分或启发式。

Verifier 至少要测试：

- **soundness**：通过是否真的意味着正确；
- **completeness**：正确解是否可能被误拒；
- **isolation**：agent 能否读取答案或修改评分器；
- **determinism**：相同状态是否得到一致结果；
- **coverage**：是否只测到表面格式。

若 agent 可以修改测试文件，奖励函数就必须从受保护环境运行；否则“全部通过”可能只是 reward hacking。

## 数据筛选

轨迹质量不等于最终奖励。应同时考虑：

$$
q(\tau)=f(
\text{success},
\text{validity},
\text{novelty},
\text{efficiency},
\text{diversity},
\text{replayability}).
$$

- 成功轨迹提供正例，失败轨迹可训练恢复和批判；
- 极短成功可能是答案泄漏，极长成功可能包含大量无效探索；
- 相似任务与相似推理应去重；
- 同一任务保留多种策略有助于组相对训练；
- 工具或环境故障应与策略失败分开标注。

## Curriculum

难度可以由最短操作数、依赖深度、状态空间、工具数量或基线成功率定义。实用 curriculum 往往按当前策略成功率采样：

$$
p_i\propto w_i\cdot g(\hat s_i),
$$

其中 $\hat s_i$ 是任务 $i$ 的近期成功率，$g$ 提高“既非必胜也非全败”的任务权重。过度追逐边界样本可能遗忘基础能力，因此需保留 replay mixture。

## 数据切分与污染

- 按仓库、题族、时间或生成模板分组切分，而非随机切轨迹；
- 防止同一 issue 的修复、fork 或镜像跨 train/test；
- 检查 benchmark 答案、隐藏测试和 verifier 源码是否进入上下文；
- 时间切分必须冻结依赖与外部服务，否则难度变化不可归因；
- 报告模型是否能访问互联网、包管理器和历史 commit。

训练算法见[数学与算法](math-algorithms.md)，安全与评测见[评测与安全](evaluation-safety.md)。
