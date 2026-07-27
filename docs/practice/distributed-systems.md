# 手撕：分布式与容错

分布式正确性来自全局 tensor 语义，而不是“每张卡 loss 都下降”。本页用紧凑代码固定 token 归一化、placement、collective、MoE permutation 与 checkpoint 原子提交。

## Ragged batch 的全局 loss

rank $r$ 上有效 token 的 loss 和为 $S_r$，数量为 $n_r$：

$$
\mathcal L=\frac{\sum_r S_r}{\sum_r n_r}.
$$

若 DDP 在 backward 后平均 $R$ 个 rank 的梯度，则每个 rank 应对

$$
\widetilde{\mathcal L}_r
=\frac{R\,S_r}{\sum_j n_j}
$$

执行 backward：

```python
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import torch
import torch.distributed as dist

def ddp_token_loss(local_sum, local_count):
    """Differentiable local_sum and scalar count -> correctly scaled DDP loss."""
    if not dist.is_initialized():
        if local_count == 0:
            raise ValueError("batch has no valid token")
        return local_sum / local_count
    denominator = local_count.detach().clone().to(local_sum)
    dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
    if denominator == 0:
        raise ValueError("global batch has no valid token")
    return local_sum * dist.get_world_size() / denominator
```

用于展示的全局 mean 还需 all-reduce detached numerator；不要把带梯度的 numerator 再归约。相同样本改变 padding、microbatch 或 rank 切分时，梯度应保持一致。

## Sharded global norm

每个参数只由一个 rank 持有时：

$$
\lVert g\rVert_2=
\sqrt{\sum_r\sum_{p\in\mathcal P_r}\lVert g_p\rVert_2^2}.
$$

```python
@torch.no_grad()
def sharded_grad_norm(parameters, collective_device=None,
                      accumulator_dtype=torch.float32):
    """Count each shard once and reduce on the process-group device."""
    gradients = [p.grad.detach() for p in parameters if p.grad is not None]
    if accumulator_dtype not in (torch.float32, torch.float64):
        raise ValueError("accumulator_dtype must be float32 or float64")
    if collective_device is None:
        if gradients:
            collective_device = gradients[0].device
        elif dist.is_initialized():
            raise ValueError("collective_device is required without local gradients")
        else:
            collective_device = torch.device("cpu")
    collective_device = torch.device(collective_device)
    if any(g.device != collective_device for g in gradients):
        raise ValueError("all local gradients must use the collective device")
    local = torch.zeros((), dtype=accumulator_dtype, device=collective_device)
    for gradient in gradients:
        local += gradient.to(accumulator_dtype).square().sum()
    if dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
    return local.sqrt()

@torch.no_grad()
def clip_sharded_grad_norm_(parameters, max_norm, collective_device=None,
                            accumulator_dtype=torch.float32):
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    parameters = list(parameters)
    norm = sharded_grad_norm(
        parameters, collective_device, accumulator_dtype,
    )
    scale = min(1.0, max_norm / (norm.item() + 1e-6))
    for p in parameters:
        if p.grad is not None:
            p.grad.mul_(scale)
    return norm
```

默认用设备侧 FP32 累加，只有确有精度需要且 collective backend 支持时才切换 FP64。没有本地梯度的 rank 无法从 tensor 推断设备，必须显式传入 `collective_device`。若参数被复制在多个 rank，必须只计一次或先按复制组归约；否则 global norm 会被重复放大。

## Reduce-scatter + all-gather

all-reduce 可分解为 reduce-scatter 与 all-gather。下面使用框架 collective 保留这条数据流，不重新实现 transport：

```python
def allreduce_via_rs_ag(x):
    """x:[P,N] logical rank chunks on each rank -> summed tensor [P,N]."""
    if not dist.is_initialized():
        return x
    p = dist.get_world_size()
    if x.size(0) != p:
        raise ValueError("leading dimension must equal world size")
    shard = torch.empty_like(x[0])
    dist.reduce_scatter_tensor(shard, x.contiguous(), op=dist.ReduceOp.SUM)
    out = torch.empty_like(x)
    dist.all_gather_into_tensor(out, shard)
    return out
```

所有 rank 必须以相同顺序调用 collective，并匹配 count、dtype 与 process group。通信异步完成前不可复用 buffer。

## Column 与 row parallel linear

对 $Y=XW^\top$，column parallel 沿输出维切 $W$，结果 concat；row parallel 沿输入维切 $X,W$，partial result 求和：

```python
def column_parallel_linear(x, weight_shards):
    """x:[...,Din], each W:[Dout/P,Din] -> [...,Dout]."""
    return torch.cat([x @ w.T for w in weight_shards], dim=-1)

def row_parallel_linear(x_shards, weight_shards):
    """each x:[...,Din/P], W:[Dout,Din/P] -> [...,Dout]."""
    if len(x_shards) != len(weight_shards):
        raise ValueError("x and weight placements disagree")
    return torch.stack([x @ w.T for x, w in zip(x_shards, weight_shards)]).sum(0)
```

```python
torch.manual_seed(2)
x = torch.randn(3, 8)
w = torch.randn(12, 8)
torch.testing.assert_close(column_parallel_linear(x, w.chunk(3, 0)), x @ w.T)
torch.testing.assert_close(row_parallel_linear(x.chunk(2, -1), w.chunk(2, -1)), x @ w.T)
```

真实 row parallel 用 all-reduce 或 reduce-scatter；column parallel 的输入与后续算子若保持分片，可以延迟 all-gather。每个 tensor 应记录 global shape、local shape、mesh dimension 与 placement。

## Pipeline bubble

平衡的 GPipe 式 pipeline，$p$ 个 stage、$m$ 个 microbatch，理想 bubble fraction：

$$
f_{\mathrm{bubble}}=\frac{p-1}{m+p-1}.
$$

```python
def pipeline_utilization(stages, microbatches):
    if stages < 1 or microbatches < 1:
        raise ValueError("stages and microbatches must be positive")
    bubble = (stages - 1) / (microbatches + stages - 1)
    return {"bubble": bubble, "utilization": 1 - bubble}
```

这个近似假设 stage 平衡且忽略通信；真实利用率还受最慢 stage、activation 传输、同步和 schedule 影响。

## MoE dispatch 与 combine

router 为每个 token 选择 top-$k$ expert。先按 expert 排序形成连续 batch，再用原 order 恢复 token：

```python
def moe_dispatch(x, router_logits, top_k):
    """x:[N,H], logits:[N,E] -> packed, expert ids, restore order, weights."""
    if not 1 <= top_k <= router_logits.size(-1):
        raise ValueError("invalid top_k")
    prob = router_logits.float().softmax(-1)
    weight, expert = prob.topk(top_k, dim=-1)
    weight = weight / weight.sum(-1, keepdim=True)
    flat_expert = expert.flatten()
    order = flat_expert.argsort(stable=True)
    packed = x[:, None, :].expand(-1, top_k, -1).reshape(-1, x.size(-1))[order]
    return packed, flat_expert[order], order, weight.flatten()

def moe_combine(packed_output, order, weight, tokens, top_k):
    """packed_output:[N*K,H] -> token output:[N,H]."""
    routed = torch.empty_like(packed_output)
    routed[order] = packed_output
    routed = routed.view(tokens, top_k, -1)
    return (routed * weight.view(tokens, top_k, 1).to(routed)).sum(1)
```

```python
x = torch.randn(5, 7)
logits = torch.randn(5, 3)
packed, expert, order, weight = moe_dispatch(x, logits, top_k=2)
y = moe_combine(packed, order, weight, tokens=5, top_k=2)
torch.testing.assert_close(y, x)
assert torch.equal(expert, expert.sort().values)
```

identity expert 下 combine 应恢复 $x$。真实系统还要执行 capacity、all-to-all、grouped GEMM 与负载统计；token 数均衡不保证 expert wall time 均衡。

## 显存账本

训练每 rank 峰值可拆为：

$$
M_{\mathrm{rank}}
=N\left(b_{\mathrm{rep}}+\frac{b_{\mathrm{shard}}}{p}\right)
+M_{\mathrm{act}}+M_{\mathrm{transient}}.
$$

```python
def training_memory_bytes(parameters, replicated_bpp, sharded_bpp, world_size,
                          activation_bytes, transient_bytes):
    """Return steady and peak bytes for one rank."""
    if min(parameters, world_size) <= 0:
        raise ValueError("parameters and world_size must be positive")
    steady = parameters * (replicated_bpp + sharded_bpp / world_size)
    return {
        "steady": steady + activation_bytes,
        "peak": steady + activation_bytes + transient_bytes,
    }
```

`transient` 包括当前 layer all-gather、collective buffer、allocator fragmentation 与异步 checkpoint staging。只报稳态参数/优化器会低估 OOM。

## Checkpoint manifest

checkpoint 只有在所有 shard 完成并校验后才能被读取。数据文件使用不可变 snapshot ID，最后原子提交 manifest：

```python
def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def resolve_shards(root, shards):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("root must be an existing directory")
    paths, relative = [], []
    for shard in shards:
        path = Path(shard).resolve()
        try:
            name = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError("every shard must stay inside root") from error
        if not path.is_file():
            raise ValueError("every shard must be a completed file")
        paths.append(path)
        relative.append(name)
    if not paths or len(set(relative)) != len(relative):
        raise ValueError("shards must be non-empty and unique")
    return root, paths, relative

def commit_manifest(root, snapshot_id, shards, metadata):
    """Publish once; an existing snapshot ID is never overwritten."""
    root, paths, relative = resolve_shards(root, shards)
    if not snapshot_id or Path(snapshot_id).name != snapshot_id:
        raise ValueError("snapshot_id must be one non-empty path component")
    manifest = {
        "snapshot_id": snapshot_id,
        "shards": [
            {"path": name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path, name in zip(paths, relative)
        ],
        "metadata": metadata,
    }
    target = root / f"{snapshot_id}.manifest.json"
    descriptor, temp_name = tempfile.mkstemp(
        dir=root, prefix=f".{snapshot_id}.", suffix=".tmp",
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return target
```

```python
def verify_manifest(root, manifest_path):
    try:
        root = Path(root).resolve()
        manifest_path = Path(manifest_path).resolve(strict=True)
        manifest_path.relative_to(root)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for shard in data["shards"]:
            path = (root / shard["path"]).resolve(strict=True)
            path.relative_to(root)
            if path.stat().st_size != shard["bytes"]:
                return False
            if file_sha256(path) != shard["sha256"]:
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
```

```python
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    shards = [root / "rank0" / "state.bin", root / "rank1" / "state.bin"]
    for rank, shard in enumerate(shards):
        shard.parent.mkdir()
        shard.write_bytes(f"checkpoint-{rank}".encode())
    manifest = commit_manifest(
        root, "step-10", shards,
        {"global_tokens": 1024, "world_size": 2},
    )
    assert verify_manifest(root, manifest)
    try:
        commit_manifest(root, "step-10", shards, {})
    except FileExistsError:
        pass
    else:
        raise AssertionError("snapshot IDs must be immutable")
    shards[0].write_bytes(b"corrupted")
    assert not verify_manifest(root, manifest)
```

`link` 在同一文件系统中以“目标不存在”为前提原子创建 manifest，因此并发提交同一 snapshot ID 时至多一个成功；临时文件先 `fsync`，但目录项与远端对象存储的持久化仍需后端专用协议。完整 metadata 还应保存 global shape/offset/dtype、optimizer、scheduler、scaler、RNG、data cursor、tokenizer/template、代码和数据版本、topology 与 sharding schema。只恢复 weights 属于 warm start，不是严格 resume。

## Checkpoint 周期

checkpoint 成本为 $C$，平均故障间隔为 $M$，周期 $\tau$ 的粗略开销：

$$
f(\tau)\approx\frac{C}{\tau}+\frac{\tau}{2M},\qquad
\tau^*\approx\sqrt{2CM}.
$$

```python
def checkpoint_interval(checkpoint_seconds, mean_time_between_failures):
    if checkpoint_seconds <= 0 or mean_time_between_failures <= 0:
        raise ValueError("time values must be positive")
    return math.sqrt(2 * checkpoint_seconds * mean_time_between_failures)
```

它忽略恢复、重启和异步 staging 竞争，只用于建立量级。真实周期要用故障分布、存储吞吐和训练抖动校准。

## 故障注入

最小测试矩阵包括：

- 任一 rank 在 collective 前后退出；
- collective count、dtype 或顺序不一致；
- activation checkpoint 前后 RNG 不一致；
- MoE 某 expert 为空或极度过载；
- shard 写完前进程退出；
- manifest 原子提交前后分别崩溃；
- 用不同 world size 恢复；
- data cursor、mixture 或 tokenizer 不匹配。

系统机制见[集合通信与状态分片](../systems/collectives-sharding.md)、[模型并行](../systems/model-parallelism.md)与[检查点](../systems/checkpointing.md)，推理侧的物理内存见[手撕：推理引擎](inference-engine.md)。

## Reference {#reference}

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [NVIDIA NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
