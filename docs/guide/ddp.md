# Multi-GPU Training & Precision

echelon3 runs multi-GPU training with **built-in DistributedDataParallel
(DDP)** — one process per GPU. You do **not** need `torchrun`: name the GPUs and
echelon3 spawns the workers itself.

## Multi-GPU in one command

```bash
echelon3-train --config-dir configs --config-name my_experiment gpus=[0,1,2,3]
```

`gpus` is a **root config key**. Set it in the config, or override it on the CLI.
Leave it out and echelon3 uses **every visible GPU on the node**:

```yaml
gpus: [0, 1, 2, 3]   # optional; default = all GPUs on this node
```

Under the hood echelon3 calls PyTorch's own launcher (`elastic_launch`) to start
one worker per GPU, wiring up `RANK` / `LOCAL_RANK` / `WORLD_SIZE` / `MASTER_*`
and the process group — exactly what `torchrun` does, without you typing it. With
a single GPU (or on CPU) nothing is spawned and training runs in-process.

!!! note "No DataParallel"
    DataParallel has been removed (0.5.0). Multiple GPUs always run as DDP, one
    process each; a single process only ever drives one GPU. The old `device_ids`
    key no longer selects multiple GPUs — use `gpus`.

## torchrun / multi-node still works

The environment-variable path is unchanged, so `torchrun` (and SLURM `srun`)
remain available for multi-node or elastic jobs. If echelon3 finds `RANK` in the
environment it assumes it is already a worker and does not spawn again:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    $(which echelon3-train) --config-dir configs --config-name my_experiment
```

For a single node, `gpus=[...]` and `torchrun` are equivalent — prefer `gpus` for
the shorter command; reach for `torchrun` when you span nodes.

## Batch size is global

`dataloaders.train.config.batch_size` is the **global** batch size — the total
across all processes. Under DDP, `create_dataloaders` divides it by the world
size and installs a `DistributedSampler`:

```yaml
dataloaders:
  train:
    module: torch.utils.data
    type: DataLoader
    config:
      batch_size: 40        # global; 40 / 4 = 10 samples per GPU
      shuffle: true         # handed to DistributedSampler; loader shuffle disabled
      num_workers: 6
      drop_last: true
```

!!! warning "It must divide evenly"
    If `batch_size` is not divisible by the number of GPUs, the run fails fast
    with a `ValueError`. Pick a global batch that is a multiple of `len(gpus)`.

The test loader is treated the same way: its `batch_size` is global and divided
per rank.

!!! warning "`num_workers` / `prefetch_factor` are per-rank"
    Unlike `batch_size`, the DataLoader worker count is **not** divided — every
    rank spins up `num_workers` workers, each prefetching `prefetch_factor`
    batches. With all ranks on one node, host RAM scales as
    `ranks × num_workers × prefetch_factor × batch`, so a value sized for a single
    GPU multiplies under DDP. If it exceeds RAM the OOM-killer strikes a worker or
    a rank; the launcher now fails fast with a hint (and prints the batch-prefetch
    total at startup) instead of hanging silently — but the fix is to lower
    `num_workers` / `prefetch_factor`. Set `ECHELON3_DDP_TIMEOUT_MIN` to shrink the
    process-group timeout backstop if a genuine desync should surface faster.

    Workers are reaped with their process (`PR_SET_PDEATHSIG` on Linux), so a
    crashed or `kill`ed run no longer leaves orphaned workers holding `/dev/shm`
    and RAM — the failure mode where a *new* run then hangs at the first batch.
    Prefer `persistent_workers: false` unless dataset init is genuinely expensive;
    it keeps workers alive between epochs and raises the orphan risk on an unclean
    stop. Rule of thumb: `num_workers` per rank ≈ cores ÷ ranks.

## What each rank does

The network is wrapped in `DistributedDataParallel` (with
`find_unused_parameters=True` by default; set
`trainer.config.ddp_find_unused_parameters: false` when your graph uses every
output every step). Each rank trains on its shard and DDP all-reduces gradients.

**Validation is symmetric and sharded.** Every rank evaluates its own
`DistributedSampler` shard through the unwrapped network.

!!! warning "Custom metrics are not aggregated across ranks"
    **torchmetrics** metrics reduce their state across ranks inside `compute()`
    (via `dist_reduce_fx`), so the aggregated value is identical on every rank and
    the keep-best decision matches everywhere — correct under DDP. **Custom metrics**
    (subclasses of `echelon3.metrics.base.Metric`) have no distributed reduction, so
    under DDP each rank computes on its own shard only, and keep-best is driven by
    rank 0's shard (rank 0 owns the checkpoint), so the selected "best" checkpoint is
    slightly noisy — per-shard numbers are not the global metric (averaging shards is
    wrong for ratio metrics like IoU). For exact DDP behaviour use a torchmetrics
    metric, or implement the **`dist_reduce()`** hook on your custom metric: the
    trainer calls it on every rank right before `compute()`, and a counter-based
    metric reduces its accumulators there — e.g. `all_reduce_sum_(self.tp, self.fp,
    self.fn)` (helper in `echelon3.metrics.base`). Summing intersections/unions across
    shards is exact. The base `dist_reduce()` is a no-op, so single-GPU runs and
    torchmetrics are unaffected.

Only rank 0 writes the checkpoint file.

## Logs and checkpoints: rank 0 only

- Rank 0 prints, shows progress bars, and owns the mlops logger; other ranks
  redirect stdout to `/dev/null`, disable `tqdm`, and get a no-op logger.
- `save_checkpoint` returns immediately on non-main ranks — only rank 0 writes.

Checkpoints store the **unwrapped** `state_dict` (no `module.` prefix), so a file
is identical whether it came from a single-GPU or a multi-GPU run and resumes
under either. Older checkpoints that still carry a `module.` prefix load fine —
it is stripped automatically.

## Mixed precision (AMP) & TF32

Training uses **bf16 automatic mixed precision by default** on GPUs that support
it (Hopper / Ampere and newer); on CPU or unsupported GPUs it stays fp32. On
modern hardware this is a large speedup at negligible quality cost.

Control it in `trainer.config`:

```yaml
trainer:
  config:
    precision: bf16        # auto (default) | bf16 | fp16 | fp32
    tf32: true             # TF32 matmul on Ampere+ (default true)
    cudnn_benchmark: true  # autotune conv algorithms for fixed input sizes (default true)
```

- `auto` / unset → bf16 when supported, else fp32.
- **`fp32` → disables autocast** (bit-for-bit the pre-0.5.0 behavior). This is how
  you keep training in plain fp32.
- `fp16` → autocast + `GradScaler`. Not supported with closure optimizers
  (`SAMOptimizer` / `LBFGS`, which do a double backward) — those fall back to
  bf16 automatically.

!!! warning "The default flipped to bf16 in 0.5.0"
    Results differ from old fp32 runs (usually negligibly, and faster). Set
    `precision: fp32` to reproduce fp32 exactly.

`echelon3-evaluate` and `echelon3-run` autocast the same way (default bf16); set
`precision: fp32` at the **config root** to force fp32 for those.

## torch.compile

bf16 only speeds up **compute-bound** work. A small network on a big GPU is often
**launch-bound** instead — dominated by per-kernel launch overhead, with the GPU
idle between many tiny kernels — and there bf16 buys nothing. The lever is
`torch.compile`, which fuses kernels and cuts the launch count:

```yaml
trainer:
  config:
    compile: true             # off by default
    compile_mode: null        # null | "reduce-overhead" | "max-autotune"
```

The network is compiled before the DDP wrapper; `ddp.unwrap()` and checkpoints
strip the resulting `_orig_mod.` prefix, so checkpoints stay interchangeable with
uncompiled runs. The first few steps recompile (warmup) — measure steady-state,
not iteration 1.

!!! note "Model-dependent speedup"
    Validated single-GPU and on 4×H200 DDP (including production image-in-image
    runs) — trains and round-trips checkpoints. How much you gain depends on the
    model: watch for shape-driven recompiles, and closure optimizers like SAM.
    Worth checking a workload is actually launch-bound first (`nvidia-smi dmon -s
    pu`: power well under TDP ⇒ the GPU is starved, and `compile` is the lever).

## Next

- [Config Schema](../reference/config-schema.md) — the `dataloaders`, `gpus`,
  and `trainer` sections.
- [First Run](../getting-started/first-run.md) — a single-GPU baseline first.
</content>
