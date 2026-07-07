# Multi-GPU Training

echelon3 trains on multiple GPUs in two ways, and it picks between them
**automatically**:

- **DistributedDataParallel (DDP)** — one process per GPU, launched with
  `torchrun`. This is the fast path.
- **DataParallel** — a single process driving several GPUs, used as a fallback
  when you run without `torchrun`.

You do not toggle a config flag. The trainer detects `torchrun` and switches
mode; the same config runs either way.

## How DDP is activated

`echelon3.ddp.init_ddp_if_needed()` runs at the top of `echelon3-train`. It
checks for the environment variables `torchrun` sets — `RANK`, `WORLD_SIZE`,
`LOCAL_RANK` — and if they are present, initializes the process group (`nccl` on
CUDA, `gloo` otherwise) and binds the process to `cuda:${LOCAL_RANK}`. No
variables, no DDP.

## Launching

Use `torchrun` and point it at the installed console script:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    $(which echelon3-train) --config-dir configs --config-name my_experiment
```

- `CUDA_VISIBLE_DEVICES` selects which physical GPUs take part.
- `--nproc_per_node` is the number of processes — one per visible GPU.
- `$(which echelon3-train)` gives `torchrun` the path to the entry point;
  everything after it is passed through to Hydra as usual.

!!! warning "`device` / `device_ids` are ignored under DDP"
    With `torchrun`, each process owns exactly one GPU chosen by `LOCAL_RANK`, so
    the `device` and `device_ids` config keys do nothing — GPU selection is
    entirely `CUDA_VISIBLE_DEVICES`. The trainer prints a yellow notice if it
    sees `device_ids` in a DDP run.

## Batch size is global

`dataloaders.train.config.batch_size` is the **global** batch size — the total
across all processes, exactly as it would be under DataParallel. Under DDP,
`create_dataloaders` divides it by the world size and installs a
`DistributedSampler`:

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
    If `batch_size` is not divisible by the world size, the run fails fast with a
    `ValueError`. Pick a global batch that is a multiple of `--nproc_per_node`.

The test loader is treated the same way: its `batch_size` is global and divided
per rank, and `num_workers` is capped at 4 per process during validation.

## What each rank does

The network is wrapped in `DistributedDataParallel` (with
`find_unused_parameters=True` by default; override via
`trainer.config.ddp_find_unused_parameters: false` when your graph uses every
output every step). Training runs the standard loop on each rank's shard, and
gradients are all-reduced by DDP.

**Validation is symmetric and sharded.** Every rank evaluates its own
`DistributedSampler` shard through the unwrapped network, and the
torchmetrics-style metrics aggregate their state across ranks inside
`compute()`. Because the aggregated values are identical on every rank, the
keep-best decision is the same everywhere; only rank 0 actually writes the file.

## Logs and checkpoints: rank 0 only

To keep output and disk writes clean, non-main ranks are muted:

- Rank 0 prints, shows progress bars, and owns the mlops logger; other ranks
  redirect stdout to `/dev/null`, disable their `tqdm` bars, and get a no-op
  logger.
- `save_checkpoint` returns immediately on non-main ranks — checkpoints are
  written only by rank 0.

Checkpoints are format-compatible between DDP and DataParallel: both store the
`state_dict` with a `module.` prefix, so you can resume a DDP run under
DataParallel and vice versa. Resuming works automatically from the highest
`checkpoint-N.tar` in `target.path`.

## The DataParallel fallback

Run the same config **without** `torchrun` and nothing distributed happens: the
trainer wraps the network in `torch.nn.DataParallel(net, device_ids=...)` using
the `device` / `device_ids` config keys, and `batch_size` is the whole batch that
DataParallel scatters across the listed GPUs.

```yaml
device: cuda
device_ids: [0, 1, 2, 3]
```

```bash
echelon3-train --config-dir configs --config-name my_experiment
```

DDP is preferred for real multi-GPU jobs — DataParallel is simpler to launch but
slower and single-process. Use it for quick two-GPU experiments; reach for
`torchrun` when throughput matters.

## Next

- [Config Schema](../reference/config-schema.md) — the `dataloaders`, `device`,
  and `trainer` sections.
- [First Run](../getting-started/first-run.md) — a single-GPU baseline first.
</content>
