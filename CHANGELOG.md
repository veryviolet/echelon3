# Changelog

All notable changes to **echelon3** are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely; versions
follow [SemVer](https://semver.org/) once 1.0.0 ships.

## 0.7.7 — 2026-07-08

### Added

- **`Metric.dist_reduce()` hook — exact custom-metric aggregation under DDP.**
  Validation is sharded per rank; torchmetrics reduce their state inside `compute()`,
  but custom `echelon3.metrics.base.Metric` subclasses used to compute on rank 0's
  shard only, making keep-best selection noisy. The base `Metric` now has a
  `dist_reduce()` method (default no-op) that the trainer calls on every rank right
  before `compute()` under DDP. A counter-based metric (e.g. IoU's
  intersection/union accumulators) implements it as a SUM all-reduce of its buffers —
  helper `all_reduce_sum_(*tensors)` provided — which is exact, because summation
  commutes with sharding (unlike averaging per-shard ratios). Single-GPU runs and
  torchmetrics are unaffected.

## 0.7.6 — 2026-07-08

### Changed

- **The per-component `config:` block is now optional everywhere, uniformly.** One
  rule across every `create_*` (net, backbone, dataset, loss, metric, optimizer,
  scheduler, dataloader, trainer, evaluator, wrapper, constructor, batch sampler,
  exporter): a component whose constructor needs no arguments may omit its `config:`
  block. Removes the previous "optional in some builders, required in others"
  inconsistency (present-`config` behaviour is unchanged).

### Fixed

- **`Metric.to()` no longer strands buffers on CPU.** The base
  `echelon3.metrics.base.Metric.to()` was a no-op that shadowed `nn.Module.to` via
  the MRO for metrics that are *also* `nn.Module`s with buffers/parameters — their
  buffers stayed on CPU and validation on CUDA raised a device-mismatch error. It
  now delegates to `nn.Module.to()` when the metric is a module; pure non-module
  metrics stay a no-op as before.

### Docs

- DDP guide: corrected the validation section — only torchmetrics reduce their state
  across ranks; custom `Metric` subclasses compute per-shard, so keep-best is driven
  by rank 0's shard (use a torchmetrics metric, or add a distributed reduce, for
  exact behaviour).

## 0.7.5 — 2026-07-08

### Fixed

- **Regression from 0.7.2: `DataLoader(multiprocessing_context="spawn")` crashed
  while pickling `worker_init_fn`.** The default PDEATHSIG `worker_init_fn` (added
  in 0.7.2 for worker reaping) was a closure, and spawn must pickle
  `worker_init_fn` to hand it to the fresh worker process — closures are not
  picklable. Only the fork path (the default) had been tested, so this shipped
  unnoticed. It is now a module-level function wrapped in `functools.partial`,
  which pickles cleanly; `spawn` works again with worker reaping intact.

## 0.7.4 — 2026-07-08

### Changed

- Revert the per-component `config:`-block optionality added in 0.7.3: `net`,
  datasets, and schedulers require their `config:` block again (as before 0.7.3).
  Making `config` optional only in some builders was inconsistent; making it
  optional everywhere would spread that leniency, and requiring it literally
  everywhere is impossible (no-arg leaf transforms such as `To01` legitimately omit
  `config`). Section-level omission (`transform` / `metrics` / `scheduler` /
  `keep_best_on`) is unchanged.

## 0.7.3 — 2026-07-08

### Changed

- **CLI output is now uniformly English and single-colour.** A handful of runtime
  messages (and raised errors) were Russian and/or highlighted in yellow, which
  looked out of place in the public package; they now read in English in the
  ambient colour like the rest of the `-->` log.
- **Warnings no longer corrupt progress bars.** Library warnings are collected
  instead of printed inline; a short summary (`--> N warning(s) since last report`
  with per-message counts) is emitted before each validation and once at the end.
  Deprecation/Future noise is dropped outright. The c10d `barrier(): using the
  device under current context` warning is silenced at the source by passing
  `device_id` to `init_process_group`.

### Fixed

- **Config parameters with sensible defaults are now optional — omitting them no
  longer crashes.** `transform` (augment/preprocess), `metrics`, and `scheduler`
  may be omitted entirely (→ ToTensorV2-only, no metrics, constant LR);
  `keep_best_on` is optional (→ save every validation); and the per-object
  `config:` block is optional for `net`, datasets, losses, and schedulers
  (constructed with no extra args). Loss `weight` defaults to `1.0`. A new pytest
  suite (`tests/`, `pip install -e .[test]`) covers these paths.

## 0.7.2 — 2026-07-08

### Fixed

- **No more orphaned DataLoader / DDP processes after a crash, `kill`, or failed
  restart.** DataLoader workers and the DDP rank processes now set
  `PR_SET_PDEATHSIG` (Linux) via a default `worker_init_fn` / at process-group
  init, so they are SIGKILL'd the instant their parent dies — for any reason,
  including the `os._exit` fast-path from 0.7.1 and an external `kill -9`.
  Previously a rank that died uncleanly left its workers running, holding
  `/dev/shm` and host RAM; a fresh run's workers then couldn't get shared memory
  and hung at the first batch (GPUs idle). `echelon3-train` / `-finetune` now also
  catch `KeyboardInterrupt` (Ctrl-C goes through the teardown path) and call a new
  `Trainer.close()` on exit to shut DataLoader workers down promptly — relevant
  with `persistent_workers: true`. The launcher additionally warns on CPU
  over-subscription (`ranks × num_workers > cores`) and on `persistent_workers`
  under DDP.

## 0.7.1 — 2026-07-08

### Fixed

- **DDP no longer hangs silently when a rank dies (typically an OOM).** Under the
  built-in launcher, a rank that OOM'd could wedge in `destroy_process_group()`
  (NCCL teardown blocks on an in-flight collective) and never exit, so
  `elastic_launch` never saw a failure and the peers blocked on the next
  collective — a silent hang up to the process-group timeout. On the error path a
  DDP rank now prints its traceback to stderr and hard-exits (`os._exit`) instead
  of relying on a clean shutdown that can block, so the launcher tears the group
  down immediately. `max_restarts=0` (fail-fast instead of silently retrying an
  OOM into a desynced group); `ChildFailedError` is surfaced with an OOM hint;
  `TORCH_NCCL_ASYNC_ERROR_HANDLING` / `TORCH_NCCL_DESYNC_DEBUG` are on so the NCCL
  watchdog aborts (and reports the stuck rank) rather than waiting; the
  process-group timeout is configurable via `ECHELON3_DDP_TIMEOUT_MIN`.
- The launcher prints the per-node DataLoader prefetch total
  (`ranks × num_workers × prefetch_factor`) and warns when it is large — these are
  **per-rank** and multiply under DDP, the common cause of the OOM above.

### Changed

- **`torch.compile` is no longer marked experimental** — validated on single-GPU
  and 4×H200 DDP including production image-in-image runs. The `compile` /
  `compile_mode` knobs are unchanged.

## 0.7.0 — 2026-07-08

### Added

- **`torch.compile` support (experimental, opt-in).** `trainer.config.compile: true`
  (with optional `compile_mode`) compiles the network — kernel fusion to cut launch
  overhead, the lever for small nets that under-use a big GPU where bf16 does
  nothing (launch-bound, not compute-bound). Compiled before the DDP wrapper;
  `ddp.unwrap()` and checkpoint save/load now also strip torch.compile's
  `_orig_mod.` prefix, so checkpoints stay interchangeable with uncompiled runs.
  Off by default; verified single-GPU **and on 4×H200 DDP** (trains + checkpoints
  round-trip). The actual speedup and any shape-driven recompiles are
  model-dependent (see `guide/ddp.md`).

### Changed

- **`PairTrainer` calls the network positionally** — `net(base, query, True)`
  instead of `net(base, query, return_features=True)`. Pair nets name the third
  argument differently (`return_features`, `return_intermediates`, …); passing it
  by position keeps the trainer agnostic to the name.

## 0.6.0 — 2026-07-08

### Added

- **`Trainer.compute_losses(source, labels, net=None)` — an extension seam for
  the forward pass and loss routing.** It runs inside the trainer's autocast and
  returns `(predictions, {name: (loss, weight)})`; both training and validation
  call it, so a subclass customizes *what the network is fed and how losses map
  to its outputs* without ever re-implementing the precision / scaler / closure /
  DDP machinery, which stays in the base. Behaviour is unchanged — the default
  `compute_losses` is the previous single-input forward.
- **`echelon3.trainers.pair.PairTrainer`** — a trainer for two-image
  ("pair" / image-in-image) inputs. Consumes `((base, query), gt)` batches
  (via `pair_collate_fn`), calls `net(base, query, return_features)` and delegates
  loss routing to `pair_losses(heatmap, features, labels)` (default: every loss on
  the heatmap). A domain-specific image-in-image trainer becomes a natural
  subclass overriding only `pair_losses` — see `guide/extending.md`.

## 0.5.2 — 2026-07-08

### Fixed

- **`GrayscaleCLAHE` is now picklable**, so it survives DataLoader workers
  started with the `spawn` start method (and spawn-only platforms). It stored a
  bare `cv2.CLAHE` C++ handle, which cannot be pickled — any run whose data
  workers pickle the dataset died with `TypeError: cannot pickle 'cv2.CLAHE'
  object`. The handle is now dropped on pickle and rebuilt from the stored
  `clip` / `grid` params on unpickle. Under the default `fork` start method the
  transform is inherited and never pickled, so single-GPU runs and the example
  smokes never triggered it — reproduced and fixed under a `spawn` DataLoader on
  4×H200.

## 0.5.1 — 2026-07-08

### Fixed

- **README (the PyPI project description) rewritten for 0.5.0.** It still
  documented the removed `DataParallel` fallback and presented `torchrun` as the
  only multi-GPU path. Now it shows the built-in `gpus=[...]` launcher (no
  `torchrun` needed), the bf16-by-default mixed precision, the full CLI set, and
  all three example smokes. Docs-only release — no code changes.

## 0.5.0 — 2026-07-08

Multi-GPU and performance release. **Breaking**: DataParallel is gone and mixed
precision defaults to bf16.

### Changed

- **Multi-GPU is built-in DDP, launched from the CLI — no `torchrun` needed.**
  Pass `gpus=[0,1,2,3]` (a root config key; default = all visible GPUs on the
  node) and echelon3 spawns one DDP worker per GPU via PyTorch's `elastic_launch`,
  wiring up `RANK`/`LOCAL_RANK`/`WORLD_SIZE`/`MASTER_*` itself. `torchrun` and
  multi-node runs still work unchanged through the environment-variable path.
  Applies to `echelon3-train` and `echelon3-finetune`.
- **Mixed precision (AMP) on by default.** Training, `evaluate` and `run` autocast
  in **bf16** on capable GPUs (fp32 on CPU / unsupported GPUs). Set
  `trainer.config.precision: fp32` to restore full fp32. `precision: fp16` uses a
  `GradScaler`; with closure optimizers (SAM/LBFGS, which double-backward) it
  falls back to bf16. TF32 matmul and `cudnn.benchmark` are on by default
  (`trainer.config.tf32`, `trainer.config.cudnn_benchmark`).
- Checkpoints save the **unwrapped** `state_dict` (no `module.` prefix); older
  `module.`-prefixed checkpoints still load (prefix stripped automatically).

### Removed

- **DataParallel.** Multiple GPUs always run as DDP; a single process drives one
  GPU. `device_ids` no longer selects multiple GPUs — use `gpus`. The
  `run` / `evaluate` / `export` CLIs load checkpoints directly instead of wrapping
  the network in `DataParallel`.

## 0.4.1 — 2026-07-07

### Fixed

- `IoU.compute` (segmentation metric) returned `nan` whenever `ignore_index` was
  unset (the default): `use_idx[None] = False` added an axis and wiped the whole
  keep-mask, so `nanmean` ran over an empty array. Now the mask is only touched
  when there are classes to ignore. Verified end to end — a `Segmenter` smoke on
  synthetic masks moves mIoU from ~0.85 to ~0.96.

### Added

- `examples/segmentation/`: a self-contained semantic-segmentation smoke
  (synthetic image/mask generator, a tiny dependency-free backbone, a `Segmenter`
  + cross-entropy + `IoU` config) mirroring `examples/detector/`.

## 0.4.0 — 2026-07-07

### Fixed

- Detection pipeline now trains end to end (CenterNet-style `HeatmapDetector`,
  YOLO-format `DetectionDataset`). Several defects along the path were fixed:
  - `Trainer.set_to_device` now stacks the per-sample image tensors that
    `VariableDataLoader` produces into one `(N, C, H, W)` batch, and leaves the
    variable-length box lists as Python lists instead of calling `.to()` on them
    (image-in-image `(base, query)` pairs are unaffected).
  - The train/validate steps no longer assume `labels` is a tensor (they read
    `labels.shape` only when it is), so list-valued detection targets work.
  - `metrics.base.Metric` gained a no-op `.to()` so custom metrics (mAP, EER,
    AUC, IoU, …) interoperate with the trainer's uniform `metric.to(device)`.
  - `HeatmapBasedDetectionLoss` uses the penalty-reduced CornerNet/CenterNet
    focal loss on the sigmoid heatmaps the head emits, instead of
    `sigmoid_focal_loss` (which double-applied a sigmoid and crushed the
    gradient, so heatmap peaks never formed).
  - `DecodeHeatmaps.decode` keeps YOLO/Albumentations boxes as normalized floats
    (they were cast to `LongTensor`, truncating every coordinate to 0).
  - Class labels are coerced to integer tensors in both the encoder and the mAP
    metric (Albumentations round-trips them through float arrays).

### Added

- `detection` extra (`faster-coco-eval`) — the `mAP` metric needs a COCO backend;
  `mAP` takes an optional `backend` argument (default `faster_coco_eval`).
- `examples/detector/`: a self-contained CenterNet-like detector smoke — synthetic
  YOLO dataset generator, a tiny dependency-free heatmap backbone, and a config
  that trains, validates (mAP rises above zero) and checkpoints on CPU or GPU.

## 0.3.1 — 2026-07-07

### Added

- Full MkDocs (Material) documentation site published to GitHub Pages at
  <https://veryviolet.github.io/echelon3/>: getting-started, concepts
  (how-it-works, run anatomy), guides (extending, DDP, ONNX export) and reference
  (config schema, built-in components, CLI). `docs` extra + a `docs` workflow that
  `mkdocs gh-deploy`s on every `v*` tag.
- `[project.urls]` with a `Documentation` link, so PyPI shows it.

## 0.3.0 — 2026-07-05

### Added

- `echelon3-finetune` CLI: warm-start from a checkpoint (`init_from`), freezing
  by regex patterns (`finetune.freeze_patterns`), head-only training and
  per-layer parameter groups (`finetune.param_groups`). With none of those
  blocks present it behaves exactly like `echelon3-train`.

## 0.2.0 — 2026-07-05

### Added

- ONNX export: `ModelExporter` base + `OnnxExporter`
  (preprocess → net → postprocess wrapped into a single graph),
  `echelon3-export` CLI, export section in the smoke example, CI step
  exporting the smoke model and verifying it with onnxruntime.
- All CLIs insert the current working directory into `sys.path`, so zoo
  repositories can reference their local packages from configs
  (`module: my_zoo.nets.foo`) when running from the repo root.

### Fixed

- `create_exporters` instantiated `torch.nn.Identity` incorrectly when no
  preprocess/postprocess is configured.

## 0.1.0 — 2026-07-05

First public release. Core of the framework extracted from the internal
echelon2 codebase and cleaned up:

### Added

- Config-driven component factory (`echelon3.creator`): every component is a
  `module` / `type` / `config` YAML triple resolved by dynamic import, with a
  file-path fallback for project-local extensions.
- Trainers: baseline `Trainer` (DataParallel + DDP via torchrun, global batch
  size semantics, multi-metric keep-best checkpointing, multiple named test
  loaders), `MultiHeadTrainer`.
- Generic datasets (folder-hive / CSV classification, segmentation pairs,
  detection, multi-head binary masks), balanced / classwise / multipart
  dataloaders, albumentations augment + torch preprocess pipeline.
- Generic losses and metrics for classification, segmentation (OHEM, boundary,
  clDice, Lovasz, multi-head BCE/IoU) and heatmap detection.
- Nets: `ClassifierNet` / `Segmenter` composition wrappers, timm backbone
  adapter, DDRNet, SegFormer-style heads, DASPP/PSP/FPN necks, CDC layers.
- Checkpoint manager, tensorboard mlops logger, weight loaders (full/partial),
  CLI entry points `echelon3-train` / `echelon3-evaluate` / `echelon3-run`
  (Hydra, `--config-dir`).
- Smoke example: synthetic dataset generator + minimal classifier config.

### Changed vs internal predecessor

- All hardcoded `'cuda'` calls removed — CPU training works.
- Vendored copies of timm and mmsegmentation dropped; external `timm` is used,
  the only needed mmseg op (`resize`) lives in `echelon3.nets.ops`.
- Heavy/optional dependencies (mosaicml SAM, segmentation-models-pytorch)
  import lazily and install via extras (`echelon3[sam]`, `echelon3[smp]`).
