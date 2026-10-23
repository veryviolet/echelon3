# Config Schema

An echelon3 run is a single YAML file. Every top-level key is one section of the
run, and every component inside it is the same `module` / `type` / `config`
triple resolved by `echelon3.creator.get_attr_from_module` (see
[How It Works](../concepts/how-it-works.md)). This page documents each section:
what goes in it, which keys are required, and which `create_*` function in
`echelon3.creator` reads it.

!!! note "`config` is just constructor kwargs"
    For nearly every component, the `config:` block is splatted into the
    constructor as `**config`. Required keys are therefore the required
    arguments of whatever class or function `module`/`type` names — echelon3
    does not validate them itself; the constructor does.

## Section overview

| Section | Read by | Purpose |
| --- | --- | --- |
| `device` / `gpus` | `cli.train` | Target device (`cuda`/`cpu`) and GPU selection (multi → DDP) |
| `net` | `create_net` | The network (nested triples for backbone/head/neck) |
| `weights_loader` | `create_universal` | How to load `net.weights` (optional) |
| `data` | `create_datasets` | Train dataset + one or many test datasets |
| `transform` | `create_augments`, `create_preprocesses` | Albumentations augment + tensor preprocess |
| `loss` | `create_loss` | List of weighted loss terms |
| `metrics` | `create_metrics` | List of validation metrics |
| `dataloaders` | `create_dataloaders` | Train + test loaders (global batch size) |
| `optimizer` | `create_optimizer` | Optimizer over `net.parameters()` |
| `scheduler` | `create_scheduler` | LR scheduler |
| `trainer` | `create_trainer` | The training loop |
| `target` | `create_checkpoint_manager` | Checkpoint directory |
| `mlops` | `create_mlops_logger` | Metric logger (optional, defaults to TensorBoard) |
| `export` | `create_exporters` | ONNX/graph export (optional) |
| `runner` | `create_universal` | Batch inference over images/video (optional) |
| `evaluator` | `create_evaluator` | Post-training evaluation (optional) |
| `init_from` / `finetune` | `cli.finetune` | Warm-start, freezing, param groups (optional) |

## `device` / `gpus`

Plain scalars/lists, not triples. `device` defaults to `cuda` (falls back to `cpu`
when CUDA is unavailable); `device: cpu` forces CPU. `gpus` is a list of GPU indices:
more than one spawns DDP (one process per GPU), a single index pins the run to that
physical GPU, and omitting it uses every visible GPU. (`device_ids` is legacy and
ignored — DataParallel was removed in 0.5.0.)

```yaml
device: cuda
gpus: [0, 1]
```

!!! warning "Ignored under DDP"
    When launched with `torchrun`, both keys are ignored — one process owns one
    GPU chosen by `LOCAL_RANK`, and visibility comes from `CUDA_VISIBLE_DEVICES`.
    See [Multi-GPU Training](../guide/ddp.md).

## `net`

`create_net` resolves the triple and calls `type(**config)`. `config` is
optional when the constructor needs no arguments. Container nets (`ClassifierNet`, `Segmenter`, `HeatmapDetector`)
receive further triples as their config values and resolve them recursively.

```yaml
net:
  module: echelon3.nets.classifier
  type: ClassifierNet
  config:
    backbone:
      module: timm
      type: create_model
      config: { model_name: mobilenetv3_small_100, pretrained: false, num_classes: 0 }
    head:
      module: echelon3.nets.heads.classifier_heads
      type: DenseClassifierHead
      config:
        num_classes: 2
        num_features: 1024
        dropout: 0.1
        activation: { module: torch.nn, type: Identity }
```

### Loading pretrained weights

Add a `weights` key (a checkpoint path) to `net`, plus a top-level
`weights_loader` section. When `net.weights` is present, `cli.train` builds the
loader with `create_universal(cfg.weights_loader)` and calls
`loader.load(net, cfg.net.weights, device)`.

```yaml
net:
  module: echelon3.nets.classifier
  type: ClassifierNet
  weights: ./pretrained/backbone.tar
  config: { ... }

weights_loader:
  module: echelon3.weightloaders.partial
  type: PartialWeightsLoader
  config: { strip_prefix: "module." }   # optional
```

`echelon3.weightloaders.basic.WeightsLoader` does a strict `load_state_dict`;
`PartialWeightsLoader` loads only tensors whose name **and** shape match,
skipping the rest — the way to warm-start a changed architecture.

## `data`

Read by `create_datasets`. Requires a `train` entry (always a single dataset)
and a `test` entry, which may be **either** a single dataset **or** a mapping of
named datasets. Each dataset triple is instantiated with
`type(**config, augment=..., preprocess=...)`, where the augment/preprocess come
from the `transform` section (train transforms for `train`, test transforms for
every `test` dataset by default). Individual test datasets and multipart train
parts may override either pipeline using a sibling `transform:` block (see below).

Single test dataset:

```yaml
data:
  train:
    module: echelon3.data.imageclassifier
    type: FoldersHiveImageClassifierDataset
    config: { classes: 2, folder: ./data/train, wildcards: ['*.png'] }
  test:
    module: echelon3.data.imageclassifier
    type: FoldersHiveImageClassifierDataset
    config: { classes: 2, folder: ./data/test, wildcards: ['*.png'] }
```

Multiple named test datasets — `create_datasets` returns a `dict[name ->
Dataset]`, which the `Trainer` validates independently:

```yaml
data:
  train: { ... }
  test:
    incidents:
      module: echelon3.data.imageclassifier
      type: FoldersHiveImageClassifierDataset
      config: { classes: 2, folder: ./data/incidents, wildcards: ['*.png'] }
    control:
      module: echelon3.data.imageclassifier
      type: FoldersHiveImageClassifierDataset
      config: { classes: 2, folder: ./data/control, wildcards: ['*.png'] }
```

!!! note "A test dataset is single vs. named by shape"
    `create_datasets` treats `test` as a single dataset if it has `module` and
    `type` keys at the top level, otherwise as a mapping of named datasets. With
    named datasets, route metrics per dataset with `trainer.config.metrics_on`
    (see below).

## `transform`

Two independent pipelines keyed by purpose (`train` / `test`):

- **`augment`** — an ordered map of [albumentations](https://albumentations.ai/)
  transforms applied to the raw HWC uint8 image. `create_augments` wraps them in
  `A.Compose([...] + [ToTensorV2()])`, so **`ToTensorV2` is appended for you** and
  the pipeline always yields a CHW tensor. If `augment` is omitted for a purpose,
  that purpose gets a bare `ToTensorV2`.
- **`preprocess`** — an ordered map of `torch.nn.Module`s assembled by
  `create_preprocesses` into a `torch.nn.Sequential`. Runs on the tensor after
  augmentation.

```yaml
transform:
  train:
    augment:
      flip:
        module: albumentations
        type: HorizontalFlip
        config: { p: 0.5 }
      jitter:
        module: albumentations
        type: ColorJitter
        config: { p: 0.3 }
    preprocess:
      to01:
        name: to01
        module: echelon3.transforms.preprocess.basic
        type: To01
  test:
    preprocess:
      to01:
        name: to01
        module: echelon3.transforms.preprocess.basic
        type: To01
```

!!! warning "`preprocess` entries need a `name`"
    `create_single_preprocess` builds an `OrderedDict` keyed by each entry's
    `name` field, so every preprocess step must carry `name:`. Augment entries
    do **not** — their map key is only for readability.

For detection, bbox-aware Compose is enabled by putting `bbox_params` under the
purpose's `config`: `transform.train.config.bbox_params: { format: yolo,
label_fields: [] }`.

### Per-dataset transforms (since 0.12.0)

Put `transform` **beside** `module`, `type`, and `config` in each test dataset spec,
or in a `MultiPartDataset` part. The existing top-level `transform.train` and
`transform.test` remain the common defaults. No `transform.test.datasets` mapping
is used. A single non-multipart training dataset still uses `transform.train`.

- Missing `augment` inherits the common augmentation; missing `preprocess` inherits
  the common preprocessing. The two keys override independently.
- A present chain replaces the entire common chain, in its listed order. Chains
  are not concatenated and operations are not merged by name.
- `augment: {}` (or `null`) removes common image operations but retains the automatic
  `ToTensorV2`. `preprocess: {}` is an identity chain; `null` disables preprocessing.
- Omitting `transform`, or using `transform: {}`, inherits both common pipelines.
- Local augmentation inherits the common bbox format/label fields. To replace or
  disable them, supply local `transform.config.bbox_params` (or `null`) together
  with the local `augment` chain.
- The same resolver is used in initial/intermediate training validation, finetuning,
  and standalone `evaluate`. The dataset applies preprocessing once; `PairEvaluator`
  does not apply it again.

```yaml
transform:
  train:
    augment:
      flip: { module: albumentations, type: HorizontalFlip, config: { p: 0.5 } }
    preprocess:
      scale: { module: echelon3.transforms.preprocess.basic, type: To01, name: scale }
  test:
    augment:
      resize: { module: albumentations, type: Resize, config: { height: 256, width: 256 } }
    preprocess:
      scale: { module: echelon3.transforms.preprocess.basic, type: To01, name: scale }
data:
  train:
    module: echelon3.data.basic
    type: MultiPartDataset
    config:
      parts:
        - name: source_a             # optional; unique within this multipart dataset
          share: 0.5
          module: my_project.data
          type: Images
          config: { split: train_a }
          transform: { augment: {} } # no random image operations; common preprocess
        - share: 0.5                # old unnamed parts remain supported
          module: my_project.data
          type: Images
          config: { split: train_b } # inherits both common train pipelines
  test:
    standard:
      module: my_project.data
      type: Images
      config: { split: test_a }      # inherits common resize and preprocessing
    native_size:
      module: my_project.data
      type: Images
      config: { split: test_b }
      transform: { augment: {} }     # original size, common preprocessing
```

Multipart parts keep their list order and `share` sampling semantics. Names are
optional non-empty, unique strings and are available as `dataset.part_names`
(`None` for unnamed parts). A part can override transforms with or without a name.
Use `MultiPartDataLoader` for the multipart train dataset and matching named test
loaders as usual. Datasets still receive and must apply `augment`/`preprocess`.

## `loss`

A **list** — each item is a single-key mapping `name -> triple`. Beyond the
usual `module`/`type`/`config`, each item requires a `weight`. `create_loss`
returns `{name: (loss_instance, weight)}`; the trainer computes
`sum(weight * loss(pred, target))`.

```yaml
loss:
  - ce:
      module: echelon3.losses.classification
      type: WeightedCrossEntropyLoss
      weight: 1.0
      config: { weight: [1.0, 1.0] }
  - margin:
      module: echelon3.losses.classification
      type: FixedMarginCrossEntropyLoss
      weight: 0.5
      config: { m: 0.35, idx: 1 }
```

`config` is optional (a loss with no constructor args can omit it); `weight` is
not.

## `metrics`

A **list** of `name -> triple` items, like `loss` but without weights. Built by
`create_metrics` via `create_universal`. Any object exposing torchmetrics-style
`update`/`compute`/`reset` works — including `torchmetrics` classes directly.

```yaml
metrics:
  - accuracy:
      module: torchmetrics
      type: Accuracy
      config: { task: multiclass, num_classes: 2 }
  - eer:
      module: echelon3.metrics.classification
      type: EER
```

## `dataloaders`

`train` (single) and `test` (single or named, mirroring `data`). The dataset is
injected as `dataset=...`; everything under `config` is passed through to the
loader constructor.

```yaml
dataloaders:
  train:
    module: torch.utils.data
    type: DataLoader
    config: { batch_size: 32, shuffle: true, num_workers: 4, drop_last: true }
  test:
    module: torch.utils.data
    type: DataLoader
    config: { batch_size: 32, shuffle: false, num_workers: 4 }
```

!!! note "`batch_size` is the global batch size"
    Under DDP, `create_dataloaders` divides `dataloaders.train.config.batch_size`
    by `world_size` and installs a `DistributedSampler`, so the number you write
    is the total across all GPUs. See [Multi-GPU Training](../guide/ddp.md).

echelon3 also ships custom loaders you can drop in here — e.g.
`echelon3.dataloaders.balance.BalancedDataLoader` for class-balanced batches.

### Worker CPU affinity (unreleased)

`dataloaders.*.config.worker_cpu_affinity` controls DataLoader worker placement
independently of the training rank. Available for train, finetune, named test
loaders and standalone evaluate on Linux:

```yaml
dataloaders:
  train:
    module: torch.utils.data
    type: DataLoader
    config:
      batch_size: 32
      num_workers: 4
      worker_cpu_affinity: launcher
  test:
    module: torch.utils.data
    type: DataLoader
    config:
      batch_size: 32
      num_workers: 2
      worker_cpu_affinity: [0, 1, 2, 3]  # example: must belong to your launcher mask
```

- Omitted or `null`: preserve inherited worker affinity (the existing default).
- `launcher`: use all CPUs allowed at CLI entry, before CUDA/DDP initialization.
  The built-in launcher passes that snapshot to every rank before rank-side NUMA
  pinning. With external `torchrun`, the snapshot is each rank's entry mask; the
  engine cannot recover a wider allocation already removed by the external launcher.
- A non-empty list of non-negative integer CPU IDs: use exactly that subset of
  the launcher mask. Booleans, unavailable IDs and invalid values raise errors.
  Duplicate IDs are harmless. All workers of that loader share the selected pool;
  this does not automatically split CPUs between ranks or jobs. Set disjoint lists
  in separate jobs when exclusive pools are desired, keeping SMT siblings together.

The setting is validated when constructing loaders, even with `num_workers: 0`
(where it has no placement effect). Affinity is set in worker-init before thread
pool setup and before a user callback, for both fork and spawn workers. Rank and
NCCL thread placement are unchanged. The kernel enforces cpuset/cgroup limits;
a failure or silently reduced mask raises a worker error. A later change to a
cgroup is not continuously monitored. Custom Python callbacks can subsequently
change their own affinity. Direct Python creator calls use the calling thread's
mask unless enclosed in an engine CLI entrypoint.

## `optimizer`

`create_optimizer` calls `type(params=net.parameters(), **config)`. `config` is
optional. A special `trainable_only: true` key (popped before construction)
restricts the optimizer to parameters with `requires_grad=True`.

```yaml
optimizer:
  module: torch.optim
  type: AdamW
  config: { lr: 0.001, weight_decay: 0.0001 }
```

## `scheduler`

`create_scheduler` calls `type(optimizer=optimizer, **config)`. `config` is
optional for a schedule with defaults. `scheduler.step()` is
called once per epoch by the trainer.

```yaml
scheduler:
  module: torch.optim.lr_scheduler
  type: CosineAnnealingLR
  config: { T_max: 50 }
```

## `trainer`

`create_trainer` injects the assembled net, optimizer, loaders, losses, metrics,
scheduler, checkpoint manager, logger and device, then splats `config`. For the
built-in `echelon3.trainers.baseline.Trainer` the useful `config` keys are:

| Key | Meaning |
| --- | --- |
| `epochs` | Number of epochs (required). |
| `keep_best_on` | Which metric(s) gate checkpoint saving (see below). |
| `times_to_validate_per_epoch` | Validation passes per epoch (default 1). |
| `float_labels` | Cast labels to float before the loss (default `false`). |
| `reset` | On resume, restart epoch/step counters instead of continuing. |
| `save_last` | Save `target.path/last.tar` after each complete epoch (default `false`). |
| `resume_from` | `latest` (default): highest numbered checkpoint; `last`: explicitly load `last.tar`. |
| `metrics_on` | Map `metric_name -> test_dataset_name` for named test sets. |

`keep_best_on` accepts a bare metric/loss name, a list, or a directional/tolerance
mapping. A loss defaults to `low`, a metric to `high`; an explicit `direction` wins.
The legacy `high_is_better` and `value` keys were removed in 0.11.0: delete
`high_is_better`, replace directional `value` with `direction`, and tolerance
`value` with `tolerance_value`.

```yaml
trainer:
  module: echelon3.trainers.baseline
  type: Trainer
  config:
    epochs: 50
    times_to_validate_per_epoch: 1
    keep_best_on:
      accuracy:
        mode: directional    # save when this metric moves the right way
        direction: high          # high | low
      accuracy_control:
        mode: tolerance      # allow small regressions within a band
        tolerance_value: 0.1%          # relative fraction (0.001) or percent ("0.1%")
        direction: high
    metrics_on:
      accuracy: incidents
      accuracy_control: control
```

**Multiple metrics are always combined with AND.** A new best checkpoint is saved
only on a validation where **every** listed metric satisfies its condition at the same
time — there is no OR, weighted, or priority mode. With several metrics this makes
saves rare (all of them must improve together on the same validation). For dict-shaped
(multi-head) predictions, use `echelon3.trainers.multihead.MultiHeadTrainer` instead.

## `target`

Passed verbatim to `CheckpointManager(**config)`.

```yaml
target:
  path: ./targets/my_experiment
  checkpoints_to_keep: 2
```

`path` is the directory for `checkpoint-N.tar` files and TensorBoard events; it
is also where the default logger writes. Training resumes automatically from the
highest-numbered checkpoint found there. `checkpoints_to_keep` must be a positive
integer; omit it or use `null` to keep all checkpoints. After each successful save,
the manager deletes older checkpoints beyond the limit. A failed save preserves
previous checkpoints; merely opening a directory does not prune it.

### Last completed epoch versus best weights

For epoch-based trainers (`train` and `finetune`), enable an independent resume
snapshot with `trainer.config.save_last: true`. After a complete epoch, including
its scheduler step, Echelon3 atomically replaces `target.path/last.tar`. This
snapshot contains model, optimizer, scheduler, scaler, completed epoch,
`global_step`, and best-metric thresholds. An interrupted or profiling-shortened
epoch does not replace it. Only rank 0 writes it under DDP.

```yaml
trainer:
  module: echelon3.trainers.baseline
  type: Trainer
  config:
    epochs: 30
    keep_best_on: accuracy
    save_last: true
    # On a subsequent launch, explicitly resume the last completed epoch:
    # resume_from: last
```

The first run can use the default `resume_from: latest`. To continue from last:

```bash
echelon3 train -cd configs -cn experiment trainer.config.resume_from=last
# The same override is supported by finetune.
```

`resume_from: last` is strict: a missing `last.tar` raises an error, with no silent
fallback to an older best checkpoint. Existing targets remain compatible through
the default `latest` mode. A last-only target is also valid for explicit last
resume. Resuming epoch N starts N+1 and preserves the saved global step, even if
the configured batch size changes. `reset: true` still loads weights only and
restarts optimizer/scheduler/counters according to the config.

`last.tar` is excluded from numbered checkpoint discovery and retention; it never
changes the default weight selection for evaluate/export. Resume from last keeps
the previous best thresholds and does not save its initial validation as a new
numbered checkpoint. Like existing snapshots, last does not persist RNG state or
partially accumulated gradients; use an epoch length divisible by
`grad_accum_steps` when continuing an accumulation run without discarding a
partial window. Exact stochastic replay is not implied.

## `mlops` (optional)

Omit it and echelon3 uses `echelon3.mlops.tensorboard.TensorboardLogger`, whose
`folder` is set to `target.path`. Provide the section to swap loggers; the
`folder` is injected for you.

```yaml
mlops:
  module: echelon3.mlops.tensorboard
  type: SegmentationTensorboardLogger
```

## `export` (optional)

Read by `create_exporters` for `echelon3 export`. Optional `preprocess` /
`postprocess` maps (same shape as `transform.*.preprocess`) are fused **around**
the network so the exported graph is `preprocess -> net -> postprocess`; then one
or more named `exporters` run. See [Exporting to ONNX](../guide/export.md).

```yaml
export:
  preprocess:
    to01: { name: to01, module: echelon3.transforms.preprocess.basic, type: To01 }
  exporters:
    onnx:
      module: echelon3.exporters.onnx
      type: OnnxExporter
      config:
        target: ./targets/my_experiment/model.onnx
        input_names: [images]
        output_names: [logits]
        input_shape: [1, 3, 64, 64]
        opset: 18
        use_tracing: true
```

## `runner` (optional)

Read by `echelon3 run` via `create_universal(cfg.runner)`. The runner processes
a folder of images or a video through `preprocess -> net -> postprocess`. It
reuses `export.preprocess` / `export.postprocess`, and an optional
`export.wrapper` triple (e.g. `echelon3.wrappers.spatial.PadToMultiplier`).

```yaml
runner:
  module: echelon3.runners.segmenter
  type: ImagesSegmenter
  config: { source: ./inputs, target: ./outputs, colormap: {1: [255, 0, 0]}, opacity: 0.5 }
```

## `evaluator` (optional)

Read by `echelon3 evaluate`. `create_evaluator` builds the evaluator with the
network, the test dataloader, and the single metric named by `evaluator.metric`
(looked up in `metrics`).

```yaml
evaluator:
  module: echelon3.evaluators.classifier
  type: ClassifierEvaluator
  metric: accuracy
  config: { scores_and_labels: ./targets/my_experiment/errors }
```

## `init_from` / `finetune` (optional)

Only `echelon3 finetune` reads these; a stock training config runs unchanged
under that entry point. See [Extending](../guide/extending.md) for the workflow.

```yaml
init_from:
  checkpoint: ./targets/base/checkpoint-40.tar
  strict: false            # tolerate missing/extra keys (default false)

finetune:
  freeze_patterns: ['backbone\.']        # freeze params whose dotted name matches
  head_only: false                       # or: train only the head
  param_groups:                          # per-layer LR multipliers
    - { pattern: 'backbone\.', lr_mult: 0.1 }
    - { pattern: 'head\.',     lr_mult: 1.0 }
```

- `init_from.checkpoint` warm-starts the network's `state_dict` (the `module.`
  prefix is stripped automatically).
- `finetune.freeze_patterns` freezes tensors whose dotted name matches any regex.
- `finetune.param_groups` builds per-group learning rates (`lr_mult` times the
  optimizer's base `lr`); `head_only: true` trains only the head.

## A complete config

This is `examples/configs/smoke.yaml` — a full classifier run with an export
section, runnable end to end (see [First Run](../getting-started/first-run.md)):

```yaml
device: cuda

net:
  module: echelon3.nets.classifier
  type: ClassifierNet
  config:
    backbone:
      module: timm
      type: create_model
      config: { model_name: mobilenetv3_small_100, pretrained: false, num_classes: 0 }
    head:
      module: echelon3.nets.heads.classifier_heads
      type: DenseClassifierHead
      config:
        num_classes: 2
        num_features: 1024
        dropout: 0.1
        activation: { module: torch.nn, type: Identity }

data:
  train:
    module: echelon3.data.imageclassifier
    type: FoldersHiveImageClassifierDataset
    config: { classes: 2, folder: ${oc.env:SMOKE_DATA_ROOT,./smoke_data}/train, wildcards: ['*.png'] }
  test:
    module: echelon3.data.imageclassifier
    type: FoldersHiveImageClassifierDataset
    config: { classes: 2, folder: ${oc.env:SMOKE_DATA_ROOT,./smoke_data}/test, wildcards: ['*.png'] }

transform:
  train:
    preprocess:
      to01: { name: to01, module: echelon3.transforms.preprocess.basic, type: To01 }
  test:
    preprocess:
      to01: { name: to01, module: echelon3.transforms.preprocess.basic, type: To01 }

loss:
  - ce:
      module: echelon3.losses.classification
      type: WeightedCrossEntropyLoss
      weight: 1.0
      config: { weight: [1.0, 1.0] }

metrics:
  - accuracy:
      module: torchmetrics
      type: Accuracy
      config: { task: multiclass, num_classes: 2 }

dataloaders:
  train:
    module: torch.utils.data
    type: DataLoader
    config: { batch_size: 16, shuffle: true, num_workers: 2, drop_last: true }
  test:
    module: torch.utils.data
    type: DataLoader
    config: { batch_size: 16, shuffle: false, num_workers: 2 }

optimizer:
  module: torch.optim
  type: AdamW
  config: { lr: 0.001 }

scheduler:
  module: torch.optim.lr_scheduler
  type: LinearLR
  config: { start_factor: 1.0, end_factor: 1.0, total_iters: 1 }

trainer:
  module: echelon3.trainers.baseline
  type: Trainer
  config:
    epochs: 6
    keep_best_on:
      accuracy: { mode: directional, direction: high }
    times_to_validate_per_epoch: 1

target:
  path: ${oc.env:SMOKE_TARGET,./targets/smoke}
  checkpoints_to_keep: 2

export:
  preprocess:
    to01: { name: to01, module: echelon3.transforms.preprocess.basic, type: To01 }
  exporters:
    onnx:
      module: echelon3.exporters.onnx
      type: OnnxExporter
      config:
        target: ${oc.env:SMOKE_TARGET,./targets/smoke}/smoke.onnx
        input_names: [images]
        output_names: [logits]
        input_shape: [1, 3, 64, 64]
        use_tracing: true
        opset: 18
```

## Next

- [Built-in Components](components.md) — the catalog of `module`/`type` values.
- [Extending](../guide/extending.md) — plug in your own classes.
