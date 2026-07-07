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
| `device` / `device_ids` | `cli.train` | Target device and `DataParallel` GPU ids |
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

## `device` / `device_ids`

Plain scalars, not triples. `device` defaults to `cuda` (falls back to `cpu`
when CUDA is unavailable). `device_ids` is a list of GPU indices passed to
`torch.nn.DataParallel`.

```yaml
device: cuda
device_ids: [0, 1]
```

!!! warning "Ignored under DDP"
    When launched with `torchrun`, both keys are ignored — one process owns one
    GPU chosen by `LOCAL_RANK`, and visibility comes from `CUDA_VISIBLE_DEVICES`.
    See [Multi-GPU Training](../guide/ddp.md).

## `net`

`create_net` resolves the triple and calls `type(**config)`. `config` is
required. Container nets (`ClassifierNet`, `Segmenter`, `HeatmapDetector`)
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
every `test` dataset).

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
required (even if empty for a schedule with defaults). `scheduler.step()` is
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
| `high_is_better` | Default direction when `keep_best_on` is a bare name (default `true`). |
| `float_labels` | Cast labels to float before the loss (default `false`). |
| `reset` | On resume, restart epoch/step counters instead of continuing. |
| `metrics_on` | Map `metric_name -> test_dataset_name` for named test sets. |

`keep_best_on` accepts a bare metric name, a list, or a directional/tolerance
mapping:

```yaml
trainer:
  module: echelon3.trainers.baseline
  type: Trainer
  config:
    epochs: 50
    times_to_validate_per_epoch: 1
    high_is_better: true
    keep_best_on:
      accuracy:
        mode: directional    # save when this metric moves the right way
        value: high          # high | low
      accuracy_control:
        mode: tolerance      # allow small regressions within a band
        value: 0.1%          # absolute (0.001) or percent ("0.1%")
        direction: high
    metrics_on:
      accuracy: incidents
      accuracy_control: control
```

A checkpoint is saved only when **all** listed metrics satisfy their condition
simultaneously. For dict-shaped (multi-head) predictions, use
`echelon3.trainers.multihead.MultiHeadTrainer` instead.

## `target`

Passed verbatim to `CheckpointManager(**config)`.

```yaml
target:
  path: ./targets/my_experiment
  checkpoints_to_keep: 2
```

`path` is the directory for `checkpoint-N.tar` files and TensorBoard events; it
is also where the default logger writes. Training resumes automatically from the
highest-numbered checkpoint found there.

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

Read by `create_exporters` for `echelon3-export`. Optional `preprocess` /
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

Read by `echelon3-run` via `create_universal(cfg.runner)`. The runner processes
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

Read by `echelon3-evaluate`. `create_evaluator` builds the evaluator with the
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

Only `echelon3-finetune` reads these; a stock training config runs unchanged
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
device_ids: [0]

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
      accuracy: { mode: directional, value: high }
    times_to_validate_per_epoch: 1
    high_is_better: true

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
</content>
</invoke>
