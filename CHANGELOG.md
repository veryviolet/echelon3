# Changelog

All notable changes to **echelon3** are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely; versions
follow [SemVer](https://semver.org/) once 1.0.0 ships.

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
