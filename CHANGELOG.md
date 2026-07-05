# Changelog

All notable changes to **echelon3** are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely; versions
follow [SemVer](https://semver.org/) once 1.0.0 ships.

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
