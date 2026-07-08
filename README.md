# echelon3

Config-driven PyTorch training framework. Every component of a training run —
network, dataset, augmentations, losses, metrics, optimizer, scheduler, trainer,
export — is described in a YAML config as a `module` / `type` / `config` triple
and instantiated dynamically:

```yaml
net:
  module: echelon3.nets.classifier   # import path (or a path to a .py file)
  type: ClassifierNet                # class or factory function in that module
  config: { ... }                    # constructor kwargs
```

There is no component registry: anything importable can be plugged in — classes
from `echelon3`, from `torch`/`timm`/`torchmetrics`/`albumentations`, or from
your own project code living next to your configs.

**Documentation: <https://veryviolet.github.io/echelon3/>**

## Install

```
pip install echelon3
```

## Train

```
echelon3-train --config-dir ./configs --config-name my_experiment
```

CLIs: `echelon3-train`, `echelon3-finetune` (warm-start / freeze / head-only),
`echelon3-evaluate`, `echelon3-run` (inference), `echelon3-export` (ONNX).

## Multi-GPU — built in, no `torchrun`

Name the GPUs and echelon3 spawns one DDP worker per GPU itself:

```
echelon3-train --config-dir ./configs --config-name my_experiment gpus=[0,1,2,3]
```

`gpus` is a root config key — leave it out and echelon3 uses every visible GPU on
the node. `dataloaders.train.config.batch_size` is the **global** batch size; it
is split across ranks automatically. `torchrun` (and SLURM `srun`) still work
unchanged for multi-node / elastic jobs.

> DataParallel was removed in 0.5.0 — multiple GPUs always run as DDP.

## Mixed precision

Training, evaluation and inference use **bf16 automatic mixed precision by
default** on capable GPUs (fp32 on CPU / unsupported GPUs) — a large speedup on
modern hardware. Force full fp32 with `precision: fp32` under `trainer.config`
(or `precision: fp32` at the config root for `evaluate` / `run`).

## Quick start

`examples/` has self-contained smoke runs — a classifier, a CenterNet-style
detector and semantic segmentation — each with a synthetic-data generator and a
minimal config that trains, validates and checkpoints on CPU or GPU.
