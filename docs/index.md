# echelon3

echelon3 is a config-driven PyTorch training framework. Every part of a training
run — the network, dataset, augmentations, losses, metrics, optimizer, scheduler,
trainer, and export — is described in a YAML file as a `module` / `type` / `config`
triple and instantiated dynamically:

```yaml
net:
  module: echelon3.nets.classifier   # import path (or a path to a .py file)
  type: ClassifierNet                # class or factory function in that module
  config: { ... }                    # constructor kwargs
```

There is **no component registry**. Anything importable can be plugged in — a class
from `echelon3`, from `torch` / `timm` / `torchmetrics` / `albumentations`, or from
your own project code living next to your configs. To add a component you write a
class and point a config at it; you never edit the framework.

## What you get

- A single factory (`echelon3.creator`) that turns a YAML tree into a live training
  run by resolving each `module` / `type` / `config` triple.
- A `Trainer` with multi-GPU support (DDP via `torchrun`, or `DataParallel`),
  multi-metric keep-best checkpointing, and multiple named validation sets.
- Batteries-included generic components: folder/CSV classification datasets,
  segmentation and detection datasets, balanced/classwise/multipart dataloaders,
  an albumentations + torch preprocessing pipeline, classification/segmentation/
  detection losses and metrics, timm-backed classifier and segmenter networks.
- CLI entry points `echelon3-train`, `echelon3-evaluate`, `echelon3-run`,
  `echelon3-export`, `echelon3-finetune` (all Hydra-based, `--config-dir`).
- ONNX export and a TensorBoard logger.

## Quickstart

```bash
pip install echelon3
```

Generate a tiny synthetic dataset and train a classifier on it:

```bash
# from a checkout of the repo (examples/ ships the smoke config)
python examples/gen_smoke_data.py --root ./smoke_data
SMOKE_DATA_ROOT=./smoke_data echelon3-train \
    --config-dir examples/configs --config-name smoke
```

See [Getting Started](getting-started/installation.md) to install and run your own
config, [How It Works](concepts/how-it-works.md) for the model behind the framework,
and [Extending](guide/extending.md) to plug in your own networks and datasets.

## Architecture zoos

The framework core ships only generic components. Ready-made architectures
(real-time segmentation networks, PIDNet, U²-Net, DFANet, classic CNN backbones)
live in the companion public repository
[veryviolet/echelon3_zoo](https://github.com/veryviolet/echelon3_zoo), which you
clone next to your configs and reference by import path — see
[Extending](guide/extending.md).
