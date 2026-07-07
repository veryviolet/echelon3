# Echelon3

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

Multi-GPU (DDP):

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    $(which echelon3-train) --config-dir ./configs --config-name my_experiment
```

`dataloaders.train.config.batch_size` is the global batch size; under DDP it is
split across ranks automatically. Without `torchrun` the trainer falls back to
`DataParallel` (`device` / `device_ids` config keys).

## Quick start

See `examples/` for a self-contained smoke run: synthetic dataset generation and
a minimal classifier config.
