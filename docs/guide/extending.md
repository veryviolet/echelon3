# Extending echelon3

echelon3 has **no component registry**. There are no decorators to apply, no
`register()` calls, no entry points to declare. A component is reached by its
import path, so *any importable class or function* — including one you just wrote
— can be named in a `module:` / `type:` triple and instantiated.

That single design choice is the whole extension model: to add a network, a
dataset, a loss, a metric, an optimizer, or a trainer, you write a normal Python
class and point a config at it.

## How resolution works

`echelon3.creator.get_attr_from_module` resolves every triple:

1. `importlib.import_module(module)` then `getattr(module, type)`.
2. If the import fails, it falls back to loading `module` as a **path to a `.py`
   file** and pulling `type` out of it.

So both of these work out of the box:

```yaml
net:
  module: my_project.nets.mynet   # an importable package path
  type: MyNet

# or, a loose file:
net:
  module: ./experiments/mynet.py  # a path to a .py file
  type: MyNet
```

The class just has to accept whatever you put under `config:` as keyword
arguments. Most echelon3 components take `**config` and a couple of injected
objects (a dataset gets `augment`/`preprocess`; a loss/metric/net gets only its
config).

## The zoo model: your code beside your configs

The recommended way to work is to keep your models, datasets and configs
together in one git repository and run echelon3 from its root:

```
my_project/
├── configs/
│   └── my_experiment.yaml
└── my_project/
    ├── __init__.py
    ├── nets/
    │   ├── __init__.py
    │   └── mynet.py
    └── data/
        └── ...
```

The CLI entry points (`echelon3 train`, `echelon3 export`, `echelon3 run`,
`echelon3 evaluate`, `echelon3 finetune`) call `add_cwd_to_sys_path()` before
echelon3 inserts the current working directory at the front of
`sys.path`, so a package sitting in your repo root resolves exactly like a
built-in one.

!!! note "Run from the repo root"
    `add_cwd_to_sys_path()` adds `os.getcwd()` — the directory you launch from,
    not where the config lives. Run `echelon3 train` from the repo root so that
    `module: my_project.nets.mynet` imports. (`python -m ...` and
    `python script.py` add the script's directory automatically; console scripts
    do not, which is exactly what this shim fixes.)

## A minimal custom network

Write a `torch.nn.Module` whose constructor accepts the config keys as kwargs.
Accept `**kwargs` so extra config keys never break construction:

```python
# my_project/nets/mynet.py
import torch
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self, num_classes: int, width: int = 32, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))
```

Point the `net` section at it — everything under `config` lands in `__init__`:

```yaml
net:
  module: my_project.nets.mynet
  type: MyNet
  config:
    num_classes: 2
    width: 64
```

The rest of the config (data, losses, optimizer, trainer, …) is unchanged; see
the [Config Schema](../reference/config-schema.md). Train it the usual way from
the repo root:

```bash
echelon3 train --config-dir configs --config-name my_experiment
```

The same pattern extends to every subsystem — a dataset subclassing
`echelon3.data.basic.AllFilesDataset`, a loss subclassing `torch.nn.Module`, a
metric with `update`/`compute`/`reset`, even a `Trainer` subclass. None of them
register with echelon3; they are just named by import path.

!!! tip "Compose with built-ins"
    Because container nets resolve their children through the same mechanism, you
    can mix your code with the built-ins: a custom head on a `timm` backbone
    inside `echelon3.nets.classifier.ClassifierNet`, or your backbone inside
    `echelon3.nets.segmenter.Segmenter`.

## Custom trainers: the `compute_losses` seam

A `Trainer` subclass rarely needs to touch the training loop. Everything that is
easy to get wrong — DDP, the autocast/precision path, the grad scaler, closure
optimizers, sharded validation, checkpointing — lives in the base and stays
there. To customize how inputs reach the network or how losses map to its
outputs, override a single method:

```python
def compute_losses(self, source, labels, net=None):
    """Runs INSIDE the trainer's autocast; return (predictions, {name: (loss, weight)})."""
    net = net if net is not None else self._net
    ...
    return predictions, losses_values
```

Both training and validation call it, so you never re-implement the precision
path. `echelon3.trainers.pair.PairTrainer` is exactly this: it consumes
`((base, query), gt)` batches, calls `net(base, query, return_features)` and
delegates loss routing to a second hook, `pair_losses(heatmap, features, labels)`
(default: every loss on the heatmap). A domain-specific image-in-image trainer is
then a natural subclass that overrides only `pair_losses`:

```python
class MyImageInImageTrainer(PairTrainer):
    def pair_losses(self, heatmap, features, labels):
        presence = (labels.flatten(1).max(1).values > 0.5).float()
        out = {}
        for name, (loss, weight) in self._losses.items():
            if "presence" in name.lower() and features is not None:
                out[name] = (loss(features, presence), weight)
            else:
                out[name] = (loss(heatmap, labels.float()), weight)
        return out   # no autocast / scaler / closure code — the base owns it
```

## Public zoo repositories

Whole architecture collections live in separate repos you clone next to your
configs and reference by import path — the same `module:` mechanism, just a
bigger library. The public
[`veryviolet/echelon3_zoo`](https://github.com/veryviolet/echelon3_zoo)
repository packages ready-to-use segmentation architectures (rtseg, PIDNet,
U2Net, DFANet) and backbones.

Clone it beside your project so it is importable from your working directory:

```
workspace/
├── echelon3_zoo/        # cloned repo (a package named echelon3_zoo)
└── my_project/
    └── configs/
```

Then reference a zoo model directly:

```yaml
net:
  module: echelon3_zoo.nets.monolithic.u2net
  type: U2Net
  config: { num_classes: 1 }
```

Run `echelon3 train` from `workspace/` (or wherever `echelon3_zoo` is
importable) and the resolver finds it. No installation, no registration — a zoo
is simply more importable code.

## Next

- [Config Schema](../reference/config-schema.md) — every section explained.
- [Built-in Components](../reference/components.md) — what ships in the box.
- [How It Works](../concepts/how-it-works.md) — the resolution model in depth.
</content>
