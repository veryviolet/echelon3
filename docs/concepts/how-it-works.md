# How It Works

echelon3 has one central idea: a training run is a tree of components, and every
component is described the same way — by **where to import it from** and **what to
pass its constructor**. There is no plugin registry, no decorators, no
`@register` calls. The framework resolves names at run time by importing them.

## The `module` / `type` / `config` triple

Every node in a config is a triple:

```yaml
module: echelon3.nets.classifier   # a Python import path...
type: ClassifierNet                # ...and a name inside it
config:                            # ...and kwargs for the constructor
  backbone: { ... }
  head: { ... }
```

The resolver (`echelon3.creator.get_attr_from_module`) does exactly what you would
by hand:

1. `importlib.import_module("echelon3.nets.classifier")`
2. `getattr(module, "ClassifierNet")`
3. call it with `**config`

If step 1 fails as an import, the resolver falls back to loading `module` as a
**path to a `.py` file** — so `module: ./experiments/mynet.py` works too.

Because `type` is resolved with `getattr`, it can be a class *or* a factory
function. When it is a function (for example `timm.create_model`), the framework
calls it to get the object. This is why you can write:

```yaml
backbone:
  module: timm
  type: create_model
  config: { model_name: mobilenetv3_small_100, pretrained: false, num_classes: 0 }
```

and get a live timm backbone, with no adapter code in echelon3.

## Composition is recursive

Container components resolve their own children. `ClassifierNet` receives
`backbone` and `head` sub-trees and calls the factory on each; `Segmenter` does the
same for `backbone` / `neck` / `head`. You compose networks by nesting triples, not
by writing glue code.

## The build order

`echelon3-train` assembles a run in a fixed order (see `echelon3.cli.train`), each
step reading one top-level section of the config:

```
transform  →  augmentations (albumentations) + preprocess (torch.nn.Sequential)
data       →  train dataset + test dataset(s)
dataloaders→  train loader + test loader(s)  (DDP-aware: see the DDP guide)
net        →  the network (+ optional weights_loader)
loss       →  { name: (loss_module, weight) }
metrics    →  { name: metric }
optimizer  →  optimizer over net.parameters()
scheduler  →  LR scheduler
target     →  checkpoint manager
mlops      →  logger (defaults to TensorBoard)
trainer    →  the training loop, then .train()
```

Each arrow is one `create_*` function in `echelon3.creator`. Nothing is special-
cased: the trainer, the losses, and the network are all resolved through the same
triple mechanism, so swapping any of them is a config edit.

## Why there is no registry

A registry couples every component to the framework — you would have to import a
package for its side effects, or add your class to a list somewhere. echelon3
resolves by import path instead, which means:

- **Your code is first-class.** A class in your repo is referenced exactly like a
  built-in one: `module: my_project.nets.foo`. See [Extending](../guide/extending.md).
- **Third-party code is first-class.** `torch.optim.AdamW`,
  `torchmetrics.Accuracy`, `albumentations.ColorJitter`, `timm.create_model` are
  all used directly from their own packages.
- **The framework stays small.** The core ships generic building blocks; whole
  architecture collections live in separate zoo repositories you opt into.

## Next

- [Anatomy of a Run](run-anatomy.md) — what the trainer actually does each step.
- [Config Schema](../reference/config-schema.md) — every section, key by key.
