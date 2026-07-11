# CLI

echelon3 ships a single executable, `echelon3`, with one subcommand per task:

```bash
echelon3 train     --config-dir configs --config-name my_experiment [overrides...]
echelon3 finetune  --config-dir configs --config-name my_experiment [overrides...]
echelon3 evaluate  --config-dir configs --config-name my_experiment [overrides...]
echelon3 export    --config-dir configs --config-name my_experiment [overrides...]
echelon3 run       --config-dir configs --config-name my_experiment [overrides...]
```

Every subcommand takes:

| Option | Short | Meaning |
| --- | --- | --- |
| `--config-dir` | `-cd` | Directory your config lives in (default: current dir). |
| `--config-name` | `-cn` | Config file name, with or without `.yaml`. **Required.** |
| `overrides` | | Any number of `key=value` overrides (see below). |

`echelon3 --help` lists the subcommands; `echelon3 train --help` shows options for
one. Config loading is powered by **OmegaConf** (not Hydra) — the override and
`defaults:` syntax below is a compatible subset, so existing configs and command
lines keep working.

!!! tip "Configs and code in the same repo"
    Each subcommand inserts the current working directory into `sys.path` before
    running, so configs can reference packages that live next to them
    (`module: my_project.nets.foo`). Run from your repo root. See
    [Extending](../guide/extending.md).

!!! note "Transitional aliases"
    The old per-command scripts `echelon3-train`, `echelon3-finetune`,
    `echelon3-evaluate`, `echelon3-export`, `echelon3-run` still exist and behave
    identically to `echelon3 <cmd>`. They are kept for compatibility and will be
    removed in a future release — prefer `echelon3 <cmd>`.

## Overrides

Any positional argument of the form `key=value` overrides a config value. Keys are
dotted paths into the config; values are typed with OmegaConf's grammar.

```bash
echelon3 train -cd configs -cn my_experiment \
    trainer.config.epochs=100 \
    optimizer.config.lr=0.0005 \
    dataloaders.train.config.batch_size=256 \
    data.train.config.root=/data/train
```

### Prefixes

| Form | Effect |
| --- | --- |
| `key=value` | Set the key (creating it if absent). |
| `+key=value` | Add the key. Same as `key=value` — the `+` is accepted for compatibility, not required. |
| `++key=value` | Add or override. Also equivalent here. |
| `~key` | **Delete** the key (and its subtree). |

There is no strict/struct mode: `key=value` freely adds a new key, so you never need
`+` just to introduce one (this removes the classic Hydra `+gpus` footgun). `+` /
`++` are still accepted so existing command lines run unchanged.

```bash
echelon3 train -cd configs -cn my_experiment \
    +trainer.config.compile=true \      # add a key that isn't in the file
    ~scheduler \                        # drop the scheduler section (→ constant LR)
    ~trainer.config.keep_best_on        # drop a nested key
```

### Value types

Values are parsed with OmegaConf's grammar (the same one Hydra used), so types are
inferred exactly as before:

| Override | Parsed as |
| --- | --- |
| `epochs=100` | `int` |
| `lr=0.0005` | `float` |
| `compile=true` | `bool` |
| `scheduler=null` | `None` |
| `gpus=[0,1,2,3]` | list |
| `compile_mode=reduce-overhead` | `str` |
| `path=${oc.env:OUT,/tmp/run}` | env interpolation, resolved at read time |

`hydra.*` overrides (e.g. `hydra.run.dir=...`) are accepted and ignored — there is no
Hydra run directory to configure.

## Config composition (`defaults:`)

A config may pull in other configs with a `defaults:` list, merged **left to right**
(later entries override earlier ones):

```yaml
# my_experiment.yaml
defaults:
  - base_experiment      # load base_experiment.yaml, merge at the root
  - net: resnet50        # load net/resnet50.yaml, merge under the `net:` key
  - _self_               # then this file's own keys (wins over the above)

trainer:
  config:
    epochs: 100
```

- **`- name`** — load `name.yaml` (may include a subdir, `- group/name`) and merge it
  at the **root**. Base configs may have their own `defaults:` (composition is
  recursive; cycles are detected).
- **`- group: option`** — a *config group*: load `group/option.yaml` and merge it
  under the **`group`** key. Nested groups work too (`- data/aug: heavy` → under
  `data.aug`). This is Hydra's default packaging.
- **`- _self_`** — where this file's own content is applied. If you omit it, it is
  applied last (this file wins), as in Hydra.

Not supported (they were the arcane, footgun-prone parts): `@package` path directives
and `override` / `optional` keywords in `defaults:`. Unsupported entries raise a clear
error.

## Subcommands

### `echelon3 train`

Train a model. Assembles the full pipeline from the config and runs the trainer's
`train()` loop. Resumes automatically if `target.path` already holds checkpoints.
Multi-GPU with `gpus=[0,1,2,3]` (built-in DDP launcher, no `torchrun` needed) — see
[Multi-GPU & Precision](../guide/ddp.md). Single-GPU honours the `gpus` index
(`gpus=[1]` runs on physical GPU 1); `device=cpu` forces CPU.

```bash
# single GPU, bf16 autocast (default on CUDA)
echelon3 train -cd configs -cn my_experiment

# 4-GPU DDP, global batch split across ranks
echelon3 train -cd configs -cn my_experiment gpus=[0,1,2,3]

# CPU smoke, 2 epochs
echelon3 train -cd configs -cn my_experiment device=cpu trainer.config.epochs=2
```

### `echelon3 finetune`

Same as `train`, plus three optional config blocks:

- `init_from.checkpoint` — warm-start the network's weights from a checkpoint (the
  `module.` prefix from DDP checkpoints is stripped; loading is non-strict by
  default).
- `finetune.freeze_patterns` — freeze parameters whose dotted name matches any regex
  in the list.
- `finetune.head_only` / `finetune.param_groups` — train only the head, or build
  per-layer parameter groups with LR multipliers.

With none of these present it behaves exactly like `echelon3 train`.

### `echelon3 evaluate`

Load the latest checkpoint under `target.path` and evaluate it against a single
metric (named by `evaluator.metric`) over the `data.test` set. Reads the `evaluator`
section of the config.

### `echelon3 export`

Load the latest checkpoint and run the exporters in the `export` section. The built-in
`OnnxExporter` wraps preprocess → network → postprocess into one ONNX graph. Needs
`pip install echelon3[export]`. See [Exporting to ONNX](../guide/export.md).

### `echelon3 run`

Run inference over images or video with a `runner` (for example a segmentation
overlay writer), using the checkpoint and the export preprocess/postprocess/wrapper
from the config.

## Interrupting a run

`Ctrl-C` stops a run cleanly — a one-line notice and exit code 130, no traceback —
including under multi-GPU DDP (the launcher, ranks, and DataLoader workers are all
reaped).
