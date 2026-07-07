# CLI

All commands are Hydra applications. They take `--config-dir` (where your configs
live) and `--config-name` (which config, without `.yaml`), and accept arbitrary
`key=value` overrides for any config value.

```bash
echelon3-train --config-dir configs --config-name my_experiment \
    trainer.config.epochs=100 optimizer.config.lr=0.0005
```

!!! tip "Configs and code in the same repo"
    Every command inserts the current working directory into `sys.path` before
    running, so configs can reference packages that live in your repo
    (`module: my_project.nets.foo`). Run from the repo root. See
    [Extending](../guide/extending.md).

## `echelon3-train`

Train a model. Assembles the full pipeline from the config and runs the trainer's
`train()` loop. Resumes automatically if `target.path` already holds checkpoints.
Supports DDP under `torchrun` and `DataParallel` otherwise — see
[Multi-GPU (DDP)](../guide/ddp.md).

## `echelon3-finetune`

Same as `echelon3-train`, plus three optional config blocks:

- `init_from.checkpoint` — warm-start the network's weights from a checkpoint
  (the `module.` prefix from DataParallel/DDP checkpoints is stripped; loading is
  non-strict by default).
- `finetune.freeze_patterns` — freeze parameters whose dotted name matches any
  regex in the list.
- `finetune.head_only` / `finetune.param_groups` — train only the head, or build
  per-layer parameter groups with LR multipliers.

With none of these present it behaves exactly like `echelon3-train`.

## `echelon3-evaluate`

Load the latest checkpoint under `target.path` and evaluate it against a single
metric (named by `evaluator.metric`) over the `data.test` set. Reads the
`evaluator` section of the config.

## `echelon3-export`

Load the latest checkpoint and run the exporters in the `export` section. The
built-in `OnnxExporter` wraps preprocess → network → postprocess into one ONNX
graph. Needs `pip install echelon3[export]`. See [Exporting to ONNX](../guide/export.md).

## `echelon3-run`

Run inference over images or video with a `runner` (for example a segmentation
overlay writer), using the checkpoint and the export preprocess/postprocess/wrapper
from the config.
