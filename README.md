# echelon3

**Describe a PyTorch training run in YAML — network, data, augmentations, losses,
metrics, optimizer, scheduler, trainer, export — and run it with one command.**

No training-loop boilerplate and no component registry: every piece is a
`module` / `type` / `config` triple that echelon3 imports and instantiates, so
anything importable drops straight in — classes from `echelon3`, from
`torch` / `timm` / `torchmetrics` / `albumentations`, or from your own project code
sitting next to your configs.

```yaml
# my_run.yaml — a whole training run
net:  { module: my_pkg.nets, type: MyNet, config: { channels: 32 } }
data:
  train: { module: my_pkg.data, type: MyDataset, config: { split: train } }
  test:  { module: my_pkg.data, type: MyDataset, config: { split: test } }
dataloaders:
  train: { module: torch.utils.data, type: DataLoader, config: { batch_size: 64, shuffle: true } }
  test:  { module: torch.utils.data, type: DataLoader, config: { batch_size: 128 } }
loss:
  - ce: { module: torch.nn, type: CrossEntropyLoss, config: {} }
optimizer: { module: torch.optim, type: AdamW, config: { lr: 3e-4 } }
trainer:   { module: echelon3.trainers.baseline, type: Trainer, config: { epochs: 50 } }
target:    { path: ./out }
```

```bash
echelon3 train -cd . -cn my_run     # trains, validates, keeps the best, checkpoints, resumes
```

## What you get

- **One CLI, five tasks** — `echelon3 train | finetune | evaluate | run | export`.
- **Multi-GPU with no `torchrun`** — `gpus=[0,1,2,3]` spawns one DDP worker per GPU;
  `batch_size` is global and split across ranks automatically.
- **bf16 mixed precision by default** on capable GPUs (`precision: fp32` to opt out),
  plus an optional `torch.compile` knob and TF32.
- **Batteries included in the trainer** — automatic resume, keep-best-checkpoint,
  validation schedule, TensorBoard / mlops logging, clean `Ctrl-C`.
- **Override anything on the CLI** — `trainer.config.epochs=100 optimizer.config.lr=1e-4`,
  `~scheduler` drops a section, `+key=…` adds one; typed values, list literals,
  `${oc.env:VAR,default}` interpolation, and `defaults:` config composition (OmegaConf).
- **Your own code, no registration** — reference nets / datasets / losses by import
  path and run from your repo root (the cwd is on `sys.path`).

## Install

```bash
pip install echelon3
# extras: echelon3[export] (ONNX)   echelon3[detection]   echelon3[smp]
```

## CLI & overrides

```bash
echelon3 train    -cd ./configs -cn my_run          # train (auto-resumes from target.path)
echelon3 finetune -cd ./configs -cn my_run          # + warm-start / freeze / head-only
echelon3 evaluate -cd ./configs -cn my_run          # score the latest checkpoint
echelon3 export   -cd ./configs -cn my_run          # preprocess→net→postprocess to ONNX
echelon3 run      -cd ./configs -cn my_run          # inference over images / video

# override any config value; +add, ~delete, typed values, multi-GPU:
echelon3 train -cd ./configs -cn my_run \
    trainer.config.epochs=100 optimizer.config.lr=5e-4 \
    dataloaders.train.config.batch_size=256 \
    gpus=[0,1] +trainer.config.compile=true ~scheduler
```

`--config-dir`/`-cd` picks the directory, `--config-name`/`-cn` the YAML.

## Quick start

`examples/` has self-contained smoke runs — a classifier, a CenterNet-style detector
and semantic segmentation — each with a synthetic-data generator and a minimal config
that trains, validates and checkpoints on CPU or GPU. For example:

```bash
cd examples/segmentation
python gen_seg_data.py --root ./seg_data
SEG_DATA=./seg_data echelon3 train -cd . -cn segmentation_smoke device=cpu
```

## Use it from your AI coding agent

Marketplace-installable plugins teach **Codex** and **Claude Code** the echelon3
config format and CLI, so the agent writes correct configs and runs them for you —
repo: <https://github.com/veryviolet/echelon3-agent-skills>.

```bash
# Codex
codex plugin marketplace add veryviolet/echelon3-agent-skills
codex plugin add echelon3@veryviolet
```
```text
# Claude Code (run inside the REPL)
/plugin marketplace add veryviolet/echelon3-agent-skills
/plugin install echelon3@veryviolet
```

## Full documentation

**<https://veryviolet.github.io/echelon3/>**
