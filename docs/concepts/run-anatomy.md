# Anatomy of a Run

Once `echelon3 train` has assembled the components (see
[How It Works](how-it-works.md)), control passes to the trainer's `train()` method.
This page describes what the built-in `echelon3.trainers.baseline.Trainer` does, so
that the `trainer` section of a config makes sense.

## Resume-or-start

`train()` first asks the checkpoint manager what already exists under `target.path`:

- **No checkpoints** → start from scratch at epoch 1, initialize the network.
- **Checkpoints present** → load the latest, recompute the global step, re-run
  validation to re-establish the "best so far" metrics, then continue.

Checkpoints are `checkpoint-<N>.tar` files holding the epoch, model `state_dict`,
optimizer and scheduler state, and the metric objects. Re-running the same command
against a populated `target.path` therefore *resumes*; point it at a fresh
directory to start over.

## The epoch loop

For each epoch the trainer iterates the train dataloader and, at each step, calls
`one_step_train`. A step:

1. zeroes gradients,
2. runs the forward pass,
3. computes each configured loss and multiplies it by its weight,
4. sums them into the total loss, backpropagates, and steps the optimizer,
5. logs per-loss values.

!!! note "Optimizers that need a closure"
    Optimizers whose class name is `SAMOptimizer` or `LBFGS` are stepped with a
    closure (they evaluate the model more than once per step); everything else uses
    the standard `loss.backward(); optimizer.step()` path. This is detected
    automatically from the optimizer you configured.

Predictions with trailing singleton dimensions are squeezed to line up with the
labels, so a network that emits `(N, C, 1, 1)` still matches `(N,)` class labels
without extra config.

At the end of each epoch the LR scheduler is stepped.

## Validation and keep-best

Validation runs several times per epoch, controlled by
`trainer.config.times_to_validate_per_epoch`. Each validation pass resets the
metrics, runs the test loader(s), and logs the results. With multiple named test
sets (see the [config schema](../reference/config-schema.md)), metrics are logged
per set as `{set_name}/{metric}`.

Checkpoints are saved based on `keep_best_on`. It can be:

- a single metric name (kept when it improves in the direction implied by
  `high_is_better`),
- a list of metric names,
- or a mapping with per-metric modes — `directional` (`value: high|low`) or
  `tolerance` (only counts as improvement if it moves past a threshold).

When several metrics are tracked, a checkpoint is saved only when **all** of them
improve together. The checkpoint manager keeps the most recent
`target.checkpoints_to_keep` files.

## A minimal `trainer` section

```yaml
trainer:
  module: echelon3.trainers.baseline
  type: Trainer
  config:
    epochs: 50
    times_to_validate_per_epoch: 1
    high_is_better: true
    keep_best_on:
      accuracy:
        mode: directional
        value: high
```

## Multi-GPU and logging

- Under `torchrun`, the network is wrapped in `DistributedDataParallel`; without
  it, in `DataParallel`. See [Multi-GPU (DDP)](../guide/ddp.md).
- Logging and checkpoint writing happen on rank 0 only. Non-main ranks get a
  no-op logger, so nothing is double-written.

## Other trainers

`echelon3.trainers.multihead.MultiHeadTrainer` handles networks that emit a dict of
per-head predictions with matching dict labels (multi-head binary segmentation).
Select it the same way — by pointing the `trainer` section at it. Custom trainers
in your own repo work identically; subclass `Trainer` and reference it by import
path.
