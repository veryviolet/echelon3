# Changelog

All notable changes to **echelon3** are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely; versions
follow [SemVer](https://semver.org/) once 1.0.0 ships.

## 0.10.0 — 2026-07-23

### Added

- **`MultiDatasetMetric` — a metric that spans several validation datasets with a single
  `compute()`.** Some metrics need a cross-dataset context that a per-loader metric cannot
  express — e.g. retrieval, where recall/mAP is defined over a *query* set matched against a
  *gallery* set. A `MultiDatasetMetric` (in `echelon3.metrics`) declares the test datasets it
  spans via `self.datasets` (typically built in its constructor from roles such as
  `query_dataset` / `gallery_dataset`), and its `update(predicted, target, dataset)` receives
  the name of the current batch's source dataset. `Trainer.validate()` orchestrates it: one
  `reset()` before all of the metric's datasets, tagged `update()`s while iterating the loaders
  (in the same pass as ordinary metrics — no extra forward), and a single `dist_reduce()` +
  `compute()` after all of them. Ordinary single-dataset metrics are unchanged. `keep_best`
  can track a multi-metric by name. The datasets a metric declares must exist among the test
  loaders and the roster must be non-empty, else validation raises a clear error. Under DDP,
  `dist_reduce()` gathers each metric's buffers across ranks (helper `all_gather_cat`) so
  `compute()` sees the full set; the console prints a short `Finalizing multi-dataset
  metrics…` / `Finalized multi-dataset metrics: …` summary after the per-loader lines.

### Changed

- **All in-code comments and docstrings translated to English.** echelon3 is a public,
  multi-language package; the source is now uniformly English. No behavior change — the sweep
  touched only comments/docstrings (and a single informational runtime line in the estimator
  entry point).

## 0.9.6 — 2026-07-18

### Changed

- **`persistent_workers` now defaults to `true` for the training DataLoader when
  `num_workers > 0`.** Previously workers were torn down and respawned every epoch; a
  Ctrl-C landing at an epoch boundary could catch a worker mid-bootstrap (under `spawn`:
  importing torch / unpickling the payload — *before* echelon3's worker-init installs
  `SIGINT → SIG_IGN`), which dumped `KeyboardInterrupt` tracebacks and leaked semaphores
  from the half-started processes. Keeping workers alive across epochs removes that
  per-epoch window (and saves the respawn cost). Applied via `setdefault`, so an explicit
  `dataloaders.train.config.persistent_workers: false` is respected, and only when
  `num_workers > 0` (torch rejects the option otherwise). Eval loaders are intentionally
  left non-persistent — validation is not a tight per-epoch loop, and transient eval
  workers avoid holding node RAM for the whole run. The DDP launcher now prints a short
  informational line noting the default is on and how to opt out.

## 0.9.5 — 2026-07-18

### Fixed

- **DDP Ctrl-C no longer hangs ~30s and leaks semaphores when a rank is interrupted
  a second time mid-shutdown.** After 0.9.4 stopped the abort, a rank that had already
  entered its interrupt handler (printed `shutting down`) could still hang: the elastic
  agent re-sends SIGINT to every rank, and that second SIGINT — delivered while inside our
  `KeyboardInterrupt` handler — re-raised `KeyboardInterrupt` *past* the `os._exit(130)`,
  landing in the `finally` where `destroy_process_group()` deadlocks on NCCL teardown. The
  rank then hung until the agent force-`SIGKILL`ed it at the grace period (~30s), which
  hard-killed the DataLoader workers without releasing their semaphores (`leaked semaphore
  objects`). The interrupt handlers now set `SIGINT → SIG_IGN` as their very first action
  (helper `_silence_sigint`, before any print/flush), so the path down to `os._exit(130)`
  can no longer be diverted; workers are reaped in the bounded window and the rank exits
  promptly. Only the interrupt path is silenced — the genuine crash path keeps Ctrl-C live.

- **albumentations no longer stalls startup and spams `UserWarning`s trying to fetch
  version info.** albumentations runs a network version-check on import (`check_version.py`)
  that, offline, blocks on an SSL handshake timeout and warns every epoch. echelon3 now sets
  `NO_ALBUMENTATIONS_UPDATE=1` at package import (via `setdefault`, before albumentations is
  imported), disabling the check; set the env var yourself to override.

## 0.9.4 — 2026-07-17

### Fixed

- **DDP Ctrl-C no longer aborts with a scary traceback / `Fatal Python error: Aborted`
  when a DataLoader worker is killed by SIGINT first.** On Ctrl-C the whole process group
  gets SIGINT and a worker can die before echelon3's worker_init installs `SIG_IGN`
  (startup race); torch then raises, in the rank, either
  `DataLoader worker ... is killed by signal: Interrupt` or — more often on recent torch —
  `DataLoader worker ... exited unexpectedly`, which the crash path reported as a failure
  (traceback + `os._exit(1)`), and teardown could C++-abort. The rank now records that a
  SIGINT was actually seen (a tiny handler layered on the default `KeyboardInterrupt`
  behaviour) and treats a worker-death `RuntimeError` as a clean interrupt (exit 130) —
  but only after a real SIGINT, so genuine worker crashes (OOM SIGKILL, segfault) and
  non-worker errors still surface loudly.

## 0.9.3 — 2026-07-16

### Fixed

- **MultiPartDataset now works under DDP.** `create_dataloaders` no longer injects an
  int-index `DistributedSampler` for a `MultiPartDataset` (its index is a `(part, sample)`
  tuple — the mismatch crashed the worker with `'int' object is not subscriptable`).
  `MultiPartBatchSampler` is now DDP-aware and rank-shards the largest part, padding to an
  **equal per-rank count** so every rank yields the same number of batches (no gradient
  all-reduce hang). A `MultiPartDataset` paired with a plain `DataLoader` (train or test) now
  raises a **clear error** naming `MultiPartDataLoader` instead of the cryptic worker crash.
- `MultiPartBatchSampler.__len__` now uses the same per-part quota as `__iter__`
  (`quants[max_part]`, not `int(share*batch_size)`): the mismatch overstated `len()` when the
  largest part was configured last with fractional shares, which silently skipped
  end-of-epoch validation and checkpoint saving.
- `MultiPartDataLoader` default `prefetch_factor` `2 -> None` (the hardcoded `2` crashed
  `num_workers=0` on modern torch).

## 0.9.2 — 2026-07-16

### Fixed

- **DDP Ctrl-C no longer leaks DataLoader-worker semaphores.** Under DDP the rank's
  KeyboardInterrupt/Exception handler calls `os._exit()`, which bypasses the `finally:`
  that runs `trainer.close()` — so the DataLoader workers were hard-killed by PDEATHSIG
  without releasing their semaphores, and the launcher's `resource_tracker` warned about
  "leaked semaphore objects" (leaking /dev/shm). `trainer.close()` now runs before every
  `os._exit` in `train`/`finetune` (via `_close_quietly`, a best-effort call bounded by a
  watchdog timeout so it can never hang the hard exit — the `_shutdown_workers`
  pin-memory-thread join is otherwise untimed), reaping the (persistent) workers cleanly.

## 0.9.1 — 2026-07-15

### Added

- **`dataloaders.*.config.collate_fn` can be a component** (`module`/`type`/`config`): the
  engine builds it into a callable and passes it to the DataLoader, instead of leaving a dict
  that would break the loader. Enables variable-size / graph batching (sets, molecular
  complexes) on the SGD path; domain-agnostic, applied to both train and test dataloaders.
  (Used by the docking components in `echelon3_zoo[docking]`.)

## 0.9.0 — 2026-07-15

### Added

- **Tabular fit/predict (estimator) trainer family** — a second, independent trainer stack
  next to the image/SGD one, for models that are fit once rather than trained by gradient
  descent: gradient-boosted trees (CatBoost/XGBoost/LightGBM/sklearn) and tabular foundation
  models (TabPFN/TabICL/TabFM/TabGPT). Same `module/type/config` idiom, same `echelon3 train`:
  a config with a `model:` section (and no `net:`) routes to the estimator assembly.
  - `trainers/estimator.py`: `EstimatorTrainer` and `MultiTargetEstimatorTrainer` (one cloned
    model per target, fit only on rows where that target is measured — NaN-masked; bundle
    `{target: model}`). No optimizer/loss/dataloaders/scheduler — the objective/loss is a
    hyperparameter of the model itself (`model.config`).
  - `data/tabular.py`: `TabularDataset` (sources: csv/parquet/feather/json/tsv, SQL, or an
    in-memory frame; single- or multi-target) and `TabularPreprocessor` (declarative sklearn
    `ColumnTransformer` as a `feature_transform`, so swapping engines stays a change of only
    `model:` even on categorical/NaN data).
  - `metrics/tabular.py`: classification AUC/Gini/KS/LogLoss/Accuracy and regression
    MAE/RMSE/R2/SpearmanR/PearsonR.
  - `inference/tabular.py`: `load_bundle` + `predict` — the saved `.tar` is a self-contained
    inference artifact (model(s) + fitted feature pipeline + feature names + target).
- Molecular/ADMET components (SMILES featurizer, molecular-graph dataset, a 2D GNN) live in the
  public `echelon3_zoo` package under the `molecular` extra — the engine stays domain-agnostic
  and free of an rdkit dependency.

## 0.8.7 — 2026-07-13

### Fixed

- **Initial validation now runs BEFORE training, for both fresh and resumed runs.**
  Previously a from-scratch run went straight into `train_epoch()` and the first
  validation (the one printed as `Initial metrics baseline`) only happened partway
  through the first epoch — so the "baseline" was measured *after* some training. The
  initial `validate_and_check_for_saving()` is now called once before the epoch loop for
  both paths (scratch and checkpoint), so the baseline is the step-0 / loaded-checkpoint
  state, and training must beat it. (Side effects: a fresh run now saves an initial
  checkpoint at step 0; a resumed run prints `Initial metrics baseline` and re-saves it.)

## 0.8.6 — 2026-07-12

### Changed

- **Reworked the training / validation console output into clean per-cycle summary
  lines.** The live `tqdm` bars (`Training epoch N …`, `Evaluating [name] …`) are now
  transient (`leave=False`); when each closes it is replaced by one past-tense summary
  line instead of a leftover bar:
  - `--> Trained epoch N: 25% (256/1000), lr=3.00e-04, loss1=…, loss2=…` — how far the
    epoch had progressed at this validation point, the current LR, and the latest losses;
  - `--> Evaluated [name]: loss1=…, metric1=…` — one line per test loader.
  Numbers use an adaptive format (trailing zeros trimmed, scientific for very small /
  large values), so metric values no longer print as `tensor(0.0161, device='cuda:0')` —
  which used to overflow the terminal and leave garbled, half-overwritten lines. The
  keep-best "Saving checkpoint" lines round the same way.

## 0.8.5 — 2026-07-12

### Fixed

- **Validation output no longer leaves garbled "Evaluating [...]" lines.** Metric
  values were pushed into the tqdm postfix as raw `tensor(0.0161, device='cuda:0')`;
  the line overflowed the terminal width, so tqdm's `\r` could not clear it and each
  evaluated test loader left two or three half-overwritten lines before the next
  epoch. The per-loader eval bar is now `leave=False` (cleared on completion) and the
  result prints as one tidy line per test set with plain floats, e.g.
  `--> [test_geoloc3] l1=0.9636  mse=1.4694`.
- Warning summary (`warncollect`) no longer cuts a message mid-word — over-long
  messages are trimmed with an ellipsis.

## 0.8.4 — 2026-07-12

### Docs

- README (PyPI front page): added a "Use it from your AI coding agent" section
  pointing at the `echelon3-agent-skills` repo, with the one-line marketplace-install
  commands for Codex and Claude Code.

## 0.8.3 — 2026-07-12

### Docs

- **Rewrote the README** (the PyPI front page) — clear value prop, a complete copy-and-run
  config + command, a "what you get" list, and the CLI/overrides — and swept the docs,
  examples and CI onto the 0.8.0 interface: `echelon3 <cmd>` everywhere (not the
  `echelon3-<cmd>` aliases), OmegaConf overrides and `defaults:` composition, built-in
  DDP via `gpus=[...]`. Fixed stale DataParallel/`torchrun` wording, dropped the legacy
  `device_ids` from the example configs, removed committed editing artifacts from doc
  pages, added the `detection` extra to the install docs, and corrected the `ddp.py`
  module docstring. Verified by a documentation-review pass and `mkdocs build --strict`.

## 0.8.2 — 2026-07-12

### Packaging

- **`CHANGELOG.md` now ships inside the package** — `echelon3/CHANGELOG.md` in the wheel
  (so it lands in `site-packages`) and in the sdist — and a **`Changelog`** project URL
  is exposed in the metadata (shown by `pip show echelon3` and on PyPI). Previously the
  changelog lived only in the repo, so tools inspecting the installed package couldn't
  find it.

## 0.8.1 — 2026-07-12

### Fixed

- **`ZeroDivisionError` on small datasets** when `times_to_validate_per_epoch` exceeds
  the number of batches in an epoch (e.g. 2 batches with `=5`). The validation trigger
  computed `... % (total_batches // times_to_validate_per_epoch)`, which is `% 0` when
  `total_batches < times_to_validate_per_epoch`. Now
  `max(1, total_batches // times_to_validate_per_epoch)`. Hit GPU and CPU alike, on any
  tiny dataset.
- **`device: cpu` now forces CPU even on a multi-GPU host.** With no `gpus` set the DDP
  launcher used every visible GPU and ignored `device: cpu`; it now returns early when
  `device` is `cpu`, so a CPU run stays on CPU.

## 0.8.0 — 2026-07-11

### Changed

- **Dropped Hydra; the CLI is now a single `echelon3` command** with subcommands
  `train | finetune | evaluate | export | run` (the old `echelon3-<cmd>` scripts stay
  as transitional aliases). Config loading and CLI overrides run on **OmegaConf**
  directly. The override syntax is compatible — `key=value`, `+key`, `++key`, `~key`
  (delete), typed values, lists, `${oc.env:...}` — with no strict/struct footgun
  (`key=value` adds a new key; `+` is optional). `defaults:` composition is supported
  (base configs + config-groups + `_self_`), so existing configs — including composed
  ones — run unchanged; `hydra:` blocks and `hydra.*` overrides are ignored. This
  removes the recurring Hydra footguns (cwd/output-dir coupling, exception wrapping,
  struct-mode prefixes) and a heavy dependency (`hydra-core` → `click`).

## 0.7.12 — 2026-07-11

### Fixed

- **Single-GPU `device=cuda` runs crashed** (regression since 0.7.9). Non-DDP device
  selection returns `torch.device('cuda')` (no index) when no `gpus=` is given, and
  `torch.cuda.set_device()` rejected it ("Expected a torch.device with a specified
  index"). `set_device` is now called only for an explicit `cuda:{index}` (from
  `gpus=`); bare `cuda` already defaults to device 0.
- **Ctrl-C now stops cleanly when DataLoader workers are used.** SIGINT reaches the
  whole process group, so a worker could die first and the main process (waiting in
  `next(iterator)`) saw "DataLoader worker exited unexpectedly" (a `RuntimeError`)
  instead of `KeyboardInterrupt`, dumping a traceback — the real cause behind the
  original Ctrl-C complaint (a race: sometimes clean, sometimes not). Workers now
  ignore SIGINT; the main process handles the interrupt and reaps them via PDEATHSIG.

## 0.7.11 — 2026-07-10

### Fixed

- **Ctrl-C now stops cleanly instead of dumping a traceback.** `KeyboardInterrupt`
  is handled separately from real errors — a one-line "Interrupted by user (Ctrl-C)"
  message and exit code 130, no traceback. Under DDP the launcher also catches
  torchelastic's `SignalException` (SIGINT reaches the whole process group) and exits
  cleanly; workers hard-exit (130) so peers and DataLoader workers are still reaped.
- **Stray training progress bar after validation.** With
  `times_to_validate_per_epoch=1` (validate at the end of each epoch) the training
  bar was re-created after the "Evaluating" bar even though the epoch had already
  finished, printing a phantom "Training epoch N" line between the validation and the
  next epoch. The bar is now re-created only when batches remain in the epoch.

## 0.7.10 — 2026-07-10

### Changed

- **DDP launcher messages now print after the banner and in the same colour.** The
  `--> DDP: launching …` / dataloader-RAM / warning lines are emitted by the parent
  before workers start; they used to appear *before* the product banner and
  uncoloured, because the banner and `Fore.CYAN` were set only in the worker. The
  banner and colour are now set in the parent (`trainer_app` / `finetune_app`) before
  the DDP launch, so the banner comes first and the launcher lines inherit the cyan
  style; workers no longer reprint the banner.

## 0.7.9 — 2026-07-10

### Fixed

- **Single-GPU runs now honour the `gpus` index instead of always landing on GPU 0.**
  Non-DDP device selection read `cfg.device` (`cuda` → `cuda:0`) and ignored `gpus`,
  so `echelon3-train … gpus=[1]` silently ran on physical GPU 0 — colliding with
  other jobs and breaking "don't touch GPU 0" reservations on shared hosts (DDP
  already honoured the index). The CLI now pins the process to `cuda:{gpus[0]}` (plus
  `torch.cuda.set_device`) via a `resolve_single_device` helper; `device: cpu`
  overrides still win and configs without `gpus` are unchanged. Done via an explicit
  device index rather than `CUDA_VISIBLE_DEVICES`, which is a no-op in-process once
  the CUDA runtime has initialised (`torch.cuda.is_available()` alone locks it).

## 0.7.8 — 2026-07-09

### Fixed

- **`MultiHeadBinaryIoU` now aggregates across ranks under DDP**, fixing noisy
  best-checkpoint selection for every project that uses it (validation is sharded
  per rank, and the metric previously computed on rank 0's shard only — inflating
  the tracked value). It implements the `dist_reduce()` hook (from 0.7.7): a SUM
  all-reduce of each head's raw `tp/fp/fn/n` counters before `compute()`. This is
  exact because the counters are additive over samples; `n` is reduced too, so the
  set of seen heads — and thus the macro-mean denominator — is consistent across
  ranks. No-op outside DDP; the counters are non-persistent buffers, so checkpoints
  are unchanged.

## 0.7.7 — 2026-07-08

### Added

- **`Metric.dist_reduce()` hook — exact custom-metric aggregation under DDP.**
  Validation is sharded per rank; torchmetrics reduce their state inside `compute()`,
  but custom `echelon3.metrics.base.Metric` subclasses used to compute on rank 0's
  shard only, making keep-best selection noisy. The base `Metric` now has a
  `dist_reduce()` method (default no-op) that the trainer calls on every rank right
  before `compute()` under DDP. A counter-based metric (e.g. IoU's
  intersection/union accumulators) implements it as a SUM all-reduce of its buffers —
  helper `all_reduce_sum_(*tensors)` provided — which is exact, because summation
  commutes with sharding (unlike averaging per-shard ratios). Single-GPU runs and
  torchmetrics are unaffected.

## 0.7.6 — 2026-07-08

### Changed

- **The per-component `config:` block is now optional everywhere, uniformly.** One
  rule across every `create_*` (net, backbone, dataset, loss, metric, optimizer,
  scheduler, dataloader, trainer, evaluator, wrapper, constructor, batch sampler,
  exporter): a component whose constructor needs no arguments may omit its `config:`
  block. Removes the previous "optional in some builders, required in others"
  inconsistency (present-`config` behaviour is unchanged).

### Fixed

- **`Metric.to()` no longer strands buffers on CPU.** The base
  `echelon3.metrics.base.Metric.to()` was a no-op that shadowed `nn.Module.to` via
  the MRO for metrics that are *also* `nn.Module`s with buffers/parameters — their
  buffers stayed on CPU and validation on CUDA raised a device-mismatch error. It
  now delegates to `nn.Module.to()` when the metric is a module; pure non-module
  metrics stay a no-op as before.

### Docs

- DDP guide: corrected the validation section — only torchmetrics reduce their state
  across ranks; custom `Metric` subclasses compute per-shard, so keep-best is driven
  by rank 0's shard (use a torchmetrics metric, or add a distributed reduce, for
  exact behaviour).

## 0.7.5 — 2026-07-08

### Fixed

- **Regression from 0.7.2: `DataLoader(multiprocessing_context="spawn")` crashed
  while pickling `worker_init_fn`.** The default PDEATHSIG `worker_init_fn` (added
  in 0.7.2 for worker reaping) was a closure, and spawn must pickle
  `worker_init_fn` to hand it to the fresh worker process — closures are not
  picklable. Only the fork path (the default) had been tested, so this shipped
  unnoticed. It is now a module-level function wrapped in `functools.partial`,
  which pickles cleanly; `spawn` works again with worker reaping intact.

## 0.7.4 — 2026-07-08

### Changed

- Revert the per-component `config:`-block optionality added in 0.7.3: `net`,
  datasets, and schedulers require their `config:` block again (as before 0.7.3).
  Making `config` optional only in some builders was inconsistent; making it
  optional everywhere would spread that leniency, and requiring it literally
  everywhere is impossible (no-arg leaf transforms such as `To01` legitimately omit
  `config`). Section-level omission (`transform` / `metrics` / `scheduler` /
  `keep_best_on`) is unchanged.

## 0.7.3 — 2026-07-08

### Changed

- **CLI output is now uniformly English and single-colour.** A handful of runtime
  messages (and raised errors) were Russian and/or highlighted in yellow, which
  looked out of place in the public package; they now read in English in the
  ambient colour like the rest of the `-->` log.
- **Warnings no longer corrupt progress bars.** Library warnings are collected
  instead of printed inline; a short summary (`--> N warning(s) since last report`
  with per-message counts) is emitted before each validation and once at the end.
  Deprecation/Future noise is dropped outright. The c10d `barrier(): using the
  device under current context` warning is silenced at the source by passing
  `device_id` to `init_process_group`.

### Fixed

- **Config parameters with sensible defaults are now optional — omitting them no
  longer crashes.** `transform` (augment/preprocess), `metrics`, and `scheduler`
  may be omitted entirely (→ ToTensorV2-only, no metrics, constant LR);
  `keep_best_on` is optional (→ save every validation); and the per-object
  `config:` block is optional for `net`, datasets, losses, and schedulers
  (constructed with no extra args). Loss `weight` defaults to `1.0`. A new pytest
  suite (`tests/`, `pip install -e .[test]`) covers these paths.

## 0.7.2 — 2026-07-08

### Fixed

- **No more orphaned DataLoader / DDP processes after a crash, `kill`, or failed
  restart.** DataLoader workers and the DDP rank processes now set
  `PR_SET_PDEATHSIG` (Linux) via a default `worker_init_fn` / at process-group
  init, so they are SIGKILL'd the instant their parent dies — for any reason,
  including the `os._exit` fast-path from 0.7.1 and an external `kill -9`.
  Previously a rank that died uncleanly left its workers running, holding
  `/dev/shm` and host RAM; a fresh run's workers then couldn't get shared memory
  and hung at the first batch (GPUs idle). `echelon3-train` / `-finetune` now also
  catch `KeyboardInterrupt` (Ctrl-C goes through the teardown path) and call a new
  `Trainer.close()` on exit to shut DataLoader workers down promptly — relevant
  with `persistent_workers: true`. The launcher additionally warns on CPU
  over-subscription (`ranks × num_workers > cores`) and on `persistent_workers`
  under DDP.

## 0.7.1 — 2026-07-08

### Fixed

- **DDP no longer hangs silently when a rank dies (typically an OOM).** Under the
  built-in launcher, a rank that OOM'd could wedge in `destroy_process_group()`
  (NCCL teardown blocks on an in-flight collective) and never exit, so
  `elastic_launch` never saw a failure and the peers blocked on the next
  collective — a silent hang up to the process-group timeout. On the error path a
  DDP rank now prints its traceback to stderr and hard-exits (`os._exit`) instead
  of relying on a clean shutdown that can block, so the launcher tears the group
  down immediately. `max_restarts=0` (fail-fast instead of silently retrying an
  OOM into a desynced group); `ChildFailedError` is surfaced with an OOM hint;
  `TORCH_NCCL_ASYNC_ERROR_HANDLING` / `TORCH_NCCL_DESYNC_DEBUG` are on so the NCCL
  watchdog aborts (and reports the stuck rank) rather than waiting; the
  process-group timeout is configurable via `ECHELON3_DDP_TIMEOUT_MIN`.
- The launcher prints the per-node DataLoader prefetch total
  (`ranks × num_workers × prefetch_factor`) and warns when it is large — these are
  **per-rank** and multiply under DDP, the common cause of the OOM above.

### Changed

- **`torch.compile` is no longer marked experimental** — validated on single-GPU
  and 4×H200 DDP including production image-in-image runs. The `compile` /
  `compile_mode` knobs are unchanged.

## 0.7.0 — 2026-07-08

### Added

- **`torch.compile` support (experimental, opt-in).** `trainer.config.compile: true`
  (with optional `compile_mode`) compiles the network — kernel fusion to cut launch
  overhead, the lever for small nets that under-use a big GPU where bf16 does
  nothing (launch-bound, not compute-bound). Compiled before the DDP wrapper;
  `ddp.unwrap()` and checkpoint save/load now also strip torch.compile's
  `_orig_mod.` prefix, so checkpoints stay interchangeable with uncompiled runs.
  Off by default; verified single-GPU **and on 4×H200 DDP** (trains + checkpoints
  round-trip). The actual speedup and any shape-driven recompiles are
  model-dependent (see `guide/ddp.md`).

### Changed

- **`PairTrainer` calls the network positionally** — `net(base, query, True)`
  instead of `net(base, query, return_features=True)`. Pair nets name the third
  argument differently (`return_features`, `return_intermediates`, …); passing it
  by position keeps the trainer agnostic to the name.

## 0.6.0 — 2026-07-08

### Added

- **`Trainer.compute_losses(source, labels, net=None)` — an extension seam for
  the forward pass and loss routing.** It runs inside the trainer's autocast and
  returns `(predictions, {name: (loss, weight)})`; both training and validation
  call it, so a subclass customizes *what the network is fed and how losses map
  to its outputs* without ever re-implementing the precision / scaler / closure /
  DDP machinery, which stays in the base. Behaviour is unchanged — the default
  `compute_losses` is the previous single-input forward.
- **`echelon3.trainers.pair.PairTrainer`** — a trainer for two-image
  ("pair" / image-in-image) inputs. Consumes `((base, query), gt)` batches
  (via `pair_collate_fn`), calls `net(base, query, return_features)` and delegates
  loss routing to `pair_losses(heatmap, features, labels)` (default: every loss on
  the heatmap). A domain-specific image-in-image trainer becomes a natural
  subclass overriding only `pair_losses` — see `guide/extending.md`.

## 0.5.2 — 2026-07-08

### Fixed

- **`GrayscaleCLAHE` is now picklable**, so it survives DataLoader workers
  started with the `spawn` start method (and spawn-only platforms). It stored a
  bare `cv2.CLAHE` C++ handle, which cannot be pickled — any run whose data
  workers pickle the dataset died with `TypeError: cannot pickle 'cv2.CLAHE'
  object`. The handle is now dropped on pickle and rebuilt from the stored
  `clip` / `grid` params on unpickle. Under the default `fork` start method the
  transform is inherited and never pickled, so single-GPU runs and the example
  smokes never triggered it — reproduced and fixed under a `spawn` DataLoader on
  4×H200.

## 0.5.1 — 2026-07-08

### Fixed

- **README (the PyPI project description) rewritten for 0.5.0.** It still
  documented the removed `DataParallel` fallback and presented `torchrun` as the
  only multi-GPU path. Now it shows the built-in `gpus=[...]` launcher (no
  `torchrun` needed), the bf16-by-default mixed precision, the full CLI set, and
  all three example smokes. Docs-only release — no code changes.

## 0.5.0 — 2026-07-08

Multi-GPU and performance release. **Breaking**: DataParallel is gone and mixed
precision defaults to bf16.

### Changed

- **Multi-GPU is built-in DDP, launched from the CLI — no `torchrun` needed.**
  Pass `gpus=[0,1,2,3]` (a root config key; default = all visible GPUs on the
  node) and echelon3 spawns one DDP worker per GPU via PyTorch's `elastic_launch`,
  wiring up `RANK`/`LOCAL_RANK`/`WORLD_SIZE`/`MASTER_*` itself. `torchrun` and
  multi-node runs still work unchanged through the environment-variable path.
  Applies to `echelon3-train` and `echelon3-finetune`.
- **Mixed precision (AMP) on by default.** Training, `evaluate` and `run` autocast
  in **bf16** on capable GPUs (fp32 on CPU / unsupported GPUs). Set
  `trainer.config.precision: fp32` to restore full fp32. `precision: fp16` uses a
  `GradScaler`; with closure optimizers (SAM/LBFGS, which double-backward) it
  falls back to bf16. TF32 matmul and `cudnn.benchmark` are on by default
  (`trainer.config.tf32`, `trainer.config.cudnn_benchmark`).
- Checkpoints save the **unwrapped** `state_dict` (no `module.` prefix); older
  `module.`-prefixed checkpoints still load (prefix stripped automatically).

### Removed

- **DataParallel.** Multiple GPUs always run as DDP; a single process drives one
  GPU. `device_ids` no longer selects multiple GPUs — use `gpus`. The
  `run` / `evaluate` / `export` CLIs load checkpoints directly instead of wrapping
  the network in `DataParallel`.

## 0.4.1 — 2026-07-07

### Fixed

- `IoU.compute` (segmentation metric) returned `nan` whenever `ignore_index` was
  unset (the default): `use_idx[None] = False` added an axis and wiped the whole
  keep-mask, so `nanmean` ran over an empty array. Now the mask is only touched
  when there are classes to ignore. Verified end to end — a `Segmenter` smoke on
  synthetic masks moves mIoU from ~0.85 to ~0.96.

### Added

- `examples/segmentation/`: a self-contained semantic-segmentation smoke
  (synthetic image/mask generator, a tiny dependency-free backbone, a `Segmenter`
  + cross-entropy + `IoU` config) mirroring `examples/detector/`.

## 0.4.0 — 2026-07-07

### Fixed

- Detection pipeline now trains end to end (CenterNet-style `HeatmapDetector`,
  YOLO-format `DetectionDataset`). Several defects along the path were fixed:
  - `Trainer.set_to_device` now stacks the per-sample image tensors that
    `VariableDataLoader` produces into one `(N, C, H, W)` batch, and leaves the
    variable-length box lists as Python lists instead of calling `.to()` on them
    (image-in-image `(base, query)` pairs are unaffected).
  - The train/validate steps no longer assume `labels` is a tensor (they read
    `labels.shape` only when it is), so list-valued detection targets work.
  - `metrics.base.Metric` gained a no-op `.to()` so custom metrics (mAP, EER,
    AUC, IoU, …) interoperate with the trainer's uniform `metric.to(device)`.
  - `HeatmapBasedDetectionLoss` uses the penalty-reduced CornerNet/CenterNet
    focal loss on the sigmoid heatmaps the head emits, instead of
    `sigmoid_focal_loss` (which double-applied a sigmoid and crushed the
    gradient, so heatmap peaks never formed).
  - `DecodeHeatmaps.decode` keeps YOLO/Albumentations boxes as normalized floats
    (they were cast to `LongTensor`, truncating every coordinate to 0).
  - Class labels are coerced to integer tensors in both the encoder and the mAP
    metric (Albumentations round-trips them through float arrays).

### Added

- `detection` extra (`faster-coco-eval`) — the `mAP` metric needs a COCO backend;
  `mAP` takes an optional `backend` argument (default `faster_coco_eval`).
- `examples/detector/`: a self-contained CenterNet-like detector smoke — synthetic
  YOLO dataset generator, a tiny dependency-free heatmap backbone, and a config
  that trains, validates (mAP rises above zero) and checkpoints on CPU or GPU.

## 0.3.1 — 2026-07-07

### Added

- Full MkDocs (Material) documentation site published to GitHub Pages at
  <https://veryviolet.github.io/echelon3/>: getting-started, concepts
  (how-it-works, run anatomy), guides (extending, DDP, ONNX export) and reference
  (config schema, built-in components, CLI). `docs` extra + a `docs` workflow that
  `mkdocs gh-deploy`s on every `v*` tag.
- `[project.urls]` with a `Documentation` link, so PyPI shows it.

## 0.3.0 — 2026-07-05

### Added

- `echelon3-finetune` CLI: warm-start from a checkpoint (`init_from`), freezing
  by regex patterns (`finetune.freeze_patterns`), head-only training and
  per-layer parameter groups (`finetune.param_groups`). With none of those
  blocks present it behaves exactly like `echelon3-train`.

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
