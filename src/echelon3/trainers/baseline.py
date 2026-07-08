from typing import Dict
from collections.abc import Mapping

from tqdm import tqdm
import torch
import re
import numpy as np
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from colorama import Fore

from echelon3.checkpoint.manager import (
    CheckpointManager,
    CHECKPOINT_EPOCH_KEYWORD,
    CHECKPOINT_MODEL_KEYWORD,
    CHECKPOINT_OPTIMIZER_KEYWORD,
    CHECKPOINT_SCHEDULER_KEYWORD,
    CHECKPOINT_METRICS_KEYWORD,
    CHECKPOINT_SCALER_KEYWORD,
)

from echelon3.mlops.basic import MLOpsLogger

from echelon3 import ddp
from echelon3 import runtime


class _NullLogger:
    """Заглушка mlops-логгера для не-главных DDP-ранков."""

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class Trainer:
    _train_loader: DataLoader = None
    _test_loader: DataLoader = None
    _test_loaders: Dict[str, DataLoader] = None  # несколько test‑датасетов
    _primary_test_name: str | None = None        # основной тест‑датасет (по умолчанию "test")

    _net: torch.nn.Module = None
    _losses: Dict[str, torch.nn.Module] = None
    _optimizer: Optimizer = None
    _scheduler = None
    _ckpt_manager: CheckpointManager = None

    _epochs = None
    _keep_best_on = None

    _times_to_validate_per_epoch = None

    _current_epoch = None

    _device = None

    _metrics = None

    _writer = None

    _tensorboard = None
    _logger: MLOpsLogger = None

    _global_step = None

    _best_metric_to_track = None
    _metric_to_track = None

    _float_labels = None

    _reset = None

    _high_is_better = None

    _qconfig = None

    _metrics_on: Dict[str, str] | None = None  # metric_name -> dataset_name

    losses_without_weights = None

    def __init__(
        self,
        epochs: int,
        keep_best_on,
        train_dataloader: DataLoader,
        test_dataloader,
        net: torch.nn.Module,
        losses: Dict[str, torch.nn.Module],
        metrics: Dict[str, torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler,
        ckpt_manager,
        mlops_logger,
        times_to_validate_per_epoch=1,
        float_labels: bool = False,
        reset: bool = False,
        high_is_better: bool = True,
        device: torch.device = torch.device("cuda"),
        device_ids: list = None,
        **kwargs,
    ):
        """
        keep_best_on может быть:
          * строкой / списком имён (старый режим)
          * словарём:
              keep_best_on:
                accuracy:
                  mode: directional
                  value: high      # high / low
                accuracy_control:
                  mode: tolerance
                  value: 0.1%      # или 0.001
                  direction: high  # low / high

        metrics_on:
          accuracy: incidents
          accuracy_control: valA
        """
        self._train_loader = train_dataloader

        # --- поддержка нескольких test‑датасетов при обратной совместимости ---
        if isinstance(test_dataloader, dict):
            self._test_loaders = dict(test_dataloader)
            self._primary_test_name = next(iter(self._test_loaders.keys()), None)
            self._test_loader = (
                self._test_loaders[self._primary_test_name]
                if self._primary_test_name is not None
                else None
            )
        else:
            self._test_loader = test_dataloader
            self._test_loaders = {"test": test_dataloader} if test_dataloader is not None else {}
            self._primary_test_name = "test" if test_dataloader is not None else None

        self._device = device
        net = net.to(device)

        # torch.compile (opt-in, EXPERIMENTAL): fuses kernels to cut launch
        # overhead — the lever for small nets that under-use big GPUs (bf16 does
        # nothing for launch-bound work). Compiled BEFORE the DDP wrapper;
        # ddp.unwrap() / checkpoints strip the resulting _orig_mod. Off by default.
        self._compiled = bool(kwargs.get("compile", False))
        if self._compiled:
            compile_mode = kwargs.get("compile_mode", None)
            net = torch.compile(net, mode=compile_mode) if compile_mode else torch.compile(net)
            if ddp.is_main():
                print(f"--> torch.compile: on (mode={compile_mode or 'default'}) — "
                      "EXPERIMENTAL; first steps recompile")

        if ddp.is_ddp():
            # DDP: один процесс = один GPU (сеть уже на устройстве).
            # find_unused_parameters=True по умолчанию: у некоторых сетей часть
            # выходов может не участвовать в лоссе на отдельных шагах.
            self._net = torch.nn.parallel.DistributedDataParallel(
                net,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=bool(kwargs.get("ddp_find_unused_parameters", True)),
            )
        else:
            # Один GPU / CPU — без обёртки. DataParallel убран; несколько GPU
            # запускаются встроенным DDP-лаунчером (см. cli.maybe_launch_ddp),
            # т.е. в каждом процессе сеть одноустройственная.
            self._net = net
        self._eval_net = None  # развёрнутая сеть для rank0-валидации в DDP
        self._losses = losses
        self._metrics = metrics
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._epochs = epochs
        self._keep_best_on = keep_best_on
        self._high_is_better = high_is_better
        self._times_to_validate_per_epoch = times_to_validate_per_epoch
        self._ckpt_manager = ckpt_manager
        # Логирует только rank 0 (в DDP не-главные ранки получают заглушку).
        self._logger = mlops_logger if (mlops_logger is not None and ddp.is_main()) \
            else _NullLogger()
        self._metrics = {name: m.to(self._device) for name, m in self._metrics.items()}
        self._float_labels = float_labels
        self._reset = reset

        # --- mixed precision (AMP) --- привязано к фактическому устройству.
        self._amp_dtype = runtime.resolve_amp_dtype(kwargs.get("precision", "auto"), device=self._device)
        # fp16 требует GradScaler, а он не сочетается с closure-оптимизаторами
        # (SAM/LBFGS сами делают двойной backward + ручную арифметику градиентов);
        # откатываемся на bf16.
        if self._amp_dtype == torch.float16 and self._optimizer_uses_closure():
            bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            self._amp_dtype = torch.bfloat16 if bf16_ok else None
            if ddp.is_main():
                print(f"--> AMP: fp16 + {type(self._optimizer).__name__} (closure) не поддержан; "
                      f"использую {runtime.precision_label(self._amp_dtype)}")
        self._amp_enabled = self._amp_dtype is not None
        # dtype для autocast; когда AMP выключен — значение игнорируется (enabled=False).
        self._autocast_dtype = self._amp_dtype if self._amp_enabled else torch.bfloat16
        self._scaler = torch.amp.GradScaler("cuda", enabled=(self._amp_dtype == torch.float16))
        if ddp.is_main():
            print(f"--> Precision: {runtime.precision_label(self._amp_dtype)} "
                  f"(autocast {'on' if self._amp_enabled else 'off'})")
            if self._amp_enabled:
                print(Fore.YELLOW + "--> AMP включён: численность отличается от fp32-прогонов "
                      "(precision: fp32 чтобы выключить)" + Fore.CYAN)

        # metrics_on может прийти как DictConfig/Mapping
        raw_metrics_on = kwargs.get("metrics_on", None)
        if raw_metrics_on is None:
            self._metrics_on = None
        elif isinstance(raw_metrics_on, Mapping):
            self._metrics_on = {str(k): str(v) for k, v in raw_metrics_on.items()}
        else:
            self._metrics_on = raw_metrics_on

        # нормализованный конфиг keep_best_on
        self._keep_best_config = self._build_keep_best_config(keep_best_on)

        # текущее и лучшее значения ВСЕХ отслеживаемых метрик
        self._current_metrics_all: Dict[str, float] | None = None
        self._best_metrics_all: Dict[str, float] | None = None

    def train(self):

        print(f"--> Initializing target...", end="")
        self._ckpt_manager.init_storage()

        ckpts_idxs = self._ckpt_manager.idxs
        if len(ckpts_idxs) == 0:
            print(" Starting from scratch.")
            self._current_epoch = 1
            self._global_step = 1
            self.initialize_network()
        else:
            print(f" Found checkpoints {ckpts_idxs}")
            print("--> Loading latest checkpoint... ", end="")
            num_loaded = self.load_from_checkpoint()
            print(f" Loaded checkpoint {num_loaded}")
            self.recalculate_start_of_epoch_global_step()
            self.initialize_network()
            # Валидация загруженного чекпойнта
            self.validate()

            if (
                self._keep_best_config is not None
                and len(self._keep_best_config) > 0
                and self._current_metrics_all is not None
                and self._best_metrics_all is None
            ):
                self._best_metrics_all = dict(self._current_metrics_all)
                self._best_metric_to_track = self._metric_to_track

        while self._current_epoch <= self._epochs:
            self.train_epoch()
            self._current_epoch += 1

    def save_checkpoint(self):
        if not ddp.is_main():
            return
        self._ckpt_manager.save_checkpoint(
            {
                CHECKPOINT_EPOCH_KEYWORD: self._current_epoch,
                CHECKPOINT_MODEL_KEYWORD: ddp.state_dict_for_save(self._net),
                CHECKPOINT_OPTIMIZER_KEYWORD: self._optimizer.state_dict(),
                CHECKPOINT_SCHEDULER_KEYWORD: self._scheduler.state_dict(),
                CHECKPOINT_METRICS_KEYWORD: self._metrics,
                CHECKPOINT_SCALER_KEYWORD: self._scaler.state_dict(),
            }
        )

    def recalculate_start_of_epoch_global_step(self):
        try:
            batches_per_epoch = len(self._train_loader)
        except TypeError:
            total_size = len(self._train_loader.dataset)
            batches_per_epoch = (
                (total_size + self._train_loader.batch_size - 1)
                // self._train_loader.batch_size
            ) if self._train_loader.batch_size is not None else \
                self._train_loader.total_batches()
        self._global_step = batches_per_epoch * (self._current_epoch - 1)

    def _optimizer_uses_closure(self) -> bool:
        name = self._optimizer.__class__.__name__
        return name in ("SAMOptimizer", "LBFGS")

    def compute_losses(self, source, labels, net=None):
        """Forward pass + per-loss values — the trainer's extension seam.

        Override this to customize how inputs are fed to the network or how losses
        map to its outputs (paired inputs, multi-output heads, name-routed losses,
        …). It runs INSIDE the trainer's autocast context; the base keeps ownership
        of scaler / backward / closure / optimizer-step and logging, so subclasses
        never re-implement the precision path. Returns
        ``(predictions, {name: (loss_tensor, weight)})``. ``net`` lets validation
        pass its (unwrapped) eval network; it defaults to the training network.
        """
        net = net if net is not None else self._net
        predictions = net(source)

        if isinstance(predictions, torch.Tensor) and isinstance(labels, torch.Tensor) \
                and len(predictions.shape) > len(labels.shape):
            while predictions.dim() > labels.dim() and predictions.size(-1) == 1:
                predictions = predictions.squeeze(-1)

        losses_values = {
            name: (
                loss[0](predictions, labels.float() if self._float_labels else labels),
                loss[1],
            )
            for name, loss in self._losses.items()
        }
        return predictions, losses_values

    def one_step_train(self, source, labels):
        def closure(**kwargs):
            self._optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=self._autocast_dtype, enabled=self._amp_enabled):
                predictions, losses_values = self.compute_losses(source, labels)
                total_loss = torch.sum(
                    torch.stack([ls[0] * ls[1] for ls in losses_values.values()])
                )
            # backward — вне autocast; при fp16 масштабируем градиенты скейлером.
            if self._scaler.is_enabled():
                self._scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            self.losses_without_weights = {m: v[0] for m, v in losses_values.items()}
            self._logger.log_train_data(self._global_step, source, labels, predictions)
            self._logger.log_train_losses(self._global_step, self.losses_without_weights)
            return total_loss

        if self._optimizer_uses_closure():
            # SAM/LBFGS: scaler всегда выключен (fp16 откатывается на bf16), поэтому
            # двойной forward/backward внутри closure идёт как раньше.
            self._optimizer.step(closure)
        else:
            _ = closure()
            if self._scaler.is_enabled():
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                self._optimizer.step()

    def initialize_network(self):
        self._net.to(self._device)

    def prepare_network_for_train(self):
        self._net.train(True)

    def prepare_network_for_validation(self):
        self._net.train(False)

    def train_epoch(self):
        # len(DataLoader) корректен и для DDP (шард DistributedSampler), и без него
        # (эквивалентен ceil(dataset/batch)); фолбэк — для кастомных лоадеров.
        try:
            total_batches = len(self._train_loader)
        except TypeError:
            total_size = len(self._train_loader.dataset)
            total_batches = int(
                np.ceil(1.0 * total_size / self._train_loader.batch_size)
            ) if self._train_loader.batch_size is not None else \
                self._train_loader.total_batches()

        sampler = getattr(self._train_loader, "sampler", None)
        if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
            sampler.set_epoch(self._current_epoch)

        train_progress = tqdm(
            initial=0,
            total=total_batches,
            desc=f"--> Training epoch {self._current_epoch}",
            ncols=0,
            dynamic_ncols=True,
            disable=not ddp.is_main(),
        )

        self.prepare_network_for_train()
        self.recalculate_start_of_epoch_global_step()

        for batch, (source, labels) in enumerate(self._train_loader):

            source, labels = self.set_to_device(source, labels)

            self.one_step_train(source, labels)

            self._global_step += 1

            if (
                (batch + 1 - (total_batches % self._times_to_validate_per_epoch)) != 0
                and (batch + 1 - (total_batches % self._times_to_validate_per_epoch))
                % (total_batches // self._times_to_validate_per_epoch)
                == 0
            ):

                train_progress.close()

                self.prepare_network_for_validation()
                self.validate_and_check_for_saving()
                self.prepare_network_for_train()

                train_progress = tqdm(
                    initial=0,
                    total=total_batches,
                    desc=f"--> Training epoch {self._current_epoch}",
                    ncols=0,
                    dynamic_ncols=True,
                    disable=not ddp.is_main(),
                )
                train_progress.n = batch

            if self.losses_without_weights is not None:
                train_progress.set_postfix(
                    {
                        l: f"{float(v.detach() if isinstance(v, torch.Tensor) else v):.4f}"
                        for l, v in self.losses_without_weights.items()
                    }
                )
            train_progress.update(1)

        train_progress.close()
        self._scheduler.step()

    # =========================
    #   ЛОГИКА KEEP_BEST_ON
    # =========================

    def _build_keep_best_config(self, keep_best_on):
        if keep_best_on is None:
            return None

        # словарь / Mapping (включая DictConfig)
        if isinstance(keep_best_on, Mapping):
            cfg = {}
            for name, v in keep_best_on.items():
                if isinstance(v, Mapping):
                    mode = str(v.get("mode", "directional")).lower()
                    value = v.get("value", None)
                    direction = v.get("direction", None)
                    direction = str(direction).lower() if isinstance(direction, str) else None
                else:
                    mode = "directional"
                    value = v
                    direction = None
                cfg[str(name)] = {"mode": mode, "value": value, "direction": direction}
            return cfg

        # строка / список имён: directional high/low по high_is_better
        names = self._normalize_keep_best_names(keep_best_on)
        direction = "high" if self._high_is_better else "low"
        return {
            n: {"mode": "directional", "value": direction, "direction": direction}
            for n in names
        }

    def _normalize_keep_best_names(self, value):
        if value is None:
            return []
        if isinstance(value, str):
            parts = re.split(r"[,\s;]+", value.strip())
            return [p for p in parts if p]
        try:
            return list(value)
        except TypeError:
            return [str(value)]

    def _parse_tolerance(self, raw):
        if isinstance(raw, str) and raw.endswith("%"):
            num = float(raw[:-1])
            return num / 100.0
        return float(raw)

    def _metric_condition(self, name: str, current: float, best: float) -> bool:
        cfg = self._keep_best_config.get(name)
        if cfg is None:
            return False

        mode = str(cfg.get("mode", "directional")).lower()
        val = cfg.get("value", None)
        direction = cfg.get("direction", None)

        if best is None:
            return True

        # ---------- tolerance ----------
        if mode == "tolerance":
            if val is None:
                raise ValueError(
                    f"keep_best_on[{name}]: mode='tolerance', но не задан value"
                )
            tol = self._parse_tolerance(val)
            denom = max(abs(best), 1e-12)

            if direction is None:
                direction = "high" if self._high_is_better else "low"
            direction = str(direction).lower()

            if direction == "low":
                if current <= best:
                    return True
                rel_inc = (current - best) / denom
                return rel_inc <= tol
            else:  # direction == "high"
                if current >= best:
                    return True
                rel_dec = (best - current) / denom
                return rel_dec <= tol

        # ---------- directional ----------
        if isinstance(val, str):
            direction = val.lower()
        if direction not in ("high", "low"):
            direction = "high" if self._high_is_better else "low"

        if direction == "high":
            return current > best
        else:
            return current < best

    def _all_metrics_improved(self) -> bool:
        if self._keep_best_config is None or not self._keep_best_config:
            return True

        if self._current_metrics_all is None:
            return False

        if self._best_metrics_all is None:
            return True

        for name in self._keep_best_config.keys():
            if name not in self._current_metrics_all:
                return False
            cur = self._current_metrics_all[name]
            best = self._best_metrics_all.get(name)
            if not self._metric_condition(name, cur, best):
                return False

        return True

    def validate_and_check_for_saving(self):
        self.validate()

        # После dist-синка метрик значения идентичны на всех ранках — keep-best
        # логика исполняется везде одинаково, сохранение гейтится в save_checkpoint.

        # если keep_best_on не задан — сохраняем каждый раз
        if self._keep_best_config is None or not self._keep_best_config:
            self.save_checkpoint()
            return

        # первая итерация: считаем текущие значения лучшими и сохраняем
        if self._best_metrics_all is None:
            self._best_metrics_all = (
                dict(self._current_metrics_all)
                if self._current_metrics_all is not None
                else None
            )
            self._best_metric_to_track = self._metric_to_track
            print(Fore.LIGHTGREEN_EX, end="")
            if self._current_metrics_all:
                metrics_str = ", ".join(
                    f"{k}={v}" for k, v in self._current_metrics_all.items()
                )
                print(
                    f"--> Initial metrics baseline for [{metrics_str}]. Saving checkpoint."
                )
            else:
                print(f"--> Initial metrics baseline. Saving checkpoint.")
            print(Fore.CYAN, end="")
            self.save_checkpoint()
            return

        # проверяем одновременное "улучшение" по всем условиям
        if self._all_metrics_improved():
            self._best_metrics_all = (
                dict(self._current_metrics_all)
                if self._current_metrics_all is not None
                else None
            )
            self._best_metric_to_track = self._metric_to_track

            print(Fore.LIGHTGREEN_EX, end="")
            if self._current_metrics_all:
                metrics_str = ", ".join(
                    f"{k}={v}" for k, v in self._current_metrics_all.items()
                )
                print(
                    f"--> Obtained better values for [{metrics_str}]. Saving checkpoint."
                )
            else:
                print(f"--> Obtained better values. Saving checkpoint.")
            print(Fore.CYAN, end="")
            self.save_checkpoint()

    def one_step_validate(self, source, labels):
        net = self._eval_net if self._eval_net is not None else self._net
        with torch.autocast("cuda", dtype=self._autocast_dtype, enabled=self._amp_enabled):
            predictions, losses_values = self.compute_losses(source, labels, net=net)
        # Метрики считаем на fp32: после autocast выходы могут быть bf16/fp16.
        predictions = runtime.to_float32(predictions)
        self._logger.log_test_data(self._global_step, source, labels, predictions)
        return predictions, losses_values

    def set_to_device(self, source, labels):
        # Sources arrive in three shapes:
        #   * a single batched tensor (classification/segmentation) -> move it;
        #   * an (base, query) tuple of *already batched* 4D tensors
        #     (image_in_image, pair_collate_fn) -> move each, keep the tuple;
        #   * a tuple of per-sample 3D image tensors (detection, VariableDataLoader
        #     collates variable-count labels, so images are not stacked) -> stack
        #     into one (N, C, H, W) batch tensor.
        if isinstance(source, (list, tuple)):
            same_sized_samples = (
                len(source) > 0
                and all(isinstance(s, torch.Tensor) and s.dim() == 3 for s in source)
                and all(s.shape == source[0].shape for s in source)
            )
            if same_sized_samples:
                source = torch.stack(list(source), dim=0).to(self._device, non_blocking=True)
            else:
                source = [
                    s.to(self._device, non_blocking=True) if isinstance(s, torch.Tensor) else s
                    for s in source
                ]
        else:
            source = source.to(self._device, non_blocking=True)

        # Labels may be a tensor, an (src, tgt) tuple of tensors, or — for detection
        # — a tuple of per-image Python lists of boxes. Only move tensors; leave the
        # box lists on CPU (the loss/metric consume them there and move as needed).
        if isinstance(labels, torch.Tensor):
            labels = labels.to(self._device, non_blocking=True)
        elif isinstance(labels, (list, tuple)):
            labels = [
                l.to(self._device, non_blocking=True) if isinstance(l, torch.Tensor) else l
                for l in labels
            ]

        return source, labels

    def validate(self):
        """
        Поддерживает:
          * один test_loader (старый режим),
          * несколько test_loaders (новый режим: dict[name -> DataLoader]).

        metrics_on:
          - если не задано -> считаем все метрики на всех датасетах;
          - если задано -> каждая метрика считается только на своём датасете.
        """
        if not self._test_loaders:
            return

        # DDP: валидация СИММЕТРИЧНА — все ранки прогоняют свой шард теста
        # (DistributedSampler в creator) развёрнутой сетью (без DDP-коллективов в
        # forward), а torchmetrics сам агрегирует состояния между ранками на
        # compute() (dist_reduce_fx="sum"). Никаких barrier/веток по рангу —
        # прошлая асимметричная схема (rank 0 считает, остальные ждут) давала
        # рассинхрон очередей NCCL-коллективов и вешала ран.
        if ddp.is_ddp():
            self._eval_net = ddp.unwrap(self._net)

        metrics_routing: Dict[str, set] | None = None
        if isinstance(self._metrics_on, Mapping) and len(self._metrics_on) > 0:
            metrics_routing = {}
            for metric_name, ds_name in self._metrics_on.items():
                if ds_name is None:
                    continue
                if ds_name not in self._test_loaders:
                    continue
                metrics_routing.setdefault(ds_name, set()).add(metric_name)

        metrics_all_loaders: Dict[str, Dict[str, float]] = {}
        loss_metrics_all_loaders: Dict[str, Dict[str, float]] = {}

        with torch.no_grad():
            for loader_name, loader in self._test_loaders.items():
                if loader is None:
                    continue

                if metrics_routing is not None:
                    active_metric_names = metrics_routing.get(loader_name, set())
                    metrics_for_loader = {
                        name: m for name, m in self._metrics.items()
                        if name in active_metric_names
                    }
                else:
                    metrics_for_loader = dict(self._metrics)

                sampler = getattr(loader, "sampler", None)
                if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
                    total_size = len(sampler)  # шард этого ранка
                else:
                    total_size = len(loader.dataset)

                train_progress = tqdm(
                    initial=0,
                    total=total_size,
                    desc=f"--> Evaluating [{loader_name}]",
                    ncols=0,
                    dynamic_ncols=True,
                    disable=not ddp.is_main(),
                )

                self._net.train(False)
                loss_metrics = {
                    name: MeanMetric().to(self._device) for name in self._losses.keys()
                }
                for m in metrics_for_loader.values():
                    m.reset()
                for m in loss_metrics.values():
                    m.reset()

                for batch, (source, labels) in enumerate(loader):
                    source, labels = self.set_to_device(source, labels)

                    predictions, losses_values = self.one_step_validate(source, labels)

                    losses_without_weights = {m: v[0] for m, v in losses_values.items()}
                    for k, m in loss_metrics.items():
                        m.update(losses_without_weights[k])

                    for k, m in metrics_for_loader.items():
                        preds_for_metric = predictions
                        if isinstance(preds_for_metric, torch.Tensor) and isinstance(
                            labels, torch.Tensor
                        ):
                            if preds_for_metric.dim() != labels.dim():
                                preds_for_metric = preds_for_metric.squeeze()
                        m.update(preds_for_metric, labels)

                    if isinstance(source, list):
                        batch_size = source[0].size(0) if isinstance(
                            source[0], torch.Tensor
                        ) else 1
                    else:
                        batch_size = source.size(0) if isinstance(
                            source, torch.Tensor
                        ) else 1

                    train_progress.update(batch_size)

                loss_metrics_values = {k: m.compute() for k, m in loss_metrics.items()}
                metrics_values = {k: m.compute() for k, m in metrics_for_loader.items()}

                train_progress.set_postfix({**loss_metrics_values, **metrics_values})
                train_progress.close()

                prefixed_losses = {f"{loader_name}/{k}": v for k, v in loss_metrics_values.items()}
                prefixed_metrics = {f"{loader_name}/{k}": v for k, v in metrics_values.items()}
                self._logger.log_test_losses(self._global_step, prefixed_losses)
                self._logger.log_test_metrics(self._global_step, prefixed_metrics)

                metrics_all_loaders[loader_name] = {
                    k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)
                    for k, v in metrics_values.items()
                }
                loss_metrics_all_loaders[loader_name] = {
                    k: float(v.detach().cpu()) if isinstance(v, torch.Tensor) else float(v)
                    for k, v in loss_metrics_values.items()
                }

        # ---- формируем словарь отслеживаемых метрик с учётом metrics_on ----
        if self._keep_best_config is not None and len(self._keep_best_config) > 0:
            current: Dict[str, float] = {}
            for name in self._keep_best_config.keys():
                ds_name = None
                if isinstance(self._metrics_on, Mapping):
                    ds_name = self._metrics_on.get(name, None)
                if ds_name is None:
                    ds_name = self._primary_test_name

                if (
                    ds_name in metrics_all_loaders
                    and name in metrics_all_loaders[ds_name]
                ):
                    current[name] = metrics_all_loaders[ds_name][name]
                elif (
                    ds_name in loss_metrics_all_loaders
                    and name in loss_metrics_all_loaders[ds_name]
                ):
                    current[name] = loss_metrics_all_loaders[ds_name][name]

            self._current_metrics_all = current if current else None
        else:
            self._current_metrics_all = None

        # ---- legacy _metric_to_track (для одного ключа и одного датасета) ----
        key_for_legacy = None
        if self._keep_best_config is not None and len(self._keep_best_config) > 0:
            key_for_legacy = next(iter(self._keep_best_config.keys()))
        elif isinstance(self._keep_best_on, str):
            key_for_legacy = self._keep_best_on

        if key_for_legacy is not None:
            ds_name = None
            if isinstance(self._metrics_on, Mapping):
                ds_name = self._metrics_on.get(key_for_legacy, None)
            if ds_name is None:
                ds_name = self._primary_test_name

            value_for_legacy = None
            if (
                ds_name in metrics_all_loaders
                and key_for_legacy in metrics_all_loaders[ds_name]
            ):
                value_for_legacy = metrics_all_loaders[ds_name][key_for_legacy]
            elif (
                ds_name in loss_metrics_all_loaders
                and key_for_legacy in loss_metrics_all_loaders[ds_name]
            ):
                value_for_legacy = loss_metrics_all_loaders[ds_name][key_for_legacy]

            if value_for_legacy is not None:
                self._metric_to_track = value_for_legacy
                if self._best_metric_to_track is None:
                    self._best_metric_to_track = self._metric_to_track

        if ddp.is_ddp():
            self._eval_net = None

    def load_from_checkpoint(self):
        ckpt, num = self._ckpt_manager.load_latest_checkpoint()
        # Снимает устаревший префикс 'module.' (старые DataParallel/DDP-чекпоинты).
        ddp.load_state_dict_flexible(self._net, ckpt[CHECKPOINT_MODEL_KEYWORD])

        if self._reset:
            print("--> Resetting.\n")
            self._current_epoch = 1
            self._global_step = 1
        else:
            self._current_epoch = ckpt[CHECKPOINT_EPOCH_KEYWORD]
            self._optimizer.load_state_dict(ckpt[CHECKPOINT_OPTIMIZER_KEYWORD])
            self._scheduler.load_state_dict(
                CHECKPOINT_SCHEDULER_KEYWORD in ckpt
                and ckpt[CHECKPOINT_SCHEDULER_KEYWORD]
                or self._scheduler.state_dict()
            )
            if ckpt.get(CHECKPOINT_SCALER_KEYWORD):
                try:
                    self._scaler.load_state_dict(ckpt[CHECKPOINT_SCALER_KEYWORD])
                except Exception:
                    pass

        return num
