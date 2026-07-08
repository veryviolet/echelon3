"""Fine-tuning entry point.

Drop-in replacement for `echelon3-train` that adds three things:

* warm-start the network's state_dict from a previously trained
  checkpoint via `cfg.init_from.*`;
* freeze parameters whose dotted name matches any of `cfg.finetune.freeze_patterns`;
* build per-layer parameter groups from `cfg.finetune.param_groups` (a
  list of `{pattern, lr_mult}` dicts) and pass them to the optimizer
  instead of `net.parameters()`.

When none of those config blocks are present the entry point behaves
exactly like `echelon3-train`, so a stock baseline config also runs
under this entry point unchanged.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import hydra
import torch
from omegaconf import DictConfig
from colorama import Fore, Style

import os
import sys

from echelon3 import __title__, __version__
from echelon3 import ddp
from echelon3 import runtime
from echelon3.cli import add_cwd_to_sys_path, maybe_launch_ddp, setup_warnings
from echelon3 import warncollect
from echelon3.creator import (
    create_datasets, create_augments, create_preprocesses, create_dataloaders,
    create_trainer, create_net, create_loss, create_optimizer, create_scheduler,
    create_checkpoint_manager, create_metrics, create_mlops_logger,
)
from echelon3.utils.param_groups import (
    freeze_by_patterns, build_param_groups, head_only_params,
)
from echelon3.checkpoint.manager import CHECKPOINT_MODEL_KEYWORD


def _load_init_weights(net: torch.nn.Module, ckpt_path: str, strict: bool = False) -> int:
    """Load a state_dict into `net`, stripping any `module.` prefix.

    Returns the number of tensors successfully loaded.
    """
    # weights_only=False is required because baseline checkpoints contain
    # torchmetrics objects (saved as part of CHECKPOINT_METRICS_KEYWORD by
    # the trainer); PyTorch >= 2.6 defaults to weights_only=True which rejects
    # any non-tensor pickled global.
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state[CHECKPOINT_MODEL_KEYWORD] if CHECKPOINT_MODEL_KEYWORD in state else state
    new_sd = {}
    for k, v in sd.items():
        new_k = k[len("module."):] if k.startswith("module.") else k
        new_sd[new_k] = v
    missing, unexpected = net.load_state_dict(new_sd, strict=strict)
    print(Fore.YELLOW, end="")
    if missing:
        print(f"--> init_from: {len(missing)} missing keys "
              f"(first 5: {missing[:5]})")
    if unexpected:
        print(f"--> init_from: {len(unexpected)} unexpected keys "
              f"(first 5: {unexpected[:5]})")
    print(Fore.CYAN, end="")
    return len(new_sd) - len(missing)


@hydra.main(version_base=None, config_path=None)
def finetune_app(cfg: DictConfig):
    # Встроенный DDP: несколько GPU — порождаем воркеры (замена torchrun) и выходим.
    if maybe_launch_ddp(cfg, _finetune):
        return
    _finetune(cfg)


def _finetune(cfg: DictConfig):
    setup_warnings()  # копить предупреждения, саммари — перед каждой валидацией
    use_ddp = ddp.init_ddp_if_needed()
    _tcfg = cfg.trainer.config if ('trainer' in cfg.keys() and 'config' in cfg.trainer.keys()) else {}
    runtime.setup_fast_matmul(tf32=_tcfg.get('tf32', True),
                              cudnn_benchmark=_tcfg.get('cudnn_benchmark', True))
    if use_ddp:
        device = torch.device(f'cuda:{ddp.local_rank()}') \
            if torch.cuda.is_available() else torch.device('cpu')
        device_ids = None
        if not ddp.is_main():
            sys.stdout = open(os.devnull, 'w')
    else:
        device = torch.device(cfg.device if 'device' in cfg.keys() else 'cuda') \
            if torch.cuda.is_available() else torch.device('cpu')
        device_ids = list(cfg.device_ids) if 'device_ids' in cfg.keys() else None

    print(Fore.CYAN)
    print(f'\n\n{__title__} {__version__}: finetune trainer.\n\n')

    _transform = cfg.transform if 'transform' in cfg.keys() else None
    train_augment, test_augment = create_augments(_transform)
    train_preprocess, test_preprocess = create_preprocesses(_transform)
    train_dataset, test_dataset = create_datasets(
        config=cfg.data,
        train_augment=train_augment, test_augment=test_augment,
        train_preprocess=train_preprocess, test_preprocess=test_preprocess,
    )
    train_dataloader, test_dataloader = create_dataloaders(
        cfg.dataloaders, train_dataset, test_dataset,
    )

    print(f'--> Initializing network... ')
    net = create_net(cfg.net)
    net.to(device)
    print(Fore.LIGHTGREEN_EX, end="")
    print(f'        {type(net).__name__}')
    print(Fore.CYAN, end="")

    # Warm-start from a baseline checkpoint, if requested.
    if 'init_from' in cfg.keys() and cfg.init_from is not None:
        ckpt_path = str(cfg.init_from.checkpoint)
        strict = bool(cfg.init_from.get('strict', False))
        print(f'--> Warm-starting from {ckpt_path} (strict={strict})... ')
        _load_init_weights(net, ckpt_path, strict=strict)

    # Freeze by regex patterns, if requested.
    if 'finetune' in cfg.keys() and 'freeze_patterns' in cfg.finetune.keys():
        patterns = list(cfg.finetune.freeze_patterns or [])
        n_frozen = freeze_by_patterns(net, patterns)
        print(Fore.YELLOW, end="")
        print(f'--> Frozen {n_frozen} parameter tensors matching {patterns}')
        print(Fore.CYAN, end="")

    losses = create_loss(cfg.loss)
    metrics = create_metrics(cfg.metrics if 'metrics' in cfg.keys() else None)

    # Optimizer: optionally with param groups.
    print(f'--> Initializing optimizer... ')
    head_only = bool(cfg.finetune.head_only) if (
        'finetune' in cfg.keys() and 'head_only' in cfg.finetune.keys()
    ) else False
    pg_spec = (
        list(cfg.finetune.param_groups) if (
            'finetune' in cfg.keys() and 'param_groups' in cfg.finetune.keys()
        ) else None
    )

    default_lr = float(cfg.optimizer.config.lr) if (
        'config' in cfg.optimizer.keys() and 'lr' in cfg.optimizer.config.keys()
    ) else 1e-4

    if head_only:
        params = head_only_params(net)
    elif pg_spec is not None:
        params = build_param_groups(net, pg_spec, default_lr=default_lr)
    else:
        params = net.parameters()

    optimizer = create_optimizer(cfg.optimizer, params)
    print(Fore.LIGHTGREEN_EX, end="")
    print(f'        {type(optimizer).__name__}({cfg.optimizer.get("config", {})})')
    print(Fore.CYAN, end="")

    scheduler = create_scheduler(cfg.scheduler, optimizer) if 'scheduler' in cfg.keys() else None
    ckpt_manager = create_checkpoint_manager(cfg.target)

    logger_config = cfg.mlops if 'mlops' in cfg.keys() else None
    mlops_logger = create_mlops_logger(cfg.target, logger_config) \
        if (not use_ddp or ddp.is_main()) else None

    print(f'--> Initializing trainer... ')
    trainer = create_trainer(
        cfg.trainer, optimizer=optimizer, net=net, losses=losses, metrics=metrics,
        train_dataloader=train_dataloader, test_dataloader=test_dataloader,
        scheduler=scheduler, ckpt_manager=ckpt_manager,
        mlops_logger=mlops_logger, devices=device, device_ids=device_ids,
    )

    print(f'--> Training... ')
    try:
        trainer.train()
    except (Exception, KeyboardInterrupt):
        # Ловим и Ctrl-C. Traceback в stderr (stdout не-главных рангов заглушён).
        # Под DDP — жёсткий выход: destroy_process_group() может зависнуть на
        # NCCL-teardown, тогда упавший ранг не выходит и elastic молча ждёт (тихий
        # вис). os._exit гарантирует смерть ранка → лаунчер снимает пиров, а
        # DataLoader-воркеров добивает PDEATHSIG.
        import traceback
        print(f'[rank {ddp.rank()}] trainer.train() interrupted/failed:', file=sys.stderr)
        traceback.print_exc()
        if ddp.is_ddp():
            sys.stderr.flush()
            os._exit(1)
        raise
    finally:
        if ddp.is_main():
            warncollect.flush()  # финальное саммари хвостовых предупреждений
        try:
            trainer.close()
        except Exception:
            pass
        ddp.shutdown()

    print(Style.RESET_ALL)


def main():
    add_cwd_to_sys_path()
    finetune_app()


if __name__ == "__main__":
    main()
