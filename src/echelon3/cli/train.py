import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import os
import sys
from omegaconf import DictConfig
from colorama import Fore, Style
from echelon3 import __title__, __version__
from echelon3 import ddp
from echelon3 import runtime
from echelon3.cli import add_cwd_to_sys_path, maybe_launch_ddp, setup_warnings, resolve_single_device, build_cli
from echelon3.cli import _close_quietly, _looks_like_interrupt, _install_sigint_flag, _silence_sigint
from echelon3 import warncollect

from echelon3.creator import create_datasets, create_augments, create_preprocesses, create_dataloaders, create_trainer
from echelon3.creator import create_net, create_loss, create_optimizer, create_scheduler, create_checkpoint_manager
from echelon3.creator import create_metrics, create_mlops_logger, create_universal
from echelon3.creator import create_tabular_datasets, create_estimator_trainer


def _is_estimator_cfg(cfg: DictConfig) -> bool:
    """Estimator (fit/predict) run vs image SGD run. Distinguished by section:
    the estimator world has `model:` (and no `net:`); SGD has `net:`."""
    return 'model' in cfg.keys() and 'net' not in cfg.keys()


def trainer_app(cfg: DictConfig):
    # We print the banner + Fore.CYAN in the PARENT before the DDP launch: then it
    # comes BEFORE the launcher messages, which inherit cyan (colorama keeps the color
    # within the process) — otherwise they would come before the banner and without
    # color. Spawned workers do not repeat the banner.
    print(Fore.CYAN)
    print(f'\n\n{__title__} {__version__}: trainer.\n\n')
    # Routing by trainer type: fit/predict (trees, tabular FM) is assembled by its OWN
    # branch (without optimizer/loss/loaders/DDP), image SGD as before.
    if _is_estimator_cfg(cfg):
        _train_estimator(cfg)
        return
    # Built-in DDP: if >1 GPU is requested and we are not a worker — spawn one process
    # per GPU (a replacement for torchrun) and exit; otherwise train in this process.
    if maybe_launch_ddp(cfg, _train):
        return
    _train(cfg)


def _train_estimator(cfg: DictConfig):
    """Assembles and runs a fit/predict model (EstimatorTrainer). Its own short setup:
    model + tabular datasets + metrics + ckpt; no optimizer/loss/loaders, no DDP."""
    setup_warnings()
    device = resolve_single_device(cfg, torch.cuda.is_available())
    print(Fore.CYAN)
    print(f'--> Estimator (fit/predict) run. device={device} (the model decides where to compute)')

    print(f'--> Initializing model... ')
    model = create_universal(cfg.model)
    _mcfg = cfg.model.config if 'config' in cfg.model.keys() else {}
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {type(model).__name__}({_mcfg})')
    print(Fore.CYAN, end='')

    print(f'--> Initializing datasets... ')
    train_data, test_data = create_tabular_datasets(cfg.data)
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        train: {train_data}')
    print(f'        test:  {test_data}')
    print(Fore.CYAN, end='')

    print(f'--> Initializing metrics... ')
    metrics = create_metrics(cfg.metrics if 'metrics' in cfg.keys() else None)
    print(Fore.LIGHTGREEN_EX, end='')
    for name, m in metrics.items():
        print(f'    {name}: {type(m).__name__}')
    print(Fore.CYAN, end='')

    feature_transform = None
    if 'feature_transform' in cfg.keys():
        print(f'--> Initializing feature_transform... ')
        feature_transform = create_universal(cfg.feature_transform)
        print(Fore.LIGHTGREEN_EX, end='')
        print(f'        {type(feature_transform).__name__}')
        print(Fore.CYAN, end='')

    print(f'--> Initializing checkpoint manager... ')
    ckpt_manager = create_checkpoint_manager(cfg.target)

    print(f'--> Initializing trainer... ')
    trainer = create_estimator_trainer(cfg.trainer, model=model, train_data=train_data,
                                       test_data=test_data, metrics=metrics, ckpt_manager=ckpt_manager,
                                       feature_transform=feature_transform)
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {type(trainer).__name__}')
    print(Fore.CYAN, end='')

    print(f'--> Training... ')
    try:
        trainer.train()
    except KeyboardInterrupt:
        print('\n--> Interrupted by user (Ctrl-C).', file=sys.stderr)
        raise SystemExit(130)
    finally:
        warncollect.flush()
        try:
            trainer.close()
        except Exception:
            pass
    print(Style.RESET_ALL)


def _train(cfg: DictConfig):

    setup_warnings()  # accumulate warnings, summary — before each validation
    _install_sigint_flag()  # flag Ctrl-C (to tell a worker death apart from OOM)
    use_ddp = ddp.init_ddp_if_needed()

    # TF32 matmul + cuDNN autotune (process-level) — before the network is created.
    _tcfg = cfg.trainer.config if ('trainer' in cfg.keys() and 'config' in cfg.trainer.keys()) else {}
    runtime.setup_fast_matmul(tf32=_tcfg.get('tf32', True),
                              cudnn_benchmark=_tcfg.get('cudnn_benchmark', True))
    if use_ddp:
        # One process = one GPU; cfg.device/device_ids are ignored, GPU visibility
        # is set via CUDA_VISIBLE_DEVICES before torchrun.
        device = torch.device(f'cuda:{ddp.local_rank()}') \
            if torch.cuda.is_available() else torch.device('cpu')
        device_ids = None
        if not ddp.is_main():
            sys.stdout = open(os.devnull, 'w')  # only rank 0 prints
    else:
        device = resolve_single_device(cfg, torch.cuda.is_available())
        if device.type == 'cuda' and device.index is not None:
            # cuda:{idx} — move the default device so that .cuda()/autocast land on the
            # right card. For bare 'cuda' (no index) the default is already 0 —
            # set_device would fail here (an index is required).
            torch.cuda.set_device(device)
        device_ids = list(cfg.device_ids) if 'device_ids' in cfg.keys() else None

    print(Fore.CYAN)  # worker: its own color (the banner is already printed by the parent)

    if use_ddp:
        print(f'--> DDP: world_size={ddp.world_size()}, backend={torch.distributed.get_backend()}, '
              f'device={device}')
        if 'device_ids' in cfg.keys():
            print('--> DDP: cfg.device_ids is ignored; use CUDA_VISIBLE_DEVICES')

    print(f'--> Initializing augmentations... ')
    _transform = cfg.transform if 'transform' in cfg.keys() else None
    train_augment, test_augment = create_augments(_transform)
    print(f'--> Train augmentations:')
    print(Fore.LIGHTGREEN_EX, end='')
    for tr in train_augment:
        print(f'        {tr}')
    print(Fore.CYAN, end='')
    print(f'--> Test augmentations:')
    print(Fore.LIGHTGREEN_EX, end='')
    if test_augment is not None:
        for tr in test_augment:
            print(f'        {tr}')
    print(Fore.CYAN, end='')

    print(f'--> Initializing preprocess... ')
    train_preprocess, test_preprocess = create_preprocesses(_transform)
    print(f'--> Train preprocess:')
    print(Fore.LIGHTGREEN_EX, end='')
    for tr in (train_preprocess or []):
        print(f'        {tr}')
    print(Fore.CYAN, end='')
    print(f'--> Test preprocess:')
    print(Fore.LIGHTGREEN_EX, end='')
    for tr in (test_preprocess or []):
        print(f'        {tr}')
    print(Fore.CYAN, end='')

    print(f'--> Initializing datasets... ')
    train_dataset, test_dataset = create_datasets(config=cfg.data,
                                                  train_augment=train_augment, test_augment=test_augment,
                                                  train_preprocess=train_preprocess, test_preprocess=test_preprocess)
    print(f'--> Train dataset:')
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {train_dataset}')
    print(Fore.CYAN, end='')
    print(f'--> Test dataset:')
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {test_dataset}')
    print(Fore.CYAN, end='')

    print(f'--> Initializing dataloaders... ')
    train_dataloader, test_dataloader = create_dataloaders(cfg.dataloaders, train_dataset, test_dataset)
    print(f'--> Train dataloader:')
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {type(train_dataloader).__name__}({cfg.dataloaders.train})')
    print(Fore.CYAN, end='')
    print(f'--> Test dataloader:')
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {type(test_dataloader).__name__}({cfg.dataloaders.test})')
    print(Fore.CYAN, end='')

    print(f'--> Initializing network... ')
    net = create_net(cfg.net)
    net.to(device)
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {type(net).__name__}({cfg.net.config})')
    print(Fore.CYAN, end='')

    if 'weights' in cfg.net.keys():
        print(f'--> Creating weights loader ... ')
        weights_loader = create_universal(cfg.weights_loader)

        print(f'--> Loading weights from {cfg.net.weights}... ')
        weights_loader.load(net, cfg.net.weights, device)

    print(f'--> Initializing losses... ')
    losses = create_loss(cfg.loss)
    print(Fore.LIGHTGREEN_EX, end='')
    for name, loss in losses.items():
        print(f'    {name}: {type(loss[0]).__name__}  with weight: {loss[1]}')
    print(Fore.CYAN, end='')

    print(f'--> Initializing metrics... ')
    metrics = create_metrics(cfg.metrics if 'metrics' in cfg.keys() else None)
    print(Fore.LIGHTGREEN_EX, end='')
    for name, m in metrics.items():
        print(f'    {name}: {type(m).__name__}')
    print(Fore.CYAN, end='')

    print(f'--> Initializing optimizer... ')
    optimizer = create_optimizer(cfg.optimizer, net.parameters())
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {type(optimizer).__name__}({cfg.optimizer.config})')
    print(Fore.CYAN, end='')

    print(f'--> Initializing learning rate scheduler... ')
    if 'scheduler' in cfg.keys():
        scheduler = create_scheduler(cfg.scheduler, optimizer)
        print(Fore.LIGHTGREEN_EX, end='')
        print(f'        {type(scheduler).__name__}({cfg.scheduler.config})')
        print(Fore.CYAN, end='')
    else:
        scheduler = None
        print(f'--> No scheduler configured (constant LR).')

    print(f'--> Initializing checkpoint manager... ')
    ckpt_manager = create_checkpoint_manager(cfg.target)
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {type(ckpt_manager).__name__}({cfg.target})')
    print(Fore.CYAN, end='')

    print(f'--> Initializing mlops logger... ')
    if 'mlops' in cfg.keys():
        logger_config = cfg.mlops
    else:
        print(f'--> No mlops backend configured. Falling back to tensorboard logger... ')
        logger_config = None
    # Under DDP only rank 0 writes events — non-main ranks do not create writers.
    mlops_logger = create_mlops_logger(cfg.target, logger_config) \
        if (not use_ddp or ddp.is_main()) else None

    print(f'--> Initializing trainer... ')
    trainer = create_trainer(cfg.trainer, optimizer=optimizer, net=net, losses=losses, metrics=metrics,
                             train_dataloader=train_dataloader,
                             test_dataloader=test_dataloader, scheduler=scheduler, ckpt_manager=ckpt_manager,
                             mlops_logger=mlops_logger, devices=device, device_ids=device_ids)
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {type(trainer).__name__}({cfg.trainer.config})')
    print(Fore.CYAN, end='')

    print(f'--> Training... ')
    try:
        trainer.train()
    except KeyboardInterrupt:
        # Ctrl-C — a normal stop, without a scary traceback.
        _silence_sigint()  # FIRST: a repeated SIGINT must not take us past os._exit into finally
        if ddp.is_main():
            print('\n--> Interrupted by user (Ctrl-C), shutting down.', file=sys.stderr)
            sys.stderr.flush()
        # IMPORTANT: shut down the DataLoader workers BEFORE os._exit. os._exit skips
        # finally, and PDEATHSIG finishes the workers off with SIGKILL without releasing
        # their semaphores — otherwise the launcher's resource_tracker complains about
        # "leaked semaphore objects" (a /dev/shm leak).
        _close_quietly(trainer)
        if ddp.is_ddp():
            os._exit(130)  # 128+SIGINT: hard exit → elastic reaps the peers
        raise SystemExit(130)
    except Exception as e:
        # Ctrl-C may have killed a dataloader worker before the main process (a race
        # before SIG_IGN) — torch raises RuntimeError 'worker ... killed by signal:
        # Interrupt'. This is an interruption, not a crash: shut down cleanly (exit 130),
        # without a scary traceback/abort.
        if _looks_like_interrupt(e):
            _silence_sigint()  # as in the KeyboardInterrupt branch: don't let a repeated SIGINT break the exit
            if ddp.is_main():
                print('\n--> Interrupted by user (Ctrl-C), shutting down.', file=sys.stderr)
                sys.stderr.flush()
            _close_quietly(trainer)
            if ddp.is_ddp():
                os._exit(130)
            raise SystemExit(130)
        # We print the traceback to stderr BEFORE shutdown (stdout of non-main ranks is silenced).
        import traceback
        print(f'[rank {ddp.rank()}] trainer.train() failed:', file=sys.stderr)
        traceback.print_exc()
        _close_quietly(trainer)  # release worker semaphores on the failure path too
        if ddp.is_ddp():
            # A clean destroy_process_group() can hang on the NCCL teardown while an
            # unfinished collective is pending in the group (the crashed rank does not
            # exit → a silent hang). A hard exit → the launcher reaps the peers.
            sys.stderr.flush()
            os._exit(1)
        raise
    finally:
        if ddp.is_main():
            warncollect.flush()  # final summary of trailing warnings
        try:
            trainer.close()  # shut down the dataloader workers on a clean exit
        except Exception:
            pass
        ddp.shutdown()

    print(Style.RESET_ALL)


main = build_cli(trainer_app)  # click CLI + OmegaConf overrides (in place of @hydra.main)


if __name__ == "__main__":
    main()
