import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import hydra
import torch
import os
import sys
from omegaconf import DictConfig
from colorama import Fore, Style
from echelon3 import __title__, __version__
from echelon3 import ddp
from echelon3 import runtime
from echelon3.cli import add_cwd_to_sys_path, maybe_launch_ddp, setup_warnings
from echelon3 import warncollect

from echelon3.creator import create_datasets, create_augments, create_preprocesses, create_dataloaders, create_trainer
from echelon3.creator import create_net, create_loss, create_optimizer, create_scheduler, create_checkpoint_manager
from echelon3.creator import create_metrics, create_mlops_logger, create_universal


@hydra.main(version_base=None, config_path=None)
def trainer_app(cfg: DictConfig):
    # Встроенный DDP: если запрошено >1 GPU и мы не воркер — порождаем по процессу
    # на GPU (замена torchrun) и выходим; иначе обучаемся в этом процессе.
    if maybe_launch_ddp(cfg, _train):
        return
    _train(cfg)


def _train(cfg: DictConfig):

    setup_warnings()  # копить предупреждения, саммари — перед каждой валидацией
    use_ddp = ddp.init_ddp_if_needed()

    # TF32 matmul + cuDNN autotune (process-level) — до создания сети.
    _tcfg = cfg.trainer.config if ('trainer' in cfg.keys() and 'config' in cfg.trainer.keys()) else {}
    runtime.setup_fast_matmul(tf32=_tcfg.get('tf32', True),
                              cudnn_benchmark=_tcfg.get('cudnn_benchmark', True))
    if use_ddp:
        # Один процесс = один GPU; cfg.device/device_ids игнорируются, видимость
        # GPU задаётся через CUDA_VISIBLE_DEVICES перед torchrun.
        device = torch.device(f'cuda:{ddp.local_rank()}') \
            if torch.cuda.is_available() else torch.device('cpu')
        device_ids = None
        if not ddp.is_main():
            sys.stdout = open(os.devnull, 'w')  # печатает только rank 0
    else:
        device = torch.device(cfg.device if 'device' in cfg.keys() else 'cuda') \
            if torch.cuda.is_available() else torch.device('cpu')

        device_ids = list(cfg.device_ids) if 'device_ids' in cfg.keys() else None

    print(Fore.CYAN)

    print(f'\n\n{__title__} {__version__}: trainer.\n\n')

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
    # В DDP события пишет только rank 0 — не-главные ранки не создают writer'ов.
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
    except (Exception, KeyboardInterrupt):
        # Ловим и KeyboardInterrupt (Ctrl-C) — иначе он шёл бы мимо этого пути.
        # Traceback печатаем ДО shutdown в stderr (stdout не-главных ранков заглушён).
        import traceback
        print(f'[rank {ddp.rank()}] trainer.train() interrupted/failed:', file=sys.stderr)
        traceback.print_exc()
        if ddp.is_ddp():
            # Чистый destroy_process_group() может сам зависнуть на NCCL-teardown,
            # пока в группе висит недоделанный коллектив — тогда упавший ранг не
            # выходит, и elastic не видит падения и не снимает пиров (тихий вис).
            # Жёсткий выход гарантирует смерть ранка → лаунчер тут же снимает пиров;
            # DataLoader-воркеры добьёт PDEATHSIG (см. ddp.set_pdeathsig).
            sys.stderr.flush()
            os._exit(1)
        raise
    finally:
        if ddp.is_main():
            warncollect.flush()  # финальное саммари хвостовых предупреждений
        try:
            trainer.close()  # погасить воркеров даталоадеров на чистом выходе
        except Exception:
            pass
        ddp.shutdown()

    print(Style.RESET_ALL)


def main():
    add_cwd_to_sys_path()
    trainer_app()


if __name__ == "__main__":
    main()
