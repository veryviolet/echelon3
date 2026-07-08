import os.path
from typing import Dict, Tuple
import inspect
from functools import partial
from collections import OrderedDict
import cv2
import torch
from torch.utils.data import DataLoader
import importlib
import importlib.util
from omegaconf import OmegaConf,open_dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf, DictConfig, ListConfig

from echelon3.checkpoint.manager import CheckpointManager
from echelon3 import ddp

PURPOSE_TRAIN = 'train'
PURPOSE_TEST = 'test'
PURPOSE_INFERENCE = 'inference'
TRANSFORM_PURPOSES = [PURPOSE_TRAIN, PURPOSE_TEST]
DATASET_PURPOSES = [PURPOSE_TRAIN, PURPOSE_TEST]


def get_attr_from_module(module, attr):

    try:
        mdl = importlib.import_module(module)
    except ImportError:
        # module может быть путём к .py-файлу (расширение из zoo/проекта пользователя)
        try:
            module_name = os.path.splitext(os.path.basename(module))[0]
            spec = importlib.util.spec_from_file_location(module_name, module)
            mdl = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mdl)
        except Exception:
            raise RuntimeError(f'failed to import module {module}')

    try:
        result = getattr(mdl, attr)
    except:
        raise RuntimeError(f'attribute {attr} not found in {module}')

    return result


def create_universal(config: DictConfig):
    par_type = get_attr_from_module(config.module, config.type)
    if inspect.isclass(par_type):
        if 'config' in config.keys():
            par = par_type(**config.config)
        else:
            par = par_type()
    else:
        if 'config' in config.keys():
            par = partial(par_type, **config.config)
        else:
            par = par_type

    return par


def create_single_augment(config: DictConfig, bbox_params: DictConfig = None):
    transforms = []

    # Если аугментаций нет (config is None), просто возвращаем ToTensorV2
    if config is not None:
        for one in config.values():
            transforms.append(create_universal(one))

    return A.Compose(transforms + [ToTensorV2()]) if bbox_params is None \
        else A.Compose(transforms + [ToTensorV2()], bbox_params={**bbox_params})

def create_single_preprocess(config: DictConfig):
    transforms = []
    for one in config.values():
        transforms.append((one.name, create_universal(one)))
    return torch.nn.Sequential(OrderedDict(transforms))


def create_augments(config: DictConfig) -> Tuple[callable, callable]:
    transforms = {k: A.Compose([ToTensorV2()]) for k in TRANSFORM_PURPOSES}

    for key, cfg in config.items():
        if key not in TRANSFORM_PURPOSES:
            raise RuntimeError(f'purpose of transform must be one of {TRANSFORM_PURPOSES}')
        if 'augment' in cfg.keys():
            bbox_params = cfg['config']['bbox_params']\
                if 'config' in cfg.keys() and 'bbox_params' in cfg['config'].keys() else None
            transforms[key] = create_single_augment(cfg.augment,  bbox_params=bbox_params)

    return transforms['train'], transforms['test']


def create_preprocesses(config: DictConfig) -> Tuple[callable, callable]:
    transforms = {k: None for k in TRANSFORM_PURPOSES}

    for key, cfg in config.items():
        if key not in TRANSFORM_PURPOSES:
            raise RuntimeError(f'purpose of transform must be one of {TRANSFORM_PURPOSES}')

        transforms[key] = create_single_preprocess(cfg.preprocess)

    return transforms['train'], transforms['test']


def create_single_dataset(config: DictConfig, augment, preprocess, **extra_kwargs):
    ds_type = get_attr_from_module(config.module, config.type)
    ds = ds_type(**config.config, augment=augment, preprocess=preprocess, **extra_kwargs)
    return ds


def create_evaluator(config: DictConfig, net, train_dataloader, test_dataloader, metric, preprocess, postprocess):
    ev_type = get_attr_from_module(config.module, config.type)
    # Для текущих классификаторов используем только валидационный (test) даталоудер
    ev = ev_type(
        **config.config,
        net=net,
        dataloader=test_dataloader,
        metric=metric,
        preprocess=preprocess,
        postprocess=postprocess,
    )
    return ev

def create_datasets(config: DictConfig, train_augment, train_preprocess, test_augment, test_preprocess):
    """
    Поддерживает:
      data:
        train: {...}  # одиночный train датасет
        test:  {...}  # одиночный test датасет (старый формат)

    а также:
      data:
        train: {...}
        test:
          incidents: {...}
          valA:      {...}

    В последнем случае возвращает:
      train_dataset: Dataset
      test_dataset: Dict[str, Dataset]
    """
    test_dataset = None
    train_dataset = None

    for purpose, cfg in config.items():
        if purpose not in DATASET_PURPOSES:
            raise RuntimeError(f'purpose must be one of {DATASET_PURPOSES}')

        if purpose == PURPOSE_TRAIN:
            # train всегда одиночный
            train_dataset = create_single_dataset(
                cfg,
                augment=train_augment,
                preprocess=train_preprocess,
            )

        elif purpose == PURPOSE_TEST:
            # Один test‑датасет (старый формат: module/type/config на верхнем уровне)
            if 'module' in cfg and 'type' in cfg:
                test_dataset = create_single_dataset(
                    cfg,
                    augment=test_augment,
                    preprocess=test_preprocess,
                )
            else:
                # Несколько test‑датасетов: ключ -> под‑конфиг
                test_dataset = {}
                for name, sub_cfg in cfg.items():
                    test_dataset[name] = create_single_dataset(
                        sub_cfg,
                        augment=test_augment,
                        preprocess=test_preprocess,
                    )

    return train_dataset, test_dataset


def create_net(config: DictConfig):
    net_type = get_attr_from_module(config.module, config.type)
    net = net_type(**config.config)
    return net


def create_backbone(config: DictConfig):
    net_type = get_attr_from_module(config.module, config.type)
    net = net_type(**config.config)
    return net


def create_loss(config: DictConfig):
    losses = {}

    for cfg in config:
        name, one = list(cfg.items())[0]
        loss_type = get_attr_from_module(one.module, one.type)
        losses[name] = (loss_type(**one.config) if 'config' in one.keys() else loss_type(), one.weight)

    return losses


def create_metrics(config):

    metrics = {}

    for cfg in config:
        name, one = list(cfg.items())[0]
        metrics[name] = create_universal(one)

    return metrics


def create_optimizer(config: DictConfig, params):
    opt_type = get_attr_from_module(config.module, config.type)
    opt_kwargs = dict(config.config) if 'config' in config.keys() else {}
    trainable_only = bool(opt_kwargs.pop('trainable_only', False))
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    opt = opt_type(params=params, **opt_kwargs)
    return opt


def create_scheduler(config: DictConfig, optimizer: torch.optim.Optimizer):
    sch_type = get_attr_from_module(config.module, config.type)
    sch = sch_type(optimizer=optimizer, **config.config)
    return sch


def create_single_dataloader(config: DictConfig, dataset):
    dataloader_type = get_attr_from_module(config.module, config.type)
    return dataloader_type(dataset=dataset, **config.config)



def _worker_init_fn(user_fn=None):
    """worker_init_fn для DataLoader: воркер получает PDEATHSIG (умирает вместе с
    рангом → не осиротеет и не держит /dev/shm/RAM), затем зовётся пользовательский
    worker_init_fn, если был."""
    def _init(worker_id):
        ddp.set_pdeathsig()
        if callable(user_fn):
            user_fn(worker_id)
    return _init


def create_dataloaders(config: DictConfig, train_dataset, test_dataset):
    """
    Поддерживает:
      dataloaders:
        train: {...}
        test:  {...}

    и:
      dataloaders:
        train: {...}
        test:
          incidents: {...}
          valA:      {...}

    В последнем случае test_dataloader будет Dict[str, DataLoader],
    что понимает доработанный Trainer.
    """
    # train — всегда одиночный
    train_dataloader_type = get_attr_from_module(config.train.module, config.train.type)
    train_cfg = OmegaConf.to_container(config.train.config, resolve=True)

    if ddp.is_ddp():
        # Семантика конфига сохраняется: batch_size — глобальный, делим на ранки.
        world = ddp.world_size()
        global_bs = int(train_cfg.get('batch_size', 1))
        if global_bs % world != 0:
            raise ValueError(
                f'DDP: dataloaders.train.config.batch_size={global_bs} '
                f'не делится на world_size={world}'
            )
        train_cfg['batch_size'] = global_bs // world
        # shuffle обеспечивает DistributedSampler (эксклюзивен с shuffle=True)
        shuffle = bool(train_cfg.pop('shuffle', True))
        train_cfg['sampler'] = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=shuffle, drop_last=bool(train_cfg.get('drop_last', False))
        )
        print(f'--> DDP dataloader: global batch {global_bs} = '
              f'{train_cfg["batch_size"]}/process x {world} processes '
              f'(num_workers={train_cfg.get("num_workers", 0)} на процесс)')

    if int(train_cfg.get('num_workers', 0) or 0) > 0:
        train_cfg['worker_init_fn'] = _worker_init_fn(train_cfg.get('worker_init_fn'))
    train_dataloader = train_dataloader_type(dataset=train_dataset, **train_cfg)

    def _test_cfg(sub_cfg, dataset):
        cfg = OmegaConf.to_container(sub_cfg, resolve=True)
        if ddp.is_ddp():
            # Симметричная валидация: каждый ранк считает свой шард, метрики
            # агрегируются torchmetrics'ом (dist_reduce_fx). Воркеры через spawn:
            # без воркеров eval
            # готовит данные одним потоком (минуты + все ранки ждут самого
            # медленного на барьере синка метрик).
            # fork-воркеры: spawn не может запиклить датасет (cv2.CLAHE внутри),
            # а fork с NCCL безопасен эмпирически — train-лоадеры так работают.
            cfg['num_workers'] = min(int(cfg.get('num_workers', 4)), 4)
            cfg.pop('shuffle', None)
            cfg['sampler'] = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=False
            )
            # batch_size конфига — глобальный (в DP его резал на карты сам
            # DataParallel; полный батч 40 на одной карте = OOM в interpolate).
            cfg['batch_size'] = max(1, int(cfg.get('batch_size', 1)) // ddp.world_size())
        if int(cfg.get('num_workers', 0) or 0) > 0:
            cfg['worker_init_fn'] = _worker_init_fn(cfg.get('worker_init_fn'))
        return cfg

    # test — один или несколько
    if isinstance(test_dataset, dict):
        # несколько тест‑датасетов
        test_dataloaders = {}
        for name, ds in test_dataset.items():
            sub_cfg = config.test[name]
            test_dataloader_type = get_attr_from_module(sub_cfg.module, sub_cfg.type)
            test_dataloaders[name] = test_dataloader_type(dataset=ds, **_test_cfg(sub_cfg.config, ds))
        return train_dataloader, test_dataloaders
    else:
        # одиночный тест‑датасет (старый формат)
        test_dataloader_type = get_attr_from_module(config.test.module, config.test.type)
        test_dataloader = test_dataloader_type(dataset=test_dataset, **_test_cfg(config.test.config, test_dataset))
        return train_dataloader, test_dataloader

def create_checkpoint_manager(config: DictConfig):
    return CheckpointManager(**config)


def create_trainer(config: DictConfig, net: torch.nn.Module, optimizer: torch.optim.Optimizer, train_dataloader,
                   test_dataloader,
                   losses, metrics, scheduler, ckpt_manager, mlops_logger, devices, device_ids):
    trainer_type = get_attr_from_module(config.module, config.type)
    trn = trainer_type(optimizer=optimizer, net=net, train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader, losses=losses, metrics=metrics,
                       scheduler=scheduler, ckpt_manager=ckpt_manager, mlops_logger=mlops_logger,
                       device=devices, device_ids=device_ids,
                       **config.config)
    return trn

def create_wrapper(config: DictConfig, net):
    ev_type = get_attr_from_module(config.module, config.type)
    ev = ev_type(**config.config, core=net)
    return ev

def create_exporters(config, net):

    preprocess = create_single_preprocess(config.preprocess) if 'preprocess' in config.keys() else torch.nn.Identity()
    postprocess = create_single_preprocess(config.postprocess) if 'postprocess' in config.keys() else torch.nn.Identity()

    exporters = {}

    for name, cfg in config.exporters.items():
        ex_type = get_attr_from_module(cfg.module, cfg.type)
        ex = ex_type(**cfg.config, net=net, preprocess=preprocess, postprocess=postprocess)
        exporters[name] = ex

    return exporters


def create_constructor(config, components):

    net_type = get_attr_from_module(config.module, config.type)
    net = net_type(**config.config, **components)

    return net


def create_batch_sampler(config: DictConfig, **kwargs):
    par_type = get_attr_from_module(config.module, config.type)
    if inspect.isclass(par_type):
        if 'config' in config.keys():
            par = par_type(**config.config, **kwargs)
        else:
            par = par_type(**kwargs)
    else:
        if 'config' in config.keys():
            par = partial(par_type, **config.config, **kwargs)
        else:
            par = par_type

    return par


def create_mlops_logger(target_config: DictConfig, mlops_config: DictConfig = None):
    if mlops_config is None:
        mlops_config = DictConfig({'module': 'echelon3.mlops.tensorboard',
                                   'type': 'TensorboardLogger'})

    with open_dict(mlops_config):
        if 'config' in mlops_config.keys():
            mlops_config.config['folder'] = target_config.path
        else:
            mlops_config['config'] = DictConfig({'folder': target_config.path})

    return create_universal(mlops_config)
