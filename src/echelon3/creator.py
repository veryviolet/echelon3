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
        # module may be a path to a .py file (an extension from the zoo/user project)
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


def _cfg_kwargs(config) -> dict:
    """kwargs from the ``config`` block. A SINGLE rule across the whole framework: the
    ``config`` block is optional for ANY component — if it is absent, the constructor
    is called without extra arguments (not "sometimes optional, sometimes not")."""
    return dict(config.config) if 'config' in config.keys() else {}


def create_universal(config: DictConfig):
    par_type = get_attr_from_module(config.module, config.type)
    kw = _cfg_kwargs(config)
    if inspect.isclass(par_type):
        return par_type(**kw)
    return partial(par_type, **kw) if kw else par_type


def create_single_augment(config: DictConfig, bbox_params: DictConfig = None):
    transforms = []

    # If there are no augmentations (config is None), just return ToTensorV2
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

    if config is None:  # transform section omitted — one ToTensorV2 for train/test
        return transforms['train'], transforms['test']

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

    if config is None:  # transform section omitted — no preprocessing
        return transforms['train'], transforms['test']

    for key, cfg in config.items():
        if key not in TRANSFORM_PURPOSES:
            raise RuntimeError(f'purpose of transform must be one of {TRANSFORM_PURPOSES}')

        if 'preprocess' in cfg.keys():  # preprocess within a purpose is optional
            transforms[key] = create_single_preprocess(cfg.preprocess)

    return transforms['train'], transforms['test']


def create_single_dataset(config: DictConfig, augment, preprocess, **extra_kwargs):
    ds_type = get_attr_from_module(config.module, config.type)
    ds = ds_type(**_cfg_kwargs(config), augment=augment, preprocess=preprocess, **extra_kwargs)
    return ds


def create_evaluator(config: DictConfig, net, train_dataloader, test_dataloader, metric, preprocess, postprocess):
    ev_type = get_attr_from_module(config.module, config.type)
    # For the current classifiers we use only the validation (test) dataloader
    ev = ev_type(
        **_cfg_kwargs(config),
        net=net,
        dataloader=test_dataloader,
        metric=metric,
        preprocess=preprocess,
        postprocess=postprocess,
    )
    return ev

def create_datasets(config: DictConfig, train_augment, train_preprocess, test_augment, test_preprocess):
    """
    Supports:
      data:
        train: {...}  # single train dataset
        test:  {...}  # single test dataset (legacy format)

    as well as:
      data:
        train: {...}
        test:
          incidents: {...}
          valA:      {...}

    In the latter case it returns:
      train_dataset: Dataset
      test_dataset: Dict[str, Dataset]
    """
    test_dataset = None
    train_dataset = None

    for purpose, cfg in config.items():
        if purpose not in DATASET_PURPOSES:
            raise RuntimeError(f'purpose must be one of {DATASET_PURPOSES}')

        if purpose == PURPOSE_TRAIN:
            # train is always single
            train_dataset = create_single_dataset(
                cfg,
                augment=train_augment,
                preprocess=train_preprocess,
            )

        elif purpose == PURPOSE_TEST:
            # A single test dataset (legacy format: module/type/config at the top level)
            if 'module' in cfg and 'type' in cfg:
                test_dataset = create_single_dataset(
                    cfg,
                    augment=test_augment,
                    preprocess=test_preprocess,
                )
            else:
                # Several test datasets: key -> sub-config
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
    net = net_type(**_cfg_kwargs(config))
    return net


def create_backbone(config: DictConfig):
    net_type = get_attr_from_module(config.module, config.type)
    net = net_type(**_cfg_kwargs(config))
    return net


def create_loss(config: DictConfig):
    losses = {}

    for cfg in config:
        name, one = list(cfg.items())[0]
        loss_type = get_attr_from_module(one.module, one.type)
        losses[name] = (loss_type(**_cfg_kwargs(one)), one.get('weight', 1.0))

    return losses


def create_metrics(config):

    metrics = {}
    if config is None:
        return metrics

    for cfg in config:
        name, one = list(cfg.items())[0]
        metrics[name] = create_universal(one)

    return metrics


def create_optimizer(config: DictConfig, params):
    opt_type = get_attr_from_module(config.module, config.type)
    opt_kwargs = _cfg_kwargs(config)
    trainable_only = bool(opt_kwargs.pop('trainable_only', False))
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    opt = opt_type(params=params, **opt_kwargs)
    return opt


def create_scheduler(config: DictConfig, optimizer: torch.optim.Optimizer):
    sch_type = get_attr_from_module(config.module, config.type)
    sch = sch_type(optimizer=optimizer, **_cfg_kwargs(config))
    return sch


def create_single_dataloader(config: DictConfig, dataset):
    dataloader_type = get_attr_from_module(config.module, config.type)
    return dataloader_type(dataset=dataset, **_cfg_kwargs(config))



def _pdeathsig_worker_init(worker_id, _user_fn=None):
    """In the worker: PDEATHSIG (dies together with its rank → won't be orphaned and won't
    hold /dev/shm/RAM) + ignore SIGINT, then the user's worker_init_fn, if any.

    SIGINT (Ctrl-C) goes to the WHOLE process group. If a worker dies from it first, the
    main process waiting for a batch in next(iterator) gets not a KeyboardInterrupt but
    "DataLoader worker exited unexpectedly" → a traceback instead of a clean shutdown.
    That's why the worker ignores SIGINT: the interrupt is handled by the main process,
    and the workers are then taken down by PDEATHSIG.

    IMPORTANT: this is a module-level function, NOT a closure — otherwise a DataLoader with
    ``multiprocessing_context='spawn'`` could not pickle it (regression 0.7.2)."""
    import signal
    ddp.set_pdeathsig()
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass
    if callable(_user_fn):
        _user_fn(worker_id)


def _worker_init_fn(user_fn=None):
    # partial of a module-level function — picklable (for spawn), unlike a closure.
    return partial(_pdeathsig_worker_init, _user_fn=user_fn)


def _resolve_collate(cfg: dict) -> dict:
    """If ``dataloaders.*.config.collate_fn`` is given as a COMPONENT (module/type/config),
    build it (``create_universal`` -> callable) and substitute it into the DataLoader kwargs.
    Needed for variable-size batching (graphs/sets, e.g. molecular docking): the default
    collate only stacks identically shaped tensors."""
    cf = cfg.get('collate_fn')
    if isinstance(cf, dict) and 'module' in cf and 'type' in cf:
        cfg['collate_fn'] = create_universal(OmegaConf.create(cf))
    return cfg


def _require_multipart_loader(dataset, loader):
    """MultiPartDataset is indexed by a (part, sample) TUPLE — it needs a MultiPartBatchSampler
    (provided by MultiPartDataLoader). With a plain torch DataLoader it receives int indices ->
    a cryptic "'int' object is not subscriptable" from the worker. Catch it upfront with a clear
    error (both train AND test)."""
    from echelon3.data.basic import MultiPartDataset
    if isinstance(dataset, MultiPartDataset):
        from echelon3.dataloaders.multipart import MultiPartBatchSampler
        if not isinstance(getattr(loader, 'batch_sampler', None), MultiPartBatchSampler):
            raise TypeError(
                "MultiPartDataset requires echelon3.dataloaders.multipart.MultiPartDataLoader "
                "(its (part, sample)-tuple index needs a MultiPartBatchSampler); a plain "
                "torch.utils.data.DataLoader sends int indices. Set the dataloader type to "
                "echelon3.dataloaders.multipart.MultiPartDataLoader.")


def create_dataloaders(config: DictConfig, train_dataset, test_dataset):
    """
    Supports:
      dataloaders:
        train: {...}
        test:  {...}

    and:
      dataloaders:
        train: {...}
        test:
          incidents: {...}
          valA:      {...}

    In the latter case test_dataloader will be a Dict[str, DataLoader],
    which the enhanced Trainer understands.
    """
    # train — always single
    train_dataloader_type = get_attr_from_module(config.train.module, config.train.type)
    train_cfg = OmegaConf.to_container(config.train.config, resolve=True) if 'config' in config.train else {}

    if ddp.is_ddp():
        # Config semantics are preserved: batch_size is global, we split it across ranks.
        world = ddp.world_size()
        global_bs = int(train_cfg.get('batch_size', 1))
        if global_bs % world != 0:
            raise ValueError(
                f'DDP: dataloaders.train.config.batch_size={global_bs} '
                f'is not divisible by world_size={world}'
            )
        train_cfg['batch_size'] = global_bs // world
        from echelon3.data.basic import MultiPartDataset
        if isinstance(train_dataset, MultiPartDataset):
            # MultiPartDataset is indexed by a (part, sample) TUPLE — the int-indexed
            # DistributedSampler is incompatible. Rank sharding is done by the DDP-aware
            # MultiPartBatchSampler (inside MultiPartDataLoader), so we don't set a sampler;
            # shuffle is also its job — we drop it so it doesn't conflict with batch_sampler.
            train_cfg.pop('shuffle', None)
            print(f'--> DDP dataloader: MultiPartDataset — per-part rank-sharding via '
                  f'MultiPartBatchSampler; per-process batch {train_cfg["batch_size"]} x {world} = {global_bs}')
        else:
            # shuffle is provided by DistributedSampler (mutually exclusive with shuffle=True)
            shuffle = bool(train_cfg.pop('shuffle', True))
            train_cfg['sampler'] = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=shuffle, drop_last=bool(train_cfg.get('drop_last', False))
            )
            print(f'--> DDP dataloader: global batch {global_bs} = '
                  f'{train_cfg["batch_size"]}/process x {world} processes '
                  f'(num_workers={train_cfg.get("num_workers", 0)} per process)')

    if int(train_cfg.get('num_workers', 0) or 0) > 0:
        train_cfg['worker_init_fn'] = _worker_init_fn(train_cfg.get('worker_init_fn'))
        # By default we keep workers alive between epochs. Otherwise they are respawned
        # every epoch, and Ctrl-C at an epoch boundary catches them mid-bootstrap (spawn:
        # import torch / pickle.load BEFORE our worker_init) — 4 KeyboardInterrupt tracebacks
        # + leaked semaphores from half-initialized processes. setdefault — we don't touch an
        # explicit user value; this branch guarantees num_workers>0 (otherwise torch fails).
        train_cfg.setdefault('persistent_workers', True)
    _resolve_collate(train_cfg)
    train_dataloader = train_dataloader_type(dataset=train_dataset, **train_cfg)

    _require_multipart_loader(train_dataset, train_dataloader)

    def _test_cfg(sub_cfg, dataset):
        cfg = OmegaConf.to_container(sub_cfg, resolve=True) if sub_cfg is not None else {}
        if ddp.is_ddp():
            # Symmetric validation: each rank computes its own shard, and metrics
            # are aggregated by torchmetrics (dist_reduce_fx). Workers via spawn:
            # without workers, eval
            # prepares data on a single thread (minutes + every rank waits for the
            # slowest one at the metric-sync barrier).
            # fork workers: spawn can't pickle the dataset (cv2.CLAHE inside),
            # while fork with NCCL is empirically safe — train loaders work this way.
            cfg['num_workers'] = min(int(cfg.get('num_workers', 4)), 4)
            cfg.pop('shuffle', None)
            # config batch_size is global (in DP it was split across cards by
            # DataParallel itself; a full batch of 40 on one card = OOM in interpolate).
            cfg['batch_size'] = max(1, int(cfg.get('batch_size', 1)) // ddp.world_size())
            from echelon3.data.basic import MultiPartDataset
            if not isinstance(dataset, MultiPartDataset):     # MultiPart shards via its own batch_sampler
                cfg['sampler'] = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        if int(cfg.get('num_workers', 0) or 0) > 0:
            cfg['worker_init_fn'] = _worker_init_fn(cfg.get('worker_init_fn'))
            # We do NOT force persistent_workers for eval: validation is not a tight per-epoch
            # loop (the "Ctrl-C at an epoch boundary" motivation is weak here), and resident
            # eval workers alongside train workers for the whole run (default 4 under DDP)
            # hold RAM for nothing. The user can enable it manually if needed.
        _resolve_collate(cfg)
        return cfg

    # test — one or several
    if isinstance(test_dataset, dict):
        # several test datasets
        test_dataloaders = {}
        for name, ds in test_dataset.items():
            sub_cfg = config.test[name]
            test_dataloader_type = get_attr_from_module(sub_cfg.module, sub_cfg.type)
            test_dataloaders[name] = test_dataloader_type(dataset=ds, **_test_cfg(sub_cfg.config if 'config' in sub_cfg else None, ds))
            _require_multipart_loader(ds, test_dataloaders[name])
        return train_dataloader, test_dataloaders
    else:
        # single test dataset (legacy format)
        test_dataloader_type = get_attr_from_module(config.test.module, config.test.type)
        test_dataloader = test_dataloader_type(dataset=test_dataset, **_test_cfg(config.test.config if 'config' in config.test else None, test_dataset))
        _require_multipart_loader(test_dataset, test_dataloader)
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
                       **_cfg_kwargs(config))
    return trn

def create_tabular_datasets(config: DictConfig):
    """Tabular datasets for the fit/predict branch: ordinary components (create_universal),
    WITHOUT injecting image augment/preprocess. Supports one or several (a dict of named)
    test datasets, just like the image-based create_datasets."""
    train_dataset = create_universal(config.train)
    if 'test' not in config:
        return train_dataset, None
    test_cfg = config.test
    if 'module' in test_cfg and 'type' in test_cfg:
        return train_dataset, create_universal(test_cfg)
    test_dataset = {name: create_universal(sub) for name, sub in test_cfg.items()}
    return train_dataset, test_dataset


def create_estimator_trainer(config: DictConfig, model, train_data, test_data, metrics,
                             ckpt_manager, feature_transform=None):
    """Fit/predict trainer (EstimatorTrainer and kin): its own injections — model + whole
    datasets + metrics + ckpt manager (+ optional feature_transform), without
    optimizer/loss/loaders/scheduler."""
    trainer_type = get_attr_from_module(config.module, config.type)
    return trainer_type(model=model, train_data=train_data, test_data=test_data,
                        metrics=metrics, ckpt_manager=ckpt_manager,
                        feature_transform=feature_transform, **_cfg_kwargs(config))


def create_wrapper(config: DictConfig, net):
    ev_type = get_attr_from_module(config.module, config.type)
    ev = ev_type(**_cfg_kwargs(config), core=net)
    return ev

def create_exporters(config, net):

    preprocess = create_single_preprocess(config.preprocess) if 'preprocess' in config.keys() else torch.nn.Identity()
    postprocess = create_single_preprocess(config.postprocess) if 'postprocess' in config.keys() else torch.nn.Identity()

    exporters = {}

    for name, cfg in config.exporters.items():
        ex_type = get_attr_from_module(cfg.module, cfg.type)
        ex = ex_type(**_cfg_kwargs(cfg), net=net, preprocess=preprocess, postprocess=postprocess)
        exporters[name] = ex

    return exporters


def create_constructor(config, components):

    net_type = get_attr_from_module(config.module, config.type)
    net = net_type(**_cfg_kwargs(config), **components)

    return net


def create_batch_sampler(config: DictConfig, **kwargs):
    par_type = get_attr_from_module(config.module, config.type)
    kw = _cfg_kwargs(config)
    if inspect.isclass(par_type):
        return par_type(**kw, **kwargs)
    return partial(par_type, **kw, **kwargs) if kw else par_type


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
