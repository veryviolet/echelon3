import os
import sys


def setup_warnings():
    """Глушить предупреждения инлайн (чтобы не рвали прогрессбары) и копить их —
    краткое саммари печатается штатным форматом echelon3 перед каждой валидацией
    (см. echelon3.warncollect.flush в трейнере). Идемпотентно; зовётся в начале
    каждого CLI."""
    from echelon3 import warncollect
    warncollect.install()


def add_cwd_to_sys_path():
    """Делает модули текущего каталога импортируемыми из конфигов.

    Рабочая модель zoo-репозиториев: пользователь работает из корня своего
    репо, конфиги ссылаются на локальные пакеты (module: my_zoo.nets.foo).
    Запуск `python script.py` кладёт каталог скрипта в sys.path автоматически,
    console-script — нет; выравниваем поведение.
    """
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def resolve_gpus(cfg):
    """Список GPU для запуска: cfg.gpus, если задан, иначе ВСЕ видимые на ноде."""
    import torch
    if "gpus" in cfg and cfg.gpus is not None:
        return [int(g) for g in cfg.gpus]
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


def resolve_single_device(cfg, cuda_available: bool):
    """Устройство для НЕ-DDP запуска.

    Приоритет: ``cfg.device=cpu`` (или CUDA недоступна) → CPU; иначе, если задан
    ``gpus``, садимся на КОНКРЕТНУЮ карту ``gpus[0]`` (раньше одиночный режим
    игнорировал индекс и всегда брал cuda:0 — тихая коллизия на шаренной ноде и
    нарушение резерва GPU 0); иначе — ``cfg.device``. DDP-путь индекс уже уважает
    (CUDA_VISIBLE_DEVICES для воркеров). Возврат — ``torch.device``."""
    import torch
    dev = str(cfg.device) if 'device' in cfg.keys() else 'cuda'
    if dev.startswith('cpu') or not cuda_available:
        return torch.device('cpu')
    if 'gpus' in cfg and cfg.gpus is not None and len(cfg.gpus) >= 1:
        return torch.device(f'cuda:{int(cfg.gpus[0])}')
    return torch.device(dev)


def maybe_launch_ddp(cfg, train_fn) -> bool:
    """Встроенный однонодовый DDP-лаунчер (замена torchrun).

    Если запрошено >1 GPU и мы ещё не воркер — порождает по одному процессу на
    GPU через штатный elastic_launch torch и возвращает True. Каждый воркер
    получает RANK/LOCAL_RANK/WORLD_SIZE/MASTER_* и уже разобранный cfg, после
    чего идёт обычным путём train_fn(cfg) → ddp.init_ddp_if_needed() включает DDP.

    Возвращает False, если запускать нечего (1 GPU / CPU / уже под лаунчером или
    внешним torchrun — тогда caller зовёт train_fn(cfg) сам).
    """
    if "RANK" in os.environ:  # уже воркер (наш спавн или внешний torchrun)
        return False

    import torch
    gpus = resolve_gpus(cfg)
    if not torch.cuda.is_available() or len(gpus) <= 1:
        return False

    # Явный список GPU ограничивает видимость воркеров; воркеры — свежие процессы
    # (spawn), поэтому CUDA_VISIBLE_DEVICES читается ими заново.
    if "gpus" in cfg and cfg.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)

    from omegaconf import OmegaConf
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch

    print(f"--> DDP: launching {len(gpus)} worker(s) on GPUs {gpus} (no torchrun needed)")

    # OOM-footgun: num_workers/prefetch_factor — ПЕР-РАНК, а лаунчер сажает все
    # ранги на одну ноду, поэтому RAM ноды ~ ранги × num_workers × prefetch ×
    # батч. Раздутое произведение — частая причина RAM-OOM (и «тихого» виса).
    try:
        _dl = cfg.dataloaders.train.config
        _nw = int(_dl.get("num_workers", 0) or 0)
        if _nw > 0:
            _pf = int(_dl.get("prefetch_factor", 2) or 2)
            _total = len(gpus) * _nw * _pf
            print(f"--> DDP dataloader RAM: {len(gpus)} ranks × {_nw} workers × "
                  f"{_pf} prefetch = {_total} batches prefetched on this node")
            if _total > 64:
                print(f"--> WARNING: {_total} prefetched batches on one node — "
                      "RAM-OOM risk; lower dataloaders.train.config.num_workers / "
                      "prefetch_factor if the run crashes or hangs.")
            _cores = os.cpu_count() or 0
            if _cores and len(gpus) * _nw > _cores:
                print(f"--> WARNING: {len(gpus)}×{_nw}={len(gpus) * _nw} DataLoader workers "
                      f"> {_cores} cores — CPU over-subscription; lower num_workers "
                      "(rule of thumb: cores / ranks per rank).")
            if _dl.get("persistent_workers"):
                print("--> WARNING: persistent_workers=true keeps workers alive between "
                      "epochs — higher risk of orphaned processes on an unclean stop.")
    except Exception:
        pass

    launch_cfg = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=len(gpus),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        run_id="echelon3",
        start_method="spawn",
        # Fail-fast: не перезапускать сдохший от OOM ранг в уже расклеенную группу
        # (дефолт 3 → тихий ретрай + зависшее ре-рандеву).
        max_restarts=0,
    )
    # Резолвим интерполяции в родителе, чтобы все воркеры получили идентичный cfg.
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
    # elastic на SIGINT/SIGTERM (Ctrl-C ушёл всей группе) кидает SignalException —
    # ловим её как штатное прерывание, а не как падёж, чтобы не было traceback.
    _interrupt = (KeyboardInterrupt,)
    try:
        from torch.distributed.elastic.multiprocessing.api import SignalException
        _interrupt = (KeyboardInterrupt, SignalException)
    except Exception:
        pass
    try:
        elastic_launch(launch_cfg, train_fn)(resolved)
    except _interrupt:
        print('\n--> Interrupted by user (Ctrl-C), workers stopped.', file=sys.stderr)
        raise SystemExit(130)
    except ChildFailedError:
        # Воркер умер (частая причина — RAM/CUDA-OOM от num_workers × prefetch ×
        # ранги). Сообщаем явно, а не выходим молча по чужому traceback.
        print("--> DDP: a worker died (see the rank traceback above). Common cause under "
              "DDP: OOM from dataloaders.*.config.num_workers / prefetch_factor.",
              file=sys.stderr)
        raise
    return True
