import os
import sys


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
    launch_cfg = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=len(gpus),
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        run_id="echelon3",
        start_method="spawn",
    )
    # Резолвим интерполяции в родителе, чтобы все воркеры получили идентичный cfg.
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    elastic_launch(launch_cfg, train_fn)(resolved)
    return True
