"""Поддержка DistributedDataParallel (DDP).

Активация автоматическая — по переменным окружения torchrun (RANK/WORLD_SIZE/
LOCAL_RANK). Запуск:

    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
        echelon3_train.py --config-name <config>

Без torchrun ничего не меняется: тренер работает через DataParallel, как раньше.

Семантика конфига сохранена: dataloaders.train.config.batch_size — ГЛОБАЛЬНЫЙ
батч (как в DataParallel); при DDP он делится на world_size (см. creator.
create_dataloaders). Чекпойнты формата DataParallel/DDP взаимозаменяемы: оба
пишут state_dict с префиксом "module.".

Валидация в DDP исполняется только на rank 0 (через развёрнутую сеть, без
коллективов), остальные ранки ждут на barrier; сохранение чекпойнтов — только
rank 0. Так keep-best логика остаётся байт-в-байт прежней при любых метриках.
"""
import os
import signal
import sys
from datetime import timedelta

import torch
import torch.distributed as dist


def set_pdeathsig():
    """Linux: текущий процесс получает SIGKILL, как только умирает его родитель.

    Ставим в ранге (родитель — агент лаунчера) и в DataLoader-воркерах (родитель —
    ранг), чтобы дерево процессов не осиротевало при os._exit / SIGKILL / краше
    предка (иначе воркеры висят, держат /dev/shm и RAM, а новый запуск зависает на
    первом батче). Best-effort, только Linux."""
    if sys.platform != "linux":
        return
    try:
        import ctypes
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
        # Гонка: родитель мог умереть до prctl — тогда мы уже репарентнуты на init.
        if os.getppid() == 1:
            os._exit(1)
    except Exception:
        pass

# Таймаут группы: бэкстоп на случай, когда ранг завис (не вышел) и elastic его не
# снимает. Дефолт щедрый (валидация/большие шаги), но конфигурируемый — уменьшите
# ECHELON3_DDP_TIMEOUT_MIN, чтобы «тихий» вис при рассинхроне падал быстрее.
_PG_TIMEOUT = timedelta(minutes=int(os.environ.get("ECHELON3_DDP_TIMEOUT_MIN", "60")))


def ddp_env_present() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_ddp_if_needed() -> bool:
    """Инициализирует process group при запуске под torchrun. Возвращает is_ddp()."""
    if ddp_env_present() and not dist.is_initialized():
        # Ранг умирает вместе с агентом лаунчера (не осиротевает при его SIGKILL).
        set_pdeathsig()
        # NCCL watchdog: аборт (а не молчаливое ожидание) при ошибке/рассинхроне +
        # отчёт, какой ранг расклеился. setdefault — юзер может переопределить.
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_DESYNC_DEBUG", "1")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        pg_kwargs = dict(backend=backend, timeout=_PG_TIMEOUT)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank())
            # device_id глушит c10d-warning "barrier(): using the device under
            # current context" в источнике (параметр есть в свежем torch).
            pg_kwargs["device_id"] = torch.device("cuda", local_rank())
        try:
            dist.init_process_group(**pg_kwargs)
        except TypeError:
            pg_kwargs.pop("device_id", None)  # старый torch без device_id
            dist.init_process_group(**pg_kwargs)
    return is_ddp()


def shutdown():
    # БЕЗ barrier: shutdown зовётся и на аварийном пути (finally), когда другие
    # ранки могут быть в несовпадающих коллективах — barrier тут даёт дедлок
    # и прячет исходный traceback.
    if is_ddp():
        dist.destroy_process_group()


def is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_ddp() else 0


def world_size() -> int:
    return dist.get_world_size() if is_ddp() else 1


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main() -> bool:
    return rank() == 0


def barrier():
    if is_ddp():
        dist.barrier()


def unwrap(net: torch.nn.Module) -> torch.nn.Module:
    """Исходная сеть под обёртками DDP/DataParallel и torch.compile
    (``OptimizedModule._orig_mod``), снятыми в любом порядке."""
    for _ in range(4):  # страховка от неожиданной вложенности
        if isinstance(net, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            net = net.module
        elif hasattr(net, "_orig_mod"):  # torch.compile OptimizedModule
            net = net._orig_mod
        else:
            break
    return net


def state_dict_for_save(net: torch.nn.Module) -> dict:
    """State dict БЕЗ префикса 'module.' — чекпоинты не зависят от обёртки
    (DDP/одиночный процесс дают одинаковый файл)."""
    return unwrap(net).state_dict()


def load_state_dict_flexible(net: torch.nn.Module, state_dict: dict, strict: bool = True):
    """Грузит веса в развёрнутый модуль, снимая с ключей префиксы обёрток —
    'module.' (DataParallel/DDP) и '_orig_mod.' (torch.compile), в любом порядке
    и вложенности, так что чекпоинты взаимозаменяемы между обёрнутыми и голыми
    прогонами."""
    _prefixes = ("module.", "_orig_mod.")

    def _strip(k: str) -> str:
        changed = True
        while changed:
            changed = False
            for p in _prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        return k

    if any(any(k.startswith(p) for p in _prefixes) for k in state_dict):
        state_dict = {_strip(k): v for k, v in state_dict.items()}
    return unwrap(net).load_state_dict(state_dict, strict=strict)
