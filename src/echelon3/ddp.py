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
from datetime import timedelta

import torch
import torch.distributed as dist

# Дефолтные 10 минут NCCL-вотчдога малы: пока rank 0 валидируется, остальные
# ранки ждут на barrier дольше таймаута на больших тестах.
_PG_TIMEOUT = timedelta(minutes=60)


def ddp_env_present() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_ddp_if_needed() -> bool:
    """Инициализирует process group при запуске под torchrun. Возвращает is_ddp()."""
    if ddp_env_present() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=_PG_TIMEOUT)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank())
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
    """Underlying module для DataParallel/DDP, иначе сама сеть."""
    if isinstance(net, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return net.module
    return net
