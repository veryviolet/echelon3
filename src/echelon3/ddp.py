"""DistributedDataParallel (DDP) support.

Activation is automatic — driven by the RANK/WORLD_SIZE/LOCAL_RANK environment
variables. They are set either by the built-in launcher (`echelon3 train ...
gpus=[0,1,2,3]` — elastic_launch spawns one process per GPU, torchrun is not
needed) or by an external torchrun/srun for multi-node runs. A single GPU or CPU
runs in-process without any wrapper (DataParallel was removed in 0.5.0).

Config semantics are preserved: dataloaders.train.config.batch_size is the GLOBAL
batch; under DDP it is divided by world_size (see creator.create_dataloaders).
Checkpoints store the UNWRAPPED state_dict (without the "module." prefix); older
files with "module." load correctly — the prefix is stripped automatically.

Validation under DDP is SYMMETRIC: each rank computes its own shard
(DistributedSampler) through the unwrapped network, and metrics are aggregated
across ranks (torchmetrics do this themselves; custom Metrics via the
dist_reduce() hook). Checkpoints are saved only by rank 0.
"""
import os
import signal
import sys
from datetime import timedelta

import torch
import torch.distributed as dist


def set_pdeathsig():
    """Linux: the current process gets SIGKILL as soon as its parent dies.

    We set this in the rank (parent is the launcher agent) and in DataLoader
    workers (parent is the rank) so the process tree is not orphaned on os._exit /
    SIGKILL / a crash of the ancestor (otherwise workers hang, hold /dev/shm and
    RAM, and a new run stalls on the first batch). Best-effort, Linux only."""
    if sys.platform != "linux":
        return
    try:
        import ctypes
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
        # Race: the parent may have died before prctl — then we are already reparented to init.
        if os.getppid() == 1:
            os._exit(1)
    except Exception:
        pass

# Process group timeout: a backstop for when a rank hangs (does not exit) and
# elastic does not reap it. The default is generous (validation/large steps) but
# configurable — lower ECHELON3_DDP_TIMEOUT_MIN so a "silent" hang on a desync
# fails faster.
_PG_TIMEOUT = timedelta(minutes=int(os.environ.get("ECHELON3_DDP_TIMEOUT_MIN", "60")))


def ddp_env_present() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_ddp_if_needed() -> bool:
    """Initializes the process group when running under torchrun. Returns is_ddp()."""
    if ddp_env_present() and not dist.is_initialized():
        # The rank dies together with the launcher agent (not orphaned on its SIGKILL).
        set_pdeathsig()
        # NCCL watchdog: abort (instead of silently waiting) on an error/desync, plus
        # a report of which rank fell out of sync. setdefault — the user may override.
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        os.environ.setdefault("TORCH_NCCL_DESYNC_DEBUG", "1")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        pg_kwargs = dict(backend=backend, timeout=_PG_TIMEOUT)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank())
            # device_id silences the c10d warning "barrier(): using the device under
            # current context" at the source (the parameter exists in recent torch).
            pg_kwargs["device_id"] = torch.device("cuda", local_rank())
        try:
            dist.init_process_group(**pg_kwargs)
        except TypeError:
            pg_kwargs.pop("device_id", None)  # old torch without device_id
            dist.init_process_group(**pg_kwargs)
    return is_ddp()


def shutdown():
    # NO barrier: shutdown is also called on the failure path (finally), when other
    # ranks may be in mismatched collectives — a barrier here would deadlock and hide
    # the original traceback.
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
    """The underlying network with the DDP/DataParallel and torch.compile
    (``OptimizedModule._orig_mod``) wrappers stripped, in any order."""
    for _ in range(4):  # safeguard against unexpected nesting
        if isinstance(net, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            net = net.module
        elif hasattr(net, "_orig_mod"):  # torch.compile OptimizedModule
            net = net._orig_mod
        else:
            break
    return net


def state_dict_for_save(net: torch.nn.Module) -> dict:
    """State dict WITHOUT the 'module.' prefix — checkpoints do not depend on the
    wrapper (DDP and single-process runs produce the same file)."""
    return unwrap(net).state_dict()


def load_state_dict_flexible(net: torch.nn.Module, state_dict: dict, strict: bool = True):
    """Loads weights into the unwrapped module, stripping wrapper prefixes from the
    keys — 'module.' (DataParallel/DDP) and '_orig_mod.' (torch.compile), in any
    order and nesting, so that checkpoints are interchangeable between wrapped and
    bare runs."""
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
