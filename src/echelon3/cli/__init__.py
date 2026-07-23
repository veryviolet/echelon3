import os
import sys


def setup_warnings():
    """Silence warnings inline (so they don't break progress bars) and collect them —
    a short summary is printed in echelon3's standard format before each validation
    (see echelon3.warncollect.flush in the trainer). Idempotent; called at the start of
    every CLI."""
    from echelon3 import warncollect
    warncollect.install()


def add_cwd_to_sys_path():
    """Make modules in the current directory importable from configs.

    Working model for zoo repositories: the user works from the root of their
    repo, and configs reference local packages (module: my_zoo.nets.foo).
    Running `python script.py` puts the script's directory on sys.path automatically,
    a console-script does not; we align the behavior.
    """
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)


def resolve_gpus(cfg):
    """List of GPUs to run on: cfg.gpus if set, otherwise ALL GPUs visible on the node."""
    import torch
    if "gpus" in cfg and cfg.gpus is not None:
        return [int(g) for g in cfg.gpus]
    return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []


def build_cli(app_fn):
    """click command in place of ``@hydra.main``: loads an OmegaConf config + Hydra-like
    overrides (``key=`` / ``+`` / ``++`` / ``~``) and calls ``app_fn(cfg)``.

    Interruption (Ctrl-C) during the load/setup stage exits cleanly, without a traceback
    (during training itself, Ctrl-C is caught deeper, in the trainer loop)."""
    import click
    from echelon3.cli.config import load_config

    @click.command(context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
        "help_option_names": ["-h", "--help"],
    })
    @click.option("--config-dir", "-cd", default=".", show_default=True,
                  help="Directory to load the config from.")
    @click.option("--config-name", "-cn", required=True,
                  help="Config file name (with or without .yaml).")
    @click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
    def _cmd(config_dir, config_name, overrides):
        add_cwd_to_sys_path()
        try:
            cfg = load_config(config_name, config_dir, overrides)
            app_fn(cfg)
        except KeyboardInterrupt:
            print("\n--> Interrupted by user (Ctrl-C).", file=sys.stderr)
            sys.exit(130)

    return _cmd


def resolve_single_device(cfg, cuda_available: bool):
    """Device for a NON-DDP run.

    Priority: ``cfg.device=cpu`` (or CUDA unavailable) → CPU; otherwise, if ``gpus``
    is set, we bind to the SPECIFIC card ``gpus[0]`` (previously single-device mode
    ignored the index and always took cuda:0 — a silent collision on a shared node and
    a violation of the GPU 0 reservation); otherwise ``cfg.device``. The DDP path
    already respects the index (CUDA_VISIBLE_DEVICES for workers). Returns a
    ``torch.device``."""
    import torch
    dev = str(cfg.device) if 'device' in cfg.keys() else 'cuda'
    if dev.startswith('cpu') or not cuda_available:
        return torch.device('cpu')
    if 'gpus' in cfg and cfg.gpus is not None and len(cfg.gpus) >= 1:
        return torch.device(f'cuda:{int(cfg.gpus[0])}')
    return torch.device(dev)


def maybe_launch_ddp(cfg, train_fn) -> bool:
    """Built-in single-node DDP launcher (replacement for torchrun).

    If >1 GPU is requested and we are not a worker yet, spawns one process per
    GPU via torch's standard elastic_launch and returns True. Each worker
    receives RANK/LOCAL_RANK/WORLD_SIZE/MASTER_* and the already-resolved cfg, then
    proceeds along the usual path train_fn(cfg) → ddp.init_ddp_if_needed() enables DDP.

    Returns False if there is nothing to launch (1 GPU / CPU / already under a launcher
    or an external torchrun — then the caller calls train_fn(cfg) itself).
    """
    if "RANK" in os.environ:  # already a worker (our spawn or an external torchrun)
        return False
    # device=cpu forces CPU: don't bring up DDP even on a multi-GPU node.
    if "device" in cfg and str(cfg.device).startswith("cpu"):
        return False

    import torch
    gpus = resolve_gpus(cfg)
    if not torch.cuda.is_available() or len(gpus) <= 1:
        return False

    # An explicit GPU list limits the workers' visibility; workers are fresh processes
    # (spawn), so CUDA_VISIBLE_DEVICES is read by them anew.
    if "gpus" in cfg and cfg.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)

    from omegaconf import OmegaConf
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch

    print(f"--> DDP: launching {len(gpus)} worker(s) on GPUs {gpus} (no torchrun needed)")

    # OOM footgun: num_workers/prefetch_factor are PER-RANK, and the launcher places all
    # ranks on a single node, so node RAM ~ ranks × num_workers × prefetch ×
    # batch. A bloated product is a common cause of RAM-OOM (and of a "silent" hang).
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
            # persistent_workers for train defaults to true when num_workers>0 (done by
            # create_dataloaders at runtime — the key may be absent from the config). We inform
            # the user when the flag is effectively on (not explicitly disabled).
            if _dl.get("persistent_workers") is not False:
                print("--> persistent_workers is on (engine default at num_workers>0): train "
                      "workers stay alive between epochs — fewer respawns and less Ctrl-C "
                      "spawn-time noise, but watch node RAM / orphaned processes on an unclean "
                      "stop (set dataloaders.train.config.persistent_workers: false to opt out).")
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
        # Fail-fast: don't restart a rank that died from OOM into an already-broken group
        # (default 3 → silent retry + a stuck re-rendezvous).
        max_restarts=0,
    )
    # Resolve interpolations in the parent so all workers get an identical cfg.
    resolved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
    # On SIGINT/SIGTERM (Ctrl-C went to the whole group) elastic raises SignalException —
    # we catch it as a normal interruption, not as a failure, to avoid a traceback.
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
        # A worker died (a common cause is RAM/CUDA-OOM from num_workers × prefetch ×
        # ranks). We report it explicitly instead of exiting silently on someone else's traceback.
        print("--> DDP: a worker died (see the rank traceback above). Common cause under "
              "DDP: OOM from dataloaders.*.config.num_workers / prefetch_factor.",
              file=sys.stderr)
        raise
    return True


def _is_sigint_worker_death(exc) -> bool:
    """Ctrl-C goes to the whole process group; sometimes SIGINT kills a DataLoader worker
    EARLIER than it managed to set SIG_IGN (our _pdeathsig_worker_init) — then the main process
    receives not a KeyboardInterrupt but torch's RuntimeError
    'DataLoader worker (pid ...) is killed by signal: Interrupt' (+ then C++ terminate →
    Fatal Python error: Aborted). This is a user interruption, not a failure — we recognize it
    to exit CLEANLY (exit 130) instead of printing a scary traceback / aborting.
    A worker death from ANOTHER signal (SIGKILL/SIGSEGV — OOM, a real crash) is not caught here."""
    s = str(exc)
    return isinstance(exc, RuntimeError) and 'killed by signal' in s and \
        ('Interrupt' in s or 'SIGINT' in s)


_SIGINT_SEEN = False


def _install_sigint_flag():
    """In the rank's main process we flag the fact of Ctrl-C and raise KeyboardInterrupt
    (standard behavior — like the default handler). The flag is needed to distinguish a
    DataLoader worker death FROM Ctrl-C from a genuine worker crash: on recent torch a SIGINT
    worker death often surfaces as 'DataLoader worker ... exited unexpectedly' (textually
    indistinguishable from OOM), but with the flag set it is specifically an interruption."""
    import signal

    def _handler(signum, frame):
        global _SIGINT_SEEN
        _SIGINT_SEEN = True
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handler)
    except Exception:
        pass


def _looks_like_interrupt(exc) -> bool:
    """True if the exception is a consequence of Ctrl-C: torch explicitly named the signal
    Interrupt/SIGINT, OR there was a SIGINT (flag) and it is a DataLoader worker death
    (including 'exited unexpectedly'). Without the flag, 'exited unexpectedly' is NOT
    considered an interruption (it could have been OOM/segfault)."""
    if _is_sigint_worker_death(exc):
        return True
    if _SIGINT_SEEN and isinstance(exc, RuntimeError):
        s = str(exc)
        return 'DataLoader worker' in s and ('exited unexpectedly' in s or 'killed by signal' in s)
    return False


def _silence_sigint():
    """Ignore repeated Ctrl-C during shutdown. Called FIRST in the interrupt handler —
    before print/flush/close/os._exit.

    Otherwise a second SIGINT (the elastic agent re-sends it to the ranks) would raise
    KeyboardInterrupt again — now INSIDE our handler, past the following os._exit(130):
    control would go to finally → ddp.shutdown()=destroy_process_group() → NCCL teardown
    deadlock → the rank hangs ~30s until a force SIGKILL (workers finished off by PDEATHSIG,
    semaphores leaking). SIG_IGN guarantees the path to os._exit is not interrupted. Idempotent.
    signal.signal is valid only in the main thread — we swallow errors (e.g. a call from non-main)."""
    import signal
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


def _close_quietly(trainer, timeout=15.0):
    """Best-effort graceful teardown of DataLoader workers before exit: frees their
    semaphores and /dev/shm (otherwise, after a hard os._exit, PDEATHSIG-SIGKILL finishes them
    off without cleanup, and the launcher's resource_tracker reports "leaked semaphore objects").

    We run it in a daemon thread with join(timeout): if close() hangs somewhere (e.g.
    _MultiProcessingDataLoaderIter._shutdown_workers waits on pin_memory_thread.join() WITHOUT
    a timeout), we won't delay the hard exit longer than timeout — the hung thread is finished
    off by os._exit itself. Idempotent; we swallow errors."""
    import threading

    def _run():
        try:
            trainer.close()
        except Exception:
            pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
