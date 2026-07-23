"""_close_quietly kills workers before the hard os._exit, but does NOT hang: if close()
stalls (e.g. pin_memory_thread.join without a timeout), the exit is bounded by a timeout."""
import signal
import time

import pytest

from echelon3.cli import _close_quietly, _silence_sigint


@pytest.fixture(autouse=True)
def _restore_sigint():
    """_silence_sigint() globally sets SIGINT=SIG_IGN. We snapshot it before and restore it
    after EVERY test — otherwise the state leaks into the rest of the pytest session (Ctrl-C on
    a hung run would stop working, and KeyboardInterrupt tests would break silently)."""
    prev = signal.getsignal(signal.SIGINT)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, prev)


def test_close_quietly_is_time_bounded():
    class SlowTrainer:
        closed = False
        def close(self):
            time.sleep(30)          # emulate a stalled _shutdown_workers
            self.closed = True

    t0 = time.time()
    _close_quietly(SlowTrainer(), timeout=1.0)
    assert time.time() - t0 < 4.0   # don't wait 30s — bounded by the timeout


def test_is_sigint_worker_death_matches_only_sigint():
    from echelon3.cli import _is_sigint_worker_death
    assert _is_sigint_worker_death(RuntimeError("DataLoader worker (pid 1) is killed by signal: Interrupt. "))
    assert _is_sigint_worker_death(RuntimeError("worker (pid 2) is killed by signal: SIGINT"))
    # NOT an interrupt: OOM-SIGKILL / segfault / generic exit / ordinary errors
    assert not _is_sigint_worker_death(RuntimeError("worker (pid 3) is killed by signal: Killed. "))
    assert not _is_sigint_worker_death(RuntimeError("worker (pid 4) is killed by signal: Segmentation fault. "))
    assert not _is_sigint_worker_death(RuntimeError("DataLoader worker (pid(s) 5) exited unexpectedly"))
    assert not _is_sigint_worker_death(ValueError("shape mismatch"))


def test_silence_sigint_ignores_further_sigint():
    """_silence_sigint() sets SIGINT=SIG_IGN so that a repeated Ctrl-C in the interrupt
    handler does not steer the thread past os._exit into finally -> ddp.shutdown() ->
    destroy_process_group() -> NCCL hang (30s). Restoration happens in the autouse fixture."""
    assert signal.getsignal(signal.SIGINT) != signal.SIG_IGN  # start: not silenced
    _silence_sigint()
    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN


def test_looks_like_interrupt_flag_gates_ambiguous():
    """'exited unexpectedly' (a common form of a worker's SIGINT death on recent torch) is
    indistinguishable from OOM by text — we treat it as an interrupt ONLY if there was a SIGINT
    (flag). An explicit 'killed by signal: Interrupt' is always an interrupt; OOM without the
    flag is a real error."""
    from echelon3 import cli
    exited = RuntimeError("DataLoader worker (pid(s) 5) exited unexpectedly")
    killed_int = RuntimeError("DataLoader worker (pid 1) is killed by signal: Interrupt. ")
    oom = RuntimeError("DataLoader worker (pid 2) is killed by signal: Killed. ")
    shape = ValueError("shape mismatch")

    prev = cli._SIGINT_SEEN
    try:
        cli._SIGINT_SEEN = False                       # no Ctrl-C happened
        assert not cli._looks_like_interrupt(exited)   # indistinguishable from OOM -> don't swallow
        assert cli._looks_like_interrupt(killed_int)   # explicit SIGINT
        assert not cli._looks_like_interrupt(oom)      # OOM -> real error

        cli._SIGINT_SEEN = True                        # Ctrl-C happened
        assert cli._looks_like_interrupt(exited)       # worker death on Ctrl-C -> interrupt
        assert cli._looks_like_interrupt(killed_int)
        assert not cli._looks_like_interrupt(shape)    # non-worker error -> don't swallow even with the flag
        assert not cli._looks_like_interrupt(RuntimeError("CUDA out of memory"))  # RuntimeError, but not a worker
    finally:
        cli._SIGINT_SEEN = prev


def test_close_quietly_calls_close_and_swallows_errors():
    calls = []

    class OkTrainer:
        def close(self):
            calls.append(1)

    class BoomTrainer:
        def close(self):
            raise RuntimeError("boom")

    _close_quietly(OkTrainer(), timeout=5.0)
    _close_quietly(BoomTrainer(), timeout=5.0)   # does not propagate
    assert calls == [1]
