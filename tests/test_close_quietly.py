"""_close_quietly гасит воркеров перед жёстким os._exit, но НЕ виснет: если close()
подвиснет (напр. pin_memory_thread.join без таймаута), выход ограничен timeout'ом."""
import signal
import time

import pytest

from echelon3.cli import _close_quietly, _silence_sigint


@pytest.fixture(autouse=True)
def _restore_sigint():
    """_silence_sigint() глобально ставит SIGINT=SIG_IGN. Снимаем снимок до и восстанавливаем
    после КАЖДОГО теста — иначе состояние протечёт в остальную pytest-сессию (Ctrl-C по
    зависшему прогону перестанет работать, а тесты на KeyboardInterrupt молча испортятся)."""
    prev = signal.getsignal(signal.SIGINT)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, prev)


def test_close_quietly_is_time_bounded():
    class SlowTrainer:
        closed = False
        def close(self):
            time.sleep(30)          # эмулируем подвисший _shutdown_workers
            self.closed = True

    t0 = time.time()
    _close_quietly(SlowTrainer(), timeout=1.0)
    assert time.time() - t0 < 4.0   # не ждём 30с — ограничены timeout'ом


def test_is_sigint_worker_death_matches_only_sigint():
    from echelon3.cli import _is_sigint_worker_death
    assert _is_sigint_worker_death(RuntimeError("DataLoader worker (pid 1) is killed by signal: Interrupt. "))
    assert _is_sigint_worker_death(RuntimeError("worker (pid 2) is killed by signal: SIGINT"))
    # НЕ прерывание: OOM-SIGKILL / segfault / общий exit / обычные ошибки
    assert not _is_sigint_worker_death(RuntimeError("worker (pid 3) is killed by signal: Killed. "))
    assert not _is_sigint_worker_death(RuntimeError("worker (pid 4) is killed by signal: Segmentation fault. "))
    assert not _is_sigint_worker_death(RuntimeError("DataLoader worker (pid(s) 5) exited unexpectedly"))
    assert not _is_sigint_worker_death(ValueError("shape mismatch"))


def test_silence_sigint_ignores_further_sigint():
    """_silence_sigint() ставит SIGINT=SIG_IGN, чтобы повторный Ctrl-C в обработчике
    прерывания не увёл поток мимо os._exit в finally -> ddp.shutdown() ->
    destroy_process_group() -> NCCL-hang (30с). Восстановление — в autouse-фикстуре."""
    assert signal.getsignal(signal.SIGINT) != signal.SIG_IGN  # старт: не заглушён
    _silence_sigint()
    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN


def test_looks_like_interrupt_flag_gates_ambiguous():
    """'exited unexpectedly' (частый вариант SIGINT-смерти воркера на свежем torch) неотличим
    от OOM по тексту — трактуем как прерывание ТОЛЬКО если был SIGINT (флаг). Явный
    'killed by signal: Interrupt' — прерывание всегда; OOM без флага — реальная ошибка."""
    from echelon3 import cli
    exited = RuntimeError("DataLoader worker (pid(s) 5) exited unexpectedly")
    killed_int = RuntimeError("DataLoader worker (pid 1) is killed by signal: Interrupt. ")
    oom = RuntimeError("DataLoader worker (pid 2) is killed by signal: Killed. ")
    shape = ValueError("shape mismatch")

    prev = cli._SIGINT_SEEN
    try:
        cli._SIGINT_SEEN = False                       # Ctrl-C не было
        assert not cli._looks_like_interrupt(exited)   # неотличимо от OOM -> не глотаем
        assert cli._looks_like_interrupt(killed_int)   # явный SIGINT
        assert not cli._looks_like_interrupt(oom)      # OOM -> реальная ошибка

        cli._SIGINT_SEEN = True                        # был Ctrl-C
        assert cli._looks_like_interrupt(exited)       # смерть воркера при Ctrl-C -> прерывание
        assert cli._looks_like_interrupt(killed_int)
        assert not cli._looks_like_interrupt(shape)    # не-worker ошибка -> не глотаем даже при флаге
        assert not cli._looks_like_interrupt(RuntimeError("CUDA out of memory"))  # RuntimeError, но не worker
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
    _close_quietly(BoomTrainer(), timeout=5.0)   # не пробрасывает
    assert calls == [1]
