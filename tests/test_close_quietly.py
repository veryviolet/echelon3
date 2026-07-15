"""_close_quietly гасит воркеров перед жёстким os._exit, но НЕ виснет: если close()
подвиснет (напр. pin_memory_thread.join без таймаута), выход ограничен timeout'ом."""
import time

from echelon3.cli import _close_quietly


def test_close_quietly_is_time_bounded():
    class SlowTrainer:
        closed = False
        def close(self):
            time.sleep(30)          # эмулируем подвисший _shutdown_workers
            self.closed = True

    t0 = time.time()
    _close_quietly(SlowTrainer(), timeout=1.0)
    assert time.time() - t0 < 4.0   # не ждём 30с — ограничены timeout'ом


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
