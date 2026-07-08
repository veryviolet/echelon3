from abc import abstractmethod
import torch


def all_reduce_sum_(*tensors):
    """In-place SUM-all-reduce тензоров по рангам под DDP (no-op вне распределённого
    контекста). Готовая реализация ``Metric.dist_reduce()`` для метрик со
    счётчиками-аккумуляторами: суммирование пересечений/объединений коммутирует с
    шардированием, поэтому даёт ТОЧНОЕ глобальное значение — в отличие от усреднения
    уже посчитанных величин (напр. IoU по шардам усреднять нельзя)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for t in tensors:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)


class Metric:

    def to(self, *args, **kwargs):
        # Custom metrics accumulate on CPU (numpy/scipy) or move tensors to the
        # right device inside update(); the trainer calls metric.to(device)
        # uniformly (torchmetrics need it), so this keeps them interchangeable.
        #
        # НО: если метрика — ещё и nn.Module с параметрами/буферами, этот no-op
        # затирал бы nn.Module.to по MRO, буферы оставались бы на CPU и validate на
        # cuda падал бы (RuntimeError device mismatch). Поэтому делегируем в
        # nn.Module.to, когда self — модуль.
        if isinstance(self, torch.nn.Module):
            return torch.nn.Module.to(self, *args, **kwargs)
        return self

    def dist_reduce(self):
        """DDP: свести накопленное состояние по рангам ПЕРЕД ``compute()`` (валидация
        шардируется по рангам через DistributedSampler). База — no-op: одиночный GPU,
        torchmetrics (сводятся сами внутри compute), метрики без распределённого
        состояния. Кастомная метрика со счётчиками-аккумуляторами должна здесь
        ``all_reduce(SUM)`` свои буферы (см. :func:`all_reduce_sum_`) — тогда
        ``compute()`` вернёт точное глобальное значение. Вызывается трейнером на ВСЕХ
        рангах симметрично (это коллективная операция)."""
        pass

    @abstractmethod
    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def reset(self):
        pass
