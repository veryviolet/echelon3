from abc import abstractmethod
import torch


def all_reduce_sum_(*tensors):
    """In-place SUM all-reduce of tensors across ranks under DDP (no-op outside a
    distributed context). A ready-made ``Metric.dist_reduce()`` for metrics with
    accumulator counters: summing intersections/unions commutes with sharding, so it
    yields the EXACT global value — unlike averaging already-computed quantities (e.g.
    per-shard IoU must not be averaged)."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for t in tensors:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)


def all_gather_cat(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather a tensor across ranks and concatenate along dim=0 (no-op outside DDP —
    returns the input unchanged). For metrics whose ``compute()`` needs the FULL set from
    all ranks (e.g. retrieval: collect the query/gallery embeddings computed on the
    DistributedSampler shards). Unlike ``all_reduce(SUM)`` it keeps the individual elements
    rather than reducing them to a scalar — the typical building block of
    ``MultiDatasetMetric.dist_reduce()``.

    Requires the tensor to have the SAME size on every rank; under DistributedSampler this
    holds (it pads the dataset to a multiple of world_size). A side effect of that padding
    is a few duplicate samples in the gathered set; drop them by id in ``compute()``."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return tensor
    world = torch.distributed.get_world_size()
    if world == 1:
        return tensor
    tensor = tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(world)]
    torch.distributed.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


class Metric:

    def to(self, *args, **kwargs):
        # Custom metrics accumulate on CPU (numpy/scipy) or move tensors to the
        # right device inside update(); the trainer calls metric.to(device)
        # uniformly (torchmetrics need it), so this keeps them interchangeable.
        #
        # BUT: if the metric is also an nn.Module with parameters/buffers, this no-op
        # would shadow nn.Module.to via the MRO, buffers would stay on CPU and validate
        # on cuda would fail (RuntimeError device mismatch). So we delegate to
        # nn.Module.to when self is a module.
        if isinstance(self, torch.nn.Module):
            return torch.nn.Module.to(self, *args, **kwargs)
        return self

    def dist_reduce(self):
        """DDP: reduce the accumulated state across ranks BEFORE ``compute()`` (validation
        is sharded across ranks via DistributedSampler). Base is a no-op: single GPU,
        torchmetrics (which reduce themselves inside compute), and metrics without
        distributed state. A custom metric with accumulator counters should ``all_reduce
        (SUM)`` its buffers here (see :func:`all_reduce_sum_`) — then ``compute()`` returns
        the exact global value. Called by the trainer on ALL ranks symmetrically (this is a
        collective operation)."""
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


class MultiDatasetMetric(Metric):
    """A metric that spans SEVERAL validation datasets at once, with a SINGLE ``compute()``
    after iterating over all of them (e.g. retrieval: queries + gallery).

    Contract with the trainer (``validate()``) — how it differs from a plain :class:`Metric`:

    * ``self.datasets`` — the list of test-dataset (loader) names the metric spans.
      Usually populated in the constructor from roles, e.g.
      ``query_dataset='queries', gallery_dataset='gallery'`` -> ``['queries', 'gallery']``.
      From it the trainer knows which loaders feed the metric and when all are done. The
      role of a given name (query vs gallery) is resolved by the metric itself.
    * ``update(predicted, target, dataset)`` — carries the NAME of the current batch's
      source dataset (unlike the plain ``update(predicted, target)``).
    * The trainer calls ``reset()`` ONCE BEFORE iterating all of the metric's datasets, and
      ``compute()`` ONCE AFTER all of them (not per loader). While a single dataset is being
      processed the mechanics are the same as a plain metric: local ``update``, accumulate
      into a buffer, NO ``compute``. Datasets are processed sequentially.
    * Under DDP ``dist_reduce()`` must gather the buffers from ALL ranks (all-gather/cat,
      see :func:`all_gather_cat`), since each rank only saw its own shard of each dataset —
      then ``compute()`` on every rank works over the full set. It is the single collective,
      called symmetrically on all ranks BEFORE ``compute()`` (not per batch).

    The design is general: it fits any metric that needs a cross-dataset context, not only
    retrieval."""

    #: names of the test datasets this metric spans (filled by the subclass constructor)
    datasets: list = []

    @abstractmethod
    def update(self, predicted: torch.Tensor, target: torch.Tensor, dataset: str):
        pass
