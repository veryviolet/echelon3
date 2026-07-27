"""PairEvaluator feeds pair inputs. The generic Evaluator calls net(single_input), which a
pair-input net can't consume. PairEvaluator mirrors PairTrainer: dataset yields
((base, query), gt), net is called net(base, query) -> prediction, and (prediction, gt) goes
to the metric. It installs pair_collate_fn and forwards the return-features flag positionally.
"""
import torch
from torch.utils.data import DataLoader, Dataset

from echelon3.evaluators.pair import PairEvaluator
from echelon3.trainers.pair import pair_collate_fn
from echelon3.metrics.base import Metric


class _PairNet(torch.nn.Module):
    def __init__(self, seen):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self._seen = seen

    def forward(self, base, query, flag=None):
        self._seen.append((tuple(base.shape), tuple(query.shape), flag))
        prediction = base.mean(dim=1, keepdim=True)   # [B, 1, H, W]
        if flag:
            return prediction, "features"
        return prediction


class _FakeMetric(Metric):
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = 0
        self.last = None

    def update(self, predicted, target):
        self.calls += 1
        self.last = (predicted, target)

    def compute(self):
        return 0.75


class _PairDS(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return (torch.zeros(3, 8, 8), torch.zeros(3, 4, 4)), torch.zeros(1, 8, 8)


def _evaluator(net, metric, n=4, batch_size=2, return_features=False):
    loader = DataLoader(_PairDS(n), batch_size=batch_size)
    return PairEvaluator(
        net=net, dataloader=loader, metric=metric,
        preprocess=None, postprocess=None, return_features=return_features,
    )


def test_installs_pair_collate():
    ev = _evaluator(_PairNet([]), _FakeMetric())
    assert ev.dataloader.collate_fn is pair_collate_fn


def test_feeds_pairs_and_returns_metric():
    seen = []
    metric = _FakeMetric()
    ev = _evaluator(_PairNet(seen), metric, n=4, batch_size=2)
    result = ev.evaluate()

    assert result == 0.75
    assert metric.calls == 2                 # 2 batches, metric fed each
    assert len(seen) == 2
    (base_shape, query_shape, flag) = seen[0]
    assert base_shape[0] == 2 and query_shape[0] == 2   # net got a PAIR of batched tensors
    assert flag is None
    pred, gt = metric.last
    assert pred.shape[0] == 2 and gt.shape[0] == 2       # metric got (prediction, gt)


def test_return_features_flag_passed_positionally():
    seen = []
    metric = _FakeMetric()
    ev = _evaluator(_PairNet(seen), metric, n=2, batch_size=2, return_features=True)
    ev.evaluate()
    assert seen[0][2] is True                # flag forwarded so features-returning nets work
    pred, _gt = metric.last                  # metric gets the unpacked prediction, not the tuple
    assert torch.is_tensor(pred)
