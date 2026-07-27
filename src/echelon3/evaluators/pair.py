import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from echelon3 import runtime
from echelon3.evaluators.basic import Evaluator
from echelon3.trainers.pair import pair_collate_fn


class PairEvaluator(Evaluator):
    """Two-image ("pair" / image-in-image) evaluator — the eval-path counterpart of
    :class:`echelon3.trainers.pair.PairTrainer`.

    The generic :class:`~echelon3.evaluators.basic.Evaluator` calls ``net(single_input)``,
    which a pair-input network can't consume. This evaluator mirrors PairTrainer's forward:
    the dataset yields ``((base, query), gt)``, the network is called ``net(base, query)``
    and returns its prediction, and ``(prediction, gt)`` is fed to the metric each batch —
    the same ``(predictions, labels)`` pairing the metric sees during training's validation.
    The metric itself (and what its score means) is supplied by config, exactly as in train.

    Like PairTrainer it installs ``pair_collate_fn`` so the two images stay paired through
    batching, and passes the return-features flag POSITIONALLY when ``return_features=True``
    (``evaluator.config.return_features``), so nets that require the flag — or name it
    ``return_features`` / ``return_intermediates`` / … — still work; the features output is
    ignored for scoring.

    Config: ``evaluator: { module: echelon3.evaluators.pair, type: PairEvaluator,
    config: { return_features: false }, metric: <name> }``.
    """

    def __init__(self, net, dataloader, metric, preprocess, postprocess,
                 return_features: bool = False, **kwargs):
        super().__init__(
            net=net, dataloader=dataloader, metric=metric,
            preprocess=preprocess, postprocess=postprocess, **kwargs,
        )
        self._return_features = bool(return_features)
        if self.dataloader is not None:
            # Keep (base, query) paired through batching, as PairTrainer does.
            self.dataloader.collate_fn = pair_collate_fn
        if hasattr(self.metric, "to"):
            self.metric = self.metric.to(self.device)

    def _forward(self, base, query):
        if self._return_features:
            prediction, _features = self.net(base, query, True)   # flag positional; net names it freely
            return prediction
        return self.net(base, query)

    def evaluate_one(self, dataloader: DataLoader, mode: str = "test"):
        if dataloader is None:
            raise RuntimeError("PairEvaluator: dataloader is None")

        self.net.to(self.device)
        self.net.eval()
        if hasattr(self.metric, "reset"):
            self.metric.reset()

        try:
            total = len(dataloader)
        except TypeError:
            total = None
        progress = tqdm(initial=0, total=total, desc=f"--> Evaluating ({mode}) ", ncols=0)

        with torch.no_grad():
            for source, gt in dataloader:
                base, query = source
                base = base.to(self.device, non_blocking=True)
                query = query.to(self.device, non_blocking=True)
                gt = gt.to(self.device, non_blocking=True)
                prediction = self._forward(base, query)
                # Feed the metric exactly as training's validate() does, so an eval score
                # matches the train-time validation score: cast to fp32 (autocast may return
                # bf16/fp16), and squeeze a trailing singleton dim mismatch vs gt.
                prediction = runtime.to_float32(prediction)
                if isinstance(prediction, torch.Tensor) and isinstance(gt, torch.Tensor) \
                        and prediction.dim() != gt.dim():
                    prediction = prediction.squeeze()
                self.metric.update(prediction, gt)
                progress.update(1)
        progress.close()

        result = self.metric.compute()
        if torch.is_tensor(result) and result.numel() == 1:
            result = float(result.cpu())
        print(f"--> Metric ({mode}): {result}")
        return result
