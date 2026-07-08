import torch

from echelon3.trainers.baseline import Trainer


def pair_collate_fn(batch):
    """Collate ``((base, query), gt)`` samples into ``((B_base, B_query), B_gt)``.

    Keeps the two images paired through batching so the network can be called as
    ``net(base, query, …)``. ``set_to_device`` already understands the resulting
    ``(base, query)`` tuple of batched tensors.
    """
    base, query, gt = [], [], []
    for (b, q), g in batch:
        base.append(b)
        query.append(q)
        gt.append(g)
    return (torch.stack(base), torch.stack(query)), torch.stack(gt)


class PairTrainer(Trainer):
    """Trainer for two-image ("pair" / image-in-image) inputs.

    The dataset yields ``((base, query), gt)``; the network is called as
    ``net(base, query, return_features)`` and returns a ``heatmap`` — or
    ``(heatmap, features)`` when ``return_features`` is set (the default). Every
    piece of the base trainer's machinery — DDP, autocast/precision, the grad
    scaler, closure optimizers (SAM/LBFGS), sharded validation, checkpointing — is
    inherited unchanged. Only the forward pass and loss routing are specialized,
    and only through :meth:`compute_losses` / :meth:`pair_losses`, so the precision
    path is never re-implemented here or in subclasses.

    :meth:`pair_losses` is the task hook: the default applies every configured
    loss to ``(heatmap, gt)``; a subclass overrides it to route specific losses
    onto the ``features`` output (presence / contrastive heads, …).
    """

    def __init__(self, *args, return_features: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._return_features = bool(return_features)
        # Pair samples must stay paired through batching.
        self._train_loader.collate_fn = pair_collate_fn
        for loader in self._test_loaders.values():
            if loader is not None:
                loader.collate_fn = pair_collate_fn

    def compute_losses(self, source, labels, net=None):
        net = net if net is not None else self._net
        base, query = source
        if self._return_features:
            heatmap, features = net(base, query, return_features=True)
        else:
            heatmap, features = net(base, query), None
        return heatmap, self.pair_losses(heatmap, features, labels)

    def pair_losses(self, heatmap, features, labels):
        """Map the configured losses onto the network's outputs.

        Default: every loss compares ``heatmap`` against ``gt``. Override to send
        presence / contrastive losses to ``features`` (with a binary label) and the
        rest to ``heatmap``. Returns ``{name: (loss_tensor, weight)}``.
        """
        return {
            name: (loss(heatmap, labels.float()), weight)
            for name, (loss, weight) in self._losses.items()
        }
