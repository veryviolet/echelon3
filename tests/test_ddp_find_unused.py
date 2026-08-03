"""find_unused_parameters defaults to False (torch's own default). True forces an extra
autograd-graph traversal every iteration — torch itself warns it hurts performance — and is
only needed for nets that leave some parameters out of the loss on a step. The value is read
into an attribute in __init__ (independent of whether DDP is active), so we can assert the
default and the opt-in without a live DDP process.
"""
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from echelon3.trainers.baseline import Trainer
from echelon3.checkpoint.manager import CheckpointManager


def _trainer(**kwargs):
    net = torch.nn.Linear(4, 3)
    loader = DataLoader(TensorDataset(torch.zeros(8, 4), torch.zeros(8, dtype=torch.long)), batch_size=4)
    return Trainer(
        epochs=1, train_dataloader=loader, test_dataloader=None, net=net,
        losses={"ce": (torch.nn.CrossEntropyLoss(), 1.0)}, metrics={},
        optimizer=torch.optim.SGD(net.parameters(), 0.01), scheduler=None,
        ckpt_manager=CheckpointManager(path=tempfile.mkdtemp(), checkpoints_to_keep=1),
        mlops_logger=None, device=torch.device("cpu"), **kwargs,
    )


def test_find_unused_parameters_defaults_false():
    assert _trainer()._ddp_find_unused_parameters is False


def test_find_unused_parameters_opt_in():
    assert _trainer(ddp_find_unused_parameters=True)._ddp_find_unused_parameters is True
