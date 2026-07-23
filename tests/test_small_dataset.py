"""Regression: times_to_validate_per_epoch greater than the number of batches in an epoch caused
a ZeroDivisionError (total_batches // ttv == 0 -> ... % 0) on small datasets.
Fix: max(1, total_batches // ttv)."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from echelon3.trainers.baseline import Trainer
from echelon3.checkpoint.manager import CheckpointManager


def test_train_survives_more_validations_than_batches(tmp_path):
    # 2 batches per epoch, asked to validate 5 times — this used to crash on batch 0.
    x = torch.zeros(128, 4)
    y = torch.zeros(128, 4)
    loader = DataLoader(TensorDataset(x, y), batch_size=64, drop_last=True)  # 2 batches
    net = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    trainer = Trainer(
        epochs=1,
        train_dataloader=loader,
        test_dataloader=None,
        net=net,
        losses={"l1": (torch.nn.L1Loss(), 1.0)},
        metrics={},
        optimizer=opt,
        scheduler=None,
        ckpt_manager=CheckpointManager(path=str(tmp_path), checkpoints_to_keep=1),
        mlops_logger=None,
        device=torch.device("cpu"),
        times_to_validate_per_epoch=5,
    )
    trainer.train()  # must not raise ZeroDivisionError
