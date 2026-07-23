"""Parameters with sensible defaults are optional in the config: create_* and Trainer
do not fail when a section/key is omitted (echelon3 0.7.3)."""
import torch
from omegaconf import OmegaConf

from echelon3 import creator
from echelon3.trainers.baseline import Trainer

from tests._fixtures import tiny_loader


def _c(d):
    return OmegaConf.create(d)


# ---- the config block is optional for ANY component (a single rule) ----
def test_create_net_without_config():
    net = creator.create_net(_c({"module": "torch.nn", "type": "Identity"}))
    assert isinstance(net, torch.nn.Identity)


def test_create_single_dataset_without_config():
    ds = creator.create_single_dataset(
        _c({"module": "tests._fixtures", "type": "TinyPairs"}),
        augment=None, preprocess=None,
    )
    assert len(ds) == 8


def test_create_scheduler_without_config():
    opt = torch.optim.SGD(torch.nn.Linear(2, 2).parameters(), lr=0.1)
    sch = creator.create_scheduler(
        _c({"module": "torch.optim.lr_scheduler", "type": "ConstantLR"}), opt,
    )
    assert sch is not None


def test_create_single_dataloader_without_config():
    dl = creator.create_single_dataloader(
        _c({"module": "torch.utils.data", "type": "DataLoader"}),
        dataset=[1, 2, 3],
    )
    assert dl is not None


# ---- create_metrics: the section is omitted ----------------------------
def test_create_metrics_none_is_empty():
    assert creator.create_metrics(None) == {}


# ---- create_augments / create_preprocesses: the transform section is omitted ---
def test_create_augments_none_defaults():
    train_aug, test_aug = creator.create_augments(None)
    assert train_aug is not None and test_aug is not None


def test_create_preprocesses_none():
    train_pp, test_pp = creator.create_preprocesses(None)
    assert train_pp is None and test_pp is None


def test_create_preprocesses_purpose_without_preprocess_key():
    train_pp, test_pp = creator.create_preprocesses(_c({"train": {}, "test": {}}))
    assert train_pp is None and test_pp is None


# ---- create_loss: weight defaults to 1.0, config is optional -----------
def test_create_loss_defaults_weight_to_one():
    losses = creator.create_loss(_c([{"main": {"module": "torch.nn", "type": "L1Loss"}}]))
    loss_fn, weight = losses["main"]
    assert isinstance(loss_fn, torch.nn.L1Loss)
    assert weight == 1.0


# ---- Trainer: keep_best_on and scheduler are optional ------------------
def test_trainer_optional_keep_best_and_scheduler():
    net = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    trainer = Trainer(
        epochs=1,
        train_dataloader=tiny_loader(),
        test_dataloader=None,
        net=net,
        losses={},
        metrics={},
        optimizer=opt,
        scheduler=None,          # optional
        ckpt_manager=None,
        mlops_logger=None,
        device=torch.device("cpu"),
        # keep_best_on omitted -> None
    )
    assert trainer._scheduler is None
    assert trainer._keep_best_config is None
