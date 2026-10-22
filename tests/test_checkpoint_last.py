"""Last-epoch snapshots are independent of best checkpoints and resume exact saved state."""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from echelon3.checkpoint.manager import CheckpointManager
from echelon3.trainers.baseline import Trainer


def trainer(path, epochs=2, **kwargs):
    net = torch.nn.Linear(2, 2)
    with torch.no_grad():
        net.weight.fill_(0.2)
        net.bias.fill_(0.1)
    loader = DataLoader(TensorDataset(torch.ones(8, 2), torch.zeros(8, 2)), batch_size=4)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    return Trainer(
        epochs=epochs, net=net, optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
        train_dataloader=loader, test_dataloader=loader,
        losses={'mse': (torch.nn.MSELoss(), 1.0)}, metrics={}, mlops_logger=None,
        ckpt_manager=CheckpointManager(path=str(path), checkpoints_to_keep=1),
        keep_best_on={'mse': {'direction': 'high'}},  # Training decreases MSE: best stays at epoch 0.
        device=torch.device('cpu'), precision='fp32', **kwargs,
    )


def assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_state_equal(a, e)
    else:
        assert actual == expected


def test_last_is_opt_in_and_does_not_change_numbered_selection(tmp_path):
    tr = trainer(tmp_path)
    tr.train()
    assert not (tmp_path / 'last.tar').exists()
    tr = trainer(tmp_path, save_last=True)
    tr.train()
    cm = CheckpointManager(path=str(tmp_path), checkpoints_to_keep=1)
    assert cm.load_last_checkpoint(cpu_only=True)['epoch'] == 2
    assert cm.load_latest_checkpoint(cpu_only=True)[0]['epoch'] == 0
    # Existing default resume still writes its initial baseline; saving last
    # must not mutate that numbered snapshot afterwards.
    assert cm.idxs == [2]
    numbered_bytes = (tmp_path / 'checkpoint-2.tar').read_bytes()
    cm.save_last_checkpoint({'epoch': 100})
    assert (tmp_path / 'checkpoint-2.tar').read_bytes() == numbered_bytes


def test_last_resume_matches_uninterrupted_training_and_preserves_best(tmp_path):
    full = trainer(tmp_path / 'full', epochs=4, save_last=True)
    full.train()
    split = trainer(tmp_path / 'split', epochs=2, save_last=True)
    split.train()
    best = (tmp_path / 'split' / 'checkpoint-1.tar').read_bytes()
    last = split._ckpt_manager.load_last_checkpoint(cpu_only=True)
    assert last['epoch'] == 2
    assert last['global_step'] == 4
    assert last['scheduler_state_dict']['last_epoch'] == 2
    assert last['scheduler_step_completed'] is True
    resumed = trainer(tmp_path / 'split', epochs=4, save_last=True, resume_from='last')
    resumed.train()
    for key in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict',
                'epoch', 'global_step', 'scaler_state_dict', 'best_metrics'):
        assert_state_equal(resumed._ckpt_manager.load_last_checkpoint(cpu_only=True)[key],
                           full._ckpt_manager.load_last_checkpoint(cpu_only=True)[key])
    assert (tmp_path / 'split' / 'checkpoint-1.tar').read_bytes() == best
    assert resumed._ckpt_manager.idxs == [1]


def test_last_only_target_and_global_step_are_restored_verbatim(tmp_path):
    tr = trainer(tmp_path, save_last=True)
    tr.train()
    state = tr._ckpt_manager.load_last_checkpoint(cpu_only=True)
    state['global_step'] = 123
    tr._ckpt_manager.save_last_checkpoint(state)
    (tmp_path / 'checkpoint-1.tar').unlink()
    resumed = trainer(tmp_path, epochs=3, save_last=True, resume_from='last')
    resumed.train()
    assert resumed._global_step == 125
    assert resumed._ckpt_manager.idxs == []
    assert resumed._ckpt_manager.load_last_checkpoint(cpu_only=True)['epoch'] == 3


def test_explicit_last_does_not_fall_back_to_best(tmp_path):
    trainer(tmp_path).train()
    with pytest.raises(FileNotFoundError, match='resume_from=last'):
        trainer(tmp_path, resume_from='last').train()


@pytest.mark.parametrize('kind', ['interrupt', 'error', 'profile_stop'])
def test_partial_epoch_cannot_replace_last(tmp_path, monkeypatch, kind):
    trainer(tmp_path, save_last=True).train()
    before = (tmp_path / 'last.tar').read_bytes()
    tr = trainer(tmp_path, epochs=3, save_last=True, resume_from='last')
    original_step = tr.one_step_train

    def partial(source, labels):
        original_step(source, labels)
        if kind == 'profile_stop':
            tr._profile_stop_requested = True
        elif kind == 'interrupt':
            raise KeyboardInterrupt
        else:
            raise RuntimeError('interrupted epoch')

    monkeypatch.setattr(tr, 'one_step_train', partial)
    if kind == 'profile_stop':
        tr.train()
    else:
        with pytest.raises(KeyboardInterrupt if kind == 'interrupt' else RuntimeError):
            tr.train()
    assert (tmp_path / 'last.tar').read_bytes() == before


@pytest.mark.parametrize('failure', ['write', 'replace'])
def test_failed_last_save_preserves_last_and_best(tmp_path, monkeypatch, failure):
    cm = CheckpointManager(path=str(tmp_path), checkpoints_to_keep=1)
    cm.save_checkpoint({'best': 1})
    cm.save_last_checkpoint({'epoch': 1})

    def fail(*args):
        raise OSError('disk full')

    if failure == 'write':
        monkeypatch.setattr(torch, 'save', fail)
    else:
        monkeypatch.setattr('echelon3.checkpoint.manager.os.replace', fail)
    with pytest.raises(OSError):
        cm.save_last_checkpoint({'epoch': 2})
    assert cm.load_last_checkpoint(cpu_only=True) == {'epoch': 1}
    assert cm.load_latest_checkpoint(cpu_only=True) == ({'best': 1}, 1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['checkpoint-1.tar', 'last.tar']


def test_last_is_not_pruned_or_used_for_numbering(tmp_path):
    cm = CheckpointManager(path=str(tmp_path), checkpoints_to_keep=1)
    cm.save_last_checkpoint({'epoch': 20})
    cm.save_checkpoint({'best': 1})
    cm.save_checkpoint({'best': 2})
    cm.init_storage()
    assert cm.idxs == [2]
    assert cm.load_last_checkpoint(cpu_only=True) == {'epoch': 20}


def test_non_main_rank_does_not_save_last(tmp_path, monkeypatch):
    tr = trainer(tmp_path, save_last=True)
    monkeypatch.setattr('echelon3.trainers.baseline.ddp.is_main', lambda: False)
    tr.save_last_checkpoint()
    assert not (tmp_path / 'last.tar').exists()


@pytest.mark.parametrize('config', [{'save_last': 'yes'}, {'resume_from': 'newest'}])
def test_invalid_last_options_fail(tmp_path, config):
    with pytest.raises(ValueError):
        trainer(tmp_path, **config)


def test_last_restores_nonempty_scaler_state_without_scheduler_catchup(tmp_path):
    tr = trainer(tmp_path, save_last=True)
    tr._scaler = torch.amp.GradScaler('cpu', init_scale=128, growth_interval=2)
    tr.train()
    state = tr._ckpt_manager.load_last_checkpoint(cpu_only=True)
    assert state['scaler_state_dict']['scale'] > 128
    # A custom scheduler need not use completed epochs as last_epoch. The new
    # phase marker must bypass the legacy catch-up heuristic entirely.
    state['scheduler_state_dict']['last_epoch'] = 0
    tr._ckpt_manager.save_last_checkpoint(state)
    resumed = trainer(tmp_path, resume_from='last')
    resumed._scaler = torch.amp.GradScaler('cpu', init_scale=16)
    resumed.load_from_checkpoint()
    assert resumed._scaler.state_dict() == state['scaler_state_dict']
    assert resumed._scheduler.last_epoch == 0
    assert resumed._optimizer.param_groups[0]['lr'] == state['optimizer_state_dict']['param_groups'][0]['lr']


def test_reset_last_loads_only_weights(tmp_path):
    tr = trainer(tmp_path, save_last=True)
    tr.train()
    reset = trainer(tmp_path, resume_from='last', reset=True)
    reset.load_from_checkpoint()
    assert_state_equal(reset._net.state_dict(), tr._net.state_dict())
    assert reset._current_epoch == 1
    assert reset._global_step == 0
    assert reset._optimizer.state_dict()['state'] == {}
    assert reset._optimizer.param_groups[0]['lr'] == 0.01
    assert reset._scheduler.last_epoch == 0
    assert reset._best_metrics_all is None


class Vectors(torch.utils.data.Dataset):
    def __init__(self, augment=None, preprocess=None):
        pass

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return torch.ones(2), torch.zeros(2)


class QuietLogger:
    def __init__(self, folder):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class FixedNet(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 2)
        with torch.no_grad():
            self.weight.fill_(0.2)
            self.bias.fill_(0.1)


def cli_config(path):
    from omegaconf import OmegaConf
    dataset = {'module': __name__, 'type': 'Vectors'}
    loader = {'module': 'torch.utils.data', 'type': 'DataLoader',
              'config': {'batch_size': 4, 'num_workers': 0}}
    return OmegaConf.create({
        'device': 'cpu',
        'net': {'module': __name__, 'type': 'FixedNet'},
        'data': {'train': dataset, 'test': dataset},
        'dataloaders': {'train': loader, 'test': loader},
        'optimizer': {'module': 'torch.optim', 'type': 'Adam', 'config': {'lr': 0.01}},
        'loss': [{'mse': {'module': 'torch.nn', 'type': 'MSELoss'}}],
        'trainer': {'module': 'echelon3.trainers.baseline', 'type': 'Trainer', 'config': {
            'epochs': 1, 'save_last': True, 'precision': 'fp32',
            'keep_best_on': {'mse': {'direction': 'high'}},
        }},
        'target': {'path': str(path), 'checkpoints_to_keep': 1},
        'mlops': {'module': __name__, 'type': 'QuietLogger'},
    })


@pytest.mark.parametrize('command', ['train', 'finetune'])
def test_cli_resumes_last(tmp_path, command):
    import signal
    from echelon3.cli.train import trainer_app
    from echelon3.cli.finetune import finetune_app
    app = trainer_app if command == 'train' else finetune_app
    cfg = cli_config(tmp_path)
    original = signal.getsignal(signal.SIGINT)
    try:
        app(cfg)
        cm = CheckpointManager(path=str(tmp_path))
        assert cm.load_last_checkpoint(cpu_only=True)['epoch'] == 1
        best = (tmp_path / 'checkpoint-1.tar').read_bytes()
        cfg.trainer.config.epochs = 2
        cfg.trainer.config.resume_from = 'last'
        app(cfg)
        assert cm.load_last_checkpoint(cpu_only=True)['epoch'] == 2
        assert cm.load_last_checkpoint(cpu_only=True)['global_step'] == 4
        assert (tmp_path / 'checkpoint-1.tar').read_bytes() == best
    finally:
        signal.signal(signal.SIGINT, original)


class InspectWeights:
    seen = []

    def __init__(self, net, **kwargs):
        self.net = net

    def export(self):
        self.seen.append(self.net.weight.detach().clone())

    def evaluate(self):
        self.export()
        return 0.0


def test_evaluate_and_export_keep_numbered_weights(tmp_path):
    from echelon3.cli.evaluate import evaluate_app
    from echelon3.cli.export import exporter_app
    cfg = cli_config(tmp_path)
    net = torch.nn.Linear(2, 2)
    cm = CheckpointManager(path=str(tmp_path))
    with torch.no_grad():
        net.weight.fill_(1)
    cm.save_checkpoint({'model_state_dict': net.state_dict()})
    with torch.no_grad():
        net.weight.fill_(99)
    cm.save_last_checkpoint({'model_state_dict': net.state_dict()})
    cfg.trainer.config.resume_from = 'last'  # Only training consumes this setting.
    cfg.export = {'exporters': {'inspect': {'module': __name__, 'type': 'InspectWeights'}}}
    cfg.metrics = [{'mse': {'module': 'torchmetrics', 'type': 'MeanSquaredError'}}]
    cfg.evaluator = {'module': __name__, 'type': 'InspectWeights', 'metric': 'mse'}
    InspectWeights.seen = []
    exporter_app(cfg)
    evaluate_app(cfg)
    assert len(InspectWeights.seen) == 2
    assert all(torch.equal(weights, torch.ones(2, 2)) for weights in InspectWeights.seen)
