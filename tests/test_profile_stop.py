"""Profiling windows must not pretend that a partial epoch finished."""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import multiprocessing as mp
from types import SimpleNamespace

from echelon3.trainers.baseline import Trainer

from .test_checkpoint_last import trainer


@pytest.mark.parametrize('steps', [1, 2])
def test_profile_stop_reports_actual_batches_without_epoch_side_effects(tmp_path, capsys, steps):
    tr = trainer(tmp_path, epochs=3, save_last=True,
                 profile={'wait': 0, 'warmup': 0, 'active': steps, 'stop_after': True})
    # Exercise the scheduling logic without spending time exporting a trace.
    tr._make_profiler = lambda: None
    tr._train_loader = DataLoader(TensorDataset(torch.ones(40, 2), torch.zeros(40, 2)), batch_size=4)
    tr.train()
    output = capsys.readouterr().out
    assert f'({steps}/10)' in output
    assert '(10/10)' not in output
    assert tr._global_step == steps
    assert tr._completed_epochs == 0
    assert tr._scheduler.last_epoch == 0
    assert not (tmp_path / 'last.tar').exists()


def test_profile_stop_false_completes_epochs(tmp_path, capsys):
    tr = trainer(tmp_path, epochs=2, save_last=True,
                 profile={'wait': 0, 'warmup': 0, 'active': 1, 'stop_after': False})
    tr._make_profiler = lambda: None
    tr.train()
    assert '(2/2)' in capsys.readouterr().out
    assert tr._global_step == 4
    assert tr._scheduler.last_epoch == 2
    assert tr._completed_epochs == 2
    assert (tmp_path / 'last.tar').exists()


def test_profile_window_crossing_epoch_keeps_last_complete_epoch(tmp_path, capsys):
    tr = trainer(tmp_path, epochs=3, save_last=True,
                 profile={'wait': 0, 'warmup': 0, 'active': 3, 'stop_after': True})
    tr._make_profiler = lambda: None
    tr.train()  # Two batches in epoch 1, one in epoch 2.
    assert '--> Trained epoch 2: 50% (1/2)' in capsys.readouterr().out
    assert tr._global_step == 3 and tr._completed_epochs == 1
    assert tr._scheduler.last_epoch == 1
    state = torch.load(tmp_path / 'last.tar', map_location='cpu', weights_only=False)
    assert state['epoch'] == 1 and state['global_step'] == 2


class CountingDataset(torch.utils.data.Dataset):
    def __init__(self, counter, fail=False):
        self.counter = counter
        self.fail = fail

    def __len__(self):
        return 100

    def __getitem__(self, index):
        if self.fail and index > 1:
            raise ValueError('broken prefetch')
        with self.counter.get_lock():
            self.counter.value += 1
        return torch.ones(64, 64)


@pytest.mark.skipif(os.name != 'posix', reason='fork/spawn process checks')
@pytest.mark.parametrize('context', ['fork', 'spawn'])
@pytest.mark.parametrize('persistent', [True, False])
def test_profile_stop_drains_only_dispatched_batches(context, persistent):
    counter = mp.get_context(context).Value('i', 0)
    loader = DataLoader(CountingDataset(counter), batch_size=1, num_workers=2,
                        prefetch_factor=2, persistent_workers=persistent,
                        multiprocessing_context=context)
    iterator = iter(loader)
    workers = list(iterator._workers)
    try:
        next(iterator)
        sent = iterator._send_idx
        Trainer._finish_profile_loader(SimpleNamespace(_train_loader=loader), iterator)
        assert counter.value == sent < len(loader)
        assert all(not w.is_alive() and w.exitcode == 0 for w in workers)
        assert loader._iterator is None
    finally:
        iterator._shutdown_workers()


def test_profile_stop_does_not_swallow_prefetch_failure():
    context = mp.get_context('spawn')
    counter = context.Value('i', 0)
    loader = DataLoader(CountingDataset(counter, fail=True), batch_size=1, num_workers=1,
                        persistent_workers=True, multiprocessing_context=context)
    iterator = iter(loader)
    try:
        next(iterator)
        with pytest.raises(ValueError, match='broken prefetch'):
            Trainer._finish_profile_loader(SimpleNamespace(_train_loader=loader), iterator)
    finally:
        iterator._shutdown_workers()
