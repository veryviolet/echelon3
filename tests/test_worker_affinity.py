"""Worker placement must not widen ranks or escape the launcher allocation."""
import os
import pickle

import pytest
import torch
from omegaconf import OmegaConf

from echelon3 import cpu_affinity as affinity
from echelon3.creator import create_single_dataloader, create_dataloaders, _worker_init_fn


@pytest.fixture
def allocation(monkeypatch):
    monkeypatch.setattr(affinity.os, 'sched_getaffinity', lambda pid: {2, 3}, raising=False)
    monkeypatch.setattr(affinity.os, 'sched_setaffinity', lambda pid, cpus: None, raising=False)
    token = affinity._launcher_cpus.set((2, 3, 6, 7))
    yield
    affinity._launcher_cpus.reset(token)


def test_resolve_uses_launcher_not_narrowed_rank(allocation):
    assert affinity.resolve_worker_affinity('launcher') == (2, 3, 6, 7)
    assert affinity.resolve_worker_affinity([7, 6, 7]) == (6, 7)
    assert affinity.resolve_worker_affinity(None) is None
    pickle.loads(pickle.dumps(_worker_init_fn(cpus=(6, 7))))


@pytest.mark.parametrize('value', [[], [True], [-1], [2.5], 'all', 2, False, [8]])
def test_invalid_or_unavailable_cpu_rejected(allocation, value):
    with pytest.raises(ValueError, match='worker_cpu_affinity'):
        affinity.resolve_worker_affinity(value)


def test_kernel_restrictions_are_not_silently_ignored(allocation):
    with pytest.raises(RuntimeError, match='cpuset/cgroup'):
        affinity.apply_worker_affinity((2, 3, 6))


def test_default_does_not_set_affinity(monkeypatch):
    def unexpected(*args):
        raise AssertionError('default must not set affinity')
    monkeypatch.setattr(affinity.os, 'sched_setaffinity', unexpected, raising=False)
    affinity.apply_worker_affinity(None)


def test_explicit_setting_requires_platform_support(monkeypatch):
    monkeypatch.delattr(affinity.os, 'sched_setaffinity', raising=False)
    assert affinity.resolve_worker_affinity(None) is None
    with pytest.raises(ValueError, match='Linux'):
        affinity.resolve_worker_affinity('launcher')


def test_capture_is_scoped_and_keeps_parent_snapshot(allocation):
    @affinity.capture_launcher_affinity
    def nested():
        return affinity.launcher_cpus()
    assert affinity.run_with_launcher_affinity(nested, (), {}, (6, 7)) == (6, 7)
    assert affinity.launcher_cpus() == (2, 3, 6, 7)


def _cfg(cpus=None, workers=1):
    return {'module': 'torch.utils.data', 'type': 'DataLoader',
            'config': {'batch_size': 2, 'num_workers': workers,
                       'worker_cpu_affinity': cpus}}


@pytest.mark.parametrize('workers', [0, 1])
def test_train_named_validation_and_evaluate_resolve_independently(allocation, workers):
    dataset = torch.utils.data.TensorDataset(torch.zeros(4))
    cfg = OmegaConf.create({'train': _cfg('launcher', workers),
                            'test': {'a': _cfg([6], workers), 'b': _cfg(None, workers)}})
    train, test = create_dataloaders(cfg, dataset, {'a': dataset, 'b': dataset})
    evaluate = create_single_dataloader(OmegaConf.create(_cfg([7], workers)), dataset)
    if workers:
        assert train.worker_init_fn.keywords['_cpus'] == (2, 3, 6, 7)
        assert test['a'].worker_init_fn.keywords['_cpus'] == (6,)
        assert test['b'].worker_init_fn.keywords['_cpus'] is None
        assert evaluate.worker_init_fn.keywords['_cpus'] == (7,)
    else:
        assert all(loader.worker_init_fn is None for loader in [train, *test.values(), evaluate])


@pytest.mark.skipif(not hasattr(os, 'sched_getaffinity'), reason='Linux affinity required')
def test_ddp_spawn_preserves_launcher_and_worker_masks(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip('at least two allowed CPUs required')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    root = Path(__file__).resolve().parents[1]
    env['PYTHONPATH'] = str(root / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('_worker_affinity_probe.py')),
                             str(tmp_path)], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    import json
    reports = [json.loads((tmp_path / f'rank-{rank}.json').read_text()) for rank in range(2)]
    for report in reports:
        assert report['launcher'] == report['wide_worker']
        assert report['default_worker'] == report['rank_before'] == report['rank_after']
        assert report['explicit_worker'] == [report['launcher'][-1]]
