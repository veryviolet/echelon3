"""Retention preserves the latest complete checkpoint, including across restarts."""
import pytest
import torch

from echelon3.checkpoint.manager import CheckpointManager


def test_retention_and_resume_keep_monotonic_numbers(tmp_path):
    manager = CheckpointManager(path=str(tmp_path), checkpoints_to_keep=2)
    for step in range(4):
        manager.save_checkpoint({'step': step})
    assert manager.idxs == [3, 4]
    assert sorted(p.name for p in tmp_path.iterdir()) == ['checkpoint-3.tar', 'checkpoint-4.tar']
    resumed = CheckpointManager(path=str(tmp_path), checkpoints_to_keep=1)
    resumed.save_checkpoint({'step': 4})
    assert resumed.idxs == [5]
    assert resumed.load_latest_checkpoint(cpu_only=True) == ({'step': 4}, 5)
    assert [p.name for p in tmp_path.iterdir()] == ['checkpoint-5.tar']


def test_omitted_limit_keeps_all_and_ignores_unrelated_files(tmp_path):
    unrelated = tmp_path / 'checkpoint-backup-17.tar'
    unrelated.write_text('unrelated')
    manager = CheckpointManager(path=str(tmp_path))
    for step in range(3):
        manager.save_checkpoint({'step': step})
    assert manager.idxs == [1, 2, 3]
    assert unrelated.read_text() == 'unrelated'


def test_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    manager = CheckpointManager(path=str(tmp_path), checkpoints_to_keep=1)
    manager.save_checkpoint({'step': 1})

    def fail_after_partial_write(obj, file):
        file.write(b'incomplete')
        raise OSError('disk full')

    monkeypatch.setattr(torch, 'save', fail_after_partial_write)
    with pytest.raises(OSError, match='disk full'):
        manager.save_checkpoint({'step': 2})
    assert manager.idxs == [1]
    assert [p.name for p in tmp_path.iterdir()] == ['checkpoint-1.tar']
    assert manager.load_latest_checkpoint(cpu_only=True) == ({'step': 1}, 1)


@pytest.mark.parametrize('limit', [0, -1, 1.5, True, '2'])
def test_invalid_limit_fails_before_writing(tmp_path, limit):
    with pytest.raises(ValueError, match='checkpoints_to_keep'):
        CheckpointManager(path=str(tmp_path / 'output'), checkpoints_to_keep=limit)
    assert not (tmp_path / 'output').exists()
