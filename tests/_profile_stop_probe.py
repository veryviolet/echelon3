"""Small native CLI profile probe; CPU/Gloo by default, no production data."""
import json
import os
from pathlib import Path
import sys
import time

import torch
from omegaconf import OmegaConf

from echelon3 import ddp
from echelon3.trainers.baseline import Trainer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, length=1280, augment=None, preprocess=None):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        time.sleep(0.001)
        width = int(os.environ.get('ECHELON3_PROFILE_TEST_WIDTH', '16'))
        return torch.full((width,), width ** -0.5), torch.zeros(2)


class ReportingTrainer(Trainer):
    def __init__(self, report_dir, **kwargs):
        super().__init__(**kwargs)
        self.report_dir = Path(report_dir)
        self.saved_workers = []

    def one_step_train(self, source, labels):
        super().one_step_train(source, labels)
        if not self.saved_workers:
            self.saved_workers = list(self._train_loader._iterator._workers)

    def close(self):
        rank = ddp.rank()
        super().close()
        report = {'rank_pid': os.getpid(), 'steps': self._global_step,
                  'completed_epochs': self._completed_epochs,
                  'scheduler_epoch': self._scheduler.last_epoch,
                  'workers': [{'pid': w.pid, 'exitcode': w.exitcode, 'alive': w.is_alive()}
                              for w in self.saved_workers]}
        (self.report_dir / f'closed-{rank}.json').write_text(json.dumps(report))


def main(folder, command):
    module = '_profile_stop_probe'
    dataset = {'module': module, 'type': 'Dataset'}
    native_gpus = os.environ.get('ECHELON3_PROFILE_TEST_GPUS')
    loader = {'module': 'torch.utils.data', 'type': 'DataLoader', 'config': {
        'batch_size': 16, 'num_workers': 2, 'persistent_workers': True,
        'prefetch_factor': 2, 'pin_memory': bool(native_gpus),
    }}
    cfg = OmegaConf.create({
        'device': 'cuda' if native_gpus else 'cpu',
        'gpus': [int(g) for g in native_gpus.split(',')] if native_gpus else [],
        'net': {'module': 'torch.nn', 'type': 'Linear',
                'config': {'in_features': int(os.environ.get('ECHELON3_PROFILE_TEST_WIDTH', '16')),
                           'out_features': 2}},
        'data': {'train': dataset, 'test': {**dataset, 'config': {'length': 16}}},
        'dataloaders': {'train': loader, 'test': loader},
        'loss': [{'mse': {'module': 'torch.nn', 'type': 'MSELoss'}}],
        'optimizer': {'module': 'torch.optim', 'type': 'SGD', 'config': {'lr': 0.01}},
        'scheduler': {'module': 'torch.optim.lr_scheduler', 'type': 'StepLR',
                      'config': {'step_size': 1}},
        'trainer': {'module': module, 'type': 'ReportingTrainer', 'config': {
            'epochs': 3, 'precision': 'fp32', 'report_dir': folder, 'save_last': True,
            'profile': {'wait': 20, 'warmup': 3, 'active': 5, 'record_shapes': False,
                        'dir': str(Path(folder) / 'trace'), 'stop_after': True},
        }},
        'target': {'path': str(Path(folder) / 'target')},
    })
    from echelon3.cli.train import _train, trainer_app
    from echelon3.cli.finetune import _finetune, finetune_app
    if native_gpus:
        assert len(cfg.gpus) == 4 and len(set(cfg.gpus)) == 4
        (trainer_app if command == 'train' else finetune_app)(cfg)
        return
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch
    launch = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=4,
                          rdzv_backend='c10d', rdzv_endpoint='127.0.0.1:0',
                          rdzv_configs={'is_host': True},
                          run_id='echelon3-profile-test', max_restarts=0,
                          start_method='spawn', monitor_interval=0.1)
    elastic_launch(launch, _train if command == 'train' else _finetune)(cfg)


if __name__ == '__main__':
    main(*sys.argv[1:])
