"""Subprocess fixture exercising train/finetune interruption under elastic DDP."""
import json
import os
from pathlib import Path
import sys
import time
from functools import partial

import torch
from omegaconf import OmegaConf

from echelon3 import ddp
from echelon3.trainers.baseline import Trainer, _NullLogger


def write_report(folder, name, value):
    path = Path(folder) / name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


class ProbeDataset(torch.utils.data.Dataset):
    def __init__(self, length=1024, augment=None, preprocess=None):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        time.sleep(0.01)
        return torch.ones(16), torch.zeros(2)


class ProbeLogger(_NullLogger):
    def __init__(self, folder):
        pass


def mark_worker_ready(worker_id, *, original, folder):
    original(worker_id)
    write_report(folder, f'worker-ready-{os.getpid()}.json', {'pid': os.getpid()})


class ProbeTrainer(Trainer):
    def __init__(self, report_dir, **kwargs):
        super().__init__(**kwargs)
        self.report_dir = report_dir
        self.retained_iterators = []
        for loader in [self._train_loader, *self._test_loaders.values()]:
            loader.worker_init_fn = partial(mark_worker_ready, original=loader.worker_init_fn,
                                            folder=report_dir)

    def one_step_train(self, source, labels):
        super().one_step_train(source, labels)
        if not self.retained_iterators:
            # A traceback in next(iterator), or caller code, can retain this
            # reference. Make that case deterministic instead of timing SIGINT
            # to land at a particular Python instruction.
            loaders = [self._train_loader, *self._test_loaders.values()]
            self.retained_iterators = [loader._iterator for loader in loaders]
            workers = [w for it in self.retained_iterators for w in it._workers]
            # A tiny validation shard may only request data from worker 0.
            # Worker 1 can still be importing modules when the first train batch
            # arrives. This test targets initialized persistent workers, so wait
            # for every initializer before allowing the controller to send SIGINT.
            deadline = time.monotonic() + 60
            while not all(Path(self.report_dir, f'worker-ready-{w.pid}.json').exists()
                          for w in workers):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Probe workers did not finish initialization')
                time.sleep(0.02)
            write_report(self.report_dir, f'ready-{ddp.rank()}.json', {
                'rank_pid': os.getpid(), 'worker_pids': [w.pid for w in workers],
                'backend': torch.distributed.get_backend(),
                'pin_memory': self._train_loader._iterator._pin_memory,
            })
        time.sleep(0.03)

    def close(self):
        workers = [w for it in self.retained_iterators for w in it._workers]
        super().close()
        write_report(self.report_dir, f'closed-{ddp.rank()}.json', [
            {'pid': w.pid, 'alive': w.is_alive(), 'exitcode': w.exitcode} for w in workers
        ])


def main(folder, command, backend):
    module = '_ddp_interrupt_probe'
    dataset = {'module': module, 'type': 'ProbeDataset'}
    loader = {'module': 'torch.utils.data', 'type': 'DataLoader', 'config': {
        'batch_size': 8, 'num_workers': 2, 'persistent_workers': True,
        'prefetch_factor': 2, 'pin_memory': backend == 'nccl',
    }}
    cfg = OmegaConf.create({
        'device': 'cuda' if backend == 'nccl' else 'cpu',
        'gpus': [0, 1],
        'net': {'module': 'torch.nn', 'type': 'Linear',
                'config': {'in_features': 16, 'out_features': 2}},
        'data': {'train': dataset, 'test': {**dataset, 'config': {'length': 8}}},
        'dataloaders': {'train': loader, 'test': loader},
        'loss': [{'mse': {'module': 'torch.nn', 'type': 'MSELoss'}}],
        'optimizer': {'module': 'torch.optim', 'type': 'SGD', 'config': {'lr': 0.01}},
        'trainer': {'module': module, 'type': 'ProbeTrainer', 'config': {
            'epochs': 100, 'precision': 'fp32', 'report_dir': folder,
        }},
        'target': {'path': str(Path(folder) / 'target')},
        'mlops': {'module': module, 'type': 'ProbeLogger'},
    })
    from echelon3.cli.train import trainer_app, _train
    from echelon3.cli.finetune import finetune_app, _finetune
    if backend == 'nccl':
        # Native Echelon3 launcher on explicitly selected physical GPUs.
        cfg.gpus = [int(g) for g in os.environ['ECHELON3_TEST_DDP_GPUS'].split(',')]
        (trainer_app if command == 'train' else finetune_app)(cfg)
    else:
        # Native launcher intentionally does not launch CPU DDP. Use the same
        # elastic launcher to exercise Echelon3's real rank/CLI shutdown path.
        from torch.distributed.launcher.api import LaunchConfig, elastic_launch
        from torch.distributed.elastic.multiprocessing.api import SignalException
        launch = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=2,
                              rdzv_backend='c10d', rdzv_endpoint='localhost:0',
                              run_id='echelon3-interrupt-test', max_restarts=0,
                              start_method='spawn', monitor_interval=0.1)
        try:
            elastic_launch(launch, _train if command == 'train' else _finetune)(cfg)
        except (KeyboardInterrupt, SignalException):
            raise SystemExit(130)


if __name__ == '__main__':
    main(*sys.argv[1:])
