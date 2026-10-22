"""Inject TensorBoard ENOSPC without filling a filesystem or using a GPU."""
import errno
import os
from pathlib import Path
import sys
import time

from omegaconf import OmegaConf

from echelon3 import cli, ddp
from echelon3.mlops.tensorboard import TensorboardLogger
from _ddp_interrupt_probe import ProbeTrainer, write_report


class BrokenStderr:
    def write(self, value):
        raise OSError(errno.ENOSPC, 'injected stderr: No space left on device')

    def flush(self):
        raise OSError(errno.ENOSPC, 'injected stderr: No space left on device')


class FailingTensorboardLogger(TensorboardLogger):
    def __init__(self, folder, report_dir, scenario):
        super().__init__(folder)
        self.report_dir = Path(report_dir)
        self.scenario = scenario

    def _disk_full(self, data):
        if self.scenario == 'stderr':
            sys.stderr = BrokenStderr()
        write_report(self.report_dir, 'failed.json', {'pid': os.getpid(), 'time': time.monotonic()})
        raise OSError(errno.ENOSPC, 'injected TensorBoard: No space left on device')

    def log_train_losses(self, step, losses):
        if step >= 2:
            while not (self.report_dir / 'inject').exists():
                time.sleep(0.01)
            # The real TensorBoard background worker fails, then add_scalar
            # re-raises that failure on the training thread on a later write.
            writer = self.train_summary_writer._get_file_writer().event_writer._async_writer
            writer._writer.write = self._disk_full
        super().log_train_losses(step, losses)


class FailureTrainer(ProbeTrainer):
    def __init__(self, scenario, **kwargs):
        super().__init__(**kwargs)
        self.scenario = scenario
        if scenario == 'gil':
            cli._FATAL_EXIT_TIMEOUT = 2.0  # Accelerate only the subprocess regression.
        if scenario == 'error_file':
            from torch.distributed.elastic.multiprocessing.errors.error_handler import ErrorHandler

            def cannot_record(self, exc):
                raise OSError(errno.ENOSPC, 'injected error file: No space left on device')

            ErrorHandler.record_exception = cannot_record

    def close(self):
        if self.scenario == 'gil' and ddp.rank() == 0:
            # Unlike time.sleep(), PyDLL keeps the GIL. A Python thread timer
            # or Thread.join(timeout) cannot regain control to exit this rank.
            import ctypes
            ctypes.PyDLL(None).sleep(60)
        super().close()


def main(folder, command, scenario):
    dataset = {'module': '_ddp_interrupt_probe', 'type': 'ProbeDataset'}
    loader = {'module': 'torch.utils.data', 'type': 'DataLoader', 'config': {
        'batch_size': 8, 'num_workers': 2, 'persistent_workers': True,
    }}
    cfg = OmegaConf.create({
        'device': 'cpu',
        'net': {'module': 'torch.nn', 'type': 'Linear', 'config': {'in_features': 16, 'out_features': 2}},
        'data': {'train': dataset, 'test': {**dataset, 'config': {'length': 8}}},
        'dataloaders': {'train': loader, 'test': loader},
        'optimizer': {'module': 'torch.optim', 'type': 'SGD', 'config': {'lr': 0.01}},
        'loss': [{'mse': {'module': 'torch.nn', 'type': 'MSELoss'}}],
        'trainer': {'module': '_ddp_failure_probe', 'type': 'FailureTrainer', 'config': {
            'epochs': 100, 'precision': 'fp32', 'report_dir': folder, 'scenario': scenario,
        }},
        'target': {'path': str(Path(folder) / 'target'), 'checkpoints_to_keep': 1},
        'mlops': {'module': '_ddp_failure_probe', 'type': 'FailingTensorboardLogger',
                  'config': {'report_dir': folder, 'scenario': scenario}},
    })
    from echelon3.cli.train import _train
    from echelon3.cli.finetune import _finetune
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch
    from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs
    launch = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=2,
                          rdzv_backend='c10d', rdzv_endpoint='localhost:0',
                          run_id='echelon3-enospc-test', max_restarts=0,
                          start_method='spawn', monitor_interval=0.1,
                          logs_specs=DefaultLogsSpecs(log_dir=str(Path(folder) / 'elastic')))
    elastic_launch(launch, _train if command == 'train' else _finetune)(cfg)


if __name__ == '__main__':
    main(*sys.argv[1:])
