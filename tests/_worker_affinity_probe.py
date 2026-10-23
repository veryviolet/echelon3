"""CPU/Gloo probe through the production launcher and real worker initializers."""
import json
import os
from pathlib import Path
import sys

# Also applies in spawned ranks: the launcher deliberately sets visibility.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from echelon3 import cpu_affinity, ddp
from echelon3.cli import maybe_launch_ddp
from echelon3.creator import create_single_dataloader


class CpuMaskDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        # The user callback below also observes the engine-applied mask.
        assert self.callback_mask == sorted(os.sched_getaffinity(0))
        return self.callback_mask


def user_init(worker_id):
    torch.utils.data.get_worker_info().dataset.callback_mask = sorted(os.sched_getaffinity(0))


def rank_main(cfg):
    wide = cpu_affinity.launcher_cpus()
    local_rank = int(os.environ['LOCAL_RANK'])
    os.sched_setaffinity(0, {wide[local_rank]})
    ddp.init_ddp_if_needed()  # CUDA hidden: real two-rank Gloo group
    before = sorted(os.sched_getaffinity(0))
    results = {}
    for method in ('fork', 'spawn'):
        for name, setting in [('wide_worker', 'launcher'), ('default_worker', None),
                              ('explicit_worker', [wide[-1]])]:
            config = OmegaConf.create({'module': 'torch.utils.data', 'type': 'DataLoader',
                                       'config': {'num_workers': 1, 'batch_size': None,
                                                  'multiprocessing_context': method,
                                                  'worker_cpu_affinity': setting}})
            loader = create_single_dataloader(config, CpuMaskDataset())
            # Preserve the real, picklable engine initializer and compose a user callback.
            loader.worker_init_fn.keywords['_user_fn'] = user_init
            observed = list(loader)[0]
            expected = list(wide) if setting == 'launcher' else before if setting is None else setting
            assert observed == expected, (method, name, observed, expected)
            results[name] = observed
    results.update(launcher=list(wide), rank_before=before,
                   rank_after=sorted(os.sched_getaffinity(0)))
    Path(cfg.folder, f'rank-{local_rank}.json').write_text(json.dumps(results))
    ddp.shutdown()


if __name__ == '__main__':
    # Exercise native launcher selection without a GPU. Spawned ranks import a
    # fresh torch with CUDA hidden; their actual process group uses Gloo.
    torch.cuda.is_available = lambda: True
    torch.cuda.device_count = lambda: 2
    cfg = OmegaConf.create({'gpus': [0, 1], 'folder': sys.argv[1]})
    assert maybe_launch_ddp(cfg, rank_main)
