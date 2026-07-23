import os
import glob
import re
import json
import torch

CHECKPOINTS_PREFIX = 'checkpoint-'
CHECKPOINTS_EXT = '.tar'

class CheckpointManager:

    _checkpoints_to_keep = None
    _path = None

    _idxs = None

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, '_' + key, val)

        self.init_storage()

    def init_storage(self):
        if os.path.exists(self._path):
            ckpt_files = list(glob.glob(os.path.join(self._path, CHECKPOINTS_PREFIX+'*'+CHECKPOINTS_EXT)))
            self._idxs = [int(re.sub(r'\D', '', os.path.basename(f))) for f in ckpt_files]
        else:
            self._idxs = []
            os.makedirs(self._path, exist_ok=True)

    def reset_storage(self):
        pass

    def load_latest_checkpoint(self, cpu_only=False):
        latest_idx = max([int(x) for x in self._idxs])
        fname = os.path.join(self._path, CHECKPOINTS_PREFIX + f'{latest_idx}' + CHECKPOINTS_EXT)

        device = torch.device('cpu') if cpu_only or not torch.cuda.is_available() else torch.device('cuda')
        with open(fname, 'rb') as f:
            ckpt = torch.load(
                f,
                map_location=device,
                weights_only=False  # IMPORTANT: allow loading full objects (model, optimizer, metrics, etc.)
            )

        return ckpt, latest_idx

    def load_checkpoint(self, idx: int, cpu_only: bool = False):
        if idx not in self._idxs:
            raise ValueError(f'Checkpoint with idx={idx} not found in {self._path}')

        fname = os.path.join(self._path, CHECKPOINTS_PREFIX + f'{idx}' + CHECKPOINTS_EXT)
        device = torch.device('cpu') if cpu_only or not torch.cuda.is_available() else torch.device('cuda')
        with open(fname, 'rb') as f:
            ckpt = torch.load(
                f,
                map_location=device,
                weights_only=False
            )
        return ckpt

    def save_checkpoint(self, ckpt: dict):
        new_number = max(self._idxs) + 1 if len(self._idxs) > 0 else 1
        with open(os.path.join(self._path, f'{CHECKPOINTS_PREFIX}{new_number}{CHECKPOINTS_EXT}'), 'wb') as f:
            torch.save(ckpt, f)
        self._idxs.append(new_number)

    @property
    def path(self):
        return self._path

    @property
    def idxs(self):
        return self._idxs


CHECKPOINT_EPOCH_KEYWORD = 'epoch'
CHECKPOINT_MODEL_KEYWORD = 'model_state_dict'
CHECKPOINT_OPTIMIZER_KEYWORD = 'optimizer_state_dict'
CHECKPOINT_SCHEDULER_KEYWORD = 'scheduler_state_dict'
CHECKPOINT_METRICS_KEYWORD = 'metrics'
CHECKPOINT_SCALER_KEYWORD = 'scaler_state_dict'