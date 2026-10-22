import os
import glob
import re
import tempfile
import torch

CHECKPOINTS_PREFIX = 'checkpoint-'
CHECKPOINTS_EXT = '.tar'
LAST_CHECKPOINT = 'last.tar'

class CheckpointManager:

    _checkpoints_to_keep = None
    _path = None

    _idxs = None

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, '_' + key, val)

        if self._checkpoints_to_keep is not None and (
            isinstance(self._checkpoints_to_keep, bool)
            or not isinstance(self._checkpoints_to_keep, int)
            or self._checkpoints_to_keep < 1
        ):
            raise ValueError('checkpoints_to_keep must be a positive integer or null (keep all)')
        self.init_storage()

    def init_storage(self):
        if os.path.exists(self._path):
            ckpt_files = list(glob.glob(os.path.join(self._path, CHECKPOINTS_PREFIX+'*'+CHECKPOINTS_EXT)))
            self._idxs = sorted(
                int(match.group(1)) for f in ckpt_files
                if (match := re.fullmatch(r'checkpoint-(\d+)\.tar', os.path.basename(f)))
            )
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
        destination = os.path.join(self._path, f'{CHECKPOINTS_PREFIX}{new_number}{CHECKPOINTS_EXT}')
        self._save_atomic(ckpt, destination)
        self._idxs.append(new_number)
        if self._checkpoints_to_keep is not None:
            for idx in sorted(self._idxs)[:-self._checkpoints_to_keep]:
                os.remove(os.path.join(self._path, f'{CHECKPOINTS_PREFIX}{idx}{CHECKPOINTS_EXT}'))
                self._idxs.remove(idx)

    def save_last_checkpoint(self, ckpt: dict):
        """Replace the completed-epoch snapshot independently of numbered checkpoints."""
        self._save_atomic(ckpt, os.path.join(self._path, LAST_CHECKPOINT))

    def load_last_checkpoint(self, cpu_only=False):
        path = os.path.join(self._path, LAST_CHECKPOINT)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'resume_from=last requires {path}; enable save_last for a completed epoch first'
            )
        device = torch.device('cpu') if cpu_only or not torch.cuda.is_available() else torch.device('cuda')
        with open(path, 'rb') as f:
            return torch.load(f, map_location=device, weights_only=False)

    def _save_atomic(self, ckpt, destination):
        # Publish a complete checkpoint before pruning any previous good copies.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self._path, suffix='.tmp', delete=False) as f:
                temporary = f.name
                torch.save(ckpt, f)
            os.replace(temporary, destination)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.remove(temporary)

    @property
    def path(self):
        return self._path

    @property
    def idxs(self):
        return self._idxs


CHECKPOINT_EPOCH_KEYWORD = 'epoch'            # COMPLETED epochs at save time (0 = initial baseline)
CHECKPOINT_MODEL_KEYWORD = 'model_state_dict'
CHECKPOINT_OPTIMIZER_KEYWORD = 'optimizer_state_dict'
CHECKPOINT_SCHEDULER_KEYWORD = 'scheduler_state_dict'
CHECKPOINT_METRICS_KEYWORD = 'metrics'
CHECKPOINT_SCALER_KEYWORD = 'scaler_state_dict'
CHECKPOINT_GLOBAL_STEP_KEYWORD = 'global_step'  # also marks the new epoch=completed format
CHECKPOINT_BEST_METRICS_KEYWORD = 'best_metrics'
CHECKPOINT_SCHEDULER_STEPPED_KEYWORD = 'scheduler_step_completed'
