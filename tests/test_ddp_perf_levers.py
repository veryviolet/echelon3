"""DDP performance levers (for bandwidth-limited interconnects — PCIe without NVLink):
gradient accumulation (fewer, bigger all-reduces), bf16/fp16 comm hook (halved payload),
gradient_as_bucket_view, and NUMA/CPU affinity. Off by default = unchanged behavior.
"""
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from echelon3 import ddp
from echelon3.trainers.baseline import Trainer
from echelon3.checkpoint.manager import CheckpointManager


class _CountingSGD(torch.optim.SGD):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.n_steps = 0

    def step(self, *a, **k):
        self.n_steps += 1
        return super().step(*a, **k)


def _trainer(tmp, n_batches=4, batch=2, **kw):
    net = torch.nn.Linear(4, 4)
    loader = DataLoader(TensorDataset(torch.zeros(n_batches * batch, 4),
                                      torch.zeros(n_batches * batch, 4)), batch_size=batch)
    opt = _CountingSGD(net.parameters(), lr=0.01)
    tr = Trainer(
        epochs=1, train_dataloader=loader, test_dataloader=None, net=net,
        losses={"l1": (torch.nn.L1Loss(), 1.0)}, metrics={}, optimizer=opt, scheduler=None,
        ckpt_manager=CheckpointManager(path=str(tmp), checkpoints_to_keep=2),
        mlops_logger=None, device=torch.device("cpu"), times_to_validate_per_epoch=1, **kw,
    )
    return tr, opt


# ---- gradient accumulation ----

def test_grad_accum_steps_optimizer_every_n(tmp_path):
    tr, opt = _trainer(tmp_path, n_batches=4, grad_accum_steps=2)
    tr.train()
    assert opt.n_steps == 2                 # 4 micro-batches / accum 2 == 2 optimizer steps


def test_grad_accum_default_steps_every_batch(tmp_path):
    tr, opt = _trainer(tmp_path, n_batches=4)   # default grad_accum_steps=1
    assert tr._grad_accum_steps == 1
    tr.train()
    assert opt.n_steps == 4                 # one optimizer step per batch


# ---- DDP comm hook (bf16/fp16) ----

class _FakeDDPNet:
    def __init__(self):
        self.registered = None

    def register_comm_hook(self, state, hook):
        self.registered = hook


def test_comm_hook_registered_for_bf16(tmp_path):
    tr, _ = _trainer(tmp_path)
    tr._net = _FakeDDPNet()
    tr._register_ddp_comm_hook("bf16")
    assert tr._net.registered is not None          # a hook was registered


def test_comm_hook_noop_for_none(tmp_path):
    tr, _ = _trainer(tmp_path)
    tr._net = _FakeDDPNet()
    tr._register_ddp_comm_hook("none")
    assert tr._net.registered is None              # default fp32 all-reduce, no hook


# ---- gradient_as_bucket_view ----

def test_gradient_as_bucket_view_default_true(tmp_path):
    tr, _ = _trainer(tmp_path)
    assert tr._ddp_gradient_as_bucket_view is True


def test_gradient_as_bucket_view_overridable(tmp_path):
    tr, _ = _trainer(tmp_path, ddp_gradient_as_bucket_view=False)
    assert tr._ddp_gradient_as_bucket_view is False


# ---- NUMA affinity helpers ----

def test_parse_cpulist():
    assert ddp._parse_cpulist("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert ddp._parse_cpulist("5") == [5]
    assert ddp._parse_cpulist("") == []


def test_numa_helpers_are_graceful():
    # Never crash: an out-of-range GPU or an unresolved mapping just returns None/False.
    assert ddp.gpu_numa_cpus(999) is None
    assert ddp.set_numa_affinity(999) is False


def test_visible_to_physical_gpu_mapping(monkeypatch):
    # A CUDA-visible index must map to the PHYSICAL GPU (pynvml/nvidia-smi ignore
    # CUDA_VISIBLE_DEVICES); else affinity pins the wrong socket for e.g. gpus=[4,5].
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    assert ddp._visible_to_physical_gpu(0) == 4
    assert ddp._visible_to_physical_gpu(3) == 7
    assert ddp._visible_to_physical_gpu(9) is None            # out of range
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcd,GPU-efgh")
    assert ddp._visible_to_physical_gpu(0) is None            # UUID/MIG — can't map to an int
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert ddp._visible_to_physical_gpu(2) == 2               # no CVD → identity
