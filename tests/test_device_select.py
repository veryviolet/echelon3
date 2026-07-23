"""resolve_single_device: single (non-DDP) mode must honour the gpus[0] index
rather than silently landing on cuda:0 (bug 0.7.8 — gpus=[1] loaded GPU 0). cfg.device=cpu
and the absence of CUDA take precedence. No GPU required: CUDA availability is passed as a
parameter, and torch.device(...) does not touch the driver."""
from omegaconf import OmegaConf

from echelon3.cli import resolve_single_device


def _c(d):
    return OmegaConf.create(d)


def test_gpus_index_is_honoured():
    assert str(resolve_single_device(_c({"device": "cuda", "gpus": [1]}), True)) == "cuda:1"


def test_gpus_index_without_device_key():
    assert str(resolve_single_device(_c({"gpus": [2]}), True)) == "cuda:2"


def test_cpu_override_wins_over_gpus():
    assert str(resolve_single_device(_c({"device": "cpu", "gpus": [1]}), True)) == "cpu"


def test_no_gpus_defaults_to_cuda():
    assert str(resolve_single_device(_c({"device": "cuda"}), True)) == "cuda"


def test_explicit_device_index_respected_without_gpus():
    assert str(resolve_single_device(_c({"device": "cuda:3"}), True)) == "cuda:3"


def test_cpu_when_cuda_unavailable():
    assert str(resolve_single_device(_c({"device": "cuda", "gpus": [1]}), False)) == "cpu"
