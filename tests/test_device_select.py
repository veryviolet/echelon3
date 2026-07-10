"""resolve_single_device: одиночный (не-DDP) режим должен уважать индекс gpus[0],
а не молча садиться на cuda:0 (баг 0.7.8 — gpus=[1] грузил GPU 0). cfg.device=cpu и
отсутствие CUDA имеют приоритет. GPU не требуется: доступность CUDA передаётся
параметром, а torch.device(...) не обращается к драйверу."""
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
