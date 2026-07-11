"""CLI-оверрайды на OmegaConf (замена Hydra): key=/+/++/~, типизация, вложенность,
игнор hydra.*. GPU/обучение не нужны."""
from omegaconf import OmegaConf

from echelon3.cli.config import apply_overrides, _parse_value


def _c(d):
    return OmegaConf.create(d)


def test_override_existing_nested():
    cfg = _c({"data": {"train": {"config": {"root": "x"}}}})
    apply_overrides(cfg, ["data.train.config.root=/home/violet/data"])
    assert cfg.data.train.config.root == "/home/violet/data"


def test_plain_key_adds_without_plus():
    cfg = _c({"a": 1})
    apply_overrides(cfg, ["trainer.config.epochs=200"])
    assert cfg.trainer.config.epochs == 200


def test_plus_adds_list():
    cfg = _c({"a": 1})
    apply_overrides(cfg, ["+gpus=[0,1,2,3]"])
    assert list(cfg.gpus) == [0, 1, 2, 3]


def test_plusplus_add_or_override():
    cfg = _c({"a": 1})
    apply_overrides(cfg, ["++trainer.config.compile=true"])
    assert cfg.trainer.config.compile is True


def test_tilde_deletes():
    cfg = _c({"scheduler": {"x": 1}, "keep": 2})
    apply_overrides(cfg, ["~scheduler"])
    assert "scheduler" not in cfg and cfg.keep == 2


def test_types_inferred():
    assert _parse_value("0.00005") == 5e-05
    assert _parse_value("true") is True
    assert _parse_value("null") is None
    assert list(_parse_value("[1,2,3]")) == [1, 2, 3]
    assert _parse_value("reduce-overhead") == "reduce-overhead"


def test_hydra_override_is_noop():
    cfg = _c({"a": 1})
    apply_overrides(cfg, ["hydra.run.dir=/tmp/x", "hydra.job.chdir=false"])
    assert "hydra" not in cfg and cfg.a == 1


def test_env_interpolation_preserved():
    cfg = _c({"p": "orig"})
    apply_overrides(cfg, ["p=${oc.env:MY_ROOT,/def}"])
    # интерполяция сохраняется как есть, резолвится позже (to_container(resolve=True))
    assert OmegaConf.to_container(cfg, resolve=True)["p"] == "/def"
