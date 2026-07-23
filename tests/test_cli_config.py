"""CLI overrides on OmegaConf (a Hydra replacement): key=/+/++/~, typing, nesting,
ignoring hydra.*. No GPU/training needed."""
import pytest
from omegaconf import OmegaConf

from echelon3.cli.config import apply_overrides, _parse_value, load_config


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
    # the interpolation is kept as-is and resolved later (to_container(resolve=True))
    assert OmegaConf.to_container(cfg, resolve=True)["p"] == "/def"


# ---- defaults: composition ---------------------------------------------
def test_defaults_base_inheritance(tmp_path):
    (tmp_path / "base.yaml").write_text("a: 1\nb:\n  x: 10\nkeep: base\n")
    (tmp_path / "child.yaml").write_text(
        "defaults:\n  - base\n  - _self_\nb:\n  x: 20\n  y: 30\nkeep: child\n")
    cfg = load_config("child", str(tmp_path))
    assert cfg.a == 1            # inherited from base
    assert cfg.b.x == 20         # child overrides
    assert cfg.b.y == 30         # child adds
    assert cfg.keep == "child"   # _self_ last → child wins


def test_defaults_self_first_base_last_wins(tmp_path):
    (tmp_path / "base.yaml").write_text("k: base\n")
    (tmp_path / "child.yaml").write_text("defaults:\n  - _self_\n  - base\nk: child\n")
    assert load_config("child", str(tmp_path)).k == "base"  # base after _self_


def test_defaults_implicit_self_last(tmp_path):
    (tmp_path / "base.yaml").write_text("k: base\nonly_base: 1\n")
    (tmp_path / "child.yaml").write_text("defaults:\n  - base\nk: child\n")
    cfg = load_config("child", str(tmp_path))
    assert cfg.k == "child" and cfg.only_base == 1  # _self_ implicitly last


def test_defaults_config_group_under_key(tmp_path):
    (tmp_path / "net").mkdir()
    (tmp_path / "net" / "resnet.yaml").write_text("module: m\ntype: ResNet\nconfig:\n  depth: 18\n")
    (tmp_path / "cfg.yaml").write_text("defaults:\n  - net: resnet\n  - _self_\nlr: 0.1\n")
    cfg = load_config("cfg", str(tmp_path))
    assert cfg.net.type == "ResNet" and cfg.net.config.depth == 18 and cfg.lr == 0.1


def test_defaults_nested_config_group(tmp_path):
    (tmp_path / "data" / "aug").mkdir(parents=True)
    (tmp_path / "data" / "aug" / "heavy.yaml").write_text("p: 0.9\n")
    (tmp_path / "cfg.yaml").write_text("defaults:\n  - data/aug: heavy\n")
    assert load_config("cfg", str(tmp_path)).data.aug.p == 0.9  # under data.aug


def test_defaults_recursive_base(tmp_path):
    (tmp_path / "root.yaml").write_text("a: 1\n")
    (tmp_path / "mid.yaml").write_text("defaults:\n  - root\n  - _self_\nb: 2\n")
    (tmp_path / "leaf.yaml").write_text("defaults:\n  - mid\n  - _self_\nc: 3\n")
    cfg = load_config("leaf", str(tmp_path))
    assert cfg.a == 1 and cfg.b == 2 and cfg.c == 3  # transitively


def test_defaults_cycle_detected(tmp_path):
    (tmp_path / "x.yaml").write_text("defaults:\n  - y\n")
    (tmp_path / "y.yaml").write_text("defaults:\n  - x\n")
    with pytest.raises(ValueError):
        load_config("x", str(tmp_path))


def test_defaults_bad_entry_raises(tmp_path):
    (tmp_path / "cfg.yaml").write_text("defaults:\n  - [1, 2]\n")
    with pytest.raises(ValueError):
        load_config("cfg", str(tmp_path))
