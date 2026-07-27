import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
from omegaconf import DictConfig
from colorama import Fore, Style

from echelon3.checkpoint.manager import CHECKPOINT_MODEL_KEYWORD
from echelon3 import __title__, __version__
from echelon3 import ddp
from echelon3 import runtime
from echelon3.cli import add_cwd_to_sys_path, build_cli

from echelon3.creator import (
    create_single_augment,
    create_universal,
    create_evaluator,
    create_net,
    create_checkpoint_manager,
    create_single_preprocess,
    create_single_dataset,
    create_single_dataloader,
)


def _resolve_test_sets(data_test, loaders_test):
    """Normalise `data.test` / `dataloaders.test` into a list of ``(name, data_cfg,
    loader_cfg)``. Accepts BOTH formats `train` accepts:

      * a single test set — `module`/`type`/`config` at the top level (legacy `evaluate`);
      * a dict of named sets — ``{test: {...}, test_ms: {...}}`` — with a matching
        `dataloaders.test` entry per name.

    Raises a clear error when named sets lack their loaders, instead of the opaque
    ``Missing key module`` that `create_single_dataset` produced for a dict `data.test`."""
    if "module" in data_test and "type" in data_test:
        return [("test", data_test, loaders_test)]
    names = list(data_test.keys())
    if not names:
        raise RuntimeError("data.test is empty — nothing to evaluate.")
    missing = [n for n in names if n not in loaders_test]
    if missing:
        raise RuntimeError(
            f"data.test declares named test sets {names}, but dataloaders.test has no loader "
            f"for {missing}. Give each named test set a matching dataloaders.test entry "
            f"(same layout as `train`)."
        )
    return [(n, data_test[n], loaders_test[n]) for n in names]


def evaluate_app(cfg: DictConfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(Fore.CYAN)
    print(f"\n\n{__title__} {__version__}: pre-export evaluate.\n\n")

    print(f"--> Initializing network... ")
    net = create_net(cfg.net)
    net.to(device)
    print(Fore.LIGHTGREEN_EX, end="")
    print(f"        {type(net).__name__}({cfg.net.config})")
    print(Fore.CYAN, end="")

    print(f"--> Initializing checkpoint manager... ")
    ckpt_manager = create_checkpoint_manager(cfg.target)
    print(Fore.LIGHTGREEN_EX, end="")
    print(f"        {type(ckpt_manager).__name__}({cfg.target})")
    print(Fore.CYAN, end="")

    print(f"--> Loading latest checkpoint... ")
    print(Fore.LIGHTGREEN_EX, end="")
    ckpt, num = ckpt_manager.load_latest_checkpoint()
    ddp.load_state_dict_flexible(net, ckpt[CHECKPOINT_MODEL_KEYWORD])
    if hasattr(net, "reread_hypers"):
        net.reread_hypers(**cfg.net.config)

    net.eval()
    print(f"--> Loaded {num} checkpoint. ")
    print(Fore.CYAN, end="")

    print(f"--> Initializing test augmentations... ")
    test_augment = create_single_augment(
        cfg.transform.test.augment if "augment" in cfg.transform.test else None
    )
    print(Fore.LIGHTGREEN_EX, end="")
    for tr in test_augment:
        print(f"        {tr}")
    print(Fore.CYAN, end="")

    print(f"--> Initializing test preprocess... ")
    test_preprocess = create_single_preprocess(cfg.transform.test.preprocess)
    print(Fore.LIGHTGREEN_EX, end="")
    for tr in test_preprocess:
        print(f"        {tr}")
    print(Fore.CYAN, end="")

    # TF32 + AMP inference: bf16 by default on supporting GPUs (precision: fp32 to disable).
    # autocast around the top-level call covers all Evaluators.
    runtime.setup_fast_matmul(
        tf32=cfg.get("tf32", True), cudnn_benchmark=cfg.get("cudnn_benchmark", True)
    )
    _dtype = runtime.resolve_amp_dtype(cfg.get("precision", "auto"), device=device)
    print(f"--> Precision: {runtime.precision_label(_dtype)}")

    def _build_metric():
        for m in cfg.metrics:
            if list(m.keys())[0] == cfg.evaluator.metric:
                return create_universal(m[cfg.evaluator.metric])
        raise RuntimeError(f"Metric '{cfg.evaluator.metric}' not found in cfg.metrics.")

    # data.test / dataloaders.test may be a single set OR a named dict of sets — the same
    # two formats `train` accepts. Evaluate each set with its own fresh metric + evaluator.
    single = "module" in cfg.data.test and "type" in cfg.data.test
    test_sets = _resolve_test_sets(cfg.data.test, cfg.dataloaders.test)

    for set_name, data_cfg, loader_cfg in test_sets:
        tag = "" if single else f" [{set_name}]"

        print(f"--> Initializing test dataset{tag} (data.test)... ")
        test_dataset = create_single_dataset(
            data_cfg, augment=test_augment, preprocess=test_preprocess
        )
        print(Fore.LIGHTGREEN_EX, end="")
        print(f"        {test_dataset}")
        print(Fore.CYAN, end="")

        print(f"--> Initializing test dataloader{tag}... ")
        test_dataloader = create_single_dataloader(loader_cfg, test_dataset)
        print(Fore.LIGHTGREEN_EX, end="")
        print(f"        {type(test_dataloader).__name__}({loader_cfg.get('config', {})})")
        print(Fore.CYAN, end="")

        metric = _build_metric()   # fresh per set — no cross-set state accumulation

        print(f"--> Initializing evaluator{tag}... ")
        evaluator = create_evaluator(
            cfg.evaluator,
            net=net,
            train_dataloader=None,
            test_dataloader=test_dataloader,
            metric=metric,
            preprocess=test_preprocess,
            postprocess=[],
        )

        print(f"--> Evaluating{tag}... ")
        with torch.autocast("cuda", dtype=_dtype or torch.bfloat16, enabled=_dtype is not None):
            val_metric = evaluator.evaluate()
        print(Fore.LIGHTGREEN_EX, end="")
        print(f"Validation{tag} {cfg.evaluator.metric}: {val_metric}")
        print(Fore.CYAN, end="")

    print(Style.RESET_ALL)


main = build_cli(evaluate_app)  # click CLI + OmegaConf overrides (in place of @hydra.main)


if __name__ == "__main__":
    main()
