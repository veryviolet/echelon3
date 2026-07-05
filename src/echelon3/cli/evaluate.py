import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import hydra
import torch
from omegaconf import DictConfig
from colorama import Fore, Style

from echelon3.checkpoint.manager import CHECKPOINT_MODEL_KEYWORD
from echelon3 import __title__, __version__

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


@hydra.main(version_base=None, config_path=None)
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
    try:
        net.load_state_dict(ckpt[CHECKPOINT_MODEL_KEYWORD])
        if hasattr(net, "reread_hypers"):
            net.reread_hypers(**cfg.net.config)
    except Exception:
        net = torch.nn.DataParallel(net)
        net.load_state_dict(ckpt[CHECKPOINT_MODEL_KEYWORD])
        if hasattr(net.module, "reread_hypers"):
            net.module.reread_hypers(**cfg.net.config)

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

    print(f"--> Initializing validation dataset (data.test)... ")
    test_dataset = create_single_dataset(
        cfg.data.test, augment=test_augment, preprocess=test_preprocess
    )
    print(Fore.LIGHTGREEN_EX, end="")
    print(f"        {test_dataset}")
    print(Fore.CYAN, end="")

    print(f"--> Initializing validation dataloader... ")
    test_dataloader = create_single_dataloader(cfg.dataloaders.test, test_dataset)
    print(Fore.LIGHTGREEN_EX, end="")
    print(f"        {type(test_dataloader).__name__}({cfg.dataloaders.test.config})")
    print(Fore.CYAN, end="")

    print(f"--> Initializing metric... ")
    metric = None
    for m in cfg.metrics:
        name = list(m.keys())[0]
        if name == cfg.evaluator.metric:
            metric = create_universal(m[cfg.evaluator.metric])
            break
    if metric is None:
        raise RuntimeError(
            f"Metric '{cfg.evaluator.metric}' not found in cfg.metrics."
        )

    print(f"--> Initializing postprocess... ")
    postprocess = []

    print(f"--> Initializing evaluator... ")
    evaluator = create_evaluator(
        cfg.evaluator,
        net=net,
        train_dataloader=None,
        test_dataloader=test_dataloader,
        metric=metric,
        preprocess=test_preprocess,
        postprocess=postprocess,
    )

    print(Fore.LIGHTGREEN_EX, end="")
    print(f"        {type(evaluator).__name__}({cfg.evaluator})")
    print(Fore.CYAN, end="")

    print(f"--> Evaluating on validation (data.test)... ")
    val_metric = evaluator.evaluate()

    print(Fore.LIGHTGREEN_EX, end="")
    print(f"Validation {cfg.evaluator.metric}: {val_metric}")
    print(Fore.CYAN, end="")

    print(Style.RESET_ALL)


def main():
    evaluate_app()


if __name__ == "__main__":
    main()
