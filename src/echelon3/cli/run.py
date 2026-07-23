import torch
from omegaconf import DictConfig
from colorama import Fore, Style

from echelon3.checkpoint.manager import CHECKPOINT_MODEL_KEYWORD
from echelon3 import __title__, __version__
from echelon3 import ddp
from echelon3 import runtime
from echelon3.cli import add_cwd_to_sys_path, build_cli

from echelon3.creator import create_net, create_checkpoint_manager
from echelon3.creator import create_single_preprocess, create_universal, create_wrapper


def runner_app(cfg: DictConfig):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(Fore.CYAN)

    print(f'\n\n{__title__} {__version__}: runner.\n\n')

    print(f'--> Initializing network... ')
    net = create_net(cfg.net)
    net.to(device)
    print(Fore.LIGHTGREEN_EX, end='')
    print(f'        {type(net).__name__}({cfg.net.config})')
    print(Fore.CYAN, end='')

    if 'target' in cfg.keys():
        print(f'--> Initializing checkpoint manager... ')
        ckpt_manager = create_checkpoint_manager(cfg.target)
        print(Fore.LIGHTGREEN_EX, end='')
        print(f'        {type(ckpt_manager).__name__}({cfg.target})')
        print(Fore.CYAN, end='')

        print(f'--> Loading latest checkpoint... ')
        print(Fore.LIGHTGREEN_EX, end='')
        ckpt, num = ckpt_manager.load_latest_checkpoint()
        # Strips the stale 'module.' prefix of old DataParallel/DDP checkpoints.
        ddp.load_state_dict_flexible(net, ckpt[CHECKPOINT_MODEL_KEYWORD])

        net.to(device)
        net.eval()
        print(f'--> Loaded {num} checkpoint. ')
        print(Fore.CYAN, end='')
    else:
        print(f'--> No target specified. Omitting checkpoint loading. ')

    print(f'--> Creating preprocess... ')
    preprocess = create_single_preprocess(cfg.export.preprocess) if 'preprocess' in cfg.export.keys() else torch.nn.Identity()
    preprocess = preprocess.to(device)
    print(f'--> Creating postprocess... ')
    postprocess = create_single_preprocess(cfg.export.postprocess) if 'postprocess' in cfg.export.keys() else torch.nn.Identity()
    postprocess = postprocess.to(device)

    print(f'--> Creating wrapper... ')
    net = create_wrapper(cfg.export.wrapper, net) if 'wrapper' in cfg.export.keys() else net

    print(f'--> Creating runner... ')
    runner = create_universal(cfg.runner)

    print(f'--> Processing ... ')
    # TF32 + AMP inference (bf16 by default; precision: fp32 to disable).
    runtime.setup_fast_matmul(
        tf32=cfg.get('tf32', True), cudnn_benchmark=cfg.get('cudnn_benchmark', True)
    )
    _dtype = runtime.resolve_amp_dtype(cfg.get('precision', 'auto'), device=device)
    print(f'--> Precision: {runtime.precision_label(_dtype)}')
    with torch.autocast('cuda', dtype=_dtype or torch.bfloat16, enabled=_dtype is not None):
        runner.process(model=net, preprocess=preprocess, postprocess=postprocess)

    print(Style.RESET_ALL)


main = build_cli(runner_app)  # click CLI + OmegaConf overrides (in place of @hydra.main)


if __name__ == "__main__":
    main()
