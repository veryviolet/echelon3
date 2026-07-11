import torch
from omegaconf import DictConfig
from colorama import Fore, Style

from echelon3.checkpoint.manager import CHECKPOINT_MODEL_KEYWORD
from echelon3 import __title__, __version__
from echelon3 import ddp
from echelon3.cli import add_cwd_to_sys_path, build_cli

from echelon3.creator import create_net, create_checkpoint_manager, create_exporters


def exporter_app(cfg: DictConfig):

    device = torch.device('cpu')

    print(Fore.CYAN)

    print(f'\n\n{__title__} {__version__}: exporter.\n\n')

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
        ckpt, num = ckpt_manager.load_latest_checkpoint(cpu_only=True)
        # Снимает устаревший префикс 'module.' старых DataParallel/DDP-чекпоинтов.
        ddp.load_state_dict_flexible(net, ckpt[CHECKPOINT_MODEL_KEYWORD])

        print(f'--> Loaded {num} checkpoint. ')
        print(Fore.CYAN, end='')
    else:
        print(f'--> No target specified. Omitting checkpoint loading. ')

    print(f'--> Creating exporters... ')
    exporters = create_exporters(cfg.export, net=net)
    print(Fore.LIGHTGREEN_EX, end='')
    for name, ex in exporters.items():
        print(f'        {name}: {type(ex).__name__}')
    print(Fore.CYAN, end='')

    for name, ex in exporters.items():
        print(f'--> Exporting {name}... ')
        ex.export()

    print(Style.RESET_ALL)


main = build_cli(exporter_app)  # click-CLI + OmegaConf-оверрайды (взамен @hydra.main)


if __name__ == "__main__":
    main()
