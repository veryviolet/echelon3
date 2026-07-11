"""Единый исполняемый ``echelon3`` — click-группа с подкомандами.

    echelon3 train    --config-dir . --config-name cfg  [key=value ...]
    echelon3 finetune --config-dir . --config-name cfg  [key=value ...]
    echelon3 run      ...
    echelon3 evaluate ...
    echelon3 export   ...

Ленивая группа: модуль подкоманды (тянущий torch и пр.) импортируется только при
её вызове, поэтому ``echelon3 --help`` быстрый. Отдельные ``echelon3-<cmd>``
оставлены временными алиасами."""
import importlib

import click

from echelon3 import __version__


class _LazyGroup(click.Group):
    _commands = {
        "train": "echelon3.cli.train:main",
        "finetune": "echelon3.cli.finetune:main",
        "run": "echelon3.cli.run:main",
        "evaluate": "echelon3.cli.evaluate:main",
        "export": "echelon3.cli.export:main",
    }

    def list_commands(self, ctx):
        return list(self._commands)

    def get_command(self, ctx, name):
        target = self._commands.get(name)
        if target is None:
            return None
        module_name, attr = target.split(":")
        return getattr(importlib.import_module(module_name), attr)


@click.group(cls=_LazyGroup, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="echelon3")
def cli():
    """echelon3 — config-driven PyTorch training (assemble nets/data/losses from YAML)."""


if __name__ == "__main__":
    cli()
