"""Friendly message for `python -m echelon3.cli` (a package can't be run directly).

echelon3 is invoked through the `echelon3` console script, not the package."""
import sys

sys.stderr.write(
    "echelon3.cli is a package and is not run directly. Use the `echelon3` command:\n"
    "  echelon3 train    -cd <config_dir> -cn <config_name> [key=value ...]\n"
    "  echelon3 finetune | evaluate | export | run\n"
    "(if `echelon3` is not on PATH: `python -m echelon3.cli.train ...`)\n"
)
sys.exit(2)
