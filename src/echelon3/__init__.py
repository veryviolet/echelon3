import os as _os
# On import / first use, albumentations reaches out to the network for "version info"
# (check_version.py). Without internet this hangs on an SSL timeout and spams UserWarnings
# every epoch. We disable it BEFORE importing albumentations (creator imports it later).
# setdefault — the user can override it if they need to for some reason.
_os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

from importlib.metadata import PackageNotFoundError, version

__title__ = 'Echelon3'

try:
    __version__ = version('echelon3')
except PackageNotFoundError:  # run from a checkout without installation
    __version__ = '0.0.0.dev0'
