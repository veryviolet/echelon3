import os as _os
# albumentations на импорте/первом использовании лезет в сеть за «version info»
# (check_version.py). Без интернета это виснет на SSL-timeout и спамит UserWarning'ами
# каждую эпоху. Отключаем ДО импорта albumentations (creator импортит его позже).
# setdefault — пользователь может переопределить, если ему это зачем-то нужно.
_os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

from importlib.metadata import PackageNotFoundError, version

__title__ = 'Echelon3'

try:
    __version__ = version('echelon3')
except PackageNotFoundError:  # запущено из чекаута без установки
    __version__ = '0.0.0.dev0'
