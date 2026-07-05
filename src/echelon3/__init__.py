from importlib.metadata import PackageNotFoundError, version

__title__ = 'Echelon3'

try:
    __version__ = version('echelon3')
except PackageNotFoundError:  # запущено из чекаута без установки
    __version__ = '0.0.0.dev0'
