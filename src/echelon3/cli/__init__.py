import os
import sys


def add_cwd_to_sys_path():
    """Делает модули текущего каталога импортируемыми из конфигов.

    Рабочая модель zoo-репозиториев: пользователь работает из корня своего
    репо, конфиги ссылаются на локальные пакеты (module: my_zoo.nets.foo).
    Запуск `python script.py` кладёт каталог скрипта в sys.path автоматически,
    console-script — нет; выравниваем поведение.
    """
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
