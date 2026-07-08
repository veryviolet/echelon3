"""Регрессия 0.7.2: PDEATHSIG-worker_init_fn обязан быть picklable, иначе
DataLoader(multiprocessing_context='spawn') падает при пикле воркер-инициализатора
(замыкание непиклится). Держим его модульной функцией + functools.partial."""
import pickle

from echelon3.creator import _worker_init_fn


def test_worker_init_fn_is_picklable():
    pickle.loads(pickle.dumps(_worker_init_fn(None)))


def _noop_user_init(worker_id):
    pass


def test_worker_init_fn_composes_picklable_user_fn():
    pickle.loads(pickle.dumps(_worker_init_fn(_noop_user_init)))
