"""Regression 0.7.2: the PDEATHSIG worker_init_fn must be picklable, otherwise
DataLoader(multiprocessing_context='spawn') fails when pickling the worker
initializer (a closure is not picklable). We keep it a module-level function +
functools.partial."""
import pickle

from echelon3.creator import _worker_init_fn


def test_worker_init_fn_is_picklable():
    pickle.loads(pickle.dumps(_worker_init_fn(None)))


def _noop_user_init(worker_id):
    pass


def test_worker_init_fn_composes_picklable_user_fn():
    pickle.loads(pickle.dumps(_worker_init_fn(_noop_user_init)))
