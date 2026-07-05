"""echelon3.data — datasets and package-wide OpenCV/BLAS thread defaults.

We force-disable OpenCV's internal thread pool here because every echelon3
dataset reads images via cv2 in DataLoader workers. With default settings,
each worker spins its own OpenCV thread pool, leading to N_workers ×
N_cv_threads contending for CPU/GIL — and, under heavy IO (large TIFFs etc.),
deadlocking the DataLoader queue.

Both fork- and spawn-style workers import this package when constructing a
dataset, so the setting propagates.
"""
import os

import cv2

cv2.setNumThreads(0)
try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def worker_init_no_cv_threads(worker_id: int) -> None:
    """Optional DataLoader worker_init_fn that re-asserts the single-thread
    cv2/BLAS settings inside each worker. The module import above usually
    suffices, but pass this as a belt-and-suspenders measure if hangs persist."""
    cv2.setNumThreads(0)
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
