"""Fork-based feature generation for the ARM competition server.

The graph model is read-only after fitting.  On Linux, ``fork`` lets workers
share those pages copy-on-write instead of loading one large pickle per shard.
"""

import multiprocessing as mp
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .io_data import TestRow, ensure_dir
from .temporal_graph import GraphFeatureModel


_MODEL = None
_ROWS = None


def worker_count(requested: int, reserve_cores: int = 8) -> int:
    available = max(1, (os.cpu_count() or 1) - max(0, int(reserve_cores)))
    return max(1, min(int(requested), available))


def _feature_bounds_worker(payload: tuple) -> str:
    shard, start, end, out_dir, prefix = payload
    if _MODEL is None or _ROWS is None:
        raise RuntimeError("fork worker did not inherit the feature model")
    features = _MODEL.feature_tensor(_ROWS[start:end], progress_every=20000)
    path = Path(out_dir) / f"{prefix}_{int(shard):03d}.npy"
    np.save(path, features.astype(np.float32, copy=False))
    return str(path)


def feature_tensor_parallel(
    model: GraphFeatureModel,
    rows: Sequence[TestRow],
    out_dir: Path,
    prefix: str,
    workers: int,
) -> np.ndarray:
    """Build an Nx100xF tensor using shared-memory fork workers on Linux."""
    out_dir = ensure_dir(out_dir)
    for path in out_dir.glob(f"{prefix}_*.npy"):
        path.unlink()
    workers = worker_count(workers)
    if len(rows) == 0:
        return np.zeros((0, 100, 0), dtype=np.float32)
    workers = min(workers, len(rows))
    if workers <= 1 or os.name != "posix":
        return model.feature_tensor(rows)

    global _MODEL, _ROWS
    _MODEL = model
    _ROWS = rows
    bounds = np.linspace(0, len(rows), workers + 1, dtype=np.int64)
    tasks = [
        (shard, int(bounds[shard]), int(bounds[shard + 1]), str(out_dir), prefix)
        for shard in range(workers)
        if int(bounds[shard + 1]) > int(bounds[shard])
    ]
    # 优化：去掉maxtasksperchild=1，提高进程复用率
    # 原因：maxtasksperchild=1导致每个worker只处理1个任务就退出，频繁进程创建/销毁
    with mp.get_context("fork").Pool(processes=len(tasks)) as pool:
        paths = pool.map(_feature_bounds_worker, tasks)
    return np.concatenate([np.load(path, mmap_mode="r") for path in paths], axis=0).astype(np.float32, copy=False)
