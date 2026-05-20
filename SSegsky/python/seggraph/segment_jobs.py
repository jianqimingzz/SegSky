from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .dumpers import dump_hnsw_L0, dump_hnsw_all_levels, dump_knn_base
from .fvecs_io import read_fvecs_slice
from .hnsw_builder import build_hnsw_faiss


@dataclass
class SegmentBuildJob:
    seg_id: int
    fvecs_path: str
    start: int
    count: int
    base_id: int
    out_dir: str
    M: int
    efC: int
    metric: str
    dump_all_levels: bool
    dump_knn_base_flag: bool
    topN: int
    efS: int
    faiss_threads: int


def _set_thread_env(num_threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)


def run_segment_job(job: SegmentBuildJob) -> dict:
    """
    Build one segment/window HNSW graph and dump requested outputs.

    Timing policy:
    - read and dump are excluded
    - build_s only measures HNSW construction time
    """
    _set_thread_env(job.faiss_threads)

    import faiss

    faiss.omp_set_num_threads(int(job.faiss_threads))

    # read vectors (not counted in build_s)
    X = read_fvecs_slice(job.fvecs_path, job.start, job.count).astype(np.float32, copy=False)

    # build HNSW only
    t0 = time.perf_counter()
    index = build_hnsw_faiss(X, job.M, job.efC, job.metric)
    build_s = time.perf_counter() - t0

    out_dir = Path(job.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adj_l0_path = out_dir / f"seg_{job.seg_id:05d}_adj_L0.txt"
    dump_hnsw_L0(index, adj_l0_path, base_id=job.base_id)

    all_levels_path: Optional[Path] = None
    if job.dump_all_levels:
        all_levels_path = out_dir / f"seg_{job.seg_id:05d}_adj_all_levels.txt"
        dump_hnsw_all_levels(index, all_levels_path, base_id=job.base_id)

    knn_path: Optional[Path] = None
    if job.dump_knn_base_flag:
        knn_path = out_dir / f"seg_{job.seg_id:05d}_knn_base_top{job.topN}.txt"
        dump_knn_base(index, X, knn_path, topN=job.topN, base_id=job.base_id, efS=job.efS)

    return {
        "seg_id": job.seg_id,
        "start": job.start,
        "count": int(index.ntotal),
        "base_id": job.base_id,
        "adj_L0": str(adj_l0_path),
        "adj_all_levels": str(all_levels_path) if all_levels_path is not None else None,
        "knn_base": str(knn_path) if knn_path is not None else None,
        "build_s": build_s,
    }