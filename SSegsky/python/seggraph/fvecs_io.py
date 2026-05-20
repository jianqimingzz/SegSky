from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

_FVECS_MEMMAP_CACHE: dict[str, np.memmap] = {}
_FVECS_META_CACHE: dict[str, Tuple[int, int, int]] = {}


def _path_key(path: str | Path) -> str:
    return str(Path(path).resolve())


def fvecs_meta(path: str | Path) -> Tuple[int, int, int]:
    """
    Return metadata of a .fvecs file as:
        (num_vectors, dim, stride_int32)

    .fvecs layout:
        [dim(int32), x1(float32), x2(float32), ..., xd(float32)] repeated
    Since float32 and int32 are both 4 bytes, we can memmap the whole file
    as int32 and reinterpret the payload as float32.
    """
    key = _path_key(path)
    cached = _FVECS_META_CACHE.get(key)
    if cached is not None:
        return cached

    p = Path(key)
    if not p.exists():
        raise FileNotFoundError(f".fvecs not found: {p}")

    size_bytes = p.stat().st_size
    if size_bytes < 4:
        raise ValueError(f"Invalid .fvecs file (too small): {p}")

    with open(p, "rb") as f:
        head = f.read(4)
    d = int(np.frombuffer(head, dtype=np.int32)[0])
    if d <= 0:
        raise ValueError(f"Invalid dimension in .fvecs header: d={d} path={p}")

    stride = d + 1
    rec_bytes = stride * 4
    if size_bytes % rec_bytes != 0:
        raise ValueError(
            f"Invalid .fvecs file: total bytes {size_bytes} is not divisible by record bytes {rec_bytes}."
        )

    n = size_bytes // rec_bytes
    meta = (n, d, stride)
    _FVECS_META_CACHE[key] = meta
    return meta


def _get_int32_memmap(path: str | Path) -> tuple[np.memmap, str]:
    key = _path_key(path)
    mm = _FVECS_MEMMAP_CACHE.get(key)
    if mm is None:
        mm = np.memmap(key, dtype="int32", mode="r")
        _FVECS_MEMMAP_CACHE[key] = mm
    return mm, key


def read_fvecs_slice(path: str | Path, start: int, count: int) -> np.ndarray:
    """
    Read vectors from [start, start+count) and return a float32 copy
    with shape [count_actual, dim].
    """
    a, key = _get_int32_memmap(path)
    n, _, stride = fvecs_meta(key)
    if start < 0 or start >= n:
        raise ValueError(f"start={start} out of range for n={n}")
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")

    end = min(start + count, n)
    b = a.reshape(n, stride)
    x = b[start:end, 1:].view("float32")  # reinterpret payload as float32
    return np.array(x, dtype=np.float32, copy=True)
