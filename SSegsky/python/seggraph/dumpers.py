from __future__ import annotations

from pathlib import Path

import numpy as np


def _ensure_parent_dir(out_path: str | Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)


def dump_hnsw_L0(index, out_path: str | Path, base_id: int = 0) -> None:
    """
    Dump only HNSW L0 neighbors.

    Output format per line:
        global_u: global_v1 global_v2 ...

    Notes:
    - IDs written to file are global IDs = local_id + base_id
    - self loops are removed if present
    """
    import faiss

    h = index.hnsw
    ntotal = int(index.ntotal)

    offsets = faiss.vector_to_array(h.offsets).astype(np.int64)
    neighbors = faiss.vector_to_array(h.neighbors).astype(np.int64)
    cum = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64)

    _ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(ntotal):
            gi = i + base_id
            base = int(offsets[i])

            # L0 occupies slots [cum[0], cum[1]) relative to each node's base
            s = base + int(cum[0])
            e = base + int(cum[1])

            neigh = neighbors[s:e]
            neigh = neigh[neigh >= 0]
            neigh = neigh[neigh != i]

            if neigh.size > 0:
                neigh = neigh + base_id
                f.write(f"{gi}: " + " ".join(map(str, neigh.tolist())) + "\n")
            else:
                f.write(f"{gi}:\n")


def dump_hnsw_all_levels(index, out_path: str | Path, base_id: int = 0) -> None:
    """
    Dump neighbors from all levels.

    Output format per line:
        global_u<TAB>L0: ...<TAB>L1: ...<TAB>...
    """
    import faiss

    h = index.hnsw
    ntotal = int(index.ntotal)

    offsets = faiss.vector_to_array(h.offsets).astype(np.int64)
    neighbors = faiss.vector_to_array(h.neighbors).astype(np.int64)
    levels = faiss.vector_to_array(h.levels).astype(np.int64)
    cum = faiss.vector_to_array(h.cum_nneighbor_per_level).astype(np.int64)

    _ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(ntotal):
            gi = i + base_id
            L = int(levels[i])
            base = int(offsets[i])

            parts = [str(gi)]
            for lev in range(L):
                s = base + int(cum[lev])
                e = base + int(cum[lev + 1])

                neigh = neighbors[s:e]
                neigh = neigh[neigh >= 0]
                neigh = neigh[neigh != i]

                if neigh.size > 0:
                    neigh = neigh + base_id
                    seg = " ".join(map(str, neigh.tolist()))
                    parts.append(f"L{lev}: {seg}")
                else:
                    parts.append(f"L{lev}:")

            f.write("\t".join(parts) + "\n")


def dump_knn_base(
    index,
    X: np.ndarray,
    out_path: str | Path,
    topN: int,
    base_id: int = 0,
    efS: int = 200,
) -> None:
    """
    Dump per-point topN neighbors within the same segment.

    Output format per line:
        global_u: (global_v,d2) (global_v,d2) ...

    Notes:
    - values follow Faiss search output (squared L2 for L2, similarity for IP)
    - self is removed
    - result order is Faiss ranking order
    """
    if topN <= 0:
        raise ValueError(f"topN must be positive, got {topN}")
    if efS <= 0:
        raise ValueError(f"efS must be positive, got {efS}")

    index.hnsw.efSearch = int(efS)
    k_search = int(topN) + 1
    D, I = index.search(X, k_search)

    _ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(X.shape[0]):
            gi = i + base_id
            parts = []
            kept = 0

            for j in range(k_search):
                nid = int(I[i, j])
                if nid < 0:
                    continue
                if nid == i:
                    continue

                value = float(D[i, j])
                parts.append(f"({nid + base_id},{value})")
                kept += 1
                if kept >= int(topN):
                    break

            f.write(f"{gi}: {' '.join(parts)}\n")
