# File Formats

This document describes the input and output file formats used in this project.

## 1. Input: `.fvecs`

The input dataset is stored in `.fvecs` format.

Each vector is stored as:

[int32 dim][float32 x1][float32 x2]...[float32 xd]

repeated for all vectors.

Notes:
- All vectors must have the same dimension.
- In the current implementation, the global vector ID is its row index in the `.fvecs` file, starting from 0.

---

## 2. Segment-level outputs (Python builder)

The segment builder writes the following files into `out_dir`:

- `seg_00000_adj_L0.txt`
- `seg_00000_knn_base_top{N}.txt`
- `segments_meta.txt`

`seg_00000_adj_all_levels.txt` is optional in the old version, but the current recommended version only keeps L0.

### 2.1 `seg_00000_adj_L0.txt`

This file stores the HNSW L0 adjacency list for one segment.

Format:

global_u: global_v1 global_v2 global_v3 ...

Example:

12: 3 8 15 21
13:
14: 2 7 19

Meaning:
- `global_u` is the global ID of the source point.
- `global_v*` are the global IDs of its L0 neighbors inside the same segment.
- An empty line after `:` means the point has no exported neighbors.

Notes:
- IDs are global IDs, not local segment IDs.
- Self-loops are removed before writing.

### 2.2 `seg_00000_knn_base_top{N}.txt`

This file stores the per-point topN neighbors found inside the same segment.

Format:

global_u: (global_v,d2) (global_v,d2) ...

Example:

12: (8,0.1375) (15,0.2401) (3,0.3012)
13: (7,0.0912) (21,0.1884)

Meaning:
- `global_u` is the source point.
- `global_v` is a neighbor inside the same segment.
- value is the raw Faiss score for the selected metric:
  - `l2`: squared L2 distance
  - `ip`: inner-product similarity

Notes:
- The point itself is removed.
- Results keep Faiss ranking order.

### 2.3 `segments_meta.txt`

This file records the segment split metadata and generated file paths.

Recommended simplified format:

seg_id    start    count    base_id    adj_L0    knn_base    build_s

Meaning:
- `seg_id`: segment / window ID
- `start`: starting global ID of this segment
- `count`: number of points in this segment
- `base_id`: same as `start` in the current implementation
- `adj_L0`: path to `seg_xxxxx_adj_L0.txt`
- `knn_base`: path to `seg_xxxxx_knn_base_topN.txt`
- `build_s`: HNSW build time in seconds

---

## 3. Cross-window outputs (C++ builder)

The cross-window stage writes:

- `cross_edges_fwd_1.txt`
- `cross_edges_bwd_1.txt`
- `knn{N}.txt`

### 3.1 `cross_edges_fwd_1.txt`

This file stores forward cross-window edges.

Format:

u: v1 v2 v3 ...

Meaning:
- `u` is the source point global ID.
- `v*` are selected cross-window neighbors from later windows.

### 3.2 `cross_edges_bwd_1.txt`

This file stores backward cross-window edges.

Format:

u: v1 v2 v3 ...

Meaning:
- `u` is the source point global ID.
- `v*` are selected cross-window neighbors from earlier windows.

Notes for both forward and backward files:
- Neighbor IDs are deduplicated before writing.
- A line may be empty after `:` if no cross-window edge is kept.

### 3.3 `knn{N}.txt`

This file stores the final merged topN neighbors after cross-window expansion.

Format:

u: (v,dist) (v,dist) ...

Example:

12: (8,0.3708) (15,0.4900) (21,0.5521)

Meaning:
- `u` is the source point.
- `v` is a final neighbor ID.
- `dist` is the final output distance.

Important:
- In `seg_xxxxx_knn_base_topN.txt`, the stored value is `d2`.
- In the final `knn{N}.txt`, the code outputs `sqrt(d2)` for L2.

---

## 4. Naming convention

- `seg_%05d_adj_L0.txt`: segment-level L0 graph
- `seg_%05d_knn_base_topN.txt`: segment-level within-window topN
- `cross_edges_fwd_1.txt`: forward cross-window edges
- `cross_edges_bwd_1.txt`: backward cross-window edges
- `knnN.txt`: final merged topN results

---

## 5. Current recommended setting

For the current pipeline:
- keep only `seg_xxxxx_adj_L0.txt`
- keep `seg_xxxxx_knn_base_topN.txt`
- do not rely on all-level adjacency export
