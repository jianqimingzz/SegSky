# SegSky: Segment-wise HNSW + Cross-window Edges (RF-ANNS)

A two-stage graph construction pipeline for ordered / segmented vector data.

---

## Requirements

### Python
* Python 3.10+
* NumPy `1.26.4`
* Faiss CPU `1.8.0.post1`

```bash
pip install -r requirements.txt
```

If you need to update dependency versions in the future, test the full Stage A/B/C pipeline first and then update `requirements.txt` with exact pinned versions.

### C++
* C++17 compiler (GCC or Clang)
* OpenMP
* CMake >= 3.16

---

## Build

```bash
mkdir -p build
cd build
cmake ..
make -j
```

Produces:
```
build/crosswin           # Stage B: cross-window edge builder
build/query_range_eval   # Stage C: query evaluator
```

---

## Stage A: Build Segment Graphs (Python)

```bash
python -m python.seggraph.cli_build_segment_hnsw \
  --fvecs      /path/to/base.fvecs \
  --out_dir    /path/to/seg_hnsw_out \
  --window_size -10 \
  --M 32 \
  --efC 400 \
  --metric l2 \
  --dump_knn_base 1 \
  --topN 10 \
  --efS 400 \
  --num_procs 2 \
  --total_threads 20
```

| Parameter | Description |
|-----------|-------------|
| `--fvecs` | Path to input `.fvecs` dataset |
| `--out_dir` | Output directory for segment files |
| `--window_size` | Segmentation rule: `> 0` = points per segment, `<= 0` = number of segments |
| `--M` | HNSW connectivity parameter |
| `--efC` | HNSW efConstruction |
| `--metric` | Distance metric: `l2` or `ip` |
| `--dump_knn_base` | Dump within-segment topN neighbors: `0` = no, `1` = yes |
| `--topN` | Number of within-segment neighbors to keep |
| `--efS` | Search width for computing segment-level knn |
| `--num_procs` | Number of parallel Python worker processes |
| `--total_threads` | Total thread budget across all workers |

`--window_size -10` means: split dataset into 10 segments, and each segment size is computed as `ceil(N/10)`.

**Outputs:** `seg_xxxxx_adj_L0.txt`, `seg_xxxxx_knn_base_topN.txt`, `segments_meta.txt`

---

## Stage B: Build Cross-window Edges (C++)

```bash
./build/crosswin \
  /path/to/base.fvecs \
  /path/to/seg_hnsw_out \
  /path/to/cross_out \
  -10 \
  10 \
  10 \
  500 \
  32 \
  32 \
  32 \
  20 \
  10 \
  1
```

| Position | Parameter | Description |
|----------|-----------|-------------|
| 1 | `fvecs_path` | Path to `.fvecs` dataset |
| 2 | `seg_dir` | Stage A output directory |
| 3 | `out_dir` | Output directory for cross-window files |
| 4 | `window_size` | Must match Stage A |
| 5 | `forward_span` | Number of later windows to probe |
| 6 | `backward_span` | Number of earlier windows to probe |
| 7 | `ef_cross` | Search expansion budget |
| 8 | `entry_samples` | Entry point samples per target window |
| 9 | `M_cross_fwd` | Max forward edges per point |
| 10 | `M_cross_bwd` | Max backward edges per point |
| 11 | `win_par_expand` | Number of threads |
| 12 | `thickness` | thickness parameter (less than `--topN` in Stage A) |
| 13 | `load_knn_base` | Load segment knn_base: `0` = no, `1` = yes |

**Outputs:** `cross_edges_fwd_1.txt`, `cross_edges_bwd_1.txt`, `knn10.txt`

---

## Stage C: Query Evaluation (C++)

```bash
./build/query_range_eval \
  --fvecs     /path/to/base.fvecs \
  --index     /path/to/seg_hnsw_out_-10 \
  --cross_dir /path/to/cross_out_-10 \
  --query     /path/to/query.fvecs \
  --k 10 \
  --t_start 50000 \
  --qnum 500
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fvecs` | required | Path to base `.fvecs` dataset |
| `--index` | required | Path to `seg_dir` (Stage A output) |
| `--cross_dir` | same as `--index` | Directory with `cross_edges_fwd/bwd_1.txt` |
| `--query` | *(none)* | External query `.fvecs`; if omitted, base vectors are used |
| `--k` | `10` | Number of nearest neighbors |
| `--t_start` | `50000` | Left boundary of the query time-range |
| `--qnum` | `500` | Number of queries to evaluate |
| `--use_isolated` | `0` | Set `1` to include isolated nodes as fallback candidates |

`T_END` and `ef` are swept automatically. Results are written to a CSV file in the current directory.

---


