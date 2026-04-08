from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .fvecs_io import fvecs_meta
from .segment_jobs import SegmentBuildJob, run_segment_job


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build per-segment HNSW graphs from .fvecs and dump adjacency lists."
    )

    p.add_argument("--fvecs", type=str, required=True, help="Input .fvecs file")
    p.add_argument("--out_dir", type=str, default="./seg_hnsw_out", help="Output directory")

    # segmentation
    p.add_argument(
        "--window_size",
        type=int,
        default=100000,
        help=(
            "Segment size. If <= 0, it means the number of segments M=abs(window_size), "
            "and actual segment size will be ceil(N/M)."
        ),
    )
    p.add_argument("--start_win", type=int, default=0, help="Start window id (inclusive)")
    p.add_argument("--end_win", type=int, default=-1, help="End window id (inclusive), -1 means last")

    # HNSW params
    p.add_argument("--M", type=int, default=32)
    p.add_argument("--efC", type=int, default=200)
    p.add_argument("--metric", type=str, default="l2", choices=["l2", "ip"])

    # dump options
    p.add_argument("--dump_all_levels", type=int, default=0, choices=[0, 1])
    p.add_argument("--dump_knn_base", type=int, default=1, choices=[0, 1])
    p.add_argument("--topN", type=int, default=10)
    p.add_argument("--efS", type=int, default=200)

    # parallel
    p.add_argument("--num_procs", type=int, default=2, help="Number of worker processes")
    p.add_argument(
        "--total_threads",
        type=int,
        default=20,
        help="Total thread budget, divided roughly evenly across processes",
    )

    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.M <= 0:
        raise ValueError("--M must be positive")
    if args.efC <= 0:
        raise ValueError("--efC must be positive")
    if args.topN <= 0:
        raise ValueError("--topN must be positive")
    if args.efS <= 0:
        raise ValueError("--efS must be positive")
    if args.num_procs <= 0:
        raise ValueError("--num_procs must be positive")
    if args.total_threads <= 0:
        raise ValueError("--total_threads must be positive")
    if args.start_win < 0:
        raise ValueError("--start_win must be >= 0")
    if args.window_size == 0:
        raise ValueError("--window_size cannot be 0")
    if args.end_win < -1:
        raise ValueError("--end_win must be -1 or >= 0")


def resolve_window_size(N: int, window_size: int) -> int:
    """
    If window_size > 0, use it directly.
    If window_size <= 0, interpret abs(window_size) as number of windows.
    """
    if window_size > 0:
        return window_size

    M = abs(window_size)
    if M < 1:
        M = 1
    if M > N:
        M = N
    return math.ceil(N / M)


def build_jobs(args: argparse.Namespace) -> tuple[list[SegmentBuildJob], int, int, int]:
    N, d, _ = fvecs_meta(args.fvecs)

    actual_window_size = resolve_window_size(N, args.window_size)
    num_windows = math.ceil(N / actual_window_size)

    end_win = num_windows - 1 if args.end_win < 0 else min(args.end_win, num_windows - 1)
    if args.start_win > end_win:
        raise ValueError(
            f"Invalid window range: start_win={args.start_win}, end_win={end_win}, num_windows={num_windows}"
        )

    faiss_threads = max(1, args.total_threads // args.num_procs)

    jobs: list[SegmentBuildJob] = []
    for seg_id in range(args.start_win, end_win + 1):
        start = seg_id * actual_window_size
        count = min(actual_window_size, N - start)
        if count <= 0:
            continue

        jobs.append(
            SegmentBuildJob(
                seg_id=seg_id,
                fvecs_path=args.fvecs,
                start=start,
                count=count,
                base_id=start,
                out_dir=args.out_dir,
                M=args.M,
                efC=args.efC,
                metric=args.metric,
                dump_all_levels=bool(args.dump_all_levels),
                dump_knn_base_flag=bool(args.dump_knn_base),
                topN=args.topN,
                efS=args.efS,
                faiss_threads=faiss_threads,
            )
        )

    return jobs, N, d, actual_window_size


def write_segments_meta(out_dir: str | Path, results: list[dict], N: int, d: int, window_size: int) -> None:
    out_dir = Path(out_dir)
    meta_path = out_dir / "segments_meta.txt"

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"# N={N} d={d} window_size={window_size}\n")
        f.write("# seg_id\tstart\tcount\tbase_id\tbuild_s\tadj_L0\tadj_all_levels\tknn_base\n")
        for r in sorted(results, key=lambda x: x["seg_id"]):
            f.write(
                f"{r['seg_id']}\t{r['start']}\t{r['count']}\t{r['base_id']}\t"
                f"{r['build_s']:.6f}\t{r['adj_L0']}\t{r['adj_all_levels']}\t{r['knn_base']}\n"
            )


def main() -> None:
    args = parse_args()
    validate_args(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs, N, d, actual_window_size = build_jobs(args)
    if not jobs:
        write_segments_meta(out_dir, [], N, d, actual_window_size)
        print("[warn] no jobs generated for the requested window range")
        print(f"[ok] wrote meta to {out_dir / 'segments_meta.txt'}")
        return

    max_workers = min(args.num_procs, len(jobs))
    faiss_threads = max(1, args.total_threads // max_workers)
    for job in jobs:
        job.faiss_threads = faiss_threads

    chunksize = max(1, len(jobs) // (max_workers * 4))

    print(f"[meta] N={N} d={d} window_size={actual_window_size} num_jobs={len(jobs)}")
    print(
        "[config] "
        f"M={args.M} efC={args.efC} metric={args.metric} "
        f"dump_all_levels={args.dump_all_levels} dump_knn_base={args.dump_knn_base} "
        f"topN={args.topN} efS={args.efS} "
        f"num_procs={args.num_procs} workers={max_workers} "
        f"total_threads={args.total_threads} faiss_threads={faiss_threads} chunksize={chunksize}"
    )

    results: list[dict] = []
    t_all = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for r in ex.map(run_segment_job, jobs, chunksize=chunksize):
            print(
                f"[done] seg={r['seg_id']:05d} start={r['start']} count={r['count']} "
                f"build_s={r['build_s']:.6f}"
            )
            results.append(r)

    write_segments_meta(out_dir, results, N, d, actual_window_size)
    elapsed = time.perf_counter() - t_all
    print(f"[ok] wrote meta to {out_dir / 'segments_meta.txt'}")
    print(f"[ok] total_elapsed_s={elapsed:.6f}")


if __name__ == "__main__":
    main()
