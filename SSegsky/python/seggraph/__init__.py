"""
Segmented HNSW graph builder.

This package provides:
- .fvecs metadata and slice loading
- Faiss HNSW construction for each segment/window
- dumping L0 adjacency / all-level adjacency / base topN
- multiprocessing job orchestration
- CLI entry for segment-wise build
"""

__all__ = [
    "fvecs_io",
    "hnsw_builder",
    "dumpers",
    "segment_jobs",
]