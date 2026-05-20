#include "cross_window_search.hpp"
#include "dataset.hpp"

#include <algorithm>
#include <climits>
#include <functional>
#include <immintrin.h>
#include <random>
#include <vector>

float l2_distance_sq_ptr(const float* a, const float* b, int dim) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;

    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    __m256 t1 = _mm256_hadd_ps(sum, sum);
    __m256 t2 = _mm256_hadd_ps(t1, t1);
    __m128 lo = _mm256_castps256_ps128(t2);
    __m128 hi = _mm256_extractf128_ps(t2, 1);
    __m128 final = _mm_add_ps(lo, hi);

    float total = _mm_cvtss_f32(final);
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        total += d * d;
    }
    return total;
}

// void ThreadCache::ensure_size(int N) {
//     if (static_cast<int>(dist_stamp.size()) != N) {
//         dist_stamp.assign(N, 0);
//         dist_cache.assign(N, 0.0f);
//         vis_stamp.assign(N, 0);
//     } else if (static_cast<int>(vis_stamp.size()) != N) {
//         vis_stamp.assign(N, 0);
//     }
// }

// void ThreadCache::next_query_epoch() {
//     ++dist_epoch;
//     if (dist_epoch == INT_MAX) {
//         std::fill(dist_stamp.begin(), dist_stamp.end(), 0);
//         dist_epoch = 1;
//     }
// }

// void ThreadCache::next_vis_epoch() {
//     ++vis_epoch;
//     if (vis_epoch == INT_MAX) {
//         std::fill(vis_stamp.begin(), vis_stamp.end(), 0);
//         vis_epoch = 1;
//     }
// }

int choose_entry_by_samples_cached(const std::vector<int>& ids_win,
                                   int entry_samples,
                                   const std::function<float(int)>& dist_to_q) {
    if (ids_win.empty()) return -1;

    int n = static_cast<int>(ids_win.size());
    if (n <= entry_samples) {
        float bd2 = INFINITY;
        int best = -1;
        for (int id : ids_win) {
            float d2 = dist_to_q(id);
            if (d2 < bd2) {
                bd2 = d2;
                best = id;
            }
        }
        return best;
    }

    static thread_local std::mt19937 rng(1234567);
    float bd2 = INFINITY;
    int best = -1;
    for (int i = 0; i < entry_samples; ++i) {
        int id = ids_win[static_cast<int>(rng() % static_cast<uint32_t>(n))];
        float d2 = dist_to_q(id);
        if (d2 < bd2) {
            bd2 = d2;
            best = id;
        }
    }
    return best;
}

std::vector<int> crosswin_range_collect_CSR(const CSRGraph& g,
                                            ThreadCache& tc,
                                            int qid,
                                            int entry,
                                            float d2_thresh,
                                            int ef_cross,
                                            const std::function<float(int)>& dist_to_q) {
    struct Node {
        float d2;
        int id;
    };

    auto comp = [](const Node& a, const Node& b) {
        return a.d2 > b.d2;  // min-heap
    };

    tc.next_vis_epoch();
    int mark = tc.vis_epoch;

    std::vector<Node> heap;
    heap.reserve(static_cast<size_t>(ef_cross) * 4);

    std::vector<int> result;
    result.reserve(256);

    heap.push_back({dist_to_q(entry), entry});
    std::push_heap(heap.begin(), heap.end(), comp);

    int processed = 0;
    while (!heap.empty() && processed < ef_cross) {
        std::pop_heap(heap.begin(), heap.end(), comp);
        Node cur = heap.back();
        heap.pop_back();

        if (tc.vis_stamp[cur.id] == mark) continue;
        tc.vis_stamp[cur.id] = mark;
        ++processed;

        if (cur.d2 < d2_thresh) result.push_back(cur.id);

        uint64_t s = g.off[static_cast<size_t>(cur.id)];
        uint64_t e = g.off[static_cast<size_t>(cur.id) + 1];
        for (uint64_t ei = s; ei < e; ++ei) {
            int nb = g.nbr[static_cast<size_t>(ei)];
            if (nb < 0) continue;
            heap.push_back({dist_to_q(nb), nb});
            std::push_heap(heap.begin(), heap.end(), comp);
        }
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    result.erase(std::remove(result.begin(), result.end(), qid), result.end());
    return result;
}