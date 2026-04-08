#include "knn_state.hpp"

#include <algorithm>
#include <limits>

int g_topN = 10;
std::vector<int> knn_cnt;
std::vector<int> knn_ids;
std::vector<float> knn_d2;
std::vector<float> g_dN_base;

void knn_init(int N, int topN) {
    g_topN = std::max(1, topN);
    knn_cnt.assign(N, 0);
    knn_ids.assign(static_cast<size_t>(N) * g_topN, -1);
    knn_d2.assign(static_cast<size_t>(N) * g_topN,
                  std::numeric_limits<float>::infinity());
}

int base_of(int u) {
    return u * g_topN;
}

void knn_try_update(int u, int vid, float d2) {
    int base = base_of(u);
    int cnt = knn_cnt[u];

    for (int i = 0; i < cnt; ++i) {
        if (knn_ids[base + i] == vid) {
            if (d2 < knn_d2[base + i]) knn_d2[base + i] = d2;
            return;
        }
    }

    if (cnt < g_topN) {
        knn_ids[base + cnt] = vid;
        knn_d2[base + cnt] = d2;
        knn_cnt[u] = cnt + 1;
        return;
    }

    int imax = 0;
    for (int i = 1; i < g_topN; ++i) {
        if (knn_d2[base + i] > knn_d2[base + imax]) {
            imax = i;
        }
    }

    if (d2 < knn_d2[base + imax]) {
        knn_ids[base + imax] = vid;
        knn_d2[base + imax] = d2;
    }
}

float current_dN_sq(int u) {
    if (knn_cnt[u] < g_topN) {
        return std::numeric_limits<float>::infinity();
    }

    int base = base_of(u);
    float mx = knn_d2[base];
    for (int i = 1; i < g_topN; ++i) {
        mx = std::max(mx, knn_d2[base + i]);
    }
    return mx;
}