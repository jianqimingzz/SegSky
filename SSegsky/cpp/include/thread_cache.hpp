#pragma once

#include <vector>
#include <climits>
#include <algorithm>

struct ThreadCache {
    std::vector<int> dist_stamp;
    std::vector<float> dist_cache;

    std::vector<int> vis_stamp;

    int dist_epoch = 1;
    int vis_epoch = 1;

    void ensure_size(int N) {
        if (static_cast<int>(dist_stamp.size()) != N) {
            dist_stamp.assign(N, 0);
            dist_cache.assign(N, 0.0f);
            vis_stamp.assign(N, 0);
        } else if (static_cast<int>(vis_stamp.size()) != N) {
            vis_stamp.assign(N, 0);
        }
    }

    void next_query_epoch() {
        ++dist_epoch;
        if (dist_epoch == INT_MAX) {
            std::fill(dist_stamp.begin(), dist_stamp.end(), 0);
            dist_epoch = 1;
        }
    }

    void next_vis_epoch() {
        ++vis_epoch;
        if (vis_epoch == INT_MAX) {
            std::fill(vis_stamp.begin(), vis_stamp.end(), 0);
            vis_epoch = 1;
        }
    }
};