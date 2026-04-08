#pragma once

#include <vector>

extern int g_topN;
extern std::vector<int> knn_cnt;     // [N]
extern std::vector<int> knn_ids;     // [N * g_topN]
extern std::vector<float> knn_d2;    // [N * g_topN], squared distance
extern std::vector<float> g_dN_base; // snapshot threshold from loaded knn_base

void knn_init(int N, int topN);

int base_of(int u);

void knn_try_update(int u, int vid, float d2);

float current_dN_sq(int u);