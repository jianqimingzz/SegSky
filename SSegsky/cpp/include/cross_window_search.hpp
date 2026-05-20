#pragma once

#include "dataset.hpp"
#include "graph_io.hpp"
#include "thread_cache.hpp"

#include <functional>
#include <vector>

float l2_distance_sq_ptr(const float* a, const float* b, int dim);

int choose_entry_by_samples_cached(const std::vector<int>& ids_win,
                                   int entry_samples,
                                   const std::function<float(int)>& dist_to_q);

std::vector<int> crosswin_range_collect_CSR(const CSRGraph& g,
                                            ThreadCache& tc,
                                            int qid,
                                            int entry,
                                            float d2_thresh,
                                            int ef_cross,
                                            const std::function<float(int)>& dist_to_q);