#pragma once

#include <string>
#include <vector>

struct Dataset {
    int d = 0;
    int N = 0;
    std::vector<float> X;          // [N * d]
    std::vector<long long> ts;     // [N], currently ts[i] = i
};

const float* vec_ptr(const Dataset& ds, int i);

bool load_fvecs_flat(const std::string& file, Dataset& ds);