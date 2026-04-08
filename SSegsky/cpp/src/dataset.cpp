#include "dataset.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

const float* vec_ptr(const Dataset& ds, int i) {
    return ds.X.data() + static_cast<size_t>(i) * ds.d;
}

bool load_fvecs_flat(const std::string& file, Dataset& ds) {
    if (!fs::exists(file)) {
        std::cerr << "Cannot open fvecs: " << file << "\n";
        return false;
    }

    size_t fsz = fs::file_size(file);
    std::ifstream fin(file, std::ios::binary);
    if (!fin) {
        std::cerr << "Cannot open fvecs: " << file << "\n";
        return false;
    }

    int32_t d = 0;
    fin.read(reinterpret_cast<char*>(&d), 4);
    if (!fin || d <= 0) {
        std::cerr << "Bad dim in fvecs header.\n";
        return false;
    }
    ds.d = static_cast<int>(d);

    size_t stride_bytes = 4ull * (static_cast<size_t>(ds.d) + 1ull);
    if (fsz % stride_bytes != 0) {
        std::cerr << "fvecs file size mismatch: fsz=" << fsz
                  << " stride_bytes=" << stride_bytes << "\n";
        return false;
    }

    size_t N = fsz / stride_bytes;
    ds.N = static_cast<int>(N);

    ds.X.resize(N * static_cast<size_t>(ds.d));
    ds.ts.resize(N);

    fin.clear();
    fin.seekg(0, std::ios::beg);

    std::vector<float> tmp(ds.d);
    for (size_t i = 0; i < N; ++i) {
        int32_t dd = 0;
        fin.read(reinterpret_cast<char*>(&dd), 4);
        if (!fin) {
            std::cerr << "Truncated fvecs at " << i << "\n";
            return false;
        }
        if (static_cast<int>(dd) != ds.d) {
            std::cerr << "Inconsistent dim at " << i
                      << ": " << dd << " vs " << ds.d << "\n";
            return false;
        }
        fin.read(reinterpret_cast<char*>(tmp.data()), sizeof(float) * ds.d);
        if (!fin) {
            std::cerr << "Truncated fvecs vec at " << i << "\n";
            return false;
        }

        std::memcpy(ds.X.data() + i * static_cast<size_t>(ds.d),
                    tmp.data(),
                    sizeof(float) * ds.d);

        ds.ts[i] = static_cast<long long>(i);  // time = id
    }

    return true;
}