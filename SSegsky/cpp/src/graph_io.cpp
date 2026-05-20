#include "graph_io.hpp"
#include "knn_state.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>

namespace fs = std::filesystem;

bool parse_int_after_colon(const std::string& line,
                           size_t colon_pos,
                           std::vector<int>* out_vals,
                           int* out_count) {
    const char* s = line.c_str();
    const char* p = s + colon_pos + 1;
    const char* e = s + line.size();

    int cnt = 0;
    while (p < e) {
        while (p < e && static_cast<unsigned char>(*p) <= ' ') ++p;
        if (p >= e) break;

        char* endp = nullptr;
        long v = std::strtol(p, &endp, 10);
        if (endp == p) break;

        if (out_vals) out_vals->push_back(static_cast<int>(v));
        ++cnt;
        p = endp;
    }

    if (out_count) *out_count = cnt;
    return true;
}

bool scan_adj_degree_file(const std::string& file, std::vector<int>& deg, int N) {
    std::ifstream fin(file);
    if (!fin) return false;

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto pos = line.find(':');
        if (pos == std::string::npos) continue;

        int u = -1;
        try {
            u = std::stoi(line.substr(0, pos));
        } catch (...) {
            continue;
        }

        if (u < 0 || u >= N) continue;

        int cnt = 0;
        parse_int_after_colon(line, pos, nullptr, &cnt);
        deg[u] += cnt;
    }
    return true;
}

bool fill_adj_file(const std::string& file,
                   const std::vector<uint64_t>& off,
                   std::vector<uint64_t>& cur,
                   std::vector<int>& nbr,
                   int N) {
    std::ifstream fin(file);
    if (!fin) return false;

    std::string line;
    std::vector<int> vals;
    vals.reserve(128);

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto pos = line.find(':');
        if (pos == std::string::npos) continue;

        int u = -1;
        try {
            u = std::stoi(line.substr(0, pos));
        } catch (...) {
            continue;
        }

        if (u < 0 || u >= N) continue;

        vals.clear();
        parse_int_after_colon(line, pos, &vals, nullptr);

        uint64_t idx = cur[u];
        uint64_t end = off[static_cast<size_t>(u) + 1];

        for (int v : vals) {
            if (v < 0 || v >= N) continue;
            if (idx < end) {
                nbr[static_cast<size_t>(idx++)] = v;
            }
        }
        cur[u] = idx;
    }
    return true;
}

CSRGraph load_all_segments_adj_CSR(const std::string& seg_dir,
                                   size_t num_windows,
                                   int N,
                                   bool verbose) {
    CSRGraph g;
    std::vector<int> deg(N, 0);

    for (size_t w = 0; w < num_windows; ++w) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "seg_%05zu_adj_L0.txt", w);
        std::string adj_file = (fs::path(seg_dir) / buf).string();

        if (!fs::exists(adj_file)) {
            if (verbose) {
                std::cerr << "[warn] missing adj file for window "
                          << w << ": " << adj_file << "\n";
            }
            continue;
        }

        scan_adj_degree_file(adj_file, deg, N);
        if (verbose && (w % 10) == 0) {
            std::cerr << " [scan deg] window " << w << "/" << num_windows << "\r";
        }
    }
    if (verbose) std::cerr << "\n";

    g.off.assign(static_cast<size_t>(N) + 1, 0);
    uint64_t E = 0;
    for (int i = 0; i < N; ++i) {
        g.off[static_cast<size_t>(i)] = E;
        E += static_cast<uint64_t>(deg[i]);
    }
    g.off[static_cast<size_t>(N)] = E;
    g.nbr.assign(static_cast<size_t>(E), -1);

    std::vector<uint64_t> cur(static_cast<size_t>(N), 0);
    for (int i = 0; i < N; ++i) {
        cur[static_cast<size_t>(i)] = g.off[static_cast<size_t>(i)];
    }

    for (size_t w = 0; w < num_windows; ++w) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "seg_%05zu_adj_L0.txt", w);
        std::string adj_file = (fs::path(seg_dir) / buf).string();
        if (!fs::exists(adj_file)) continue;

        fill_adj_file(adj_file, g.off, cur, g.nbr, N);

        if (verbose && (w % 10) == 0) {
            std::cerr << " [fill csr] window " << w << "/" << num_windows << "\r";
        }
    }
    if (verbose) std::cerr << "\n";

    return g;
}

void fill_knn_line_direct(const char* line, int N) {
    char* endp = nullptr;
    long u = std::strtol(line, &endp, 10);
    if (endp == line || u < 0 || u >= N) return;

    const char* p = std::strchr(line, ':');
    if (!p) return;
    ++p;

    int base = base_of(static_cast<int>(u));
    int cnt = 0;

    while (*p && cnt < g_topN) {
        if (*p == '(') {
            long vid = std::strtol(p + 1, &endp, 10);
            if (endp && *endp == ',') {
                char* endp2 = nullptr;
                float d2 = std::strtof(endp + 1, &endp2);
                if (endp2 && *endp2 == ')') {
                    if (vid >= 0 && vid < N && std::isfinite(d2)) {
                        knn_ids[base + cnt] = static_cast<int>(vid);
                        knn_d2[base + cnt] = d2;
                        ++cnt;
                    }
                    p = endp2 + 1;
                    continue;
                }
            }
        }
        ++p;
    }

    knn_cnt[static_cast<int>(u)] = cnt;
}

bool load_knn_base_one_file_fast(const std::string& file, int N) {
    FILE* fp = std::fopen(file.c_str(), "r");
    if (!fp) return false;

    static const size_t BUF_SZ = 1 << 20;
    static thread_local std::vector<char> bufv(BUF_SZ);
    setvbuf(fp, bufv.data(), _IOFBF, BUF_SZ);

    static const int LINE_SZ = 1 << 20;
    char* line = static_cast<char*>(std::malloc(LINE_SZ));
    if (!line) {
        std::fclose(fp);
        return false;
    }

    while (std::fgets(line, LINE_SZ, fp)) {
        fill_knn_line_direct(line, N);
    }

    std::free(line);
    std::fclose(fp);
    return true;
}

void load_all_segments_knn_base(const std::string& seg_dir,
                                size_t num_windows,
                                int N,
                                int topN,
                                int num_threads,
                                bool verbose) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1) num_threads(num_threads)
#endif
    for (ptrdiff_t wi = 0; wi < static_cast<ptrdiff_t>(num_windows); ++wi) {
        size_t w = static_cast<size_t>(wi);

        char buf[256];
        std::snprintf(buf, sizeof(buf), "seg_%05zu_knn_base_top%d.txt", w, topN);
        std::string knn_file = (fs::path(seg_dir) / buf).string();

        if (!fs::exists(knn_file)) {
            if (verbose) {
#ifdef _OPENMP
#pragma omp critical
#endif
                std::cerr << "[warn] missing knn_base file for window "
                          << w << ": " << knn_file << "\n";
            }
            continue;
        }

        if (verbose) {
#ifdef _OPENMP
#pragma omp critical
#endif
            std::cerr << " [load knn_base] begin window "
                      << w << "/" << num_windows << "  " << knn_file << "\n";
        }

        load_knn_base_one_file_fast(knn_file, N);

        if (verbose) {
#ifdef _OPENMP
#pragma omp critical
#endif
            std::cerr << " [load knn_base] done  window "
                      << w << "/" << num_windows << "\n";
        }
    }
}