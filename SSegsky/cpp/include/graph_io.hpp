#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct CSRGraph {
    std::vector<uint64_t> off;   // [N+1]
    std::vector<int> nbr;        // [E]
};

bool parse_int_after_colon(const std::string& line,
                           size_t colon_pos,
                           std::vector<int>* out_vals,
                           int* out_count);

bool scan_adj_degree_file(const std::string& file,
                          std::vector<int>& deg,
                          int N);

bool fill_adj_file(const std::string& file,
                   const std::vector<uint64_t>& off,
                   std::vector<uint64_t>& cur,
                   std::vector<int>& nbr,
                   int N);

CSRGraph load_all_segments_adj_CSR(const std::string& seg_dir,
                                   size_t num_windows,
                                   int N,
                                   bool verbose = true);

void fill_knn_line_direct(const char* line, int N);

bool load_knn_base_one_file_fast(const std::string& file, int N);

void load_all_segments_knn_base(const std::string& seg_dir,
                                size_t num_windows,
                                int N,
                                int topN,
                                int num_threads,
                                bool verbose = true);