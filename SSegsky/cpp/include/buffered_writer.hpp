#pragma once

#include <cstdio>
#include <string>
#include <vector>

struct BufferedWriter {
    FILE* fp = nullptr;
    std::string buf;
    size_t flush_bytes = 1 << 20; // 1MB

    void open(const std::string& path);
    void write_line_u_list(int u, const std::vector<int>& nbrs);
    void flush();
    void close();
    ~BufferedWriter();
};

void concat_parts(const std::string& out_path,
                  const std::vector<std::string>& parts);