#include "buffered_writer.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <vector>

void BufferedWriter::open(const std::string& path) {
    fp = std::fopen(path.c_str(), "w");
    if (!fp) {
        std::cerr << "[fatal] cannot open for write: "
                  << path << " err=" << std::strerror(errno) << "\n";
        std::exit(1);
    }
    buf.reserve(flush_bytes * 2);
}

void BufferedWriter::write_line_u_list(int u, const std::vector<int>& nbrs) {
    buf.append(std::to_string(u));
    buf.push_back(':');
    for (int v : nbrs) {
        buf.push_back(' ');
        buf.append(std::to_string(v));
    }
    buf.push_back('\n');

    if (buf.size() >= flush_bytes) flush();
}

void BufferedWriter::flush() {
    if (!fp || buf.empty()) return;
    std::fwrite(buf.data(), 1, buf.size(), fp);
    buf.clear();
}

void BufferedWriter::close() {
    flush();
    if (fp) {
        std::fclose(fp);
        fp = nullptr;
    }
}

BufferedWriter::~BufferedWriter() {
    close();
}

void concat_parts(const std::string& out_path, const std::vector<std::string>& parts) {
    FILE* out = std::fopen(out_path.c_str(), "w");
    if (!out) {
        std::cerr << "[fatal] cannot open final output: "
                  << out_path << " err=" << std::strerror(errno) << "\n";
        std::exit(1);
    }

    static const size_t BUF = 1 << 20;
    std::vector<char> b(BUF);

    for (const auto& p : parts) {
        FILE* in = std::fopen(p.c_str(), "r");
        if (!in) {
            std::cerr << "[fatal] cannot open part file: "
                      << p << " err=" << std::strerror(errno) << "\n";
            std::exit(1);
        }

        while (true) {
            size_t n = std::fread(b.data(), 1, BUF, in);
            if (n == 0) break;
            std::fwrite(b.data(), 1, n, out);
        }

        std::fclose(in);
    }

    std::fclose(out);
}