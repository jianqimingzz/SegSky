#include "buffered_writer.hpp"
#include "cross_window_search.hpp"
#include "dataset.hpp"
#include "graph_io.hpp"
#include "knn_state.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::milliseconds;

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::string fvecs_path = (argc >= 2) ? std::string(argv[1]) : "./data.fvecs";
    std::string seg_dir = (argc >= 3) ? std::string(argv[2]) : "./seg_hnsw_out";
    std::string out_dir = (argc >= 4) ? std::string(argv[3]) : "./cross_out";

    long long window_size = (argc >= 5) ? std::stoll(argv[4]) : 100000;
    int forward_span = (argc >= 6) ? std::stoi(argv[5]) : 10;
    int backward_span = (argc >= 7) ? std::stoi(argv[6]) : 10;
    int ef_cross = (argc >= 8) ? std::stoi(argv[7]) : 500;
    int entry_samples = (argc >= 9) ? std::stoi(argv[8]) : 32;
    int M_cross_fwd = (argc >= 10) ? std::stoi(argv[9]) : 32;
    int M_cross_bwd = (argc >= 11) ? std::stoi(argv[10]) : 32;
    int win_par_expand = (argc >= 12) ? std::stoi(argv[11]) : 20;
    int topNParam = (argc >= 13) ? std::stoi(argv[12]) : 10;
    int load_knn_base = (argc >= 14) ? std::stoi(argv[13]) : 1;

    fs::create_directories(out_dir);

    auto T_all0 = Clock::now();

    auto T0 = Clock::now();
    Dataset ds;
    if (!load_fvecs_flat(fvecs_path, ds)) return 1;
    int N = ds.N;

    if (window_size <= 0) {
        long long M = std::llabs(window_size);
        if (M < 1) M = 1;
        if (M > static_cast<long long>(N)) M = N;
        window_size = (static_cast<long long>(N) + M - 1) / M;
    }
    size_t num_windows =
        static_cast<size_t>((static_cast<long long>(N) + window_size - 1) / window_size);

    auto T1 = Clock::now();
    std::cerr << "[meta] N=" << N
              << " dim=" << ds.d
              << " window_size=" << window_size
              << " num_windows=" << num_windows << "\n";

#ifdef _OPENMP
    std::cerr << "OpenMP max threads: " << omp_get_max_threads() << "\n";
#endif

    std::vector<std::vector<int>> win_ids(num_windows);
    for (int i = 0; i < N; ++i) {
        size_t w = static_cast<size_t>(static_cast<long long>(i) / window_size);
        win_ids[w].push_back(i);
    }

    knn_init(N, topNParam);

    auto T2 = Clock::now();
    CSRGraph g = load_all_segments_adj_CSR(seg_dir, num_windows, N, true);
    auto T3 = Clock::now();
    std::cerr << "[adj] CSR edges E=" << static_cast<unsigned long long>(g.nbr.size()) << "\n";

    auto T4 = Clock::now();
    if (load_knn_base) {
        load_all_segments_knn_base(seg_dir, num_windows, N, g_topN, win_par_expand, true);
    } else {
        std::cerr << "[warn] load_knn_base=0 => g_dN_base may be INF\n";
    }
    auto T5 = Clock::now();

    auto T6 = Clock::now();
    g_dN_base.assign(static_cast<size_t>(N), std::numeric_limits<float>::infinity());
#pragma omp parallel for schedule(static) num_threads(win_par_expand)
    for (int i = 0; i < N; ++i) {
        g_dN_base[static_cast<size_t>(i)] = current_dN_sq(i);
    }
    auto T7 = Clock::now();

    int T = 1;
#ifdef _OPENMP
    T = win_par_expand;
#endif

    std::vector<std::string> parts_fwd(T), parts_bwd(T);
    for (int tid = 0; tid < T; ++tid) {
        parts_fwd[tid] =
            (fs::path(out_dir) / ("cross_edges_fwd_1.part" + std::to_string(tid) + ".txt")).string();
        parts_bwd[tid] =
            (fs::path(out_dir) / ("cross_edges_bwd_1.part" + std::to_string(tid) + ".txt")).string();
    }

    std::atomic<long long> xf_sum(0), xb_sum(0);
    std::atomic<long long> xf_max(0), xb_max(0);

    auto T8 = Clock::now();

#pragma omp parallel num_threads(win_par_expand)
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif

        BufferedWriter wf, wb;
        wf.open(parts_fwd[tid]);
        wb.open(parts_bwd[tid]);

        thread_local ThreadCache tc;
        tc.ensure_size(N);

        std::vector<int> out_fwd_u;
        std::vector<int> out_bwd_u;

#pragma omp for schedule(dynamic,512)
        for (int u = 0; u < N; ++u) {
            tc.next_query_epoch();
            int epoch = tc.dist_epoch;

            const float* qu = vec_ptr(ds, u);
            auto dist_to_q = [&](int x) -> float {
                if (tc.dist_stamp[static_cast<size_t>(x)] == epoch)
                    return tc.dist_cache[static_cast<size_t>(x)];

                float d2 = l2_distance_sq_ptr(qu, vec_ptr(ds, x), ds.d);
                tc.dist_stamp[static_cast<size_t>(x)] = epoch;
                tc.dist_cache[static_cast<size_t>(x)] = d2;
                return d2;
            };

            size_t w = static_cast<size_t>(static_cast<long long>(u) / window_size);

            out_fwd_u.clear();
            out_bwd_u.clear();

            float d2_thr_f = g_dN_base[static_cast<size_t>(u)];
            if (!std::isfinite(d2_thr_f))
                d2_thr_f = std::numeric_limits<float>::infinity();

            for (int off = 1; off <= forward_span; ++off) {
                size_t tw = w + off;
                if (tw >= win_ids.size()) break;
                const auto& tgt_ids = win_ids[tw];
                if (tgt_ids.empty()) continue;

                int entry = choose_entry_by_samples_cached(tgt_ids, entry_samples, dist_to_q);
                if (entry < 0) continue;

                auto under = crosswin_range_collect_CSR(
                    g, tc, u, entry, d2_thr_f, ef_cross, dist_to_q);
                if (under.empty()) continue;

                std::vector<std::pair<float, int>> cand_all;
                cand_all.reserve(under.size());
                for (int v : under) {
                    float d2 = dist_to_q(v);
                    cand_all.emplace_back(d2, v);
                }
                if (cand_all.empty()) continue;

                for (auto& p : cand_all) {
                    knn_try_update(u, p.second, p.first);
                }

                float theta_new = current_dN_sq(u);
                if (!std::isfinite(theta_new))
                    theta_new = std::numeric_limits<float>::infinity();

                std::vector<std::pair<float, int>> cand_edges;
                cand_edges.reserve(cand_all.size());
                for (auto& p : cand_all) {
                    if (p.first < theta_new) cand_edges.push_back(p);
                }

                if (static_cast<int>(cand_edges.size()) > M_cross_fwd) {
                    std::nth_element(
                        cand_edges.begin(),
                        cand_edges.begin() + M_cross_fwd,
                        cand_edges.end(),
                        [](const auto& a, const auto& b) { return a.first < b.first; });
                    cand_edges.resize(M_cross_fwd);
                }

                std::sort(cand_edges.begin(), cand_edges.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });

                for (auto& p : cand_edges) {
                    out_fwd_u.push_back(p.second);
                }

                d2_thr_f = theta_new;
            }

            float d2_thr_b = g_dN_base[static_cast<size_t>(u)];
            if (!std::isfinite(d2_thr_b))
                d2_thr_b = std::numeric_limits<float>::infinity();

            for (int off = 1; off <= backward_span; ++off) {
                if (static_cast<long long>(w) - off < 0) break;
                size_t tw = w - off;
                const auto& tgt_ids = win_ids[tw];
                if (tgt_ids.empty()) continue;

                int entry = choose_entry_by_samples_cached(tgt_ids, entry_samples, dist_to_q);
                if (entry < 0) continue;

                auto under = crosswin_range_collect_CSR(
                    g, tc, u, entry, d2_thr_b, ef_cross, dist_to_q);
                if (under.empty()) continue;

                std::vector<std::pair<float, int>> cand_all;
                cand_all.reserve(under.size());
                for (int v : under) {
                    float d2 = dist_to_q(v);
                    cand_all.emplace_back(d2, v);
                }
                if (cand_all.empty()) continue;

                for (auto& p : cand_all) {
                    knn_try_update(u, p.second, p.first);
                }

                float theta_new = current_dN_sq(u);
                if (!std::isfinite(theta_new))
                    theta_new = std::numeric_limits<float>::infinity();

                std::vector<std::pair<float, int>> cand_edges;
                cand_edges.reserve(cand_all.size());
                for (auto& p : cand_all) {
                    if (p.first < theta_new) cand_edges.push_back(p);
                }

                if (static_cast<int>(cand_edges.size()) > M_cross_bwd) {
                    std::nth_element(
                        cand_edges.begin(),
                        cand_edges.begin() + M_cross_bwd,
                        cand_edges.end(),
                        [](const auto& a, const auto& b) { return a.first < b.first; });
                    cand_edges.resize(M_cross_bwd);
                }

                std::sort(cand_edges.begin(), cand_edges.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });

                for (auto& p : cand_edges) {
                    out_bwd_u.push_back(p.second);
                }

                d2_thr_b = theta_new;
            }

            if (!out_fwd_u.empty()) {
                std::sort(out_fwd_u.begin(), out_fwd_u.end());
                out_fwd_u.erase(std::unique(out_fwd_u.begin(), out_fwd_u.end()), out_fwd_u.end());
            }

            if (!out_bwd_u.empty()) {
                std::sort(out_bwd_u.begin(), out_bwd_u.end());
                out_bwd_u.erase(std::unique(out_bwd_u.begin(), out_bwd_u.end()), out_bwd_u.end());
            }

            wf.write_line_u_list(u, out_fwd_u);
            wb.write_line_u_list(u, out_bwd_u);

            long long lf = static_cast<long long>(out_fwd_u.size());
            long long lb = static_cast<long long>(out_bwd_u.size());

            xf_sum.fetch_add(lf, std::memory_order_relaxed);
            xb_sum.fetch_add(lb, std::memory_order_relaxed);

            long long curm = xf_max.load(std::memory_order_relaxed);
            while (lf > curm &&
                   !xf_max.compare_exchange_weak(curm, lf, std::memory_order_relaxed)) {
            }

            curm = xb_max.load(std::memory_order_relaxed);
            while (lb > curm &&
                   !xb_max.compare_exchange_weak(curm, lb, std::memory_order_relaxed)) {
            }
        }

        wf.close();
        wb.close();
    }

    auto T9 = Clock::now();

    auto T10 = Clock::now();
    std::string fwd_final = (fs::path(out_dir) / "cross_edges_fwd_1.txt").string();
    std::string bwd_final = (fs::path(out_dir) / "cross_edges_bwd_1.txt").string();
    concat_parts(fwd_final, parts_fwd);
    concat_parts(bwd_final, parts_bwd);
    std::cerr << "Saved -> " << fwd_final << "\n";
    std::cerr << "Saved -> " << bwd_final << "\n";
    auto T11 = Clock::now();

    auto T12 = Clock::now();
    {
        std::ofstream fk(fs::path(out_dir) / (std::string("knn") + std::to_string(g_topN) + ".txt"));
        for (int i = 0; i < N; ++i) {
            fk << i << ":";

            std::vector<std::pair<float, int>> tmp;
            tmp.reserve(knn_cnt[static_cast<size_t>(i)]);

            int base = base_of(i);
            for (int t = 0; t < knn_cnt[static_cast<size_t>(i)]; ++t) {
                int id = knn_ids[static_cast<size_t>(base) + t];
                float d2 = knn_d2[static_cast<size_t>(base) + t];
                if (id >= 0 && std::isfinite(d2)) {
                    tmp.emplace_back(d2, id);
                }
            }

            std::sort(tmp.begin(), tmp.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            for (auto& p : tmp) {
                fk << " (" << p.second << "," << std::sqrt(p.first) << ")";
            }
            fk << "\n";
        }

        std::cerr << "Saved -> "
                  << (fs::path(out_dir) / (std::string("knn") + std::to_string(g_topN) + ".txt"))
                  << "\n";
    }
    auto T13 = Clock::now();

    std::cerr << "[cross fwd] avg_out=" << static_cast<double>(xf_sum.load()) / static_cast<double>(N)
              << ", max_out=" << xf_max.load() << "\n";
    std::cerr << "[cross bwd] avg_out=" << static_cast<double>(xb_sum.load()) / static_cast<double>(N)
              << ", max_out=" << xb_max.load() << "\n";

    auto T_all1 = Clock::now();

    auto ms = [&](Clock::time_point a, Clock::time_point b) {
        return static_cast<long long>(std::chrono::duration_cast<Ms>(b - a).count());
    };

    std::cerr << "================ time breakdown (ms) ================\n";
    std::cerr << "[time] load_fvecs_ms      = " << ms(T0, T1) << "\n";
    std::cerr << "[time] build_winids_ms    = " << ms(T1, T2) << "\n";
    std::cerr << "[time] load_adj_CSR_ms    = " << ms(T2, T3) << "\n";
    std::cerr << "[time] load_knn_base_ms   = " << ms(T4, T5) << "\n";
    std::cerr << "[time] build_gdN_base_ms  = " << ms(T6, T7) << "\n";
    std::cerr << "[time] cross_expand_ms    = " << ms(T8, T9) << "\n";
    std::cerr << "[time] merge_parts_ms     = " << ms(T10, T11) << "\n";
    std::cerr << "[time] output_knn_ms      = " << ms(T12, T13) << "\n";
    std::cerr << "[time] total_ms           = " << ms(T_all0, T_all1) << "\n";
    std::cerr << "=====================================================\n";

    return 0;
}