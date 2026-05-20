// query_seg_hnsw_slot_backend_unify_v3.cpp
//
// ✅ 版本目标（你提到的“想要 + 坑修复”一次性解决）
// 1) 支持 --query query.fvecs 读入独立查询集（不再默认用 base 的前 0..500）
// 2) 修复“坑”：以前文件名写着支持 text backend，但实际直接 return 3
//    -> v3 同时支持：
//       - SLOT(.bin) 后端：按激活 slots 遍历邻接（UNIFY-style 点级过滤）
//       - TEXT(CSR) 后端：加载全图 CSR（seg_XXXXX_adj_L0 + cross_edges），查询时同样点级过滤（不构建 per-range 子图）
// 3) 修复“坑”：外部 query 集不应 exclude_id=qid（因为 qid 不再是 base id）
//    -> 外部 query 时 exclude_id = -1
// 4) 参数改成 --query 风格，并提供更清晰的 --help/usage
//
// [优化更新]: 
// - 直接读入连续内存，解决加载阶段峰值内存翻倍爆炸问题。
// - CSRGraph offsets 升级为 uint64_t，彻底解决超大图（边数 > 21.4亿）int 溢出导致的 Segfault 坑。
//
// Build:
//   g++ -O3 -march=native -std=c++17 -fopenmp query_seg_hnsw_slot_backend_unify_v3.cpp -o query_unify_v3
//
// Example (SLOT backend):
//   ./query_unify_v3 --fvecs /path/base.fvecs --index /path/preprocessed_index_T10.bin --k 10 --qnum 500 --t_start 50000
//   ./query_unify_v3 --fvecs /path/base.fvecs --index /path/preprocessed_index_T10.bin --query /path/query.fvecs
//
// Example (TEXT backend):
//   ./query_unify_v3 --fvecs /home/teamli/桌面/last_new/base.100M.fvecs --index /home/teamli/桌面/last_new/seg_hnsw_out_-10 --cross_dir /home/teamli/桌面/last_new/cross_out_-10 --query /media/teamli/android/data/deep1B/tail_5000.fvecs
//     ./query_unify_v3 --fvecs /media/teamli/android/sample_data/deep/deep1M_clean.fvecs --index /home/teamli/桌面/last/deep1M/seg_hnsw_out_-10 --cross_dir /home/teamli/桌面/last/deep1M/cross_out_-10 --query /media/teamli/android/data/deep1B/tail_5000.fvecs

#include <bits/stdc++.h>
#include <immintrin.h>
#include <mm_malloc.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <regex>
#include <queue>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
namespace fs = std::filesystem;

// ===================== 细粒度计时开关 =====================
#ifndef ENABLE_FINE_TIMING
#define ENABLE_FINE_TIMING 0
#endif

#if ENABLE_FINE_TIMING
  #define TIME_BLOCK(acc, CODE) do { \
    auto __t0 = chrono::high_resolution_clock::now(); \
    { CODE } \
    auto __t1 = chrono::high_resolution_clock::now(); \
    acc += (uint64_t)chrono::duration_cast<chrono::nanoseconds>(__t1 - __t0).count(); \
  } while(0)
#else
  #define TIME_BLOCK(acc, CODE) do { CODE } while(0)
#endif

// ===================== sweep 参数（保持你原本的 grid） =====================
static const int GRID_EF[] = {
    10, 15, 20, 25, 30, 35, 40, 45, 50, 55,
    60, 70, 80, 90, 100, 120, 140, 160, 180, 200,
    250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,
    1400, 1700
};
static const int GRID_ADDK[] = {1};

// UNIFY-style entry selection: pick best STARTS among centers by distance
static const int STARTS = 8;

// ===================== CSV / naming =====================
static inline bool file_exists(const string& path){
    ifstream f(path);
    return f.good();
}

static string extract_tag_from_path(const string& p){
    fs::path pp(p);
    string base = pp.filename().string();

    const string p1 = "seg_hnsw_out_";
    const string p2 = "cross_out_";
    if (base.rfind(p1, 0) == 0) return "ws" + base.substr(p1.size());
    if (base.rfind(p2, 0) == 0) return "ws" + base.substr(p2.size());

    // preprocessed_index_deep_T10.bin -> slotT10
    {
        std::regex re(R"(T(\d+))");
        std::smatch m;
        if (std::regex_search(base, m, re)){
            return string("slotT") + m[1].str();
        }
    }
    return "tagNA";
}

static string dataset_name_from_fvecs(const string& fvecs_path){
    return fs::path(fvecs_path).stem().string();
}

static string build_csv_path(const string& fvecs_path,
                             const string& seg_dir_or_bin,
                             const string& cross_dir,
                             const string& query_path)
{
    string dataset = dataset_name_from_fvecs(fvecs_path);
    string tag  = extract_tag_from_path(seg_dir_or_bin);
    if (tag == "tagNA") tag = extract_tag_from_path(cross_dir);

    string qtag = "qInBase";
    if (!query_path.empty()) qtag = "qExternal";

    return "grid_results_top10_" + dataset + "_" + tag + "_" + qtag + ".csv";
}

static void csv_write_header_if_needed(const string& path){
    if (!file_exists(path)){
        ofstream fout(path, ios::app);
        fout <<
            "dataset_name,dataset_path,query_path,backend,index_path,cross_dir,"
            "T_START,T_END,ef,add_k,k,q_lo,q_hi,queries,avg_visited,"
            "recall_micro,recall_macro,total_ms,setup_ms,graph_ms,dist_ms,qps,"
            "load_index_ms,build_range_ms,build_gt_ms\n";
    }
}

static void csv_append_row(
    const string& path,
    const string& dataset_name,
    const string& dataset_path,
    const string& query_path,
    const string& backend,
    const string& index_path,
    const string& cross_dir,
    int t_start, int t_end,
    int ef, int addk, int k,
    int q_lo, int q_hi,
    long long queries,
    double avg_visited,
    double recall_micro,
    double recall_macro,
    long long total_ms,
    long long setup_ms,
    long long graph_ms,
    long long dist_ms,
    double qps,
    long long load_index_ms,
    long long build_range_ms,
    long long build_gt_ms
){
    ofstream fout(path, ios::app);
    fout.setf(std::ios::fixed); fout<<setprecision(6);
    fout
        << dataset_name << ","
        << dataset_path << ","
        << (query_path.empty() ? "": query_path) << ","
        << backend << ","
        << index_path << ","
        << cross_dir << ","
        << t_start << "," << t_end << ","
        << ef << "," << addk << "," << k << ","
        << q_lo << "," << q_hi << ","
        << queries << ","
        << avg_visited << ","
        << recall_micro << ","
        << recall_macro << ","
        << total_ms << ","
        << setup_ms << ","
        << graph_ms << ","
        << dist_ms << ","
        << qps << ","
        << load_index_ms << ","
        << build_range_ms << ","
        << build_gt_ms
        << "\n";
}

// ===================== 压成连续内存 =====================
struct DenseDataset {
    int N=0;
    int D=0;
    float* data=nullptr;   // [N*D]
    float* norms=nullptr;  // [N] (L2^2)

    void free_all(){
        if (data)  _mm_free(data);
        if (norms) _mm_free(norms);
        data=nullptr;
        norms=nullptr;
        N=0; D=0;
    }
};

static inline const float* vec_ptr(const DenseDataset& ds, int idx){
    return ds.data + (size_t)idx * ds.D;
}

// ===================== 直接读取 fvecs，避开内存峰值翻倍 =====================
static DenseDataset load_fvecs_direct(const string& file) {
    ifstream fin(file, ios::binary | ios::ate);
    if (!fin) {
        cerr << "[error] cannot open fvecs: " << file << "\n";
        exit(1);
    }
    
    // 1. 获取文件总大小
    size_t file_size = fin.tellg();
    fin.seekg(0, ios::beg);

    // 2. 读第一个向量获取维度 D
    int32_t D = 0;
    fin.read(reinterpret_cast<char*>(&D), 4);
    if (D <= 0) {
        cerr << "[error] bad dim in fvecs.\n";
        exit(1);
    }

    // 3. 计算 N
    size_t bytes_per_vec = 4 + (size_t)D * 4;
    if (file_size % bytes_per_vec != 0) {
        cerr << "[warn] file size is not a perfect multiple of vector size.\n";
    }
    int N = file_size / bytes_per_vec;

    // 4. 一次性分配连续内存
    DenseDataset ds;
    ds.N = N;
    ds.D = D;
    size_t total = (size_t)N * D;
    ds.data  = (float*)_mm_malloc(total * sizeof(float), 64);
    ds.norms = (float*)_mm_malloc((size_t)N * sizeof(float), 64);
    if (!ds.data || !ds.norms) {
        cerr << "[error] OOM in load_fvecs_direct\n";
        exit(1);
    }

    // 5. 倒回开头，直接读取并计算 norm
    fin.seekg(0, ios::beg);
    for (int i = 0; i < N; i++) {
        int32_t d;
        fin.read(reinterpret_cast<char*>(&d), 4);
        float* dst = ds.data + (size_t)i * D;
        fin.read(reinterpret_cast<char*>(dst), sizeof(float) * D);
        
        float acc = 0.0f;
        for (int j = 0; j < D; j++) {
            acc += dst[j] * dst[j];
        }
        ds.norms[i] = acc;
    }
    
    return ds;
}

// ===================== AVX2 FMA 点积 =====================
static inline float dot_product_avx_unrolled(
    const float* __restrict a,
    const float* __restrict b,
    size_t dim)
{
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 31 < dim; i += 32){
        __m256 va0 = _mm256_loadu_ps(a + i +  0);
        __m256 vb0 = _mm256_loadu_ps(b + i +  0);
        sum0 = _mm256_fmadd_ps(va0, vb0, sum0);

        __m256 va1 = _mm256_loadu_ps(a + i +  8);
        __m256 vb1 = _mm256_loadu_ps(b + i +  8);
        sum1 = _mm256_fmadd_ps(va1, vb1, sum1);

        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        sum2 = _mm256_fmadd_ps(va2, vb2, sum2);

        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        sum3 = _mm256_fmadd_ps(va3, vb3, sum3);
    }

    __m256 sum = _mm256_add_ps(
        _mm256_add_ps(sum0, sum1),
        _mm256_add_ps(sum2, sum3)
    );

    for (; i + 7 < dim; i += 8){
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 v  = _mm_add_ps(lo, hi);
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    float total = _mm_cvtss_f32(v);

    for (; i < dim; ++i){
        total += a[i] * b[i];
    }
    return total;
}

static inline float l2_sq_cached(
    const float* __restrict q_ptr_local,
    float q_norm2,
    const DenseDataset& ds,
    int vid
){
    const float* xb = vec_ptr(ds, vid);
    float dot = dot_product_avx_unrolled(q_ptr_local, xb, ds.D);
    float dist2 = q_norm2 + ds.norms[vid] - 2.0f * dot;
    return dist2;
}

// ===================== CSR 图（TEXT 后端） =====================
struct CSRGraph {
    vector<uint64_t> offsets; // N+1  ✅ 升级为 64位 防溢出
    vector<int> degs;         // N
    vector<int> nbrs;         // E
};

static inline void parse_line_u_and_neighbors(
    const char* line,
    int& u_out,
    vector<int>& neigh_out
){
    neigh_out.clear();
    u_out = -1;

    char* endp = nullptr;
    long u = strtol(line, &endp, 10);
    if (endp == line) return;

    const char* p_colon = strchr(line, ':');
    if (!p_colon) return;
    u_out = (int)u;

    const char* p = p_colon + 1;
    int cur = 0;
    bool in_num = false;
    bool neg = false;

    auto flush = [&](){
        if (in_num){
            int v = neg ? -cur : cur;
            neigh_out.push_back(v);
            cur = 0; in_num = false; neg = false;
        }
    };

    for (; *p; ++p){
        char c = *p;
        if (c=='-' && !in_num){
            neg = true;
            in_num = true;
            cur = 0;
        } else if (isdigit((unsigned char)c)){
            if (!in_num){
                in_num = true;
                neg = false;
                cur = 0;
            }
            cur = cur*10 + (c - '0');
        } else {
            flush();
        }
    }
    flush();
}

static void scan_deg_one_file(
    const string& path,
    vector<int>& degs,
    bool undirected,
    int N
){
    FILE* fp = fopen(path.c_str(), "r");
    if (!fp){
        cerr << "[warn] cannot open: " << path << "\n";
        return;
    }
    static const size_t BUF_SZ = 1<<20;
    static thread_local vector<char> buf(BUF_SZ);
    setvbuf(fp, buf.data(), _IOFBF, BUF_SZ);

    const int LINE_SZ = 1<<20;
    char* line = (char*)malloc(LINE_SZ);
    vector<int> tmp;
    tmp.reserve(256);

    while (fgets(line, LINE_SZ, fp)){
        int u;
        parse_line_u_and_neighbors(line, u, tmp);
        if (u < 0 || u >= N) continue;
        for (int v: tmp){
            if (v < 0 || v >= N) continue;
            degs[u]++;
            if (undirected) degs[v]++;
        }
    }
    free(line);
    fclose(fp);
}

// ✅ 升级 offsets 和 cursor 的签名类型防溢出
static void fill_csr_one_file(
    const string& path,
    const vector<uint64_t>& offsets,
    vector<uint64_t>& cursor,
    vector<int>& nbrs,
    bool undirected,
    int N
){
    FILE* fp = fopen(path.c_str(), "r");
    if (!fp){
        cerr << "[warn] cannot open: " << path << "\n";
        return;
    }
    static const size_t BUF_SZ = 1<<20;
    static thread_local vector<char> buf(BUF_SZ);
    setvbuf(fp, buf.data(), _IOFBF, BUF_SZ);

    const int LINE_SZ = 1<<20;
    char* line = (char*)malloc(LINE_SZ);
    vector<int> tmp;
    tmp.reserve(256);

    while (fgets(line, LINE_SZ, fp)){
        int u;
        parse_line_u_and_neighbors(line, u, tmp);
        if (u < 0 || u >= N) continue;
        for (int v: tmp){
            if (v < 0 || v >= N) continue;
            nbrs[cursor[u]++] = v;
            if (undirected) nbrs[cursor[v]++] = u;
        }
    }
    free(line);
    fclose(fp);
}

static vector<int> detect_segment_ids(const string& seg_dir){
    vector<int> ids;
    std::regex re(R"(seg_(\d{5})_adj_L0\.txt)");
    for (auto& e : fs::directory_iterator(seg_dir)){
        if (!e.is_regular_file()) continue;
        auto name = e.path().filename().string();
        std::smatch m;
        if (std::regex_match(name, m, re)){
            ids.push_back(std::stoi(m[1].str()));
        }
    }
    sort(ids.begin(), ids.end());
    ids.erase(unique(ids.begin(), ids.end()), ids.end());
    return ids;
}

static CSRGraph load_graph_from_seg_outputs(
    const string& seg_dir,
    const string& cross_fwd_path,
    const string& cross_bwd_path,
    int N,
    long long& out_load_ms,
    bool verbose=true
){
    auto t0 = chrono::high_resolution_clock::now();

    auto seg_ids = detect_segment_ids(seg_dir);
    if (seg_ids.empty()){
        cerr << "[error] no seg_XXXXX_adj_L0.txt under: " << seg_dir << "\n";
        exit(1);
    }
    size_t num_windows = seg_ids.size();

    const bool SEG_UNDIRECTED = true;
    const bool CROSS_UNDIRECTED = true;

    vector<int> degs(N, 0);

    if (verbose) cerr << "[meta] num_windows=" << num_windows << " seg_dir=" << seg_dir << "\n";
    for (size_t i=0;i<num_windows;i++){
        int w = seg_ids[i];
        char buf[256];
        snprintf(buf, sizeof(buf), "seg_%05d_adj_L0.txt", w);
        string f = (fs::path(seg_dir) / buf).string();
        if (!fs::exists(f)){
            cerr << "[warn] missing: " << f << "\n";
            continue;
        }
        if (verbose) cerr << " [scan deg] window " << i << "/" << num_windows << "\n";
        scan_deg_one_file(f, degs, SEG_UNDIRECTED, N);
    }

    if (fs::exists(cross_fwd_path)){
        if (verbose) cerr << " [scan deg] cross_fwd\n";
        scan_deg_one_file(cross_fwd_path, degs, CROSS_UNDIRECTED, N);
    } else {
        cerr << "[warn] missing cross_fwd: " << cross_fwd_path << "\n";
    }

    if (fs::exists(cross_bwd_path)){
        if (verbose) cerr << " [scan deg] cross_bwd\n";
        scan_deg_one_file(cross_bwd_path, degs, CROSS_UNDIRECTED, N);
    } else {
        cerr << "[warn] missing cross_bwd: " << cross_bwd_path << "\n";
    }

    CSRGraph g;
    g.degs = std::move(degs);
    g.offsets.assign(N+1, 0ULL);  // ✅ 初始化为 64位 0
    for (int i=0;i<N;i++){
        g.offsets[i+1] = g.offsets[i] + (uint64_t)g.degs[i]; // ✅ 防止加和过程溢出
    }
    uint64_t E = g.offsets[N]; // ✅ 使用 uint64_t 存储边数
    g.nbrs.resize((size_t)E);

    vector<uint64_t> cursor = g.offsets; // ✅ 游标也需是 64位

    for (size_t i=0;i<num_windows;i++){
        int w = seg_ids[i];
        char buf[256];
        snprintf(buf, sizeof(buf), "seg_%05d_adj_L0.txt", w);
        string f = (fs::path(seg_dir) / buf).string();
        if (!fs::exists(f)) continue;
        if (verbose) cerr << " [fill csr] window " << i << "/" << num_windows << "\n";
        fill_csr_one_file(f, g.offsets, cursor, g.nbrs, SEG_UNDIRECTED, N);
    }

    if (fs::exists(cross_fwd_path)){
        if (verbose) cerr << " [fill csr] cross_fwd\n";
        fill_csr_one_file(cross_fwd_path, g.offsets, cursor, g.nbrs, CROSS_UNDIRECTED, N);
    }
    if (fs::exists(cross_bwd_path)){
        if (verbose) cerr << " [fill csr] cross_bwd\n";
        fill_csr_one_file(cross_bwd_path, g.offsets, cursor, g.nbrs, CROSS_UNDIRECTED, N);
    }

    auto t1 = chrono::high_resolution_clock::now();
    out_load_ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    cerr << "[adj] CSR edges E=" << E << " load_ms=" << out_load_ms << "ms\n";

    return g;
}

// ===================== SLOT 索引（preprocess_slot_index.cpp 输出） =====================
struct SlotIndex {
    uint32_t N = 0;
    uint32_t T = 0;
    std::vector<uint64_t> slot_ptr; // len = N*(T+1)+1
    std::vector<uint32_t> nbrs;     // len = slot_ptr.back()
};

static inline bool is_slot_index_file(const std::string& path){
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[8];
    in.read(magic, 8);
    if (!in) return false;
    const char expect[8] = {'S','L','O','T','I','D','X','1'};
    return std::memcmp(magic, expect, 8) == 0;
}

static SlotIndex load_slot_index_bin(const std::string& path, long long& out_load_ms){
    auto t0 = std::chrono::high_resolution_clock::now();

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open slot index bin: " + path);

    char magic[8];
    in.read(magic, 8);
    const char expect[8] = {'S','L','O','T','I','D','X','1'};
    if (!in || std::memcmp(magic, expect, 8) != 0){
        throw std::runtime_error("bad slot index magic (expect SLOTIDX1): " + path);
    }

    SlotIndex idx;
    in.read((char*)&idx.N, 4);
    in.read((char*)&idx.T, 4);
    if (!in) throw std::runtime_error("read N/T failed: " + path);

    const uint64_t ptr_len = (uint64_t)idx.N * (uint64_t)(idx.T + 1) + 1;
    idx.slot_ptr.resize((size_t)ptr_len);
    in.read((char*)idx.slot_ptr.data(), (std::streamsize)(ptr_len * sizeof(uint64_t)));
    if (!in) throw std::runtime_error("read slot_ptr failed: " + path);

    uint64_t Eprime = idx.slot_ptr.back();
    idx.nbrs.resize((size_t)Eprime);
    in.read((char*)idx.nbrs.data(), (std::streamsize)(Eprime * sizeof(uint32_t)));
    if (!in) throw std::runtime_error("read nbrs failed: " + path);

    auto t1 = std::chrono::high_resolution_clock::now();
    out_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cerr << "[slot] loaded bin: " << path
              << " N=" << idx.N << " T=" << idx.T
              << " ptr_len=" << idx.slot_ptr.size()
              << " Eprime=" << idx.slot_ptr.back()
              << " load_ms=" << out_load_ms << "ms\n";
    return idx;
}

// ===================== Range helper =====================
struct RangeFilter {
    vector<char> in_range; // N
    vector<int>  nodes;    // ids in [T_START,T_END]
};

static RangeFilter build_range_filter(int N, int T_START, int T_END){
    RangeFilter rf;
    rf.in_range.assign(N, 0);
    if (T_START < 0) T_START = 0;
    if (T_END >= N) T_END = N-1;
    rf.nodes.reserve((size_t)max(0, T_END - T_START + 1));
    for (int i=T_START;i<=T_END;i++){
        rf.in_range[i] = 1;
        rf.nodes.push_back(i);
    }
    return rf;
}

// ===================== GT: base-internal vs external query =====================
static vector<int> exact_topk_in_range_baseqid(
    const DenseDataset& base,
    const vector<char>& in_range,
    int qid_base,
    int k)
{
    float q_norm2 = base.norms[qid_base];
    const float* qptr = vec_ptr(base, qid_base);

    vector<pair<float,int>> dist;
    dist.reserve(1024);

    for (int i=0;i<base.N; ++i){
        if (!in_range[i] || i==qid_base) continue;
        float dot = dot_product_avx_unrolled(qptr, vec_ptr(base,i), base.D);
        float d2  = q_norm2 + base.norms[i] - 2.0f * dot;
        dist.push_back({d2, i});
    }

    if ((int)dist.size() > k){
        nth_element(dist.begin(), dist.begin()+k, dist.end(),
                    [](const auto& a, const auto& b){ return a.first < b.first; });
        dist.resize(k);
    }
    sort(dist.begin(), dist.end(),
         [](const auto& a, const auto& b){ return a.first < b.first; });

    vector<int> ids;
    ids.reserve(dist.size());
    for (auto& p : dist) ids.push_back(p.second);
    return ids;
}

static vector<int> exact_topk_in_range_queryvec(
    const DenseDataset& base,
    const vector<char>& in_range,
    const float* qptr,
    float q_norm2,
    int k)
{
    vector<pair<float,int>> dist;
    dist.reserve(1024);

    for (int i=0;i<base.N; ++i){
        if (!in_range[i]) continue;
        float dot = dot_product_avx_unrolled(qptr, vec_ptr(base,i), base.D);
        float d2  = q_norm2 + base.norms[i] - 2.0f * dot;
        dist.push_back({d2, i});
    }

    if ((int)dist.size() > k){
        nth_element(dist.begin(), dist.begin()+k, dist.end(),
                    [](const auto& a, const auto& b){ return a.first < b.first; });
        dist.resize(k);
    }
    sort(dist.begin(), dist.end(),
         [](const auto& a, const auto& b){ return a.first < b.first; });

    vector<int> ids;
    ids.reserve(dist.size());
    for (auto& p : dist) ids.push_back(p.second);
    return ids;
}

static pair<int,int> recall_at_k_counts(
    const vector<pair<int,float>>& found,
    const vector<int>& gt_ids,
    int k,
    int exclude_id) // external query => pass -1
{
    unordered_set<int> gt(gt_ids.begin(), gt_ids.end());
    int matched = 0, used = 0;
    for (const auto& pr : found){
        int id = pr.first;
        if (exclude_id >= 0 && id==exclude_id) continue;
        if (used >= k) break;
        if (gt.count(id)) matched++;
        used++;
    }
    int denom = (int)gt_ids.size();
    denom = min(denom, k);
    if (denom == 0) return {matched, 1};
    return {matched, denom};
}

// ===================== 小根堆 candidate（按距离升序 pop） =====================
struct MinHeap {
    vector<pair<float,int>> heap;

    inline void init(size_t cap_hint){
        heap.clear();
        heap.reserve(cap_hint);
    }
    inline bool empty() const { return heap.empty(); }
    inline const pair<float,int>& top() const { return heap[0]; }

    inline void sift_up(size_t idx){
        while (idx > 0){
            size_t p = (idx - 1) >> 1;
            if (heap[p].first <= heap[idx].first) break;
            std::swap(heap[p], heap[idx]);
            idx = p;
        }
    }
    inline void sift_down(size_t idx){
        size_t n = heap.size();
        for(;;){
            size_t l = (idx<<1) + 1;
            size_t r = l + 1;
            size_t best = idx;
            if (l < n && heap[l].first < heap[best].first) best = l;
            if (r < n && heap[r].first < heap[best].first) best = r;
            if (best == idx) break;
            std::swap(heap[idx], heap[best]);
            idx = best;
        }
    }

    inline void push(const pair<float,int>& val){
        heap.push_back(val);
        sift_up(heap.size()-1);
    }
    inline void pop(){
        size_t n = heap.size();
        heap[0] = heap[n-1];
        heap.pop_back();
        if (!heap.empty()) sift_down(0);
    }
};

// ===================== BestK: 固定容量 top-(k+add_k) 最大堆 =====================
struct BestK {
    vector<float> dists;
    vector<int>   ids;
    vector<int>   heap_pos;

    int size = 0;
    int cap  = 0;
    float worst_dist = std::numeric_limits<float>::infinity();

    inline void init(int capacity_hint){
        cap = max(1, capacity_hint);
        dists.clear(); ids.clear(); heap_pos.clear();
        dists.reserve(cap);
        ids.reserve(cap);
        heap_pos.reserve(cap);
        size = 0;
        worst_dist = std::numeric_limits<float>::infinity();
    }

    inline bool worse(int a_idx, int b_idx) const {
        return dists[a_idx] > dists[b_idx];
    }

    inline void heap_sift_up(size_t idx){
        while (idx > 0){
            size_t p = (idx - 1) >> 1;
            if (!worse(heap_pos[idx], heap_pos[p])) break;
            std::swap(heap_pos[idx], heap_pos[p]);
            idx = p;
        }
    }

    inline void heap_sift_down(size_t idx){
        size_t n = heap_pos.size();
        for(;;){
            size_t l = (idx<<1) + 1;
            size_t r = l + 1;
            size_t best = idx;
            if (l < n && worse(heap_pos[l], heap_pos[best])) best = l;
            if (r < n && worse(heap_pos[r], heap_pos[best])) best = r;
            if (best == idx) break;
            std::swap(heap_pos[idx], heap_pos[best]);
            idx = best;
        }
    }

    inline void push_new(float distv, int nid){
        dists.push_back(distv);
        ids.push_back(nid);
        int idx = size;
        size++;
        heap_pos.push_back(idx);
        heap_sift_up(heap_pos.size()-1);
        worst_dist = dists[heap_pos[0]];
    }

    inline void consider(float distv, int nid){
        if (size < cap){
            push_new(distv, nid);
            return;
        }
        if (distv >= worst_dist) return;

        int p = heap_pos[0];
        dists[p] = distv;
        ids[p]   = nid;
        heap_sift_down(0);
        worst_dist = dists[heap_pos[0]];
    }
};

// ===================== 通用 QueryStats（main 用来聚合计时） =====================
struct QueryStats {
    uint64_t last_dist_ns  = 0;
    uint64_t last_setup_ns = 0;
    uint64_t last_graph_ns = 0;
};

// ===================== SLOT 后端（UNIFY-style per-query filtering） =====================
struct QueryIndexSlot {
    struct NodeState { uint32_t gen; float dist; };

    const DenseDataset& base;
    const SlotIndex& sidx;
    const vector<char>& in_range; // fixed range bitmap
    int t_start;
    int t_end;

    uint32_t segL;
    uint32_t segR;

    vector<int> centers;

    mutable vector<NodeState> node_state;
    mutable uint32_t cur_gen;

    mutable uint64_t dist_ns_query;
    mutable uint64_t setup_ns_query;
    mutable uint64_t graph_ns_query;

    mutable QueryStats stats;

    QueryIndexSlot(const DenseDataset& base_,
                   const SlotIndex& sidx_,
                   const vector<char>& in_range_,
                   int t_start_, int t_end_)
        : base(base_), sidx(sidx_), in_range(in_range_), t_start(t_start_), t_end(t_end_)
    {
        size_t N = (size_t)base.N;
        node_state.resize(N);
        for (size_t i=0;i<N;i++){ node_state[i].gen = 0u; node_state[i].dist = 0.0f; }
        cur_gen = 1u;

        segL = (uint32_t)((uint64_t)(uint32_t)t_start * (uint64_t)sidx.T / (uint64_t)sidx.N);
        segR = (uint32_t)((uint64_t)(uint32_t)t_end   * (uint64_t)sidx.T / (uint64_t)sidx.N);
        if (segR >= sidx.T) segR = sidx.T - 1;

        dist_ns_query = setup_ns_query = graph_ns_query = 0;
    }

    inline void set_centers(const vector<int>& c){ centers = c; }
    inline bool already_seen(int id) const { return node_state[id].gen == cur_gen; }

    inline float ensure_distance(const float* __restrict q_ptr_local,
                                 float q_norm2,
                                 int vid,
                                 size_t& visited_count) const
    {
        NodeState& st = node_state[vid];
        if (st.gen != cur_gen){
            TIME_BLOCK(dist_ns_query, {
                float d2 = l2_sq_cached(q_ptr_local, q_norm2, base, vid);
                st.dist = d2;
                st.gen  = cur_gen;
            });
            visited_count++;
        }
        return st.dist;
    }

    pair<vector<pair<int,float>>, size_t>
    query_by_vec(const float* q_ptr_local, float q_norm2, int k_for_search, int ef,
                 const vector<int>* isolated_nodes=nullptr) const
    {
        dist_ns_query = setup_ns_query = graph_ns_query = 0;

        cur_gen++;
        if (cur_gen == 0){
            for (auto &ns : node_state) ns.gen = 0u;
            cur_gen = 1u;
        }

        auto t_setup0 = chrono::high_resolution_clock::now();

        MinHeap candidate;
        candidate.init((size_t)ef + 64);

        std::priority_queue<pair<float,int>> top_candidates;
        BestK bestk;
        bestk.init(k_for_search);

        size_t visited_count = 0;

        // pick best STARTS among centers by distance
        vector<pair<float,int>> cands;
        cands.reserve(centers.size());
        for (int c : centers){
            if (c < 0 || c >= base.N) continue;
            if (!in_range[c]) continue;
            if (already_seen(c)) continue;
            float d = ensure_distance(q_ptr_local, q_norm2, c, visited_count);
            cands.push_back({d, c});
        }
        if (!cands.empty()){
            int take = min((int)cands.size(), STARTS);
            nth_element(cands.begin(), cands.begin()+take, cands.end(),
                        [](auto& a, auto& b){ return a.first < b.first; });
            cands.resize(take);
            sort(cands.begin(), cands.end(), [](auto& a, auto& b){ return a.first < b.first; });
            for (auto& pr : cands){
                float d = pr.first;
                int s = pr.second;
                candidate.push({d, s});
                top_candidates.push({d, s});
                bestk.consider(d, s);
                if ((int)top_candidates.size() > ef) top_candidates.pop();
            }
        } else {
            // fallback: first in-range id
            int s = t_start;
            float d = ensure_distance(q_ptr_local, q_norm2, s, visited_count);
            candidate.push({d, s});
            top_candidates.push({d, s});
            bestk.consider(d, s);
        }

        float lowerBound = top_candidates.empty()
            ? std::numeric_limits<float>::infinity()
            : top_candidates.top().first;

        auto t_setup1 = chrono::high_resolution_clock::now();
        setup_ns_query += (uint64_t)chrono::duration_cast<chrono::nanoseconds>(t_setup1 - t_setup0).count();

        auto t_graph0 = chrono::high_resolution_clock::now();

        while (!candidate.empty()){
            float best_cand = candidate.top().first;
            if (best_cand > lowerBound) break;

            auto cur_node = candidate.top();
            candidate.pop();
            int u = cur_node.second;

            uint64_t base_off = (uint64_t)(uint32_t)u * (uint64_t)(sidx.T + 1);

            for (uint32_t ss = segL; ss <= segR; ++ss){
                uint64_t l = sidx.slot_ptr[(size_t)base_off + ss];
                uint64_t r = sidx.slot_ptr[(size_t)base_off + ss + 1];
                for (uint64_t p = l; p < r; ++p){
                    int v = (int)sidx.nbrs[(size_t)p];
                    if (v < t_start || v > t_end) continue; // point-level filter
                    if (already_seen(v)) continue;

                    _mm_prefetch((const char*)vec_ptr(base, v), _MM_HINT_T0);

                    float distv = ensure_distance(q_ptr_local, q_norm2, v, visited_count);
                    bestk.consider(distv, v);

                    if ((int)top_candidates.size() < ef || distv < lowerBound){
                        candidate.push({distv, v});
                        top_candidates.push({distv, v});
                        if ((int)top_candidates.size() > ef) top_candidates.pop();
                        lowerBound = top_candidates.top().first;
                    }
                }
                if (ss == segR) break; // avoid uint wrap
            }
        }

        auto t_graph1 = chrono::high_resolution_clock::now();
        graph_ns_query += (uint64_t)chrono::duration_cast<chrono::nanoseconds>(t_graph1 - t_graph0).count();

        // graph_time = (graph loop) - (distance time) to isolate overhead
        if (graph_ns_query > dist_ns_query) graph_ns_query -= dist_ns_query;
        else graph_ns_query = 0;

        // optional isolated fallback
        if (isolated_nodes && !isolated_nodes->empty()){
            for (int u : *isolated_nodes){
                if (u < 0 || u >= base.N) continue;
                if (!in_range[u]) continue;
                if (already_seen(u)) continue;
                float distv = ensure_distance(q_ptr_local, q_norm2, u, visited_count);
                bestk.consider(distv, u);
            }
        }

        vector<pair<int,float>> res;
        res.reserve(bestk.size);
        for (int i=0;i<bestk.size;i++){
            res.push_back({bestk.ids[i], bestk.dists[i]});
        }
        sort(res.begin(), res.end(), [](auto& A, auto& B){ return A.second < B.second; });

        stats.last_dist_ns  = dist_ns_query;
        stats.last_setup_ns = setup_ns_query;
        stats.last_graph_ns = graph_ns_query;

        return {res, visited_count};
    }
};

// ===================== TEXT(CSR) 后端（同样 UNIFY-style 点级过滤，不构建 per-range 子图） =====================
struct QueryIndexCSR {
    struct NodeState { uint32_t gen; float dist; };

    const DenseDataset& base;
    const CSRGraph& g;
    const vector<char>& in_range;
    int t_start;
    int t_end;

    vector<int> centers;

    mutable vector<NodeState> node_state;
    mutable uint32_t cur_gen;

    mutable uint64_t dist_ns_query;
    mutable uint64_t setup_ns_query;
    mutable uint64_t graph_ns_query;

    mutable QueryStats stats;

    QueryIndexCSR(const DenseDataset& base_,
                  const CSRGraph& g_,
                  const vector<char>& in_range_,
                  int t_start_, int t_end_)
        : base(base_), g(g_), in_range(in_range_), t_start(t_start_), t_end(t_end_)
    {
        size_t N = (size_t)base.N;
        node_state.resize(N);
        for (size_t i=0;i<N;i++){ node_state[i].gen = 0u; node_state[i].dist = 0.0f; }
        cur_gen = 1u;

        dist_ns_query = setup_ns_query = graph_ns_query = 0;
    }

    inline void set_centers(const vector<int>& c){ centers = c; }
    inline bool already_seen(int id) const { return node_state[id].gen == cur_gen; }

    inline float ensure_distance(const float* __restrict q_ptr_local,
                                 float q_norm2,
                                 int vid,
                                 size_t& visited_count) const
    {
        NodeState& st = node_state[vid];
        if (st.gen != cur_gen){
            TIME_BLOCK(dist_ns_query, {
                float d2 = l2_sq_cached(q_ptr_local, q_norm2, base, vid);
                st.dist = d2;
                st.gen  = cur_gen;
            });
            visited_count++;
        }
        return st.dist;
    }

    pair<vector<pair<int,float>>, size_t>
    query_by_vec(const float* q_ptr_local, float q_norm2, int k_for_search, int ef,
                 const vector<int>* isolated_nodes=nullptr) const
    {
        dist_ns_query = setup_ns_query = graph_ns_query = 0;

        cur_gen++;
        if (cur_gen == 0){
            for (auto &ns : node_state) ns.gen = 0u;
            cur_gen = 1u;
        }

        auto t_setup0 = chrono::high_resolution_clock::now();

        MinHeap candidate;
        candidate.init((size_t)ef + 64);

        std::priority_queue<pair<float,int>> top_candidates;
        BestK bestk;
        bestk.init(k_for_search);

        size_t visited_count = 0;

        // entry
        vector<pair<float,int>> cands;
        cands.reserve(centers.size());
        for (int c : centers){
            if (c < 0 || c >= base.N) continue;
            if (!in_range[c]) continue;
            if (already_seen(c)) continue;
            float d = ensure_distance(q_ptr_local, q_norm2, c, visited_count);
            cands.push_back({d, c});
        }
        if (!cands.empty()){
            int take = min((int)cands.size(), STARTS);
            nth_element(cands.begin(), cands.begin()+take, cands.end(),
                        [](auto& a, auto& b){ return a.first < b.first; });
            cands.resize(take);
            sort(cands.begin(), cands.end(), [](auto& a, auto& b){ return a.first < b.first; });
            for (auto& pr : cands){
                float d = pr.first;
                int s = pr.second;
                candidate.push({d, s});
                top_candidates.push({d, s});
                bestk.consider(d, s);
                if ((int)top_candidates.size() > ef) top_candidates.pop();
            }
        } else {
            int s = t_start;
            float d = ensure_distance(q_ptr_local, q_norm2, s, visited_count);
            candidate.push({d, s});
            top_candidates.push({d, s});
            bestk.consider(d, s);
        }

        float lowerBound = top_candidates.empty()
            ? std::numeric_limits<float>::infinity()
            : top_candidates.top().first;

        auto t_setup1 = chrono::high_resolution_clock::now();
        setup_ns_query += (uint64_t)chrono::duration_cast<chrono::nanoseconds>(t_setup1 - t_setup0).count();

        auto t_graph0 = chrono::high_resolution_clock::now();

        // expand from CSR neighbors, still point-level filter by [t_start,t_end]
        while (!candidate.empty()){
            float best_cand = candidate.top().first;
            if (best_cand > lowerBound) break;

            auto cur_node = candidate.top();
            candidate.pop();
            int u = cur_node.second;

            // ✅ CSR graph 左右边界修改为 uint64_t
            uint64_t l = g.offsets[u];
            uint64_t r = g.offsets[u+1];
            for (uint64_t p = l; p < r; ++p){
                int v = g.nbrs[p];
                if (v < t_start || v > t_end) continue;
                if (already_seen(v)) continue;

                _mm_prefetch((const char*)vec_ptr(base, v), _MM_HINT_T0);

                float distv = ensure_distance(q_ptr_local, q_norm2, v, visited_count);
                bestk.consider(distv, v);

                if ((int)top_candidates.size() < ef || distv < lowerBound){
                    candidate.push({distv, v});
                    top_candidates.push({distv, v});
                    if ((int)top_candidates.size() > ef) top_candidates.pop();
                    lowerBound = top_candidates.top().first;
                }
            }
        }

        auto t_graph1 = chrono::high_resolution_clock::now();
        graph_ns_query += (uint64_t)chrono::duration_cast<chrono::nanoseconds>(t_graph1 - t_graph0).count();
        if (graph_ns_query > dist_ns_query) graph_ns_query -= dist_ns_query;
        else graph_ns_query = 0;

        if (isolated_nodes && !isolated_nodes->empty()){
            for (int u : *isolated_nodes){
                if (u < 0 || u >= base.N) continue;
                if (!in_range[u]) continue;
                if (already_seen(u)) continue;
                float distv = ensure_distance(q_ptr_local, q_norm2, u, visited_count);
                bestk.consider(distv, u);
            }
        }

        vector<pair<int,float>> res;
        res.reserve(bestk.size);
        for (int i=0;i<bestk.size;i++){
            res.push_back({bestk.ids[i], bestk.dists[i]});
        }
        sort(res.begin(), res.end(), [](auto& A, auto& B){ return A.second < B.second; });

        stats.last_dist_ns  = dist_ns_query;
        stats.last_setup_ns = setup_ns_query;
        stats.last_graph_ns = graph_ns_query;

        return {res, visited_count};
    }
};

// ===================== Args parsing: --query 风格 =====================
struct Args {
    string fvecs_path;
    string index_path;   // --index : SLOT bin or seg_dir
    string cross_dir;    // optional when text backend
    string query_path;   // --query optional
    int k = 10;
    int qnum = 500;      // means q_hi = min(qnum, QN-1)
    int t_start = 50000;
    int use_isolated = 0;
};

static void print_usage(const char* prog){
    cerr <<
    "Usage:\n"
    "  " << prog << " --fvecs <base.fvecs> --index <slot.bin|seg_dir> [--cross_dir <cross_dir>]\n"
    "         [--query <query.fvecs>] [--k 10] [--qnum 500] [--t_start 50000] [--use_isolated 0]\n"
    "\n"
    "Notes:\n"
    "  - If --index is SLOTIDX1 .bin => SLOT backend.\n"
    "  - Else treat --index as seg_dir (TEXT backend), load seg_XXXXX_adj_L0.txt + cross_edges_* from --cross_dir.\n"
    "  - Query set:\n"
    "      * default: use base vectors [0..qnum]\n"
    "      * with --query: load query vectors from query.fvecs\n"
    "\n";
}

static bool match_opt(const string& a, const char* opt){
    return a == opt;
}

static Args parse_args(int argc, char** argv){
    Args args;
    for (int i=1;i<argc;i++){
        string a = argv[i];
        if (match_opt(a, "--help") || match_opt(a, "-h")){
            print_usage(argv[0]);
            exit(0);
        } else if (match_opt(a, "--fvecs") && i+1<argc){
            args.fvecs_path = argv[++i];
        } else if (match_opt(a, "--index") && i+1<argc){
            args.index_path = argv[++i];
        } else if (match_opt(a, "--cross_dir") && i+1<argc){
            args.cross_dir = argv[++i];
        } else if (match_opt(a, "--query") && i+1<argc){
            args.query_path = argv[++i];
        } else if (match_opt(a, "--k") && i+1<argc){
            args.k = stoi(argv[++i]);
        } else if (match_opt(a, "--qnum") && i+1<argc){
            args.qnum = stoi(argv[++i]);
        } else if (match_opt(a, "--t_start") && i+1<argc){
            args.t_start = stoi(argv[++i]);
        } else if (match_opt(a, "--use_isolated") && i+1<argc){
            args.use_isolated = stoi(argv[++i]);
        } else {
            cerr << "[error] unknown or incomplete option: " << a << "\n";
            print_usage(argv[0]);
            exit(2);
        }
    }

    if (args.fvecs_path.empty() || args.index_path.empty()){
        cerr << "[error] missing required: --fvecs or --index\n";
        print_usage(argv[0]);
        exit(2);
    }
    return args;
}

// ===================== 主程序 =====================
int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Args args = parse_args(argc, argv);

#ifdef _OPENMP
    cerr << "OpenMP max threads: " << omp_get_max_threads() << "\n";
#endif

    // 1) load base vectors ✅ 换成优化后的直接加载方式
    DenseDataset base = load_fvecs_direct(args.fvecs_path);
    int N = base.N;
    int D = base.D;
    cerr << "[base] N="<<N<<", D="<<D<<"\n";

    // 2) load external queries (optional)
    DenseDataset qds;
    bool USE_EXTERNAL_QUERIES = !args.query_path.empty();
    if (USE_EXTERNAL_QUERIES){
        // ✅ 外部 query 也直接加载
        qds = load_fvecs_direct(args.query_path);
        if (qds.D != base.D){
            cerr << "[error] query dim mismatch: qD="<<qds.D<<" baseD="<<base.D<<"\n";
            return 2;
        }
        cerr << "[query] external Q="<<qds.N<<", D="<<qds.D<<"\n";
    } else {
        cerr << "[query] use base as query set (qid in base)\n";
    }

    // 3) build dynamic GRID_T_END (same idea, but allow override t_start)
    int T_START = args.t_start;
    vector<int> GRID_T_END_DYNAMIC;
    {
        static const int PCTS[] = {2,5,10,20,30,40,50,60,70,80,90};
        GRID_T_END_DYNAMIC.reserve(sizeof(PCTS)/sizeof(int));

        for (int p : PCTS){
            long long te = (long long)floor((long double)N * p / 100.0L);
            te += (long long)T_START;
            te = std::max((long long)T_START, std::min(te, (long long)N - 1));
            GRID_T_END_DYNAMIC.push_back((int)te);
        }
        sort(GRID_T_END_DYNAMIC.begin(), GRID_T_END_DYNAMIC.end());
        GRID_T_END_DYNAMIC.erase(
            std::unique(GRID_T_END_DYNAMIC.begin(), GRID_T_END_DYNAMIC.end()),
            GRID_T_END_DYNAMIC.end()
        );
    }

    // 4) detect backend and load index
    long long load_index_ms = 0;
    bool USE_SLOT_BACKEND = is_slot_index_file(args.index_path);

    SlotIndex slot_idx;
    CSRGraph g_all;

    string backend_name;

    if (USE_SLOT_BACKEND){
        backend_name = "SLOT";
        try {
            slot_idx = load_slot_index_bin(args.index_path, load_index_ms);
        } catch (const std::exception& e){
            cerr << "[error] slot index load failed: " << e.what() << "\n";
            return 2;
        }
        if ((int)slot_idx.N != N){
            cerr << "[error] slot index N mismatch: slot.N=" << slot_idx.N << " but base N=" << N << "\n";
            return 2;
        }
    } else {
        backend_name = "TEXT";
        if (args.cross_dir.empty()) args.cross_dir = args.index_path;
        string cross_fwd = (fs::path(args.cross_dir) / "cross_edges_fwd_1.txt").string();
        string cross_bwd = (fs::path(args.cross_dir) / "cross_edges_bwd_1.txt").string();
        g_all = load_graph_from_seg_outputs(args.index_path, cross_fwd, cross_bwd, N, load_index_ms, true);
    }

    // 5) CSV
    string dataset_name = dataset_name_from_fvecs(args.fvecs_path);
    string csv_path = build_csv_path(args.fvecs_path, args.index_path, args.cross_dir, args.query_path);
    cerr << "[meta] dataset_name=" << dataset_name << "\n";
    cerr << "[meta] backend=" << backend_name << "\n";
    cerr << "[meta] CSV => " << csv_path << "\n";
    csv_write_header_if_needed(csv_path);

    const int nEF = (int)(sizeof(GRID_EF)/sizeof(int));
    const int nAK = (int)(sizeof(GRID_ADDK)/sizeof(int));
    int k = args.k;

    // query count range
    int QN = USE_EXTERNAL_QUERIES ? qds.N : base.N;
    int q_hi_target = min(args.qnum, QN-1);
    int q_lo = 0;

    // sweep T_END
    for (int T_END : GRID_T_END_DYNAMIC){
        if (T_END < T_START) continue;

        // build range filter
        auto t_rf0 = chrono::high_resolution_clock::now();
        RangeFilter rf = build_range_filter(N, T_START, T_END);
        auto t_rf1 = chrono::high_resolution_clock::now();
        long long build_range_ms = chrono::duration_cast<chrono::milliseconds>(t_rf1 - t_rf0).count();
        cerr << "\n[range] ["<<T_START<<","<<T_END<<"] size=" << rf.nodes.size()
             << " build_range_ms=" << build_range_ms << "\n";

        // optional isolated nodes detection (simple, SLOT uses activated slots; TEXT uses deg in-range)
        vector<int> isolated_nodes;
        if (args.use_isolated){
            isolated_nodes.reserve(rf.nodes.size());
            if (USE_SLOT_BACKEND){
                QueryIndexSlot tmp(base, slot_idx, rf.in_range, T_START, T_END);
                for (int u : rf.nodes){
                    bool has = false;
                    uint64_t base_off = (uint64_t)(uint32_t)u * (uint64_t)(slot_idx.T + 1);
                    for (uint32_t ss = tmp.segL; ss <= tmp.segR; ++ss){
                        uint64_t l = slot_idx.slot_ptr[(size_t)base_off + ss];
                        uint64_t r = slot_idx.slot_ptr[(size_t)base_off + ss + 1];
                        for (uint64_t p = l; p < r; ++p){
                            int v = (int)slot_idx.nbrs[(size_t)p];
                            if (v >= T_START && v <= T_END){ has = true; break; }
                        }
                        if (ss == tmp.segR) break;
                        if (has) break;
                    }
                    if (!has) isolated_nodes.push_back(u);
                }
            } else {
                // TEXT: treat node as isolated if it has no neighbor in [T_START,T_END]
                for (int u : rf.nodes){
                    bool has = false;
                    // ✅ 升级为 64位 防溢出
                    uint64_t l = g_all.offsets[u], r = g_all.offsets[u+1];
                    for (uint64_t p=l;p<r;p++){
                        int v = g_all.nbrs[p];
                        if (v>=T_START && v<=T_END){ has=true; break; }
                    }
                    if (!has) isolated_nodes.push_back(u);
                }
            }
            cerr << "[iso] ENABLED isolated_nodes=" << isolated_nodes.size()
                 << " / in-range=" << rf.nodes.size() << "\n";
        } else {
            cerr << "[iso] DISABLED\n";
        }

        // build GT cache once per range
        auto t_gt0 = chrono::high_resolution_clock::now();
        vector<vector<int>> gt_cache(q_hi_target - q_lo + 1);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
        for (int qi = q_lo; qi <= q_hi_target; ++qi){
            if (!USE_EXTERNAL_QUERIES){
                gt_cache[qi - q_lo] = exact_topk_in_range_baseqid(base, rf.in_range, qi, k);
            } else {
                const float* qptr = vec_ptr(qds, qi);
                float qn2 = qds.norms[qi];
                gt_cache[qi - q_lo] = exact_topk_in_range_queryvec(base, rf.in_range, qptr, qn2, k);
            }
        }

        auto t_gt1 = chrono::high_resolution_clock::now();
        long long build_gt_ms = chrono::duration_cast<chrono::milliseconds>(t_gt1 - t_gt0).count();
        cerr << "[gt] build_gt_ms=" << build_gt_ms << "\n";

        // build query engine + centers (per range)
        unique_ptr<QueryIndexSlot> eng_slot;
        unique_ptr<QueryIndexCSR>  eng_csr;

        // centers: sample 64 in-range nodes (then pick best STARTS per query)
        vector<int> centers;
        if ((int)rf.nodes.size() <= 64) centers = rf.nodes;
        else {
            std::mt19937 gen(1234567u + (uint32_t)T_END);
            vector<int> sh = rf.nodes;
            std::shuffle(sh.begin(), sh.end(), gen);
            centers.assign(sh.begin(), sh.begin() + 64);
        }
        cerr << "[centers] " << centers.size() << "\n";

        if (USE_SLOT_BACKEND){
            eng_slot.reset(new QueryIndexSlot(base, slot_idx, rf.in_range, T_START, T_END));
            eng_slot->set_centers(centers);
        } else {
            eng_csr.reset(new QueryIndexCSR(base, g_all, rf.in_range, T_START, T_END));
            eng_csr->set_centers(centers);
        }

        // ef/add_k sweep
        for (int i_ef=0;i_ef<nEF;i_ef++){
            for (int i_ak=0;i_ak<nAK;i_ak++){
                int ef = GRID_EF[i_ef];
                int add_k = GRID_ADDK[i_ak];

                cerr << "\n==== [RUN] T_END="<<T_END
                     << "  ef="<<ef
                     << "  add_k="<<add_k
                     << " ====\n";

                long long total_visited_sum = 0;
                long long q_cnt = 0;

                long long matched_sum = 0;
                long long denom_sum   = 0;
                double macro_recall_sum = 0.0;

                long long total_ns_sum = 0;
                long long setup_ns_sum = 0;
                long long graph_ns_sum = 0;
                long long dist_ns_sum  = 0;

                const vector<int>* iso_ptr = (args.use_isolated ? &isolated_nodes : nullptr);

                for (int qi = q_lo; qi <= q_hi_target; ++qi){
                    // choose query vec
                    const float* qptr = USE_EXTERNAL_QUERIES ? vec_ptr(qds, qi) : vec_ptr(base, qi);
                    float qn2 = USE_EXTERNAL_QUERIES ? qds.norms[qi] : base.norms[qi];

                    auto tq0 = chrono::high_resolution_clock::now();

                    vector<pair<int,float>> res;
                    size_t visited = 0;

                    if (USE_SLOT_BACKEND){
                        tie(res, visited) = eng_slot->query_by_vec(qptr, qn2, k+add_k, ef, iso_ptr);
                    } else {
                        tie(res, visited) = eng_csr->query_by_vec(qptr, qn2, k+add_k, ef, iso_ptr);
                    }

                    auto tq1 = chrono::high_resolution_clock::now();

                    long long this_total_ns =
                        chrono::duration_cast<chrono::nanoseconds>(tq1 - tq0).count();
                    total_ns_sum += this_total_ns;

                    if (USE_SLOT_BACKEND){
                        setup_ns_sum += (long long)eng_slot->stats.last_setup_ns;
                        graph_ns_sum += (long long)eng_slot->stats.last_graph_ns;
                        dist_ns_sum  += (long long)eng_slot->stats.last_dist_ns;
                    } else {
                        setup_ns_sum += (long long)eng_csr->stats.last_setup_ns;
                        graph_ns_sum += (long long)eng_csr->stats.last_graph_ns;
                        dist_ns_sum  += (long long)eng_csr->stats.last_dist_ns;
                    }

                    const auto& gt_ids = gt_cache[qi - q_lo];
                    int exclude_id = USE_EXTERNAL_QUERIES ? -1 : qi; // ✅ 坑修复：外部 query 不排除 qi
                    auto [matched, denom] = recall_at_k_counts(res, gt_ids, k, exclude_id);
                    double recall = (double)matched / (double)denom;

                    total_visited_sum += (long long)visited;
                    q_cnt += 1;

                    matched_sum += matched;
                    denom_sum   += denom;
                    macro_recall_sum += recall;
                }

                double avg_visited = q_cnt ? (double)total_visited_sum / (double)q_cnt : 0.0;
                double recall_micro = (denom_sum > 0) ? (double)matched_sum / (double)denom_sum : 0.0;
                double recall_macro = (q_cnt > 0) ? (double)macro_recall_sum / (double)q_cnt : 0.0;

                long long total_ms = (long long)llround(total_ns_sum / 1e6);
                long long setup_ms = (long long)llround(setup_ns_sum / 1e6);
                long long graph_ms = (long long)llround(graph_ns_sum / 1e6);
                long long dist_ms  = (long long)llround(dist_ns_sum  / 1e6);

                double total_s = (double)total_ns_sum / 1e9;
                double qps = (total_s > 0.0) ? (double)q_cnt / total_s : 0.0;

                cout << "==== Summary over queries [" << q_lo << "," << q_hi_target << "] ====\n";
                cout << "[cfg] backend="<<backend_name
                     << " T_START=" << T_START << ", T_END=" << T_END
                     << " | ef="<<ef<<" add_k="<<add_k<<" k="<<k<<"\n";
                cout << "Queries: " << q_cnt
                     << " | AvgVisited: " << fixed << setprecision(1) << avg_visited << "\n";
                cout << "Recall@"<< k << " (micro): " << fixed << setprecision(4) << recall_micro << "\n";
                cout << "Recall@"<< k << " (macro): " << fixed << setprecision(4) << recall_macro << "\n";
                cout << "Time breakdown (sum over queries, ms):\n"
                     << "  total_ms=" << total_ms
                     << " | setup_ms=" << setup_ms
                     << " | graph_ms=" << graph_ms
                     << " | dist_ms="  << dist_ms << "\n";
                cout << "QPS: " << fixed << setprecision(2) << qps << " q/s\n\n";

                csv_append_row(
                    csv_path,
                    dataset_name, args.fvecs_path,
                    args.query_path,
                    backend_name,
                    args.index_path, args.cross_dir,
                    T_START, T_END,
                    ef, add_k, k,
                    q_lo, q_hi_target,
                    q_cnt,
                    avg_visited,
                    recall_micro,
                    recall_macro,
                    total_ms,
                    setup_ms,
                    graph_ms,
                    dist_ms,
                    qps,
                    load_index_ms,
                    build_range_ms,
                    build_gt_ms
                );
            }
        }
    }

    base.free_all();
    if (USE_EXTERNAL_QUERIES) qds.free_all();
    return 0;
}