#include "taco.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace taco;
using namespace std::chrono;

namespace {

struct TimingStats {
    double total_ms = 0.0;
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double stddev_ms = 0.0;
    double variance_pct = 0.0;
};

static std::string normalize_format(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

static bool parse_positive_int(const char* text, int* out) {
    char* end = nullptr;
    long parsed = std::strtol(text, &end, 10);
    if (end == text || (end && *end != '\0')) return false;
    if (parsed <= 0 || parsed > static_cast<long>(std::numeric_limits<int>::max())) return false;
    *out = static_cast<int>(parsed);
    return true;
}

static TimingStats compute_stats(const std::vector<double>& iter_times) {
    TimingStats stats;
    if (iter_times.empty()) return stats;
    stats.min_ms = iter_times[0];
    stats.max_ms = iter_times[0];
    for (double value : iter_times) {
        stats.total_ms += value;
        if (value < stats.min_ms) stats.min_ms = value;
        if (value > stats.max_ms) stats.max_ms = value;
    }
    stats.avg_ms = stats.total_ms / static_cast<double>(iter_times.size());
    double variance = 0.0;
    for (double value : iter_times) {
        double diff = value - stats.avg_ms;
        variance += diff * diff;
    }
    variance /= static_cast<double>(iter_times.size());
    stats.stddev_ms = std::sqrt(variance);
    stats.variance_pct = (stats.avg_ms > 0.0) ? (stats.stddev_ms / stats.avg_ms * 100.0) : 0.0;
    return stats;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <S.mtx> [K] [iterations] [csr|csc]\n";
        return 1;
    }

    int K = 64;
    if (argc >= 3 && !parse_positive_int(argv[2], &K)) {
        std::cerr << "Error: K must be > 0\n";
        return 1;
    }
    int iterations = 100;
    if (argc >= 4 && !parse_positive_int(argv[3], &iterations)) {
        std::cerr << "Error: iterations must be > 0\n";
        return 1;
    }
    std::string sparse_format = (argc >= 5) ? normalize_format(argv[4]) : "csr";
    if (sparse_format != "csr" && sparse_format != "csc") {
        std::cerr << "Error: format must be 'csr' or 'csc'\n";
        return 1;
    }

    const Format matrix_format = (sparse_format == "csr") ? CSR : CSC;
    Format dense2({Dense, Dense});

    Tensor<double> S = read(argv[1], matrix_format);
    S.pack();
    const int rows = S.getDimension(0);
    const int cols = S.getDimension(1);

    Tensor<double> D({rows, K}, dense2);
    Tensor<double> E({K, cols}, dense2);
    srand(42);
    std::vector<double> d_vals(static_cast<size_t>(rows) * static_cast<size_t>(K), 0.0);
    std::vector<double> e_vals(static_cast<size_t>(K) * static_cast<size_t>(cols), 0.0);
    for (int i = 0; i < rows; i++) {
        for (int k = 0; k < K; k++) {
            size_t idx = static_cast<size_t>(i) * static_cast<size_t>(K) + static_cast<size_t>(k);
            d_vals[idx] = static_cast<double>(rand()) / RAND_MAX;
            D.insert({i, k}, d_vals[idx]);
        }
    }
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < cols; j++) {
            size_t idx = static_cast<size_t>(k) * static_cast<size_t>(cols) + static_cast<size_t>(j);
            e_vals[idx] = static_cast<double>(rand()) / RAND_MAX;
            E.insert({k, j}, e_vals[idx]);
        }
    }
    D.pack();
    E.pack();

    Tensor<double> Tmp({rows, cols}, dense2);
    Tensor<double> C({rows, cols}, matrix_format);
    IndexVar i, j, k;
    Tmp(i, j) = D(i, k) * E(k, j);
    C(i, j) = S(i, j) * Tmp(i, j);
    Tmp.compile();
    Tmp.assemble();
    C.compile();
    C.assemble();

    auto c_values_array = C.getStorage().getValues();
    const size_t c_nnz = c_values_array.getSize();
    auto clear_output = [&]() {
        auto tmp_values = Tmp.getStorage().getValues();
        double* tmp_data = static_cast<double*>(tmp_values.getData());
        std::fill(tmp_data, tmp_data + tmp_values.getSize(), 0.0);
        auto values = C.getStorage().getValues();
        double* data = static_cast<double*>(values.getData());
        std::fill(data, data + values.getSize(), 0.0);
    };

    clear_output();
    Tmp.setNeedsCompute(true);
    Tmp.compute();
    C.setNeedsCompute(true);
    C.compute();

    std::vector<double> iter_times;
    iter_times.reserve(static_cast<size_t>(iterations));
    for (int iter = 0; iter < iterations; iter++) {
        clear_output();
        Tmp.setNeedsCompute(true);
        auto start = high_resolution_clock::now();
        Tmp.compute();
        C.setNeedsCompute(true);
        C.compute();
        auto end = high_resolution_clock::now();
        iter_times.push_back(duration<double, std::milli>(end - start).count());
    }
    TimingStats stats = compute_stats(iter_times);

    Tensor<double> C_ref_dense({rows, cols}, dense2);
    C_ref_dense(i, j) = S(i, j) * D(i, k) * E(k, j);
    C_ref_dense.compile();
    C_ref_dense.assemble();
    C_ref_dense.setNeedsCompute(true);
    C_ref_dense.compute();
    auto c_ref_dense_arr = C_ref_dense.getStorage().getValues();
    const double* c_ref_dense_vals = static_cast<const double*>(c_ref_dense_arr.getData());

    std::vector<double> c_ref(c_nnz, 0.0);
    if (sparse_format == "csr") {
        int *c_ptr, *c_idx;
        double* unused_c_vals;
        getCSRArrays<double>(C, &c_ptr, &c_idx, &unused_c_vals);
        for (int row = 0; row < rows; row++) {
            for (int p = c_ptr[row]; p < c_ptr[row + 1]; p++) {
                const int col = c_idx[p];
                const size_t dense_idx = static_cast<size_t>(row) * static_cast<size_t>(cols) +
                                         static_cast<size_t>(col);
                c_ref[static_cast<size_t>(p)] = c_ref_dense_vals[dense_idx];
            }
        }
    } else {
        int *c_ptr, *c_idx;
        double* unused_c_vals;
        getCSCArrays<double>(C, &c_ptr, &c_idx, &unused_c_vals);
        for (int col = 0; col < cols; col++) {
            for (int p = c_ptr[col]; p < c_ptr[col + 1]; p++) {
                const int row = c_idx[p];
                const size_t dense_idx = static_cast<size_t>(row) * static_cast<size_t>(cols) +
                                         static_cast<size_t>(col);
                c_ref[static_cast<size_t>(p)] = c_ref_dense_vals[dense_idx];
            }
        }
    }

    double max_error = 0.0;
    auto final_values = C.getStorage().getValues();
    const double* c_values = static_cast<const double*>(final_values.getData());
    for (size_t p = 0; p < c_nnz; p++) {
        double err = c_values[p] - c_ref[p];
        if (err < 0.0) err = -err;
        if (err > max_error) max_error = err;
    }

    const int nnz_s = static_cast<int>(S.getStorage().getValues().getSize());

    std::cout << "Matrix S: " << argv[1] << " (" << rows << " x " << cols << ", " << nnz_s << " nnz)\n";
    std::cout << "Format: " << sparse_format << "\n";
    std::cout << "K dimension: " << K << "\n";
    std::cout << "Output C nnz: " << c_nnz << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << stats.total_ms << " ms\n";
    std::cout << "Avg time per iteration: " << std::fixed << std::setprecision(4) << stats.avg_ms << " ms\n";
    std::cout << "Min time per iteration: " << std::fixed << std::setprecision(4) << stats.min_ms << " ms\n";
    std::cout << "Max time per iteration: " << std::fixed << std::setprecision(4) << stats.max_ms << " ms\n";
    std::cout << "Stddev time: " << std::fixed << std::setprecision(4) << stats.stddev_ms << " ms\n";
    std::cout << "Variance: " << std::fixed << std::setprecision(2) << stats.variance_pct << "%\n";
    std::cout << "Max error vs reference: " << std::scientific << max_error << "\n";
    std::cout << "Implementation: TACO\n";
    return 0;
}
