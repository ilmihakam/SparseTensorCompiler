#include "taco.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
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
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <A.mtx> <B.mtx> [iterations] [csr|csc]\n";
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

    Format compute_format = (sparse_format == "csr") ? CSR : CSC;
    const bool csc_safe_fallback = (sparse_format == "csc");
    if (csc_safe_fallback) {
        compute_format = CSR;
    }
    Tensor<double> A = read(argv[1], compute_format);
    Tensor<double> B = read(argv[2], compute_format);
    A.setName("A");
    B.setName("B");
    A.pack();
    B.pack();

    if (A.getDimension(1) != B.getDimension(0)) {
        std::cerr << "Error: A.cols must equal B.rows for SpGEMM\n";
        return 1;
    }

    const int rows = A.getDimension(0);
    const int cols = B.getDimension(1);
    Tensor<double> C({rows, cols}, compute_format);

    IndexVar i, j, k;
    C(i, j) = A(i, k) * B(k, j);
    C.compile();
    C.assemble();

    auto c_values_array = C.getStorage().getValues();
    const size_t c_nnz = c_values_array.getSize();
    auto clear_output = [&]() {
        auto values = C.getStorage().getValues();
        double* data = static_cast<double*>(values.getData());
        std::fill(data, data + values.getSize(), 0.0);
    };

    clear_output();
    C.setNeedsCompute(true);
    C.compute();

    std::vector<double> iter_times;
    iter_times.reserve(static_cast<size_t>(iterations));
    for (int iter = 0; iter < iterations; iter++) {
        clear_output();
        auto start = high_resolution_clock::now();
        C.setNeedsCompute(true);
        C.compute();
        auto end = high_resolution_clock::now();
        iter_times.push_back(duration<double, std::milli>(end - start).count());
    }
    TimingStats stats = compute_stats(iter_times);

    std::unordered_map<long long, double> ref_map;
    ref_map.reserve(static_cast<size_t>(rows) * 4);
    if (compute_format == CSR) {
        int *a_ptr, *a_idx, *b_ptr, *b_idx, *c_ptr, *c_idx;
        double *a_vals, *b_vals, *unused_c_vals;
        getCSRArrays<double>(A, &a_ptr, &a_idx, &a_vals);
        getCSRArrays<double>(B, &b_ptr, &b_idx, &b_vals);
        getCSRArrays<double>(C, &c_ptr, &c_idx, &unused_c_vals);

        std::vector<double> acc(static_cast<size_t>(cols), 0.0);
        std::vector<unsigned char> marked(static_cast<size_t>(cols), 0);
        std::vector<int> touched;
        touched.reserve(static_cast<size_t>(cols));

        for (int row = 0; row < rows; row++) {
            touched.clear();
            for (int pA = a_ptr[row]; pA < a_ptr[row + 1]; pA++) {
                const int kk = a_idx[pA];
                const double a_val = a_vals[pA];
                for (int pB = b_ptr[kk]; pB < b_ptr[kk + 1]; pB++) {
                    const int col = b_idx[pB];
                    if (!marked[static_cast<size_t>(col)]) {
                        marked[static_cast<size_t>(col)] = 1;
                        touched.push_back(col);
                    }
                    acc[static_cast<size_t>(col)] += a_val * b_vals[pB];
                }
            }
            for (int col : touched) {
                const long long key = static_cast<long long>(row) * static_cast<long long>(cols) + col;
                ref_map[key] = acc[static_cast<size_t>(col)];
            }
            for (int col : touched) {
                acc[static_cast<size_t>(col)] = 0.0;
                marked[static_cast<size_t>(col)] = 0;
            }
        }
    } else {
        int *a_ptr, *a_idx, *b_ptr, *b_idx, *c_ptr, *c_idx;
        double *a_vals, *b_vals, *unused_c_vals;
        getCSCArrays<double>(A, &a_ptr, &a_idx, &a_vals);
        getCSCArrays<double>(B, &b_ptr, &b_idx, &b_vals);
        getCSCArrays<double>(C, &c_ptr, &c_idx, &unused_c_vals);

        std::vector<double> acc(static_cast<size_t>(rows), 0.0);
        std::vector<unsigned char> marked(static_cast<size_t>(rows), 0);
        std::vector<int> touched;
        touched.reserve(static_cast<size_t>(rows));

        for (int col = 0; col < cols; col++) {
            touched.clear();
            for (int pB = b_ptr[col]; pB < b_ptr[col + 1]; pB++) {
                const int kk = b_idx[pB];
                const double b_val = b_vals[pB];
                for (int pA = a_ptr[kk]; pA < a_ptr[kk + 1]; pA++) {
                    const int row = a_idx[pA];
                    if (!marked[static_cast<size_t>(row)]) {
                        marked[static_cast<size_t>(row)] = 1;
                        touched.push_back(row);
                    }
                    acc[static_cast<size_t>(row)] += a_vals[pA] * b_val;
                }
            }
            for (int row : touched) {
                const long long key = static_cast<long long>(row) * static_cast<long long>(cols) + col;
                ref_map[key] = acc[static_cast<size_t>(row)];
            }
            for (int row : touched) {
                acc[static_cast<size_t>(row)] = 0.0;
                marked[static_cast<size_t>(row)] = 0;
            }
        }
    }

    std::unordered_map<long long, double> actual_map;
    actual_map.reserve(c_nnz * 2 + 1);
    auto final_values = C.getStorage().getValues();
    const double* c_values = static_cast<const double*>(final_values.getData());
    if (compute_format == CSR) {
        int *c_ptr, *c_idx;
        double* unused_c_vals;
        getCSRArrays<double>(C, &c_ptr, &c_idx, &unused_c_vals);
        for (int row = 0; row < rows; row++) {
            for (int p = c_ptr[row]; p < c_ptr[row + 1]; p++) {
                long long key = static_cast<long long>(row) * static_cast<long long>(cols) + c_idx[p];
                actual_map[key] += c_values[p];
            }
        }
    } else {
        int *c_ptr, *c_idx;
        double* unused_c_vals;
        getCSCArrays<double>(C, &c_ptr, &c_idx, &unused_c_vals);
        for (int col = 0; col < cols; col++) {
            for (int p = c_ptr[col]; p < c_ptr[col + 1]; p++) {
                long long key = static_cast<long long>(c_idx[p]) * static_cast<long long>(cols) + col;
                actual_map[key] += c_values[p];
            }
        }
    }

    double max_error = 0.0;
    for (const auto& kv : ref_map) {
        const double actual = actual_map.count(kv.first) ? actual_map[kv.first] : 0.0;
        double err = actual - kv.second;
        if (err < 0.0) err = -err;
        if (err > max_error) max_error = err;
    }
    for (const auto& kv : actual_map) {
        if (ref_map.count(kv.first)) continue;
        double err = kv.second;
        if (err < 0.0) err = -err;
        if (err > max_error) max_error = err;
    }

    const int nnz_a = static_cast<int>(A.getStorage().getValues().getSize());
    const int nnz_b = static_cast<int>(B.getStorage().getValues().getSize());

    std::cout << "Matrix A: " << argv[1] << " (" << A.getDimension(0) << " x " << A.getDimension(1) << ", "
              << nnz_a << " nnz)\n";
    std::cout << "Matrix B: " << argv[2] << " (" << B.getDimension(0) << " x " << B.getDimension(1) << ", "
              << nnz_b << " nnz)\n";
    std::cout << "Format: " << sparse_format << "\n";
    if (csc_safe_fallback) {
        std::cout << "Kernel mode: csc_safe_fallback_via_csr\n";
    }
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
