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

    const Format matrix_format = (sparse_format == "csr") ? CSR : CSC;
    Tensor<double> A = read(argv[1], matrix_format);
    Tensor<double> B = read(argv[2], matrix_format);
    A.setName("A");
    B.setName("B");
    A.pack();
    B.pack();

    if (A.getDimension(0) != B.getDimension(0) || A.getDimension(1) != B.getDimension(1)) {
        std::cerr << "Error: dimension mismatch for SpElMul\n";
        return 1;
    }

    const int rows = A.getDimension(0);
    const int cols = A.getDimension(1);
    Tensor<double> C({rows, cols}, matrix_format);

    IndexVar i, j;
    C(i, j) = A(i, j) * B(i, j);
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
        C.setNeedsCompute(true);
        auto start = high_resolution_clock::now();
        C.compute();
        auto end = high_resolution_clock::now();
        iter_times.push_back(duration<double, std::milli>(end - start).count());
    }
    TimingStats stats = compute_stats(iter_times);

    std::vector<double> c_ref(c_nnz, 0.0);
    if (sparse_format == "csr") {
        int *a_ptr, *a_idx, *b_ptr, *b_idx, *c_ptr, *c_idx;
        double *a_vals, *b_vals, *unused_c_vals;
        getCSRArrays<double>(A, &a_ptr, &a_idx, &a_vals);
        getCSRArrays<double>(B, &b_ptr, &b_idx, &b_vals);
        getCSRArrays<double>(C, &c_ptr, &c_idx, &unused_c_vals);
        for (int row = 0; row < rows; row++) {
            int pA = a_ptr[row];
            int pB = b_ptr[row];
            int endA = a_ptr[row + 1];
            int endB = b_ptr[row + 1];
            for (int pC = c_ptr[row]; pC < c_ptr[row + 1]; pC++) {
                const int col = c_idx[pC];
                double av = 0.0;
                double bv = 0.0;
                while (pA < endA && a_idx[pA] < col) pA++;
                while (pB < endB && b_idx[pB] < col) pB++;
                if (pA < endA && a_idx[pA] == col) {
                    av = a_vals[pA];
                    pA++;
                }
                if (pB < endB && b_idx[pB] == col) {
                    bv = b_vals[pB];
                    pB++;
                }
                c_ref[static_cast<size_t>(pC)] = av * bv;
            }
        }
    } else {
        int *a_ptr, *a_idx, *b_ptr, *b_idx, *c_ptr, *c_idx;
        double *a_vals, *b_vals, *unused_c_vals;
        getCSCArrays<double>(A, &a_ptr, &a_idx, &a_vals);
        getCSCArrays<double>(B, &b_ptr, &b_idx, &b_vals);
        getCSCArrays<double>(C, &c_ptr, &c_idx, &unused_c_vals);
        for (int col = 0; col < cols; col++) {
            int pA = a_ptr[col];
            int pB = b_ptr[col];
            int endA = a_ptr[col + 1];
            int endB = b_ptr[col + 1];
            for (int pC = c_ptr[col]; pC < c_ptr[col + 1]; pC++) {
                const int row = c_idx[pC];
                double av = 0.0;
                double bv = 0.0;
                while (pA < endA && a_idx[pA] < row) pA++;
                while (pB < endB && b_idx[pB] < row) pB++;
                if (pA < endA && a_idx[pA] == row) {
                    av = a_vals[pA];
                    pA++;
                }
                if (pB < endB && b_idx[pB] == row) {
                    bv = b_vals[pB];
                    pB++;
                }
                c_ref[static_cast<size_t>(pC)] = av * bv;
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

    const int nnz_a = static_cast<int>(A.getStorage().getValues().getSize());
    const int nnz_b = static_cast<int>(B.getStorage().getValues().getSize());

    std::cout << "Matrix A: " << argv[1] << " (" << rows << " x " << cols << ", " << nnz_a << " nnz)\n";
    std::cout << "Matrix B: " << argv[2] << " (" << rows << " x " << cols << ", " << nnz_b << " nnz)\n";
    std::cout << "Format: " << sparse_format << "\n";
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
