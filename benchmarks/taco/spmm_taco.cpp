#include "taco.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cctype>
#include <limits>

using namespace taco;
using namespace std::chrono;

namespace {

struct RefCSRMatrix {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    std::vector<int> rowptr;
    std::vector<int> colidx;
    std::vector<double> vals;
};

static void skip_comments_and_blank(std::istream& in, std::string& line) {
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (!line.empty() && line[0] == '%') continue;
        return;
    }
    line.clear();
}

static RefCSRMatrix read_matrix_market_as_csr(const char* path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error(std::string("Failed to open matrix file: ") + path);
    }

    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("Empty matrix file");
    }
    if (line.rfind("%%MatrixMarket", 0) != 0) {
        throw std::runtime_error("Unsupported MatrixMarket banner");
    }

    std::istringstream banner(line);
    std::string mm, object, format, field, symmetry;
    banner >> mm >> object >> format >> field >> symmetry;
    if (object != "matrix" || format != "coordinate") {
        throw std::runtime_error("Only coordinate MatrixMarket matrices are supported");
    }

    const bool is_pattern = (field == "pattern");
    const bool is_integer = (field == "integer");
    const bool is_real = (field == "real" || field == "double");
    if (!is_pattern && !is_integer && !is_real) {
        throw std::runtime_error("Unsupported MatrixMarket field type");
    }

    const bool symmetric = (symmetry == "symmetric");

    skip_comments_and_blank(in, line);
    if (line.empty()) {
        throw std::runtime_error("Missing MatrixMarket size line");
    }

    int M = 0, N = 0, nnz = 0;
    {
        std::istringstream sizes(line);
        sizes >> M >> N >> nnz;
        if (M <= 0 || N <= 0 || nnz < 0) {
            throw std::runtime_error("Invalid MatrixMarket size line");
        }
    }

    std::vector<std::vector<std::pair<int, double>>> rows(static_cast<size_t>(M));

    for (int entry_idx = 0; entry_idx < nnz; entry_idx++) {
        if (!std::getline(in, line)) {
            throw std::runtime_error("Unexpected EOF while reading MatrixMarket entries");
        }
        if (line.empty() || line[0] == '%') {
            entry_idx--;
            continue;
        }

        std::istringstream entry(line);
        int row = 0;
        int col = 0;
        double value = 1.0;
        entry >> row >> col;
        if (row <= 0 || col <= 0) {
            throw std::runtime_error("Invalid MatrixMarket entry index");
        }

        if (!is_pattern) {
            if (is_integer) {
                long long int_value = 0;
                entry >> int_value;
                value = static_cast<double>(int_value);
            } else {
                entry >> value;
            }
        }

        row -= 1;
        col -= 1;
        if (row < 0 || row >= M || col < 0 || col >= N) {
            throw std::runtime_error("MatrixMarket entry out of bounds");
        }

        rows[static_cast<size_t>(row)].push_back({col, value});
        if (symmetric && row != col) {
            if (col >= 0 && col < M && row >= 0 && row < N) {
                rows[static_cast<size_t>(col)].push_back({row, value});
            }
        }
    }

    RefCSRMatrix matrix;
    matrix.rows = M;
    matrix.cols = N;
    matrix.rowptr.resize(static_cast<size_t>(M + 1), 0);

    int out_nnz = 0;
    for (int row = 0; row < M; row++) {
        auto& entries = rows[static_cast<size_t>(row)];
        if (entries.empty()) continue;

        std::sort(entries.begin(), entries.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

        int write_idx = 0;
        for (size_t idx = 0; idx < entries.size(); idx++) {
            if (write_idx == 0 || entries[idx].first != entries[static_cast<size_t>(write_idx - 1)].first) {
                entries[static_cast<size_t>(write_idx++)] = entries[idx];
            } else {
                entries[static_cast<size_t>(write_idx - 1)].second += entries[idx].second;
            }
        }
        entries.resize(static_cast<size_t>(write_idx));
        out_nnz += static_cast<int>(entries.size());
    }

    matrix.nnz = out_nnz;
    matrix.colidx.reserve(static_cast<size_t>(out_nnz));
    matrix.vals.reserve(static_cast<size_t>(out_nnz));

    int offset = 0;
    for (int row = 0; row < M; row++) {
        matrix.rowptr[static_cast<size_t>(row)] = offset;
        for (const auto& [col, value] : rows[static_cast<size_t>(row)]) {
            matrix.colidx.push_back(col);
            matrix.vals.push_back(value);
            offset++;
        }
    }
    matrix.rowptr[static_cast<size_t>(M)] = offset;

    return matrix;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx> [iterations] [csr|csc]\n";
        return 1;
    }

    int iterations = 100;
    std::string sparse_format = "csr";

    auto normalize_format = [](std::string value) {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    };
    auto parse_iterations = [](const char* text, int* out) {
        char* end = nullptr;
        long parsed = std::strtol(text, &end, 10);
        if (end == text || (end && *end != '\0')) return false;
        if (parsed <= 0 || parsed > static_cast<long>(std::numeric_limits<int>::max())) return false;
        *out = static_cast<int>(parsed);
        return true;
    };

    bool arg2_is_iterations = false;
    if (argc >= 3) {
        int parsed_iterations = 0;
        if (parse_iterations(argv[2], &parsed_iterations)) {
            iterations = parsed_iterations;
            arg2_is_iterations = true;
        } else {
            sparse_format = normalize_format(argv[2]);
        }
    }
    if (argc >= 4) {
        if (arg2_is_iterations) {
            sparse_format = normalize_format(argv[3]);
        } else {
            int parsed_iterations = 0;
            if (!parse_iterations(argv[3], &parsed_iterations)) {
                std::cerr << "Error: iterations must be > 0\n";
                std::cerr << "Usage: " << argv[0] << " <matrix.mtx> [iterations] [csr|csc]\n";
                return 1;
            }
            iterations = parsed_iterations;
        }
    }

    if (argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx> [iterations] [csr|csc]\n";
        return 1;
    }
    if (iterations <= 0) {
        std::cerr << "Error: iterations must be > 0\n";
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx> [iterations] [csr|csc]\n";
        return 1;
    }
    if (sparse_format != "csr" && sparse_format != "csc") {
        std::cerr << "Error: format must be 'csr' or 'csc'\n";
        std::cerr << "Usage: " << argv[0] << " <matrix.mtx> [iterations] [csr|csc]\n";
        return 1;
    }

    const int N = 32;  // Dense output columns for correctness-check SpMM.
    Format dense({Dense, Dense});
    const Format matrix_format = (sparse_format == "csr") ? CSR : CSC;

    Tensor<double> A = read(argv[1], matrix_format);
    A.pack();
    const int M = A.getDimension(0);
    const int K = A.getDimension(1);

    Tensor<double> B({K, N}, dense);
    Tensor<double> C({M, N}, dense);

    srand(42);
    std::vector<double> B_vals(static_cast<size_t>(K) * static_cast<size_t>(N), 0.0);
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            const size_t idx = static_cast<size_t>(k) * static_cast<size_t>(N) + static_cast<size_t>(n);
            B_vals[idx] = static_cast<double>(rand()) / RAND_MAX;
            B.insert({k, n}, B_vals[idx]);
        }
    }
    B.pack();

    IndexVar i, j, k;
    C(i, j) = A(i, k) * B(k, j);
    C.compile();
    C.assemble();

    // Build reference C = A * B using a MatrixMarket-based CSR representation.
    RefCSRMatrix A_ref = read_matrix_market_as_csr(argv[1]);
    if (A_ref.rows != M || A_ref.cols != K) {
        throw std::runtime_error("MatrixMarket dims mismatch between TACO and reference loader");
    }

    std::vector<double> C_ref(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0);
    for (int row = 0; row < M; row++) {
        const size_t c_row_offset = static_cast<size_t>(row) * static_cast<size_t>(N);
        for (int p = A_ref.rowptr[static_cast<size_t>(row)];
             p < A_ref.rowptr[static_cast<size_t>(row + 1)];
             p++) {
            const int col = A_ref.colidx[static_cast<size_t>(p)];
            const double a_val = A_ref.vals[static_cast<size_t>(p)];
            const size_t b_row_offset = static_cast<size_t>(col) * static_cast<size_t>(N);
            for (int n = 0; n < N; n++) {
                C_ref[c_row_offset + static_cast<size_t>(n)] +=
                    a_val * B_vals[b_row_offset + static_cast<size_t>(n)];
            }
        }
    }

    auto clear_C = [&C, M, N]() {
        auto c_vals = C.getStorage().getValues();
        if (c_vals.getSize() != static_cast<size_t>(M) * static_cast<size_t>(N)) {
            throw std::runtime_error("Unexpected TACO dense output storage size for C");
        }
        double* c_data = static_cast<double*>(c_vals.getData());
        std::fill(c_data, c_data + static_cast<size_t>(M) * static_cast<size_t>(N), 0.0);
    };

    clear_C();
    C.setNeedsCompute(true);
    C.compute();

    std::vector<double> iter_times;
    iter_times.reserve(static_cast<size_t>(iterations));
    for (int iter = 0; iter < iterations; iter++) {
        clear_C();
        C.setNeedsCompute(true);
        auto start = high_resolution_clock::now();
        C.compute();
        auto end = high_resolution_clock::now();
        iter_times.push_back(duration<double, std::milli>(end - start).count());
    }

    double total_ms = 0.0;
    double min_ms = iter_times[0];
    double max_ms = iter_times[0];
    for (double value : iter_times) {
        total_ms += value;
        if (value < min_ms) min_ms = value;
        if (value > max_ms) max_ms = value;
    }
    const double avg_ms = total_ms / iterations;

    double variance = 0.0;
    for (double value : iter_times) {
        const double diff = value - avg_ms;
        variance += diff * diff;
    }
    variance /= iterations;
    const double stddev = std::sqrt(variance);
    const double variance_pct = (avg_ms > 0.0) ? (stddev / avg_ms * 100.0) : 0.0;

    int nnz = 0;
    auto a_vals = A.getStorage().getValues();
    nnz = a_vals.getSize();

    auto c_vals = C.getStorage().getValues();
    if (c_vals.getSize() != static_cast<size_t>(M) * static_cast<size_t>(N)) {
        throw std::runtime_error("Unexpected TACO dense output storage size for C");
    }
    const double* c_data = static_cast<const double*>(c_vals.getData());

    double max_error = 0.0;
    for (size_t idx = 0; idx < C_ref.size(); idx++) {
        double err = c_data[idx] - C_ref[idx];
        if (err < 0) err = -err;
        if (err > max_error) max_error = err;
    }

    std::cout << "Matrix: " << argv[1]
              << " (" << M << " x " << K << ", "
              << nnz << " nnz)" << std::endl;
    std::cout << "Format: " << sparse_format << std::endl;
    std::cout << "Output cols: " << N << std::endl;
    std::cout << "Warmup iterations: 1" << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2)
              << total_ms << " ms" << std::endl;
    std::cout << "Avg time per iteration: " << std::fixed << std::setprecision(4)
              << avg_ms << " ms" << std::endl;
    std::cout << "Min time per iteration: " << std::fixed << std::setprecision(4)
              << min_ms << " ms" << std::endl;
    std::cout << "Max time per iteration: " << std::fixed << std::setprecision(4)
              << max_ms << " ms" << std::endl;
    std::cout << "Stddev time: " << std::fixed << std::setprecision(4)
              << stddev << " ms" << std::endl;
    std::cout << "Variance: " << std::fixed << std::setprecision(2)
              << variance_pct << "%" << std::endl;
    std::cout << "Max error vs reference: " << std::scientific << max_error << std::endl;
    std::cout << "Implementation: TACO" << std::endl;

    return 0;
}
