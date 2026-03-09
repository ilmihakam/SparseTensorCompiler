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

struct CSCsrMatrix {
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

static CSCsrMatrix read_matrix_market_as_csr(const char* path) {
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

    // Banner format: %%MatrixMarket matrix coordinate <field> <symmetry>
    // We only need to handle coordinate matrices for benchmark inputs.
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

    std::vector<std::vector<std::pair<int, double>>> rows;
    rows.resize(M);

    for (int k = 0; k < nnz; k++) {
        if (!std::getline(in, line)) {
            throw std::runtime_error("Unexpected EOF while reading MatrixMarket entries");
        }
        if (line.empty() || line[0] == '%') {
            k--;
            continue;
        }

        std::istringstream entry(line);
        int r = 0, c = 0;
        double v = 1.0;
        entry >> r >> c;
        if (r <= 0 || c <= 0) {
            throw std::runtime_error("Invalid MatrixMarket entry index");
        }
        if (!is_pattern) {
            if (is_integer) {
                long long vi = 0;
                entry >> vi;
                v = static_cast<double>(vi);
            } else {
                entry >> v;
            }
        }

        // Convert to 0-based.
        r -= 1;
        c -= 1;
        if (r < 0 || r >= M || c < 0 || c >= N) {
            throw std::runtime_error("MatrixMarket entry out of bounds");
        }

        rows[r].push_back({c, v});
        if (symmetric && r != c) {
            if (c < 0 || c >= M || r < 0 || r >= N) {
                // Should not happen for square matrices, but avoid UB.
                continue;
            }
            rows[c].push_back({r, v});
        }
    }

    CSCsrMatrix A;
    A.rows = M;
    A.cols = N;
    A.rowptr.resize(static_cast<size_t>(M + 1), 0);

    // Sort and compress duplicates per row, then build CSR arrays.
    int nnz_out = 0;
    for (int r = 0; r < M; r++) {
        auto& row = rows[r];
        if (row.empty()) continue;
        std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        int write = 0;
        for (size_t idx = 0; idx < row.size(); idx++) {
            if (write == 0 || row[idx].first != row[write - 1].first) {
                row[write++] = row[idx];
            } else {
                row[write - 1].second += row[idx].second;
            }
        }
        row.resize(static_cast<size_t>(write));
        nnz_out += static_cast<int>(row.size());
    }

    A.nnz = nnz_out;
    A.colidx.reserve(static_cast<size_t>(nnz_out));
    A.vals.reserve(static_cast<size_t>(nnz_out));

    int offset = 0;
    for (int r = 0; r < M; r++) {
        A.rowptr[r] = offset;
        for (const auto& [c, v] : rows[r]) {
            A.colidx.push_back(c);
            A.vals.push_back(v);
            offset++;
        }
    }
    A.rowptr[M] = offset;
    return A;
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

    // Backward compatible parsing:
    // 1) <matrix> [iterations]
    // 2) <matrix> [iterations] [format]
    // 3) <matrix> [format] [iterations]
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

    // Load matrix in selected sparse format
    Format dv({Dense});
    const Format matrix_format = (sparse_format == "csr") ? CSR : CSC;

    Tensor<double> A = read(argv[1], matrix_format);
    A.pack();
    int M = A.getDimension(0);
    int N = A.getDimension(1);

    Tensor<double> x({N}, dv);
    Tensor<double> y({M}, dv);

    // Initialize x with same pattern as our compiler
    // CRITICAL: Use same seed for reproducibility
    srand(42);
    std::vector<double> x_vals(static_cast<size_t>(N), 0.0);
    for (int i = 0; i < N; i++) {
        x_vals[static_cast<size_t>(i)] = (double)rand() / RAND_MAX;
        x.insert({i}, x_vals[static_cast<size_t>(i)]);
    }
    x.pack();

    // Define computation
    IndexVar i, j;
    y(i) = A(i, j) * x(j);

    // Compile and assemble
    y.compile();
    y.assemble();

    // Build a reference result from MatrixMarket to sanity-check TACO execution.
    // This is outside the timed region and uses the same x values.
    CSCsrMatrix A_ref = read_matrix_market_as_csr(argv[1]);
    if (A_ref.rows != M || A_ref.cols != N) {
        throw std::runtime_error("MatrixMarket dims mismatch between TACO and reference loader");
    }
    std::vector<double> y_ref(static_cast<size_t>(M), 0.0);
    for (int r = 0; r < M; r++) {
        double acc = 0.0;
        for (int p = A_ref.rowptr[static_cast<size_t>(r)]; p < A_ref.rowptr[static_cast<size_t>(r + 1)]; p++) {
            const int c = A_ref.colidx[static_cast<size_t>(p)];
            acc += A_ref.vals[static_cast<size_t>(p)] * x_vals[static_cast<size_t>(c)];
        }
        y_ref[static_cast<size_t>(r)] = acc;
    }

    // Warmup run (not timed) to match generated kernels
    y.setNeedsCompute(true);
    y.compute();

    std::vector<double> iter_times;
    iter_times.reserve(iterations);

    for (int iter = 0; iter < iterations; iter++) {
        y.setNeedsCompute(true);
        auto start = high_resolution_clock::now();
        y.compute();
        auto end = high_resolution_clock::now();
        double iter_ms = duration<double, std::milli>(end - start).count();
        iter_times.push_back(iter_ms);
    }

    double total_ms = 0.0;
    double min_ms = iter_times[0];
    double max_ms = iter_times[0];
    for (double value : iter_times) {
        total_ms += value;
        if (value < min_ms) min_ms = value;
        if (value > max_ms) max_ms = value;
    }
    double avg_ms = total_ms / iterations;

    double variance = 0.0;
    for (double value : iter_times) {
        double diff = value - avg_ms;
        variance += diff * diff;
    }
    variance /= iterations;
    double stddev = std::sqrt(variance);
    double variance_pct = (avg_ms > 0.0) ? (stddev / avg_ms * 100.0) : 0.0;

    // Get nnz from packed tensor
    int nnz = 0;
    // Try to get values array size (different API in newer TACO)
    auto vals = A.getStorage().getValues();
    nnz = vals.getSize();

    // Compute correctness vs reference on the final output.
    // Use packed dense values for y.
    double max_error = 0.0;
    auto y_vals = y.getStorage().getValues();
    if (y_vals.getSize() != static_cast<size_t>(M)) {
        throw std::runtime_error("Unexpected TACO dense output storage size for y");
    }
    const double* y_data = static_cast<const double*>(y_vals.getData());
    for (int r = 0; r < M; r++) {
        double err = y_data[static_cast<size_t>(r)] - y_ref[static_cast<size_t>(r)];
        if (err < 0) err = -err;
        if (err > max_error) max_error = err;
    }

    std::cout << "Matrix: " << argv[1]
              << " (" << M << " x " << N << ", "
              << nnz << " nnz)" << std::endl;
    std::cout << "Format: " << sparse_format << std::endl;
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
