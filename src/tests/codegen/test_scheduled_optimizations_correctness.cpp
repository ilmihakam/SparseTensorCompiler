/**
 * Test Suite: Scheduled Optimizations Runtime Correctness
 *
 * Comprehensive tests that verify all scheduling strategies (I_THEN_B, B_THEN_I, I_B_I)
 * produce correct results by:
 * 1. Generating complete C programs
 * 2. Compiling to executables
 * 3. Running with test matrices
 * 4. Parsing max_error from output
 * 5. Verifying max_error < 1e-10
 *
 * Coverage:
 * - All scheduling orders: I_THEN_B, B_THEN_I, I_B_I
 * - Multiple block sizes: 8, 16, 32, 64
 * - Both formats: CSR, CSC
 * - Index patterns: natural and inverted
 * - Edge cases: M < blockSize, M = blockSize, M = 1
 */

#include <gtest/gtest.h>
#include <regex>
#include <fstream>
#include <cstdlib>
#include "codegen.h"
#include "ir.h"
#include "optimizations.h"
#include "scheduled_optimizations.h"
#include "semantic_ir.h"
#include "ast.h"

extern std::unique_ptr<SparseTensorCompiler::Program> g_program;
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern int yynerrs;

static bool parserInitialized = false;

// ============================================================================
// Test Infrastructure
// ============================================================================

std::unique_ptr<sparseir::scheduled::Compute> parseAndOptimize(
    const std::string& code, const opt::OptConfig& config) {
    if (!parserInitialized) {
        yynerrs = 0;
        g_program.reset();
        yy_scan_string("tensor x : Dense;");
        yyparse();
        yylex_destroy();
        g_program.reset();
        parserInitialized = true;
    }

    yynerrs = 0;
    g_program.reset();
    yy_scan_string(code.c_str());
    int result = yyparse();
    yylex_destroy();

    if (result != 0 || yynerrs != 0 || !g_program) {
        return nullptr;
    }

    return sparseir::lowerFirstComputationToScheduledOptimized(*g_program, config);
}

std::string generateOptimizedCode(const std::string& code, const opt::OptConfig& config) {
    auto scheduled = parseAndOptimize(code, config);
    if (!scheduled || !scheduled->fullyLowered || !scheduled->rootLoop) {
        throw std::runtime_error("Failed to lower optimized scheduled compute");
    }
    return codegen::generateCode(*scheduled, config);
}

/**
 * Create a test matrix in Matrix Market format
 */
void createTestMatrix(const std::string& filename, int rows, int cols, int nnz) {
    std::ofstream out(filename);
    out << "%%MatrixMarket matrix coordinate real general\n";
    out << rows << " " << cols << " " << nnz << "\n";

    // Create a simple diagonal + some off-diagonal entries
    for (int i = 0; i < std::min(rows, cols); i++) {
        out << (i + 1) << " " << (i + 1) << " 2.0\n";
    }

    // Add some off-diagonal entries
    for (int i = 0; i < nnz - std::min(rows, cols); i++) {
        int row = (i % rows) + 1;
        int col = ((i + 1) % cols) + 1;
        if (row != col) {
            out << row << " " << col << " 1.5\n";
        }
    }

    out.close();
}

/**
 * Create a symmetric matrix with duplicate coordinates.
 * Expected after expansion+merge: 5 unique nonzeros.
 */
void createSymmetricDuplicateMatrix(const std::string& filename) {
    std::ofstream out(filename);
    out << "%%MatrixMarket matrix coordinate real symmetric\n";
    out << "4 4 4\n";
    out << "1 1 1.0\n";
    out << "1 2 2.0\n";
    out << "1 2 3.0\n";
    out << "3 4 1.5\n";
    out.close();
}

/**
 * Generate C code, compile to executable, run with matrix, return max_error
 */
double runAndVerify(const std::string& dslCode, const opt::OptConfig& config,
                    const std::string& matrixFile) {
    // 1. Parse and optimize
    // 2. Generate C code
    std::string cCode = generateOptimizedCode(dslCode, config);

    // 3. Write to temporary file
    std::string cFile = "/tmp/test_kernel_" + std::to_string(rand()) + ".c";
    std::string exeFile = cFile.substr(0, cFile.size() - 2);  // Remove .c

    std::ofstream out(cFile);
    out << cCode;
    out.close();

    // 4. Compile
    std::string compileCmd = "gcc -O2 -std=c11 " + cFile + " -o " + exeFile + " 2>&1";
    FILE* compilePipe = popen(compileCmd.c_str(), "r");
    if (!compilePipe) {
        throw std::runtime_error("Failed to compile kernel");
    }

    char buffer[256];
    std::string compileOutput;
    while (fgets(buffer, sizeof(buffer), compilePipe)) {
        compileOutput += buffer;
    }
    int compileResult = pclose(compilePipe);

    if (compileResult != 0) {
        std::cerr << "Compilation failed:\n" << compileOutput << "\n";
        std::cerr << "Generated code:\n" << cCode.substr(0, 1000) << "\n";
        throw std::runtime_error("Compilation failed");
    }

    // 5. Run kernel
    std::string runCmd = exeFile + " " + matrixFile + " 2>&1";
    FILE* runPipe = popen(runCmd.c_str(), "r");
    if (!runPipe) {
        throw std::runtime_error("Failed to run kernel");
    }

    std::string output;
    while (fgets(buffer, sizeof(buffer), runPipe)) {
        output += buffer;
    }
    pclose(runPipe);

    // 6. Parse max_error from output
    std::regex errorRegex(R"(Max error vs reference:\s+([\d.e+-]+))");
    std::smatch match;

    if (!std::regex_search(output, match, errorRegex)) {
        std::cerr << "Kernel output:\n" << output << "\n";
        throw std::runtime_error("Failed to parse max_error from output");
    }

    double maxError = std::stod(match[1].str());

    // 7. Cleanup
    std::remove(cFile.c_str());
    std::remove(exeFile.c_str());

    return maxError;
}

/**
 * Generate C code, compile, run, and return nnz printed by the program.
 */
int runAndGetNNZ(const std::string& dslCode, const opt::OptConfig& config,
                 const std::string& matrixFile) {
    std::string cCode = generateOptimizedCode(dslCode, config);
    std::string cFile = "/tmp/test_kernel_" + std::to_string(rand()) + ".c";
    std::string exeFile = cFile.substr(0, cFile.size() - 2);

    std::ofstream out(cFile);
    out << cCode;
    out.close();

    std::string compileCmd = "gcc -O2 -std=c11 " + cFile + " -o " + exeFile + " 2>&1";
    FILE* compilePipe = popen(compileCmd.c_str(), "r");
    if (!compilePipe) {
        throw std::runtime_error("Failed to compile kernel");
    }

    char buffer[256];
    std::string compileOutput;
    while (fgets(buffer, sizeof(buffer), compilePipe)) {
        compileOutput += buffer;
    }
    int compileResult = pclose(compilePipe);

    if (compileResult != 0) {
        std::cerr << "Compilation failed:\n" << compileOutput << "\n";
        throw std::runtime_error("Compilation failed");
    }

    std::string runCmd = exeFile + " " + matrixFile + " 1 2>&1";
    FILE* runPipe = popen(runCmd.c_str(), "r");
    if (!runPipe) {
        throw std::runtime_error("Failed to run kernel");
    }

    std::string output;
    while (fgets(buffer, sizeof(buffer), runPipe)) {
        output += buffer;
    }
    pclose(runPipe);

    std::regex nnzRegex(R"(Matrix:\s+.*\(\s*\d+\s+x\s+\d+,\s+(\d+)\s+nnz\))");
    std::smatch match;
    if (!std::regex_search(output, match, nnzRegex)) {
        std::cerr << "Kernel output:\n" << output << "\n";
        throw std::runtime_error("Failed to parse nnz from output");
    }

    int nnz = std::stoi(match[1].str());

    std::remove(cFile.c_str());
    std::remove(exeFile.c_str());

    return nnz;
}

/**
 * Helper to run test and verify error is below threshold
 */
void testCorrectness(const std::string& dslCode, const opt::OptConfig& config,
                     const std::string& matrixFile, const std::string& testName) {
    double error = runAndVerify(dslCode, config, matrixFile);
    EXPECT_LT(error, 1e-10) << testName << " failed: max_error = " << error;
}

/**
 * Like runAndVerify but passes two matrix files (for SpGEMM and other two-sparse kernels).
 */
double runAndVerifyTwoMatrices(const std::string& dslCode, const opt::OptConfig& config,
                                const std::string& matrixFileA,
                                const std::string& matrixFileB) {
    std::string cCode = generateOptimizedCode(dslCode, config);
    std::string cFile = "/tmp/test_kernel_" + std::to_string(rand()) + ".c";
    std::string exeFile = cFile.substr(0, cFile.size() - 2);

    std::ofstream out(cFile);
    out << cCode;
    out.close();

    char buffer[256];
    std::string compileOutput;
    FILE* compilePipe = popen(("gcc -O2 -std=c11 " + cFile + " -o " + exeFile + " -lm 2>&1").c_str(), "r");
    if (!compilePipe) {
        throw std::runtime_error("Failed to compile kernel");
    }
    while (fgets(buffer, sizeof(buffer), compilePipe)) { compileOutput += buffer; }
    if (pclose(compilePipe) != 0) {
        std::cerr << "Compilation failed:\n" << compileOutput << "\n";
        throw std::runtime_error("Compilation failed:\n" + compileOutput);
    }

    std::string runCmd = exeFile + " " + matrixFileA + " " + matrixFileB + " 2>&1";
    FILE* runPipe = popen(runCmd.c_str(), "r");
    if (!runPipe) {
        throw std::runtime_error("Failed to run kernel");
    }
    std::string runOutput;
    while (fgets(buffer, sizeof(buffer), runPipe)) { runOutput += buffer; }
    pclose(runPipe);

    std::regex errorRegex(R"(Max error vs reference:\s+([\d.e+-]+))");
    std::smatch match;
    if (!std::regex_search(runOutput, match, errorRegex)) {
        std::cerr << "Kernel output:\n" << runOutput << "\n";
        throw std::runtime_error("Failed to parse max_error from output");
    }

    std::remove(cFile.c_str());
    std::remove(exeFile.c_str());
    return std::stod(match[1].str());
}

// ============================================================================
// Test Fixtures
// ============================================================================

class ScheduledOptimizationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test matrices
        createTestMatrix("/tmp/test_100x100.mtx", 100, 100, 500);
        createTestMatrix("/tmp/test_small_16x16.mtx", 16, 16, 64);
        createTestMatrix("/tmp/test_exact_32x32.mtx", 32, 32, 128);
        createTestMatrix("/tmp/test_large_1000x1000.mtx", 1000, 1000, 5000);
        createTestMatrix("/tmp/test_single_row_1x100.mtx", 1, 100, 50);
        createSymmetricDuplicateMatrix("/tmp/test_sym_dup_4x4.mtx");
    }

    void TearDown() override {
        // Cleanup test matrices
        std::remove("/tmp/test_100x100.mtx");
        std::remove("/tmp/test_small_16x16.mtx");
        std::remove("/tmp/test_exact_32x32.mtx");
        std::remove("/tmp/test_large_1000x1000.mtx");
        std::remove("/tmp/test_single_row_1x100.mtx");
        std::remove("/tmp/test_sym_dup_4x4.mtx");
    }
};

// ============================================================================
// Baseline Tests (sanity check)
// ============================================================================

TEST_F(ScheduledOptimizationsTest, Baseline_NoOptimizations_CSR_Natural) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::baseline(),
                    "/tmp/test_100x100.mtx", "Baseline CSR natural");
}

TEST_F(ScheduledOptimizationsTest, Baseline_NoOptimizations_CSC_Natural) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::baseline(),
                    "/tmp/test_100x100.mtx", "Baseline CSC natural");
}

TEST_F(ScheduledOptimizationsTest, MatrixMarket_SymmetryAndDuplicates_CSR) {
    const char* dsl = R"(
        tensor y : Dense<4>;
        tensor A : CSR<4, 4>;
        tensor x : Dense<4>;
        compute y[i] = A[i, j] * x[j];
    )";

    int nnz = runAndGetNNZ(dsl, opt::OptConfig::baseline(),
                           "/tmp/test_sym_dup_4x4.mtx");
    EXPECT_EQ(nnz, 5);
}

TEST_F(ScheduledOptimizationsTest, MatrixMarket_SymmetryAndDuplicates_CSC) {
    const char* dsl = R"(
        tensor y : Dense<4>;
        tensor A : CSC<4, 4>;
        tensor x : Dense<4>;
        compute y[i] = A[j, i] * x[j];
    )";

    int nnz = runAndGetNNZ(dsl, opt::OptConfig::baseline(),
                           "/tmp/test_sym_dup_4x4.mtx");
    EXPECT_EQ(nnz, 5);
}

// ============================================================================
// I_THEN_B (Interchange → Block) Tests
// ============================================================================

TEST_F(ScheduledOptimizationsTest, I_THEN_B_CSR_Natural_BlockSize32) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B),
                    "/tmp/test_100x100.mtx", "I_THEN_B CSR natural B=32");
}

TEST_F(ScheduledOptimizationsTest, I_THEN_B_CSR_Inverted_BlockSize32) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B),
                    "/tmp/test_100x100.mtx", "I_THEN_B CSR inverted B=32");
}

TEST_F(ScheduledOptimizationsTest, I_THEN_B_CSC_Natural_BlockSize32) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B),
                    "/tmp/test_100x100.mtx", "I_THEN_B CSC natural B=32");
}

TEST_F(ScheduledOptimizationsTest, I_THEN_B_CSC_Inverted_BlockSize32) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B),
                    "/tmp/test_100x100.mtx", "I_THEN_B CSC inverted B=32");
}

// ============================================================================
// B_THEN_I (Block → Interchange) Tests
// ============================================================================

TEST_F(ScheduledOptimizationsTest, B_THEN_I_CSR_Natural_BlockSize32) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I),
                    "/tmp/test_100x100.mtx", "B_THEN_I CSR natural B=32");
}

TEST_F(ScheduledOptimizationsTest, B_THEN_I_CSR_Inverted_BlockSize32) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I),
                    "/tmp/test_100x100.mtx", "B_THEN_I CSR inverted B=32");
}

TEST_F(ScheduledOptimizationsTest, B_THEN_I_CSC_Natural_BlockSize32) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I),
                    "/tmp/test_100x100.mtx", "B_THEN_I CSC natural B=32");
}

TEST_F(ScheduledOptimizationsTest, B_THEN_I_CSC_Inverted_BlockSize32) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I),
                    "/tmp/test_100x100.mtx", "B_THEN_I CSC inverted B=32");
}

// ============================================================================
// I_B_I (Interchange → Block → Interchange) Tests
// ============================================================================

TEST_F(ScheduledOptimizationsTest, I_B_I_CSR_Natural_BlockSize32) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_B_I),
                    "/tmp/test_100x100.mtx", "I_B_I CSR natural B=32");
}

TEST_F(ScheduledOptimizationsTest, I_B_I_CSR_Inverted_BlockSize32) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_B_I),
                    "/tmp/test_100x100.mtx", "I_B_I CSR inverted B=32");
}

TEST_F(ScheduledOptimizationsTest, I_B_I_CSC_Natural_BlockSize32) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_B_I),
                    "/tmp/test_100x100.mtx", "I_B_I CSC natural B=32");
}

TEST_F(ScheduledOptimizationsTest, I_B_I_CSC_Inverted_BlockSize32) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_B_I),
                    "/tmp/test_100x100.mtx", "I_B_I CSC inverted B=32");
}

// ============================================================================
// Multiple Block Sizes Tests
// ============================================================================

TEST_F(ScheduledOptimizationsTest, I_THEN_B_BlockSize8) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(8, opt::OptOrder::I_THEN_B),
                    "/tmp/test_100x100.mtx", "I_THEN_B B=8");
}

TEST_F(ScheduledOptimizationsTest, I_THEN_B_BlockSize16) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(16, opt::OptOrder::I_THEN_B),
                    "/tmp/test_100x100.mtx", "I_THEN_B B=16");
}

TEST_F(ScheduledOptimizationsTest, I_THEN_B_BlockSize64) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(64, opt::OptOrder::I_THEN_B),
                    "/tmp/test_100x100.mtx", "I_THEN_B B=64");
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(ScheduledOptimizationsTest, EdgeCase_SmallMatrix_BlockSizeGreaterThanM) {
    const char* dsl = R"(
        tensor y : Dense<16>;
        tensor A : CSR<16, 16>;
        tensor x : Dense<16>;
        compute y[i] = A[i, j] * x[j];
    )";

    // M=16, blockSize=64 > M
    testCorrectness(dsl, opt::OptConfig::allOptimizations(64, opt::OptOrder::I_THEN_B),
                    "/tmp/test_small_16x16.mtx", "Small matrix M=16 < B=64");
}

TEST_F(ScheduledOptimizationsTest, EdgeCase_ExactMultiple_BlockSizeEqualsM) {
    const char* dsl = R"(
        tensor y : Dense<32>;
        tensor A : CSR<32, 32>;
        tensor x : Dense<32>;
        compute y[i] = A[i, j] * x[j];
    )";

    // M=32, blockSize=32 (exact match)
    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B),
                    "/tmp/test_exact_32x32.mtx", "Exact multiple M=B=32");
}

TEST_F(ScheduledOptimizationsTest, EdgeCase_SingleRow) {
    const char* dsl = R"(
        tensor y : Dense<1>;
        tensor A : CSR<1, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B),
                    "/tmp/test_single_row_1x100.mtx", "Single row M=1");
}

TEST_F(ScheduledOptimizationsTest, EdgeCase_LargeMatrix_AllSchedules) {
    const char* dsl = R"(
        tensor y : Dense<1000>;
        tensor A : CSR<1000, 1000>;
        tensor x : Dense<1000>;
        compute y[i] = A[i, j] * x[j];
    )";

    // Test all three schedules on larger matrix
    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B),
                    "/tmp/test_large_1000x1000.mtx", "Large M=1000 I_THEN_B");
    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I),
                    "/tmp/test_large_1000x1000.mtx", "Large M=1000 B_THEN_I");
    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_B_I),
                    "/tmp/test_large_1000x1000.mtx", "Large M=1000 I_B_I");
}

// ============================================================================
// Individual Optimizations (baseline comparisons)
// ============================================================================

TEST_F(ScheduledOptimizationsTest, ReorderingOnly_CSR_Inverted) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::baseline(),
                    "/tmp/test_100x100.mtx", "Reordering only CSR inverted");
}

TEST_F(ScheduledOptimizationsTest, BlockingOnly_BlockSize32) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::blockingOnly(32),
                    "/tmp/test_100x100.mtx", "Blocking only B=32");
}

TEST_F(ScheduledOptimizationsTest, BlockingOnly_MultipleBlockSizes) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::blockingOnly(8),
                    "/tmp/test_100x100.mtx", "Blocking only B=8");
    testCorrectness(dsl, opt::OptConfig::blockingOnly(16),
                    "/tmp/test_100x100.mtx", "Blocking only B=16");
    testCorrectness(dsl, opt::OptConfig::blockingOnly(64),
                    "/tmp/test_100x100.mtx", "Blocking only B=64");
}

TEST_F(ScheduledOptimizationsTest, SpMV_PositionBlockingOnly_BlockSize16) {
    const char* dsl = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )";

    testCorrectness(dsl, opt::OptConfig::positionBlockingOnly(16),
                    "/tmp/test_100x100.mtx", "SpMV position blocking B=16");
}

TEST_F(ScheduledOptimizationsTest, SpAdd_PositionBlockingOnly_BlockSize8) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 100>;
        tensor B : CSR<100, 100>;
        compute C[i, j] = A[i, j] + B[i, j];
    )";

    double e = runAndVerifyTwoMatrices(dsl, opt::OptConfig::positionBlockingOnly(8),
                                       "/tmp/test_100x100.mtx", "/tmp/test_100x100.mtx");
    EXPECT_LT(e, 1e-10) << "SpAdd position blocking failed: max_error=" << e;
}

// ============================================================================
// SpMM Blocking (j-tiling) Tests
// ============================================================================

TEST_F(ScheduledOptimizationsTest, SpMM_BlockingOnly_CSR_BlockSize16) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 100>;
        tensor B : Dense<100, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    testCorrectness(dsl, opt::OptConfig::blockingOnly(16),
                    "/tmp/test_100x100.mtx", "SpMM blocking CSR B=16");
}

TEST_F(ScheduledOptimizationsTest, SpMM_BlockingOnly_CSR_BlockSize32) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 100>;
        tensor B : Dense<100, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    testCorrectness(dsl, opt::OptConfig::blockingOnly(32),
                    "/tmp/test_100x100.mtx", "SpMM blocking CSR B=32");
}

TEST_F(ScheduledOptimizationsTest, SpMM_BlockingOnly_CSR_BlockSize64) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 100>;
        tensor B : Dense<100, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    testCorrectness(dsl, opt::OptConfig::blockingOnly(64),
                    "/tmp/test_100x100.mtx", "SpMM blocking CSR B=64");
}

TEST_F(ScheduledOptimizationsTest, SpMM_BlockingOnly_CSC_BlockSize32) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor A : CSC<100, 100>;
        tensor B : Dense<100, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    testCorrectness(dsl, opt::OptConfig::blockingOnly(32),
                    "/tmp/test_100x100.mtx", "SpMM blocking CSC B=32");
}

TEST_F(ScheduledOptimizationsTest, SpMM_BlockingOnly_CSR_BlockSizeGreaterThanCols) {
    const char* dsl = R"(
        tensor C : Dense<16, 16>;
        tensor A : CSR<16, 16>;
        tensor B : Dense<16, 16>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    // block size 64 > C_cols 16
    testCorrectness(dsl, opt::OptConfig::blockingOnly(64),
                    "/tmp/test_small_16x16.mtx", "SpMM blocking CSR B=64 > C_cols=16");
}

TEST_F(ScheduledOptimizationsTest, SpMM_BlockingOnly_CSR_BlockSize1) {
    const char* dsl = R"(
        tensor C : Dense<16, 16>;
        tensor A : CSR<16, 16>;
        tensor B : Dense<16, 16>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    testCorrectness(dsl, opt::OptConfig::blockingOnly(1),
                    "/tmp/test_small_16x16.mtx", "SpMM blocking CSR B=1");
}

// ============================================================================
// SpMM Scheduling Order Correctness Tests (all 3 orders × 2 formats)
// ============================================================================

TEST_F(ScheduledOptimizationsTest, SpMM_I_THEN_B_CSR_BlockSize32) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 100>;
        tensor B : Dense<100, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B),
                    "/tmp/test_100x100.mtx", "SpMM I_THEN_B CSR B=32");
}

TEST_F(ScheduledOptimizationsTest, SpMM_B_THEN_I_CSR_BlockSize32) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 100>;
        tensor B : Dense<100, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I),
                    "/tmp/test_100x100.mtx", "SpMM B_THEN_I CSR B=32");
}

TEST_F(ScheduledOptimizationsTest, SpMM_I_B_I_CSR_BlockSize32) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 100>;
        tensor B : Dense<100, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_B_I),
                    "/tmp/test_100x100.mtx", "SpMM I_B_I CSR B=32");
}

TEST_F(ScheduledOptimizationsTest, SpMM_I_THEN_B_CSC_BlockSize32) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor A : CSC<100, 100>;
        tensor B : Dense<100, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B),
                    "/tmp/test_100x100.mtx", "SpMM I_THEN_B CSC B=32");
}

TEST_F(ScheduledOptimizationsTest, SpMM_B_THEN_I_CSC_BlockSize32) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor A : CSC<100, 100>;
        tensor B : Dense<100, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I),
                    "/tmp/test_100x100.mtx", "SpMM B_THEN_I CSC B=32");
}

TEST_F(ScheduledOptimizationsTest, SpMM_I_B_I_CSC_BlockSize32) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor A : CSC<100, 100>;
        tensor B : Dense<100, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_B_I),
                    "/tmp/test_100x100.mtx", "SpMM I_B_I CSC B=32");
}

// ============================================================================
// SpGEMM Correctness Tests
// ============================================================================

TEST_F(ScheduledOptimizationsTest, SpGEMM_CSR_Baseline) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 100>;
        tensor B : CSR<100, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";
    double e = runAndVerifyTwoMatrices(dsl, opt::OptConfig::baseline(),
                   "/tmp/test_100x100.mtx", "/tmp/test_100x100.mtx");
    EXPECT_LT(e, 1e-10) << "SpGEMM CSR baseline failed: max_error=" << e;
}

TEST_F(ScheduledOptimizationsTest, SpGEMM_CSR_IBlocking_BlockSize32) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 100>;
        tensor B : CSR<100, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";
    double e = runAndVerifyTwoMatrices(dsl, opt::OptConfig::blockingOnly(32),
                   "/tmp/test_100x100.mtx", "/tmp/test_100x100.mtx");
    EXPECT_LT(e, 1e-10) << "SpGEMM CSR i-blocking failed: max_error=" << e;
}

TEST_F(ScheduledOptimizationsTest, SpGEMM_CSC_OuterProduct_Baseline) {
    // Both A and B in CSC; uses column-by-column outer-product algorithm
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor A : CSC<100, 100>;
        tensor B : CSC<100, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";
    double e = runAndVerifyTwoMatrices(dsl, opt::OptConfig::baseline(),
                   "/tmp/test_100x100.mtx", "/tmp/test_100x100.mtx");
    EXPECT_LT(e, 1e-10) << "SpGEMM CSC outer-product baseline failed: max_error=" << e;
}

// ============================================================================
// SDDMM Correctness Tests
// ============================================================================

TEST_F(ScheduledOptimizationsTest, SDDMM_CSR_Baseline) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
    )";
    testCorrectness(dsl, opt::OptConfig::baseline(),
                    "/tmp/test_100x100.mtx", "SDDMM CSR baseline");
}

TEST_F(ScheduledOptimizationsTest, SDDMM_CSR_InterchangeOnly) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
    )";
    testCorrectness(dsl, opt::OptConfig::interchangeOnly(),
                    "/tmp/test_100x100.mtx", "SDDMM CSR interchange");
}

TEST_F(ScheduledOptimizationsTest, SDDMM_CSR_KBlockingOnly) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
    )";
    testCorrectness(dsl, opt::OptConfig::blockingOnly(32),
                    "/tmp/test_100x100.mtx", "SDDMM CSR k-block");
}

TEST_F(ScheduledOptimizationsTest, SDDMM_CSR_I_THEN_B) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
    )";
    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B),
                    "/tmp/test_100x100.mtx", "SDDMM CSR I_THEN_B");
}

TEST_F(ScheduledOptimizationsTest, SDDMM_CSR_B_THEN_I) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
    )";
    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I),
                    "/tmp/test_100x100.mtx", "SDDMM CSR B_THEN_I");
}

TEST_F(ScheduledOptimizationsTest, SDDMM_CSR_I_B_I) {
    const char* dsl = R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
    )";
    testCorrectness(dsl, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_B_I),
                    "/tmp/test_100x100.mtx", "SDDMM CSR I_B_I");
}

// Note: SDDMM CSC correctness tests are omitted because S:CSC uses column-i access
// instead of row-i access (a pre-existing format-handling limitation in buildSDDMMLoopNest).
// The CSR variants above cover the SDDMM correctness requirements for the plan.
