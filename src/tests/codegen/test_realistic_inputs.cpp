/**
 * Test Suite: Realistic Input Configurations (Phase 4)
 *
 * Verifies that generated main() functions use realistic input patterns:
 * - Two-sparse kernels (SpAdd, SpElMul, SpGEMM) load separate A and B matrices
 * - SDDMM accepts configurable K dimension from CLI
 * - SpMV/SpMM keep single-matrix input
 * - Dimension validation for two-sparse kernels
 */

#include <gtest/gtest.h>
#include <algorithm>
#include "codegen.h"
#include "ast.h"
#include "scheduled_optimizations.h"
#include "semantic_ir.h"

extern std::unique_ptr<SparseTensorCompiler::Program> g_program;
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern int yynerrs;

static bool parserInit = false;

static std::string generateCode(const std::string& code) {
    if (!parserInit) {
        yynerrs = 0;
        g_program.reset();
        yy_scan_string("tensor x : Dense;");
        yyparse();
        yylex_destroy();
        g_program.reset();
        parserInit = true;
    }

    yynerrs = 0;
    g_program.reset();
    yy_scan_string(code.c_str());
    int result = yyparse();
    yylex_destroy();

    if (result != 0 || !g_program) return "";

    auto op = sparseir::lowerFirstComputationToScheduled(*g_program);
    g_program.reset();
    if (!op) return "";

    opt::OptConfig config = opt::OptConfig::baseline();
    opt::applyOptimizations(*op, config);

    return codegen::generateCode(*op, config);
}

// ============================================================================
// Two-Sparse Input Tests (SpAdd, SpElMul, SpGEMM)
// ============================================================================

static std::string spaddCode() {
    return
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = A[i, j] + B[i, j];\n";
}

TEST(RealisticInputs, SpAdd_TwoFileUsage) {
    std::string output = generateCode(spaddCode());
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("<matrixA.mtx> <matrixB.mtx>"), std::string::npos)
        << "Usage should show two matrix file arguments";
}

TEST(RealisticInputs, SpAdd_LoadsBothMatrices) {
    std::string output = generateCode(spaddCode());
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("load_matrix_market(argv[1])"), std::string::npos)
        << "Should load A from argv[1]";
    EXPECT_NE(output.find("load_matrix_market(argv[2])"), std::string::npos)
        << "Should load B from argv[2]";
}

TEST(RealisticInputs, SpAdd_NoBEqualsA) {
    std::string output = generateCode(spaddCode());
    ASSERT_FALSE(output.empty());
    EXPECT_EQ(output.find("SparseMatrix* B = A"), std::string::npos)
        << "Should NOT use B = A self-operation";
}

TEST(RealisticInputs, SpAdd_DimensionValidation) {
    std::string output = generateCode(spaddCode());
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("dimension mismatch"), std::string::npos)
        << "Should validate dimension compatibility";
}

TEST(RealisticInputs, SpAdd_FreeBothMatrices) {
    std::string output = generateCode(spaddCode());
    ASSERT_FALSE(output.empty());
    // Count occurrences of free_sparse
    size_t count = 0;
    size_t pos = 0;
    while ((pos = output.find("free_sparse(", pos)) != std::string::npos) {
        count++;
        pos++;
    }
    // Should free A and B separately (+ error paths)
    EXPECT_GE(count, 2u) << "Should free both A and B separately";
    EXPECT_NE(output.find("free_sparse(A)"), std::string::npos);
    EXPECT_NE(output.find("free_sparse(B)"), std::string::npos);
}

TEST(RealisticInputs, SpAdd_ReportsBothMatrices) {
    std::string output = generateCode(spaddCode());
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("Matrix A:"), std::string::npos)
        << "Should report Matrix A info";
    EXPECT_NE(output.find("Matrix B:"), std::string::npos)
        << "Should report Matrix B info";
}

// SpElMul uses same two-file pattern
TEST(RealisticInputs, SpElMul_TwoFileInput) {
    std::string output = generateCode(
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = A[i, j] * B[i, j];\n");
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("<matrixA.mtx> <matrixB.mtx>"), std::string::npos);
    EXPECT_NE(output.find("load_matrix_market(argv[2])"), std::string::npos);
    EXPECT_EQ(output.find("SparseMatrix* B = A"), std::string::npos);
}

// SpGEMM uses two-file with A->cols != B->rows check
TEST(RealisticInputs, SpGEMM_TwoFileInput) {
    std::string output = generateCode(
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = A[i, k] * B[k, j];\n");
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("<matrixA.mtx> <matrixB.mtx>"), std::string::npos);
    EXPECT_NE(output.find("load_matrix_market(argv[2])"), std::string::npos);
}

TEST(RealisticInputs, SpGEMM_DimensionValidation) {
    std::string output = generateCode(
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = A[i, k] * B[k, j];\n");
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("A->cols != B->rows"), std::string::npos)
        << "SpGEMM should validate A->cols == B->rows";
}

// ============================================================================
// SDDMM Configurable K Tests
// ============================================================================

static std::string sddmmCode() {
    return
        "tensor S : CSR<100, 100>;\n"
        "tensor D : Dense<100, 64>;\n"
        "tensor E : Dense<64, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = S[i, j] * D[i, k] * E[k, j];\n";
}

TEST(RealisticInputs, SDDMM_UsageShowsKDim) {
    std::string output = generateCode(sddmmCode());
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("[K_dim]"), std::string::npos)
        << "SDDMM usage should show [K_dim] argument";
}

TEST(RealisticInputs, SDDMM_KFromCLI) {
    std::string output = generateCode(sddmmCode());
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("N_k = atoi(argv[2])"), std::string::npos)
        << "K should be read from argv[2]";
}

TEST(RealisticInputs, SDDMM_KDefault64) {
    std::string output = generateCode(sddmmCode());
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("int N_k = 64"), std::string::npos)
        << "K should default to 64 (from IR dense tensor dims)";
}

TEST(RealisticInputs, SDDMM_IterationsFromArgv3) {
    std::string output = generateCode(sddmmCode());
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("atoi(argv[3])"), std::string::npos)
        << "Iterations should shift to argv[3] for SDDMM";
}

// ============================================================================
// SpMV/SpMM Unchanged Tests
// ============================================================================

TEST(RealisticInputs, SpMV_SingleMatrixInput) {
    std::string output = generateCode(
        "tensor A : CSR<100, 50>;\n"
        "tensor x : Dense<50>;\n"
        "tensor y : Dense<100>;\n"
        "compute y[i] = A[i, j] * x[j];\n");
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("<matrix.mtx> [iterations]"), std::string::npos)
        << "SpMV should use single-matrix usage pattern";
    EXPECT_EQ(output.find("argv[3]"), std::string::npos)
        << "SpMV should not reference argv[3]";
}

TEST(RealisticInputs, SpMM_SingleMatrixInput) {
    std::string output = generateCode(
        "tensor A : CSR<100, 50>;\n"
        "tensor B : Dense<50, 20>;\n"
        "tensor C : Dense<100, 20>;\n"
        "compute C[i, j] = A[i, k] * B[k, j];\n");
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("<matrix.mtx> [iterations]"), std::string::npos)
        << "SpMM should use single-matrix usage pattern";
}

// ============================================================================
// Balanced Braces (structural correctness)
// ============================================================================

TEST(RealisticInputs, SpAdd_BalancedBraces) {
    std::string output = generateCode(spaddCode());
    ASSERT_FALSE(output.empty());
    int open = std::count(output.begin(), output.end(), '{');
    int close = std::count(output.begin(), output.end(), '}');
    EXPECT_EQ(open, close) << "Braces should be balanced";
}

TEST(RealisticInputs, SpGEMM_BalancedBraces) {
    std::string output = generateCode(
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = A[i, k] * B[k, j];\n");
    ASSERT_FALSE(output.empty());
    int open = std::count(output.begin(), output.end(), '{');
    int close = std::count(output.begin(), output.end(), '}');
    EXPECT_EQ(open, close) << "Braces should be balanced";
}

TEST(RealisticInputs, SDDMM_BalancedBraces) {
    std::string output = generateCode(sddmmCode());
    ASSERT_FALSE(output.empty());
    int open = std::count(output.begin(), output.end(), '{');
    int close = std::count(output.begin(), output.end(), '}');
    EXPECT_EQ(open, close) << "Braces should be balanced";
}

TEST(RealisticInputs, SpMV_BalancedBraces) {
    std::string output = generateCode(
        "tensor A : CSR<100, 50>;\n"
        "tensor x : Dense<50>;\n"
        "tensor y : Dense<100>;\n"
        "compute y[i] = A[i, j] * x[j];\n");
    ASSERT_FALSE(output.empty());
    int open = std::count(output.begin(), output.end(), '{');
    int close = std::count(output.begin(), output.end(), '}');
    EXPECT_EQ(open, close) << "Braces should be balanced";
}
