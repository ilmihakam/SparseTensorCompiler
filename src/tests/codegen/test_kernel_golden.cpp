/**
 * Test Suite: Golden Artifact Tests
 *
 * Compares generateKernelCode() output against .kernel.c.txt golden files
 * for all 16 (kernel × format × config) combinations.
 */

#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>

#include "codegen.h"
#include "optimizations.h"
#include "scheduled_optimizations.h"
#include "semantic_ir.h"
#include "ast.h"

extern std::unique_ptr<SparseTensorCompiler::Program> g_program;
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern int yynerrs;

// ============================================================================
// Helpers
// ============================================================================

namespace {

bool parserInitialized = false;

std::unique_ptr<sparseir::scheduled::Compute> parseForGolden(
    const std::string& code) {
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
    return sparseir::lowerFirstComputationToScheduled(*g_program);
}

std::string readGoldenFile(const std::string& relativePath) {
    std::string fullPath = std::string(SOURCE_DIR) + "/" + relativePath;
    std::ifstream file(fullPath);
    if (!file.is_open()) {
        return "ERROR: Cannot open golden file: " + fullPath;
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// DSL inputs
const char* SPMV_CSR =
    "tensor y : Dense<100>; tensor A : CSR<100, 100>; tensor x : Dense<100>; "
    "compute y[i] = A[i, j] * x[j];";
const char* SPMV_CSC =
    "tensor y : Dense<100>; tensor A : CSC<100, 100>; tensor x : Dense<100>; "
    "compute y[i] = A[i, j] * x[j];";
const char* SPMM_CSR =
    "tensor C : Dense<100, 50>; tensor A : CSR<100, 80>; tensor B : Dense<80, 50>; "
    "compute C[i, j] = A[i, k] * B[k, j];";
const char* SPMM_CSC =
    "tensor C : Dense<100, 50>; tensor A : CSC<100, 80>; tensor B : Dense<80, 50>; "
    "compute C[i, j] = A[i, k] * B[k, j];";

std::string generateKernel(const char* dsl, opt::OptConfig config) {
    auto compute = parseForGolden(dsl);
    if (!compute) return "ERROR: parse failed";
    opt::applyOptimizations(*compute, config);
    return codegen::generateKernelCode(*compute, config);
}

} // anonymous namespace

// ============================================================================
// Baseline tests (4)
// ============================================================================

TEST(KernelGoldenTest, SpMV_CSR_Baseline) {
    std::string actual = generateKernel(SPMV_CSR, opt::OptConfig::baseline());
    std::string expected = readGoldenFile("artifacts/baseline/spmv_csr_baseline.kernel.c.txt");
    EXPECT_EQ(actual, expected);
}

TEST(KernelGoldenTest, SpMV_CSC_Baseline) {
    std::string actual = generateKernel(SPMV_CSC, opt::OptConfig::baseline());
    std::string expected = readGoldenFile("artifacts/baseline/spmv_csc_baseline.kernel.c.txt");
    EXPECT_EQ(actual, expected);
}

TEST(KernelGoldenTest, SpMM_CSR_Baseline) {
    std::string actual = generateKernel(SPMM_CSR, opt::OptConfig::baseline());
    std::string expected = readGoldenFile("artifacts/baseline/spmm_csr_baseline.kernel.c.txt");
    EXPECT_EQ(actual, expected);
}

TEST(KernelGoldenTest, SpMM_CSC_Baseline) {
    std::string actual = generateKernel(SPMM_CSC, opt::OptConfig::baseline());
    std::string expected = readGoldenFile("artifacts/baseline/spmm_csc_baseline.kernel.c.txt");
    EXPECT_EQ(actual, expected);
}

// ============================================================================
// Interchange tests (2 — SpMM only)
// ============================================================================

TEST(KernelGoldenTest, SpMM_CSR_Interchange) {
    std::string actual = generateKernel(SPMM_CSR, opt::OptConfig::interchangeOnly());
    std::string expected = readGoldenFile("artifacts/interchange/spmm_csr_interchange.kernel.c.txt");
    EXPECT_EQ(actual, expected);
}

TEST(KernelGoldenTest, SpMM_CSC_Interchange) {
    std::string actual = generateKernel(SPMM_CSC, opt::OptConfig::interchangeOnly());
    std::string expected = readGoldenFile("artifacts/interchange/spmm_csc_interchange.kernel.c.txt");
    EXPECT_EQ(actual, expected);
}

// ============================================================================
// Block-only tests (4)
// ============================================================================

TEST(KernelGoldenTest, SpMV_CSR_BlockOnly) {
    std::string actual = generateKernel(SPMV_CSR, opt::OptConfig::blockingOnly(32));
    std::string expected = readGoldenFile("artifacts/block_only/spmv_csr_block_only.kernel.c.txt");
    EXPECT_EQ(actual, expected);
}

TEST(KernelGoldenTest, SpMV_CSC_BlockOnly) {
    std::string actual = generateKernel(SPMV_CSC, opt::OptConfig::blockingOnly(32));
    std::string expected = readGoldenFile("artifacts/block_only/spmv_csc_block_only.kernel.c.txt");
    EXPECT_EQ(actual, expected);
}

TEST(KernelGoldenTest, SpMM_CSR_BlockOnly) {
    std::string actual = generateKernel(SPMM_CSR, opt::OptConfig::blockingOnly(32));
    std::string expected = readGoldenFile("artifacts/block_only/spmm_csr_block_only.kernel.c.txt");
    EXPECT_EQ(actual, expected);
}

TEST(KernelGoldenTest, SpMM_CSC_BlockOnly) {
    std::string actual = generateKernel(SPMM_CSC, opt::OptConfig::blockingOnly(32));
    std::string expected = readGoldenFile("artifacts/block_only/spmm_csc_block_only.kernel.c.txt");
    EXPECT_EQ(actual, expected);
}

// ============================================================================
// Schedule tests (6 — SpMM only)
// ============================================================================

TEST(KernelGoldenTest, SpMM_CSR_IThenB) {
    std::string actual = generateKernel(SPMM_CSR,
        opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B));
    std::string expected = readGoldenFile("artifacts/schedules/spmm_csr_i_then_b.kernel.c.txt");
    EXPECT_EQ(actual, expected);
}

TEST(KernelGoldenTest, SpMM_CSC_IThenB) {
    std::string actual = generateKernel(SPMM_CSC,
        opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B));
    std::string expected = readGoldenFile("artifacts/schedules/spmm_csc_i_then_b.kernel.c.txt");
    EXPECT_EQ(actual, expected);
}

TEST(KernelGoldenTest, SpMM_CSR_BThenI) {
    std::string actual = generateKernel(SPMM_CSR,
        opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I));
    std::string expected = readGoldenFile("artifacts/schedules/spmm_csr_b_then_i.kernel.c.txt");
    EXPECT_EQ(actual, expected);
}

TEST(KernelGoldenTest, SpMM_CSC_BThenI) {
    std::string actual = generateKernel(SPMM_CSC,
        opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I));
    std::string expected = readGoldenFile("artifacts/schedules/spmm_csc_b_then_i.kernel.c.txt");
    EXPECT_EQ(actual, expected);
}

TEST(KernelGoldenTest, SpMM_CSR_IBI) {
    std::string actual = generateKernel(SPMM_CSR,
        opt::OptConfig::allOptimizations(32, opt::OptOrder::I_B_I));
    std::string expected = readGoldenFile("artifacts/schedules/spmm_csr_i_b_i.kernel.c.txt");
    EXPECT_EQ(actual, expected);
}

TEST(KernelGoldenTest, SpMM_CSC_IBI) {
    std::string actual = generateKernel(SPMM_CSC,
        opt::OptConfig::allOptimizations(32, opt::OptOrder::I_B_I));
    std::string expected = readGoldenFile("artifacts/schedules/spmm_csc_i_b_i.kernel.c.txt");
    EXPECT_EQ(actual, expected);
}
