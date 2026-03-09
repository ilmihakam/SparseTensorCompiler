/**
 * Test Suite: Output Format (Phase 3)
 *
 * Verifies that OutputStrategy is correctly set during IR lowering.
 */

#include <gtest/gtest.h>
#include "ast.h"
#include "semantic_ir.h"

extern std::unique_ptr<SparseTensorCompiler::Program> g_program;
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern int yynerrs;

static bool parserInit = false;

static std::unique_ptr<sparseir::scheduled::Compute> lowerFromCode(const std::string& code) {
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

    if (result != 0 || !g_program) return nullptr;

    auto op = sparseir::lowerFirstComputationToScheduled(*g_program);
    g_program.reset();
    return op;
}

TEST(OutputFormat, SpGEMM_OutputStrategy_DenseArray) {
    auto op = lowerFromCode(
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = A[i, k] * B[k, j];\n");
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->outputStrategy, ir::OutputStrategy::DenseArray);
}

TEST(OutputFormat, SpGEMM_OutputFormat_Dense) {
    auto op = lowerFromCode(
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = A[i, k] * B[k, j];\n");
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->output.format, ir::Format::Dense);
}

TEST(OutputFormat, SpMV_OutputStrategy_DenseArray) {
    auto op = lowerFromCode(
        "tensor A : CSR<100, 50>;\n"
        "tensor x : Dense<50>;\n"
        "tensor y : Dense<100>;\n"
        "compute y[i] = A[i, j] * x[j];\n");
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->outputStrategy, ir::OutputStrategy::DenseArray);
}

TEST(OutputFormat, SpAdd_OutputStrategy_DenseArray) {
    auto op = lowerFromCode(
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = A[i, j] + B[i, j];\n");
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->outputStrategy, ir::OutputStrategy::DenseArray);
}

TEST(OutputFormat, SDDMM_OutputStrategy_DenseArray) {
    auto op = lowerFromCode(
        "tensor S : CSR<100, 100>;\n"
        "tensor D : Dense<100, 64>;\n"
        "tensor E : Dense<64, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = S[i, j] * D[i, k] * E[k, j];\n");
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->outputStrategy, ir::OutputStrategy::DenseArray);
}

TEST(OutputFormat, SpAdd_OutputStrategy_SparseFixedPattern) {
    auto op = lowerFromCode(
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : CSR<100, 100>;\n"
        "compute C[i, j] = A[i, j] + B[i, j];\n");
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->outputStrategy, ir::OutputStrategy::SparseFixedPattern);
    EXPECT_EQ(op->outputPattern, sparseir::OutputPatternKind::Union);
}

TEST(OutputFormat, SpElMul_OutputStrategy_SparseFixedPattern) {
    auto op = lowerFromCode(
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : CSR<100, 100>;\n"
        "compute C[i, j] = A[i, j] * B[i, j];\n");
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->outputStrategy, ir::OutputStrategy::SparseFixedPattern);
    EXPECT_EQ(op->outputPattern, sparseir::OutputPatternKind::Intersection);
}

TEST(OutputFormat, SpGEMM_OutputStrategy_HashPerRow_WhenSparseOutput) {
    auto op = lowerFromCode(
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : CSR<100, 100>;\n"
        "compute C[i, j] = A[i, k] * B[k, j];\n");
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->outputStrategy, ir::OutputStrategy::HashPerRow);
    EXPECT_EQ(op->outputPattern, sparseir::OutputPatternKind::DynamicRowAccumulator);
}

TEST(OutputFormat, SDDMM_OutputStrategy_SparseFixedPattern) {
    auto op = lowerFromCode(
        "tensor S : CSR<100, 100>;\n"
        "tensor D : Dense<100, 64>;\n"
        "tensor E : Dense<64, 100>;\n"
        "tensor C : CSR<100, 100>;\n"
        "compute C[i, j] = S[i, j] * D[i, k] * E[k, j];\n");
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->outputStrategy, ir::OutputStrategy::SparseFixedPattern);
    EXPECT_EQ(op->outputPattern, sparseir::OutputPatternKind::Sampled);
}
