/**
 * Test Suite: SpElMul Code Generation
 *
 * Tests generation of SpElMul (Sparse Element-wise Multiplication) kernels.
 * Kernel: C[i,j] = A[i,j] * B[i,j]
 * where A, B are sparse (CSR or CSC), C is dense.
 */

#include <gtest/gtest.h>
#include <sstream>
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

static bool parserInitialized = false;

static std::unique_ptr<sparseir::scheduled::Compute> parseForSpElMulCodegen(
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

static std::string generate(sparseir::scheduled::Compute& compute,
                            const opt::OptConfig& config) {
    opt::applyOptimizations(compute, config);
    return codegen::generateCode(compute, config);
}

// ============================================================================
// SpElMul CSR Tests
// ============================================================================

TEST(SpElMulCodegenTest, SpElMul_CSR_ScheduledCompute) {
    auto compute = parseForSpElMulCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 100>;
        tensor B : CSR<100, 100>;
        compute C[i, j] = A[i, j] * B[i, j];
    )");
    ASSERT_NE(compute, nullptr);
    EXPECT_TRUE(compute->fullyLowered);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::DenseArray);
    EXPECT_EQ(compute->exprInfo.numSparseInputs, 2);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("void compute("), std::string::npos);
    EXPECT_EQ(output.find("void spelmul("), std::string::npos);
}

TEST(SpElMulCodegenTest, SpElMul_CSR_Signature) {
    auto compute = parseForSpElMulCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 100>;
        tensor B : CSR<100, 100>;
        compute C[i, j] = A[i, j] * B[i, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("void compute("), std::string::npos);
    EXPECT_NE(output.find("const SparseMatrix* A"), std::string::npos);
    EXPECT_NE(output.find("const SparseMatrix* B"), std::string::npos);
}

TEST(SpElMulCodegenTest, SpElMul_CSR_MergeIntersection) {
    auto compute = parseForSpElMulCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 100>;
        tensor B : CSR<100, 100>;
        compute C[i, j] = A[i, j] * B[i, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    // Generic intersection merge uses min/max alignment across merged tensors.
    EXPECT_NE(output.find("while"), std::string::npos);
    EXPECT_NE(output.find("min_idx == max_idx"), std::string::npos);
    EXPECT_NE(output.find("A->vals[pA] * B->vals[pB]"), std::string::npos);
}

TEST(SpElMulCodegenTest, SpElMul_CSR_ReferenceKernel) {
    auto compute = parseForSpElMulCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 100>;
        tensor B : CSR<100, 100>;
        compute C[i, j] = A[i, j] * B[i, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("void reference("), std::string::npos);
    EXPECT_EQ(output.find("spelmul_reference"), std::string::npos);
}

TEST(SpElMulCodegenTest, SpElMul_CSR_FullProgram) {
    auto compute = parseForSpElMulCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 100>;
        tensor B : CSR<100, 100>;
        compute C[i, j] = A[i, j] * B[i, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("#include"), std::string::npos);
    EXPECT_NE(output.find("int main"), std::string::npos);
    EXPECT_NE(output.find("return"), std::string::npos);
}

TEST(SpElMulCodegenTest, SpElMul_CSR_BalancedBraces) {
    auto compute = parseForSpElMulCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 100>;
        tensor B : CSR<100, 100>;
        compute C[i, j] = A[i, j] * B[i, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    int braceCount = 0;
    for (char c : output) {
        if (c == '{') braceCount++;
        if (c == '}') braceCount--;
    }
    EXPECT_EQ(braceCount, 0);

    int parenCount = 0;
    for (char c : output) {
        if (c == '(') parenCount++;
        if (c == ')') parenCount--;
    }
    EXPECT_EQ(parenCount, 0);
}

// ============================================================================
// SpElMul CSC Tests
// ============================================================================

TEST(SpElMulCodegenTest, SpElMul_CSC_CodePatterns) {
    auto compute = parseForSpElMulCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSC<100, 100>;
        tensor B : CSC<100, 100>;
        compute C[i, j] = A[i, j] * B[i, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    // CSC patterns: col_ptr, row_idx, generic intersection merge.
    EXPECT_NE(output.find("col_ptr"), std::string::npos);
    EXPECT_NE(output.find("row_idx"), std::string::npos);
    EXPECT_NE(output.find("min_idx == max_idx"), std::string::npos);
}

TEST(SpElMulCodegenTest, SpElMul_CSC_BalancedBraces) {
    auto compute = parseForSpElMulCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSC<100, 100>;
        tensor B : CSC<100, 100>;
        compute C[i, j] = A[i, j] * B[i, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    int braceCount = 0;
    for (char c : output) {
        if (c == '{') braceCount++;
        if (c == '}') braceCount--;
    }
    EXPECT_EQ(braceCount, 0);
}
