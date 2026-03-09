/**
 * Test Suite: SpGEMM Code Generation
 *
 * Tests generation of SpGEMM (Sparse-Sparse Matrix Multiplication) kernels.
 * Kernel: C[i,j] = A[i,k] * B[k,j]
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

static std::unique_ptr<sparseir::scheduled::Compute> parseForSpGEMMCodegen(
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
// SpGEMM CSR Tests
// ============================================================================

TEST(SpGEMMCodegenTest, SpGEMM_CSR_ScheduledCompute) {
    auto compute = parseForSpGEMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 80>;
        tensor B : CSR<80, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(compute, nullptr);
    EXPECT_TRUE(compute->fullyLowered);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::DenseArray);
    EXPECT_EQ(compute->exprInfo.numSparseInputs, 2);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("void compute("), std::string::npos);
    EXPECT_EQ(output.find("void spgemm("), std::string::npos);
}

TEST(SpGEMMCodegenTest, SpGEMM_CSR_Signature) {
    auto compute = parseForSpGEMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 80>;
        tensor B : CSR<80, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("void compute("), std::string::npos);
    EXPECT_NE(output.find("const SparseMatrix* A"), std::string::npos);
    EXPECT_NE(output.find("const SparseMatrix* B"), std::string::npos);
}

TEST(SpGEMMCodegenTest, SpGEMM_CSR_GustavsonPattern) {
    auto compute = parseForSpGEMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 80>;
        tensor B : CSR<80, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    // Nested sparse product pattern should traverse A rows, then B rows via k.
    EXPECT_NE(output.find("B->row_ptr[k]"), std::string::npos);
    EXPECT_NE(output.find("A->row_ptr"), std::string::npos);
    EXPECT_NE(output.find("A->vals[pA] * B->vals[pB]"), std::string::npos);
}

TEST(SpGEMMCodegenTest, SpGEMM_CSR_ReferenceKernel) {
    auto compute = parseForSpGEMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 80>;
        tensor B : CSR<80, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("void reference("), std::string::npos);
    EXPECT_EQ(output.find("spgemm_reference"), std::string::npos);
}

TEST(SpGEMMCodegenTest, SpGEMM_CSR_FullProgram) {
    auto compute = parseForSpGEMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 80>;
        tensor B : CSR<80, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("#include"), std::string::npos);
    EXPECT_NE(output.find("int main"), std::string::npos);
    EXPECT_NE(output.find("return"), std::string::npos);
}

TEST(SpGEMMCodegenTest, SpGEMM_CSR_BalancedBraces) {
    auto compute = parseForSpGEMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 80>;
        tensor B : CSR<80, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
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
// SpGEMM CSC Tests
// ============================================================================

TEST(SpGEMMCodegenTest, SpGEMM_CSC_CodePatterns) {
    auto compute = parseForSpGEMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSC<100, 80>;
        tensor B : CSC<80, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    // CSC patterns: column traversals for both inputs and direct sparse value multiply.
    EXPECT_NE(output.find("col_ptr"), std::string::npos);
    EXPECT_NE(output.find("row_idx"), std::string::npos);
    EXPECT_NE(output.find("A->col_ptr[k]"), std::string::npos);
    EXPECT_NE(output.find("A->vals[pA] * B->vals[pB]"), std::string::npos);
}

TEST(SpGEMMCodegenTest, SpGEMM_CSC_BalancedBraces) {
    auto compute = parseForSpGEMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSC<100, 80>;
        tensor B : CSC<80, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
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

// ============================================================================
// SpGEMM Blocking and Outer-Product Tests
// ============================================================================

TEST(SpGEMMCodegenTest, SpGEMM_CSR_IBlocking) {
    auto compute = parseForSpGEMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 80>;
        tensor B : CSR<80, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(compute, nullptr);
    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    std::string output = generate(*compute, config);
    EXPECT_NE(output.find("i_block"), std::string::npos);
    EXPECT_EQ(output.find("j_block"), std::string::npos);
    EXPECT_EQ(output.find("k_block"), std::string::npos);
    EXPECT_EQ(compute->optimizations.tiledIndex, "i");
}

TEST(SpGEMMCodegenTest, SpGEMM_CSC_OuterProduct_Baseline) {
    // A:CSC,B:CSC → column-by-column outer-product: j(Dense) → k(Sparse,B-CSC) → i(Sparse,A-CSC)
    auto compute = parseForSpGEMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSC<100, 80>;
        tensor B : CSC<80, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(compute, nullptr);
    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);
    // Column-by-column outer-product: both A and B traversed column-wise
    EXPECT_NE(output.find("A->col_ptr"), std::string::npos);
    EXPECT_NE(output.find("B->col_ptr"), std::string::npos);
    EXPECT_NE(output.find("C[i][j] +="), std::string::npos);
    int braceCount = 0;
    for (char c : output) { if (c=='{') braceCount++; if (c=='}') braceCount--; }
    EXPECT_EQ(braceCount, 0);
}
