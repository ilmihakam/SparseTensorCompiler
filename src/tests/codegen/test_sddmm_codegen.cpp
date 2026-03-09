/**
 * Test Suite: SDDMM Code Generation
 *
 * Tests generation of SDDMM (Sampled Dense-Dense Matrix Multiplication) kernels.
 * Kernel: C[i,j] = S[i,j] * D[i,k] * E[k,j]
 * where S is sparse (CSR or CSC), D and E are dense, C is dense.
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

static std::unique_ptr<sparseir::scheduled::Compute> parseForSDDMMCodegen(
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
// SDDMM CSR Tests
// ============================================================================

TEST(SDDMMCodegenTest, SDDMM_CSR_ScheduledCompute) {
    auto compute = parseForSDDMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
    )");
    ASSERT_NE(compute, nullptr);
    EXPECT_TRUE(compute->fullyLowered);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::DenseArray);
    EXPECT_EQ(compute->exprInfo.numSparseInputs, 1);
    EXPECT_EQ(compute->exprInfo.numDenseInputs, 2);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("void compute("), std::string::npos);
    EXPECT_EQ(output.find("void sddmm("), std::string::npos);
}

TEST(SDDMMCodegenTest, SDDMM_CSR_Signature) {
    auto compute = parseForSDDMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("void compute("), std::string::npos);
    EXPECT_NE(output.find("const SparseMatrix* S"), std::string::npos);
    EXPECT_NE(output.find("double** D"), std::string::npos);
    EXPECT_NE(output.find("double** E"), std::string::npos);
}

TEST(SDDMMCodegenTest, SDDMM_CSR_CodePatterns) {
    auto compute = parseForSDDMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    // SDDMM CSR patterns: sparse sampling + dense contraction
    EXPECT_NE(output.find("S->row_ptr"), std::string::npos);
    EXPECT_NE(output.find("S->col_idx"), std::string::npos);
    EXPECT_NE(output.find("sum"), std::string::npos);
    EXPECT_NE(output.find("D[i][k]"), std::string::npos);
    EXPECT_NE(output.find("S->vals[pS] * sum"), std::string::npos);
}

TEST(SDDMMCodegenTest, SDDMM_CSR_ReferenceKernel) {
    auto compute = parseForSDDMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("void reference("), std::string::npos);
    EXPECT_EQ(output.find("sddmm_reference"), std::string::npos);
}

TEST(SDDMMCodegenTest, SDDMM_CSR_FullProgram) {
    auto compute = parseForSDDMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("#include"), std::string::npos);
    EXPECT_NE(output.find("int main"), std::string::npos);
    EXPECT_NE(output.find("return"), std::string::npos);
}

TEST(SDDMMCodegenTest, SDDMM_CSR_BalancedBraces) {
    auto compute = parseForSDDMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
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
// SDDMM CSC Tests
// ============================================================================

TEST(SDDMMCodegenTest, SDDMM_CSC_CodePatterns) {
    auto compute = parseForSDDMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor S : CSC<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    // CSC patterns: col_ptr, row_idx
    EXPECT_NE(output.find("S->col_ptr"), std::string::npos);
    EXPECT_NE(output.find("S->row_idx"), std::string::npos);
}

TEST(SDDMMCodegenTest, SDDMM_CSC_BalancedBraces) {
    auto compute = parseForSDDMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor S : CSC<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
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
// SDDMM Schedule Tests
// ============================================================================

TEST(SDDMMCodegenTest, SDDMM_CSR_Interchange_FusesAccumulator) {
    auto compute = parseForSDDMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
    )");
    ASSERT_NE(compute, nullptr);
    opt::OptConfig config = opt::OptConfig::interchangeOnly();
    std::string output = generate(*compute, config);
    // fuseAccumulatorPattern eliminates the sum variable in the optimized kernel.
    // The reference kernel still uses sum; only check the optimized section.
    size_t refPos = output.find("void reference(");
    ASSERT_NE(refPos, std::string::npos);
    std::string optimizedSection = output.substr(0, refPos);
    EXPECT_EQ(optimizedSection.find("double sum"), std::string::npos);
    EXPECT_NE(output.find("C[i][j] +="), std::string::npos);
}

TEST(SDDMMCodegenTest, SDDMM_CSR_KBlocking) {
    auto compute = parseForSDDMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
    )");
    ASSERT_NE(compute, nullptr);
    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    std::string output = generate(*compute, config);
    EXPECT_NE(output.find("k_block"), std::string::npos);
    EXPECT_EQ(output.find("i_block"), std::string::npos);
    EXPECT_EQ(compute->optimizations.tiledIndex, "k");
}

TEST(SDDMMCodegenTest, SDDMM_CSR_I_THEN_B) {
    auto compute = parseForSDDMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
    )");
    ASSERT_NE(compute, nullptr);
    opt::OptConfig config = opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B);
    std::string output = generate(*compute, config);
    // Check optimized kernel only (reference still has double sum)
    size_t refPos = output.find("void reference(");
    ASSERT_NE(refPos, std::string::npos);
    std::string optimizedSection = output.substr(0, refPos);
    EXPECT_EQ(optimizedSection.find("double sum"), std::string::npos);  // fused by interchange
    EXPECT_NE(output.find("k_block"), std::string::npos);
    EXPECT_TRUE(compute->optimizations.blockingApplied);
    EXPECT_TRUE(compute->optimizations.interchangeApplied);
}

TEST(SDDMMCodegenTest, SDDMM_CSR_B_THEN_I) {
    auto compute = parseForSDDMMCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 64>;
        tensor E : Dense<64, 100>;
        compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
    )");
    ASSERT_NE(compute, nullptr);
    opt::OptConfig config = opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I);
    std::string output = generate(*compute, config);
    // Block-pair interchange: k_block+k move before j; sum is fused in optimized kernel
    size_t refPos = output.find("void reference(");
    ASSERT_NE(refPos, std::string::npos);
    std::string optimizedSection = output.substr(0, refPos);
    EXPECT_EQ(optimizedSection.find("double sum"), std::string::npos);
    EXPECT_NE(output.find("k_block"), std::string::npos);
    EXPECT_NE(output.find("C[i][j] +="), std::string::npos);
    EXPECT_TRUE(compute->optimizations.blockingApplied);
    EXPECT_TRUE(compute->optimizations.interchangeApplied);
}
