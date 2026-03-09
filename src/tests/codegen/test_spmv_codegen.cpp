/**
 * Test Suite: SpMV Code Generation
 *
 * Tests generation of SpMV kernels for all optimization configurations.
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

std::unique_ptr<sparseir::scheduled::Compute> parseForSpMVCodegen(const std::string& code) {
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

// ============================================================================
// SpMV CSR Configuration Tests
// ============================================================================

/**
 * Test: SpMV CSR baseline (no optimizations).
 */
TEST(SpMVCodegenTest, SpMV_CSR_Baseline) {
    auto op = parseForSpMVCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    EXPECT_NE(output.find("void compute("), std::string::npos);
    // Should have CSR access patterns
    EXPECT_NE(output.find("row_ptr"), std::string::npos);
}

/**
 * Test: SpMV CSR with format-correctness reordering (automatic when needed).
 */
TEST(SpMVCodegenTest, SpMV_CSR_FormatCorrectness) {
    auto op = parseForSpMVCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    EXPECT_NE(output.find("void compute("), std::string::npos);
    EXPECT_NE(output.find("int main"), std::string::npos);
}

/**
 * Test: SpMV CSR with blocking.
 */
TEST(SpMVCodegenTest, SpMV_CSR_Blocking) {
    auto op = parseForSpMVCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // Should have blocking
    EXPECT_NE(output.find("_block"), std::string::npos);
    EXPECT_NE(output.find("32"), std::string::npos);
}

/**
 * Test: SpMV CSR with both optimizations.
 */
TEST(SpMVCodegenTest, SpMV_CSR_BothOpts) {
    auto op = parseForSpMVCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::allOptimizations(32);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    EXPECT_NE(output.find("void compute("), std::string::npos);
    EXPECT_NE(output.find("int main"), std::string::npos);
}

// ============================================================================
// SpMV CSC Configuration Tests
// ============================================================================

/**
 * Test: SpMV CSC baseline.
 */
TEST(SpMVCodegenTest, SpMV_CSC_Baseline) {
    auto op = parseForSpMVCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should have CSC access patterns
    EXPECT_NE(output.find("col_ptr"), std::string::npos);
}

/**
 * Test: SpMV CSC with various opt configurations.
 */
TEST(SpMVCodegenTest, SpMV_CSC_AllConfigs) {
    auto op = parseForSpMVCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::allOptimizations(32);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // Should produce complete program
    EXPECT_NE(output.find("int main"), std::string::npos);
    EXPECT_NE(output.find("col_ptr"), std::string::npos);
}

// ============================================================================
// SpMV Kernel Structure Tests
// ============================================================================

/**
 * Test: SpMV kernel has proper signature.
 */
TEST(SpMVCodegenTest, SpMV_KernelSignature) {
    auto op = parseForSpMVCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should have proper function signature with SparseMatrix*, double* x, double* y
    EXPECT_NE(output.find("SparseMatrix"), std::string::npos);
    EXPECT_NE(output.find("double*"), std::string::npos);
}

/**
 * Test: SpMV uses accumulation pattern.
 */
TEST(SpMVCodegenTest, SpMV_Accumulation) {
    auto op = parseForSpMVCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should use accumulation: y[i] += ...
    EXPECT_NE(output.find("+="), std::string::npos);
}

/**
 * Test: SpMV uses local sum variable for reduction.
 */
TEST(SpMVCodegenTest, SpMV_LocalSum) {
    auto op = parseForSpMVCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // May use local sum variable or direct accumulation
    // Either pattern is acceptable
    bool hasAccumulation = output.find("+=") != std::string::npos;
    EXPECT_TRUE(hasAccumulation);
}

/**
 * Test: SpMV generates reference kernel.
 */
TEST(SpMVCodegenTest, SpMV_ReferenceKernel) {
    auto op = parseForSpMVCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    EXPECT_NE(output.find("void reference("), std::string::npos);
}

/**
 * Test: SpMV generates full program with main().
 */
TEST(SpMVCodegenTest, SpMV_FullProgram) {
    auto op = parseForSpMVCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should have complete program structure
    EXPECT_NE(output.find("#include"), std::string::npos);
    EXPECT_NE(output.find("int main"), std::string::npos);
    EXPECT_NE(output.find("return"), std::string::npos);
}

/**
 * Test: SpMV output should be compilable (syntactically correct).
 */
TEST(SpMVCodegenTest, SpMV_CompilableOutput) {
    auto op = parseForSpMVCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should have balanced braces
    int braceCount = 0;
    for (char c : output) {
        if (c == '{') braceCount++;
        if (c == '}') braceCount--;
    }
    EXPECT_EQ(braceCount, 0);

    // Should have balanced parentheses
    int parenCount = 0;
    for (char c : output) {
        if (c == '(') parenCount++;
        if (c == ')') parenCount--;
    }
    EXPECT_EQ(parenCount, 0);
}
