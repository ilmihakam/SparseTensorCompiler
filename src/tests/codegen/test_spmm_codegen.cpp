/**
 * Test Suite: SpMM Code Generation
 *
 * Tests generation of SpMM (Sparse Matrix-Matrix Multiplication) kernels
 * for all optimization configurations.
 *
 * Kernel: C[i,j] = A[i,k] * B[k,j]
 * where A is sparse (CSR or CSC), B and C are dense.
 */

#include <gtest/gtest.h>
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

std::unique_ptr<sparseir::scheduled::Compute> parseForSpMMCodegen(const std::string& code) {
    if (!parserInitialized) {
        yynerrs = 0;
        g_program.reset();
        yy_scan_string("tensor C : Dense;");
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
// SpMM CSR Configuration Tests
// ============================================================================

/**
 * Test: SpMM CSR baseline (no optimizations).
 * Verifies basic code generation for CSR SpMM.
 */
TEST(SpMMCodegenTest, SpMM_CSR_Baseline) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSR<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    EXPECT_NE(output.find("void compute("), std::string::npos);

    // Should have CSR access patterns
    EXPECT_NE(output.find("row_ptr"), std::string::npos);
    EXPECT_NE(output.find("col_idx"), std::string::npos);

    // Should have 3-loop structure (i, k, j)
    // Note: exact loop structure depends on codegen implementation
    EXPECT_NE(output.find("for"), std::string::npos);

    // Should have complete program structure
    EXPECT_NE(output.find("int main"), std::string::npos);
    EXPECT_NE(output.find("#include"), std::string::npos);
}

/**
 * Test: SpMM CSR with blocking (j-tiling).
 * Verifies that j-tiled blocking codegen is used instead of i-blocking.
 */
TEST(SpMMCodegenTest, SpMM_CSR_Blocking) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSR<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly();
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // Should have j-tiled block loop (j_block variable)
    EXPECT_NE(output.find("j_block"), std::string::npos);

    // Should NOT have i_block (i is not the tiled index)
    EXPECT_EQ(output.find("i_block"), std::string::npos);

    // Should still have main program
    EXPECT_NE(output.find("int main"), std::string::npos);
}

/**
 * Test: SpMM CSR with all optimizations.
 */
TEST(SpMMCodegenTest, SpMM_CSR_AllOptimizations) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSR<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::allOptimizations();
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // Should have valid generated code
    EXPECT_NE(output.find("void compute("), std::string::npos);
    EXPECT_NE(output.find("int main"), std::string::npos);
}

// ============================================================================
// SpMM CSC Configuration Tests
// ============================================================================

/**
 * Test: SpMM CSC baseline.
 * CSC should create k-outer, i-sparse, j-inner loop structure.
 */
TEST(SpMMCodegenTest, SpMM_CSC_Baseline) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSC<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    EXPECT_NE(output.find("void compute("), std::string::npos);

    // Should have CSC access patterns
    EXPECT_NE(output.find("col_ptr"), std::string::npos);
    EXPECT_NE(output.find("row_idx"), std::string::npos);

    // Should have complete program
    EXPECT_NE(output.find("int main"), std::string::npos);
}

/**
 * Test: SpMM CSC with blocking (j-tiling).
 */
TEST(SpMMCodegenTest, SpMM_CSC_Blocking) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSC<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly();
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // Should have j-tiled block loop (j_block variable)
    EXPECT_NE(output.find("j_block"), std::string::npos);

    // Should use CSC access patterns
    EXPECT_NE(output.find("col_ptr"), std::string::npos);
    EXPECT_NE(output.find("row_idx"), std::string::npos);

    EXPECT_NE(output.find("void compute("), std::string::npos);
    EXPECT_NE(output.find("int main"), std::string::npos);
}

// ============================================================================
// Matrix Market Support Tests
// ============================================================================

/**
 * Test: Generated code includes Matrix Market loader.
 */
TEST(SpMMCodegenTest, IncludesMatrixMarketLoader) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSR<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should have Matrix Market loader function
    EXPECT_NE(output.find("load_matrix_market"), std::string::npos);
    EXPECT_NE(output.find("SparseMatrix"), std::string::npos);
}

/**
 * Test: Generated code includes timing utilities.
 */
TEST(SpMMCodegenTest, IncludesTimingUtilities) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSR<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should have timing utilities
    EXPECT_NE(output.find("get_time_ms"), std::string::npos);
    EXPECT_NE(output.find("clock_gettime"), std::string::npos);
}

/**
 * Test: Generated code includes reference implementation for verification.
 */
TEST(SpMMCodegenTest, IncludesReferenceImplementation) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSR<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    EXPECT_NE(output.find("void reference("), std::string::npos);
}

// ============================================================================
// Different Matrix Sizes Tests
// ============================================================================

/**
 * Test: Different matrix dimensions.
 */
TEST(SpMMCodegenTest, DifferentMatrixSizes) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<50, 30>;
        tensor A : CSR<50, 40>;
        tensor B : Dense<40, 30>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    EXPECT_NE(output.find("void compute("), std::string::npos);
    EXPECT_NE(output.find("int main"), std::string::npos);

    // Dimensions should be reflected structurally in the generic scheduled backend.
    EXPECT_NE(output.find("int N_j = 30"), std::string::npos);
    EXPECT_NE(output.find("malloc(N_j * sizeof(double))"), std::string::npos);
}

/**
 * Test: Large matrix dimensions.
 */
TEST(SpMMCodegenTest, LargeMatrixDimensions) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<1000, 500>;
        tensor A : CSR<1000, 800>;
        tensor B : Dense<800, 500>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    EXPECT_NE(output.find("void compute("), std::string::npos);
    EXPECT_NE(output.find("int main"), std::string::npos);
}

// ============================================================================
// Code Structure Tests
// ============================================================================

/**
 * Test: Generated code has proper header comments.
 */
TEST(SpMMCodegenTest, HasProperHeaderComments) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSR<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should have header comments
    EXPECT_NE(output.find("Generated by SparseTensorCompiler"), std::string::npos);
    EXPECT_NE(output.find("Kernel:"), std::string::npos);
}

/**
 * Test: Generated code shows which optimizations are applied.
 */
TEST(SpMMCodegenTest, ShowsOptimizationsInComments) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSR<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly();
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // Should mention blocking in comments
    EXPECT_NE(output.find("blocking"), std::string::npos);
}

/**
 * Test: Baseline shows no optimizations.
 */
TEST(SpMMCodegenTest, BaselineShowsNoOptimizations) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSR<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should indicate no optimizations (OFF or disabled)
    // Exact format depends on codegen implementation
    EXPECT_TRUE(
        output.find("OFF") != std::string::npos ||
        output.find("baseline") != std::string::npos
    );
}

/**
 * Test: Generated code includes necessary headers.
 */
TEST(SpMMCodegenTest, IncludesNecessaryHeaders) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSR<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should include necessary C headers
    EXPECT_NE(output.find("#include <stdio.h>"), std::string::npos);
    EXPECT_NE(output.find("#include <stdlib.h>"), std::string::npos);
    EXPECT_NE(output.find("#include <time.h>"), std::string::npos);
}

// ============================================================================
// Schedule Order Loop Structure Tests
// ============================================================================

/**
 * Helper: extract the body of the spmm() kernel function from generated code.
 */
static std::string extractSpMMKernelBody(const std::string& code) {
    auto sigPos = code.find("void compute(");
    if (sigPos == std::string::npos) return "";
    auto openBrace = code.find('{', sigPos);
    if (openBrace == std::string::npos) return "";
    int depth = 0;
    for (size_t i = openBrace; i < code.size(); i++) {
        if (code[i] == '{') depth++;
        else if (code[i] == '}') { depth--; if (depth == 0) return code.substr(openBrace + 1, i - openBrace - 1); }
    }
    return "";
}

/**
 * Test: B_THEN_I CSR produces i -> j_block -> sparse(k) -> j loop structure.
 * The sparse CSR loop (row_ptr) comes before the inner j loop.
 */
TEST(SpMMCodegenTest, SpMM_CSR_B_THEN_I_HasJJOuter) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSR<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    std::string body = extractSpMMKernelBody(output);
    ASSERT_FALSE(body.empty());

    // B_THEN_I: i -> j_block -> k(sparse) -> j
    // row_ptr (sparse k) should appear before inner j loop
    ASSERT_NE(body.find("j_block"), std::string::npos) << "Missing j_block loop in B_THEN_I CSR";
    auto rowPtrPos = body.find("row_ptr");
    auto innerJPos = body.find("for (int j =");
    ASSERT_NE(rowPtrPos, std::string::npos) << "Missing row_ptr in B_THEN_I CSR";
    ASSERT_NE(innerJPos, std::string::npos) << "Missing inner j loop in B_THEN_I CSR";
    EXPECT_LT(rowPtrPos, innerJPos) << "B_THEN_I CSR: sparse loop should precede inner j";
}

/**
 * Test: B_THEN_I CSC produces k -> j_block -> sparse(i) -> j loop structure.
 * The sparse CSC loop (col_ptr) comes before the inner j loop.
 */
TEST(SpMMCodegenTest, SpMM_CSC_B_THEN_I_HasJJOuter) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSC<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    std::string body = extractSpMMKernelBody(output);
    ASSERT_FALSE(body.empty());

    // B_THEN_I: k -> j_block -> i(sparse) -> j
    // col_ptr (sparse i) should appear before inner j loop
    ASSERT_NE(body.find("j_block"), std::string::npos) << "Missing j_block loop in B_THEN_I CSC";
    auto colPtrPos = body.find("col_ptr");
    auto innerJPos = body.find("for (int j =");
    ASSERT_NE(colPtrPos, std::string::npos) << "Missing col_ptr in B_THEN_I CSC";
    ASSERT_NE(innerJPos, std::string::npos) << "Missing inner j loop in B_THEN_I CSC";
    EXPECT_LT(colPtrPos, innerJPos) << "B_THEN_I CSC: sparse loop should precede inner j";
}

/**
 * Test: I_THEN_B CSR produces i -> j_block -> j -> sparse(k) structure.
 * i is outermost, j_block appears before the inner sparse k loop.
 */
TEST(SpMMCodegenTest, SpMM_CSR_I_THEN_B_HasIOuter) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSR<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    std::string body = extractSpMMKernelBody(output);
    ASSERT_FALSE(body.empty());

    // I_THEN_B: i -> j_block -> j -> k(sparse)
    // i loop before j_block, and j loop before row_ptr (sparse k)
    auto iPos = body.find("for (int i");
    auto jBlockPos = body.find("j_block");
    auto innerJPos = body.find("for (int j =");
    auto rowPtrPos = body.find("row_ptr");
    ASSERT_NE(iPos, std::string::npos) << "Missing i loop in I_THEN_B CSR";
    ASSERT_NE(jBlockPos, std::string::npos) << "Missing j_block in I_THEN_B CSR";
    ASSERT_NE(innerJPos, std::string::npos) << "Missing inner j loop in I_THEN_B CSR";
    ASSERT_NE(rowPtrPos, std::string::npos) << "Missing row_ptr in I_THEN_B CSR";
    EXPECT_LT(iPos, jBlockPos) << "I_THEN_B CSR: i should be before j_block";
    EXPECT_LT(innerJPos, rowPtrPos) << "I_THEN_B CSR: j should precede sparse k";
}

/**
 * Test: I_THEN_B CSC produces k -> j_block -> j -> sparse(i) structure.
 * k is outermost, j_block appears before the inner sparse i loop.
 */
TEST(SpMMCodegenTest, SpMM_CSC_I_THEN_B_HasKOuter) {
    auto op = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSC<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    std::string body = extractSpMMKernelBody(output);
    ASSERT_FALSE(body.empty());

    // I_THEN_B: k -> j_block -> j -> i(sparse)
    // k loop before j_block, and j loop before col_ptr (sparse i)
    auto kPos = body.find("for (int k");
    auto jBlockPos = body.find("j_block");
    auto innerJPos = body.find("for (int j =");
    auto colPtrPos = body.find("col_ptr");
    ASSERT_NE(kPos, std::string::npos) << "Missing k loop in I_THEN_B CSC";
    ASSERT_NE(jBlockPos, std::string::npos) << "Missing j_block in I_THEN_B CSC";
    ASSERT_NE(innerJPos, std::string::npos) << "Missing inner j loop in I_THEN_B CSC";
    ASSERT_NE(colPtrPos, std::string::npos) << "Missing col_ptr in I_THEN_B CSC";
    EXPECT_LT(kPos, jBlockPos) << "I_THEN_B CSC: k should be before j_block";
    EXPECT_LT(innerJPos, colPtrPos) << "I_THEN_B CSC: j should precede sparse i";
}

/**
 * Test: B_THEN_I produces different code than I_THEN_B for CSR SpMM.
 */
TEST(SpMMCodegenTest, SpMM_CSR_DifferentSchedules_DifferentCode) {
    auto opITB = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSR<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(opITB, nullptr);

    auto opBTI = parseForSpMMCodegen(R"(
        tensor C : Dense<100, 50>;
        tensor A : CSR<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(opBTI, nullptr);

    opt::OptConfig configITB = opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B);
    opt::applyOptimizations(*opITB, configITB);
    std::string outputITB = codegen::generateCode(*opITB, configITB);

    opt::OptConfig configBTI = opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I);
    opt::applyOptimizations(*opBTI, configBTI);
    std::string outputBTI = codegen::generateCode(*opBTI, configBTI);

    // The kernel bodies should differ
    std::string bodyITB = extractSpMMKernelBody(outputITB);
    std::string bodyBTI = extractSpMMKernelBody(outputBTI);
    ASSERT_FALSE(bodyITB.empty());
    ASSERT_FALSE(bodyBTI.empty());
    EXPECT_NE(bodyITB, bodyBTI) << "I_THEN_B and B_THEN_I should produce different SpMM kernel code";
}
