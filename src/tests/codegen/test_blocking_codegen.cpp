/**
 * Test Suite: Blocking Code Generation
 *
 * Tests generation of blocked (tiled) loop constructs.
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

std::unique_ptr<sparseir::scheduled::Compute> parseForBlockingCodegen(const std::string& code) {
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
// Block Loop Structure Tests
// ============================================================================

/**
 * Test: Block loop generates outer tiling loop.
 */
TEST(BlockingCodegenTest, BlockLoopOuter) {
    auto op = parseForBlockingCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // Should have block loop variable (i_block or similar)
    EXPECT_NE(output.find("_block"), std::string::npos);
}

/**
 * Test: Block loop bounds use ceiling division.
 */
TEST(BlockingCodegenTest, BlockLoopBounds) {
    auto op = parseForBlockingCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // Should have ceiling division pattern: (N + B - 1) / B or equivalent
    // OR a precomputed constant block count when dimensions are compile-time.
    bool hasCeiling = output.find("+ 32 - 1") != std::string::npos ||
                      output.find("+ 31") != std::string::npos ||
                      output.find("/ 32") != std::string::npos ||
                      output.find("< 4") != std::string::npos;  // ceil(100/32) = 4
    EXPECT_TRUE(hasCeiling);
}

/**
 * Test: Blocked inner loop calculates start index.
 */
TEST(BlockingCodegenTest, BlockedInnerStart) {
    auto op = parseForBlockingCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // Should calculate start: i_start = i_block * B
    bool hasStart = output.find("_start") != std::string::npos ||
                    output.find("* 32") != std::string::npos;
    EXPECT_TRUE(hasStart);
}

/**
 * Test: Blocked inner loop calculates end index with min.
 */
TEST(BlockingCodegenTest, BlockedInnerEnd) {
    auto op = parseForBlockingCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // Should have min calculation or ternary for end bound
    bool hasMin = output.find("_end") != std::string::npos ||
                  output.find("? ") != std::string::npos ||
                  output.find("min(") != std::string::npos;
    EXPECT_TRUE(hasMin);
}

/**
 * Test: Blocked inner loop iterates within block range.
 */
TEST(BlockingCodegenTest, BlockedInnerLoop) {
    auto op = parseForBlockingCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // Should have inner for loop
    size_t block_loop = output.find("_block");
    ASSERT_NE(block_loop, std::string::npos);

    // Should have another for loop after block loop
    size_t inner_for = output.find("for (", block_loop);
    EXPECT_NE(inner_for, std::string::npos);
}

/**
 * Test: Default block size 32 is used.
 */
TEST(BlockingCodegenTest, BlockSize32) {
    auto op = parseForBlockingCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // Should use 32 as block size
    EXPECT_NE(output.find("32"), std::string::npos);
}

/**
 * Test: Custom block size 64 is used.
 */
TEST(BlockingCodegenTest, BlockSize64) {
    auto op = parseForBlockingCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(64);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // Should use 64 as block size
    EXPECT_NE(output.find("64"), std::string::npos);
}

/**
 * Test: Blocking with sparse inner loop preserved.
 */
TEST(BlockingCodegenTest, BlockingWithSparseInner) {
    auto op = parseForBlockingCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // Should still have CSR sparse iteration
    EXPECT_NE(output.find("row_ptr"), std::string::npos);
    EXPECT_NE(output.find("col_idx"), std::string::npos);
}

/**
 * Test: Blocking metadata comment in output.
 */
TEST(BlockingCodegenTest, BlockingMetadataComment) {
    auto op = parseForBlockingCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // Should have comment about blocking
    EXPECT_NE(output.find("blocking"), std::string::npos);
}

// ============================================================================
// 2D Blocking Code Generation Tests
// ============================================================================

/**
 * Test: SpMM 2D blocking generates both i_block and j_block loops.
 */
TEST(BlockingCodegenTest, SpMM_2D_BothBlockLoops) {
    auto op = parseForBlockingCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blocking2D(32, 64);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    EXPECT_NE(output.find("i_block"), std::string::npos);
    EXPECT_NE(output.find("j_block"), std::string::npos);
}

/**
 * Test: SpMM 2D blocking uses different block sizes for each dimension.
 */
TEST(BlockingCodegenTest, SpMM_2D_AsymmetricBlockSizes) {
    auto op = parseForBlockingCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blocking2D(32, 64);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // i_block should use 32
    EXPECT_NE(output.find("i_start = i_block * 32"), std::string::npos);
    // j_block should use 64
    EXPECT_NE(output.find("j_start = j_block * 64"), std::string::npos);
}

/**
 * Test: SpMM 2D blocking header comment shows 2D info.
 */
TEST(BlockingCodegenTest, SpMM_2D_HeaderComment) {
    auto op = parseForBlockingCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blocking2D(32, 64);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    EXPECT_NE(output.find("2D"), std::string::npos);
}

/**
 * Test: SpMV with 2D flag only generates 1D blocking (i_block only).
 */
TEST(BlockingCodegenTest, SpMV_2D_StillOnlyIBlock) {
    auto op = parseForBlockingCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blocking2D(32, 64);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    EXPECT_NE(output.find("i_block"), std::string::npos);
    EXPECT_EQ(output.find("j_block"), std::string::npos);
}

/**
 * Test: SpMM 2D blocking preserves sparse CSR iteration.
 */
TEST(BlockingCodegenTest, SpMM_2D_PreservesCSRIteration) {
    auto op = parseForBlockingCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blocking2D(32, 32);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    EXPECT_NE(output.find("row_ptr"), std::string::npos);
    EXPECT_NE(output.find("col_idx"), std::string::npos);
    EXPECT_NE(output.find("vals"), std::string::npos);
}

/**
 * Test: SpMM 2D blocking preserves computation body.
 */
TEST(BlockingCodegenTest, SpMM_2D_PreservesBody) {
    auto op = parseForBlockingCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blocking2D(32, 32);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    EXPECT_NE(output.find("+="), std::string::npos);
    EXPECT_NE(output.find("C["), std::string::npos);
    EXPECT_NE(output.find("B["), std::string::npos);
}

/**
 * Test: Blocking preserves inner loop body.
 */
TEST(BlockingCodegenTest, BlockingPreservesBody) {
    auto op = parseForBlockingCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    // Should still have accumulation and array accesses
    EXPECT_NE(output.find("+="), std::string::npos);
    EXPECT_NE(output.find("y["), std::string::npos);
    EXPECT_NE(output.find("x["), std::string::npos);
}

TEST(BlockingCodegenTest, PositionBlockingIteratorEmitsPointerChunks) {
    auto op = parseForBlockingCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::positionBlockingOnly(16);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    EXPECT_NE(output.find("sparse position blocking"), std::string::npos);
    EXPECT_NE(output.find("for (int pA_block = 0;"), std::string::npos);
    EXPECT_NE(output.find("for (int pA = pA_start; pA < pA_end; pA++)"), std::string::npos);
}

TEST(BlockingCodegenTest, PositionBlockingMergeEmitsChunkCounter) {
    auto op = parseForBlockingCodegen(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 100>;
        tensor B : CSR<100, 100>;
        compute C[i, j] = A[i, j] + B[i, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::positionBlockingOnly(8);
    opt::applyOptimizations(*op, config);
    std::string output = codegen::generateCode(*op, config);

    EXPECT_NE(output.find("sparse position blocking"), std::string::npos);
    EXPECT_NE(output.find("j_chunk_steps"), std::string::npos);
    EXPECT_NE(output.find("while ((pA < endA || pB < endB) && j_chunk_steps < 8)"), std::string::npos);
}
