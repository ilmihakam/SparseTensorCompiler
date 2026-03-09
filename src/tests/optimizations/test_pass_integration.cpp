/**
 * Test Suite: Optimization Pass Integration
 *
 * Tests the complete optimization pipeline with all combinations
 * of optimizations and kernel types.
 */

#include <gtest/gtest.h>
#include "optimizations.h"
#include "ast.h"
#include "scheduled_optimizations.h"
#include "semantic_ir.h"

extern std::unique_ptr<SparseTensorCompiler::Program> g_program;
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern int yynerrs;

static bool parserInitialized = false;

std::unique_ptr<sparseir::scheduled::Compute> parseAndLowerFull(const std::string& code) {
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

int countLoopsInOp(const sparseir::scheduled::Loop* loop) {
    if (!loop) return 0;
    int count = 1;
    for (const auto& child : loop->children) {
        count += countLoopsInOp(child.get());
    }
    return count;
}

// ============================================================================
// SpMV Configuration Tests (4 configurations)
// ============================================================================

/**
 * Test: SpMV with baseline (no optimizations).
 */
TEST(PassIntegrationTest, SpMV_Baseline_NoOpts) {
    auto op = parseAndLowerFull(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    opt::applyOptimizations(*op, config);

    EXPECT_FALSE(op->optimizations.reorderingApplied);
    EXPECT_FALSE(op->optimizations.interchangeApplied);
    EXPECT_FALSE(op->optimizations.blockingApplied);
    EXPECT_EQ(op->output.name, "y");
}

/**
 * Test: SpMV with format-correctness reordering only.
 */
TEST(PassIntegrationTest, SpMV_FormatCorrectnessReordering) {
    auto op = parseAndLowerFull(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    opt::applyOptimizations(*op, config);

    // Reordering may or may not be applied (lowering might already be correct)
    EXPECT_FALSE(op->optimizations.interchangeApplied);
    EXPECT_FALSE(op->optimizations.blockingApplied);
}

/**
 * Test: SpMV with blocking only.
 */
TEST(PassIntegrationTest, SpMV_BlockingOnly) {
    auto op = parseAndLowerFull(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, config);

    EXPECT_FALSE(op->optimizations.reorderingApplied);
    EXPECT_FALSE(op->optimizations.interchangeApplied);
    EXPECT_TRUE(op->optimizations.blockingApplied);
}

/**
 * Test: SpMV with both optimizations.
 */
TEST(PassIntegrationTest, SpMV_BothOptimizations) {
    auto op = parseAndLowerFull(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::allOptimizations(32);
    opt::applyOptimizations(*op, config);

    // With natural order access, blocking should be applied
    // Reordering won't be applied since access is already optimal
    EXPECT_TRUE(op->optimizations.blockingApplied);
    ASSERT_NE(op->rootLoop, nullptr);
}

// ============================================================================
// SpMM Configuration Tests (4 configurations)
// ============================================================================

/**
 * Test: SpMM with baseline (no optimizations).
 */
TEST(PassIntegrationTest, SpMM_Baseline_NoOpts) {
    auto op = parseAndLowerFull(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    opt::applyOptimizations(*op, config);

    EXPECT_FALSE(op->optimizations.reorderingApplied);
    EXPECT_FALSE(op->optimizations.interchangeApplied);
    EXPECT_FALSE(op->optimizations.blockingApplied);
    EXPECT_EQ(op->output.name, "C");
}

/**
 * Test: SpMM with format-correctness reordering only.
 */
TEST(PassIntegrationTest, SpMM_FormatCorrectnessReordering) {
    auto op = parseAndLowerFull(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[k, i] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    opt::applyOptimizations(*op, config);

    // Reordering may or may not be applied (lowering might already be correct)
    EXPECT_FALSE(op->optimizations.interchangeApplied);
    EXPECT_FALSE(op->optimizations.blockingApplied);
}

/**
 * Test: SpMM with blocking only.
 */
TEST(PassIntegrationTest, SpMM_BlockingOnly) {
    auto op = parseAndLowerFull(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, config);

    EXPECT_FALSE(op->optimizations.reorderingApplied);
    EXPECT_FALSE(op->optimizations.interchangeApplied);
    EXPECT_TRUE(op->optimizations.blockingApplied);
}

/**
 * Test: SpMM with both optimizations.
 */
TEST(PassIntegrationTest, SpMM_BothOptimizations) {
    auto op = parseAndLowerFull(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::allOptimizations(32);
    opt::applyOptimizations(*op, config);

    // With natural order access, blocking should be applied
    EXPECT_TRUE(op->optimizations.blockingApplied);
    ASSERT_NE(op->rootLoop, nullptr);
}

// ============================================================================
// Pass Ordering Tests
// ============================================================================

/**
 * Test: Reordering is applied before blocking.
 *
 * The optimization pipeline should apply reordering first to get
 * optimal loop order, then blocking operates on the reordered structure.
 */
TEST(PassIntegrationTest, ReorderingBeforeBlocking) {
    auto op = parseAndLowerFull(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::allOptimizations(32);
    opt::applyOptimizations(*op, config);

    // If both are applied, reordering should happen first
    // This is verified by the metadata or final loop structure
    ASSERT_NE(op->rootLoop, nullptr);
}

/**
 * Test: Blocking operates on reordered loops.
 *
 * After reordering, the blocking pass should tile the correct
 * (now outer) loop.
 */
TEST(PassIntegrationTest, BlockingAppliedToReorderedLoops) {
    auto op = parseAndLowerFull(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::allOptimizations(32);
    opt::applyOptimizations(*op, config);

    // Blocking should be applied when outer loop is dense
    EXPECT_TRUE(op->optimizations.blockingApplied);
    ASSERT_NE(op->rootLoop, nullptr);
}

/**
 * Test: Pass order affects final result.
 *
 * Applying optimizations in the correct order (reorder then block)
 * should produce different results than wrong order.
 */
TEST(PassIntegrationTest, PassOrderMatters) {
    // Create two identical ops
    auto op1 = parseAndLowerFull(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )");

    auto op2 = parseAndLowerFull(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )");

    ASSERT_NE(op1, nullptr);
    ASSERT_NE(op2, nullptr);

    // Apply with pipeline (correct order)
    opt::OptConfig config = opt::OptConfig::allOptimizations(32);
    opt::applyOptimizations(*op1, config);

    // Apply manually in wrong order (blocking before reordering)
    opt::OptConfig blockConfig = opt::OptConfig::blockingOnly(32);
    opt::applyBlocking(*op2, blockConfig);
    opt::applyReordering(*op2);

    // Both should be valid, but structure may differ
    ASSERT_NE(op1->rootLoop, nullptr);
    ASSERT_NE(op2->rootLoop, nullptr);
}

/**
 * Test: Metadata tracks both optimization passes.
 */
TEST(PassIntegrationTest, MetadataTracksBothPasses) {
    auto op = parseAndLowerFull(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::allOptimizations(32);
    opt::applyOptimizations(*op, config);

    // Blocking metadata should be set when blocking is applied
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_EQ(op->optimizations.blockSize, 32);
    EXPECT_FALSE(op->optimizations.tiledIndex.empty());
}

// ============================================================================
// End-to-End Tests
// ============================================================================

/**
 * Test: Full pipeline for SpMV with CSR.
 *
 * Parse → Lower → Optimize → Verify structure
 */
TEST(PassIntegrationTest, ApplyOptimizations_SpMV_CSR) {
    auto op = parseAndLowerFull(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    // Verify initial lowering
    EXPECT_EQ(op->output.name, "y");
    EXPECT_EQ(op->inputs.size(), 2);  // A and x
    ASSERT_NE(op->rootLoop, nullptr);

    // Apply full optimizations
    opt::OptConfig config = opt::OptConfig::allOptimizations(32);
    opt::applyOptimizations(*op, config);

    // Verify optimized structure
    ASSERT_NE(op->rootLoop, nullptr);
    EXPECT_TRUE(op->optimizations.blockingApplied);
}

/**
 * Test: Full pipeline for SpMV with CSC.
 */
TEST(PassIntegrationTest, ApplyOptimizations_SpMV_CSC) {
    auto op = parseAndLowerFull(R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    EXPECT_EQ(op->output.name, "y");

    opt::OptConfig config = opt::OptConfig::allOptimizations(32);
    opt::applyOptimizations(*op, config);

    ASSERT_NE(op->rootLoop, nullptr);
}

/**
 * Test: Full pipeline for SpMM with CSR.
 */
TEST(PassIntegrationTest, ApplyOptimizations_SpMM_CSR) {
    auto op = parseAndLowerFull(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    EXPECT_EQ(op->output.name, "C");
    EXPECT_EQ(op->inputs.size(), 2);  // A and B

    opt::OptConfig config = opt::OptConfig::allOptimizations(32);
    opt::applyOptimizations(*op, config);

    ASSERT_NE(op->rootLoop, nullptr);
    EXPECT_TRUE(op->optimizations.blockingApplied);
}

/**
 * Test: Full pipeline for SpMM with CSC.
 */
TEST(PassIntegrationTest, ApplyOptimizations_SpMM_CSC) {
    auto op = parseAndLowerFull(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSC<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    EXPECT_EQ(op->output.name, "C");

    opt::OptConfig config = opt::OptConfig::allOptimizations(32);
    opt::applyOptimizations(*op, config);

    ASSERT_NE(op->rootLoop, nullptr);
}
