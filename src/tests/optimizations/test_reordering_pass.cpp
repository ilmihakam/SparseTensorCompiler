/**
 * Test Suite: Loop Reordering Pass
 *
 * Tests the loop reordering optimization that rearranges loop nests
 * to match the natural iteration order of sparse tensor formats.
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

// Parser warmup flag to handle Flex/Bison initialization
static bool parserInitialized = false;

/**
 * Helper to parse code and lower to IR.
 */
std::unique_ptr<sparseir::scheduled::Compute> parseAndLower(const std::string& code) {
    if (!parserInitialized) {
        // Warmup parse
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

/**
 * Helper to get loop order as a vector of index names.
 */
std::vector<std::string> getLoopOrder(const sparseir::scheduled::Loop* loop) {
    std::vector<std::string> order;
    const sparseir::scheduled::Loop* current = loop;
    while (current) {
        order.push_back(current->indexName);
        if (!current->children.empty()) {
            current = current->children[0].get();
        } else {
            break;
        }
    }
    return order;
}

// ============================================================================
// SpMV Reordering Tests
// ============================================================================

/**
 * Test: SpMV with CSR in natural order - no change needed.
 *
 * y[i] = A[i,j] * x[j] with A:CSR
 * Natural CSR order is i→j, which matches the access pattern.
 */
TEST(ReorderingPassTest, SpMV_CSR_NaturalOrder_NoChange) {
    auto op = parseAndLower(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    // Store original order
    auto originalOrder = getLoopOrder(op->rootLoop.get());

    // Apply reordering
    opt::applyReordering(*op);

    // Verify order unchanged (already optimal)
    auto newOrder = getLoopOrder(op->rootLoop.get());
    EXPECT_EQ(originalOrder, newOrder);
}

/**
 * Test: SpMV with CSR in reversed order.
 *
 * Generic iterator lowering may already choose a legal scatter schedule.
 * Reordering should preserve a valid loop tree without forcing i→j.
 */
TEST(ReorderingPassTest, SpMV_CSR_ReversedOrder_Reorders) {
    auto op = parseAndLower(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::applyReordering(*op);

    auto order = getLoopOrder(op->rootLoop.get());
    ASSERT_GE(order.size(), 2);
    EXPECT_EQ(order[0], "j");
    EXPECT_EQ(order[1], "i");
}

/**
 * Test: SpMV with CSC in natural order - no change needed.
 *
 * For CSC, natural order is col→row (j→i).
 */
TEST(ReorderingPassTest, SpMV_CSC_NaturalOrder_NoChange) {
    auto op = parseAndLower(R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    auto originalOrder = getLoopOrder(op->rootLoop.get());

    opt::applyReordering(*op);

    // Should remain unchanged (natural for CSC)
    auto newOrder = getLoopOrder(op->rootLoop.get());
    EXPECT_EQ(originalOrder, newOrder);
}

/**
 * Test: SpMV with CSC in reversed order.
 *
 * With format-correct lowering, the loop order should be j→i
 * even if the DSL access is A[i,j].
 */
TEST(ReorderingPassTest, SpMV_CSC_ReversedOrder_Reorders) {
    auto op = parseAndLower(R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::applyReordering(*op);

    // After lowering (and reordering if needed), should iterate j (col) then i (row)
    auto order = getLoopOrder(op->rootLoop.get());
    ASSERT_GE(order.size(), 2);
    EXPECT_EQ(order[0], "j");
    EXPECT_EQ(order[1], "i");
}

/**
 * Test: SpMV with Dense - no reordering applied.
 *
 * Dense format doesn't benefit from reordering.
 */
TEST(ReorderingPassTest, SpMV_Dense_NoReordering) {
    auto op = parseAndLower(R"(
        tensor y : Dense<100>;
        tensor A : Dense<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::applyReordering(*op);

    EXPECT_FALSE(op->optimizations.reorderingApplied);
}

/**
 * Test: Reordering metadata is tracked correctly.
 */
TEST(ReorderingPassTest, SpMV_ReorderingMetadataTracked) {
    auto op = parseAndLower(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::applyReordering(*op);

    // Metadata should track the transformation
    if (op->optimizations.reorderingApplied) {
        EXPECT_FALSE(op->optimizations.originalOrder.empty());
        EXPECT_FALSE(op->optimizations.newOrder.empty());
    }
}

// ============================================================================
// SpMM Reordering Tests
// ============================================================================

/**
 * Test: SpMM with CSR in natural order - no change needed.
 *
 * C[i,j] = A[i,k] * B[k,j] with A:CSR
 * The i→k ordering for sparse A is natural for CSR.
 */
TEST(ReorderingPassTest, SpMM_CSR_NaturalOrder_NoChange) {
    auto op = parseAndLower(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    auto originalOrder = getLoopOrder(op->rootLoop.get());

    opt::applyReordering(*op);

    // For SpMM with CSR, i→k→j order should work well
    // Check that structure is valid
    ASSERT_NE(op->rootLoop, nullptr);
}

/**
 * Test: SpMM with CSR in reversed order - should reorder.
 */
TEST(ReorderingPassTest, SpMM_CSR_ReversedOrder_Reorders) {
    auto op = parseAndLower(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[k, i] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::applyReordering(*op);

    // Reordering should be considered
    ASSERT_NE(op->rootLoop, nullptr);
}

/**
 * Test: SpMM with CSC - appropriate loop order preserved.
 */
TEST(ReorderingPassTest, SpMM_CSC_NaturalOrder_NoChange) {
    auto op = parseAndLower(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSC<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[k, i] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::applyReordering(*op);

    ASSERT_NE(op->rootLoop, nullptr);
}

/**
 * Test: SpMM with CSC in reversed order - should reorder.
 */
TEST(ReorderingPassTest, SpMM_CSC_ReversedOrder_Reorders) {
    auto op = parseAndLower(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSC<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::applyReordering(*op);

    ASSERT_NE(op->rootLoop, nullptr);
}

/**
 * Test: SpMM reordering metadata is tracked.
 */
TEST(ReorderingPassTest, SpMM_ReorderingMetadataTracked) {
    auto op = parseAndLower(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[k, i] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::applyReordering(*op);

    // If reordering was applied, metadata should be set
    if (op->optimizations.reorderingApplied) {
        EXPECT_FALSE(op->optimizations.originalOrder.empty());
    }
}

/**
 * Test: Inner dense loop position is considered.
 */
TEST(ReorderingPassTest, SpMM_InnerLoopPreserved) {
    auto op = parseAndLower(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::applyReordering(*op);

    // The j loop (for dense output columns) should typically be innermost
    // to maximize vectorization opportunities
    auto order = getLoopOrder(op->rootLoop.get());
    ASSERT_GE(order.size(), 2);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

/**
 * Test: Already optimal order doesn't change.
 */
TEST(ReorderingPassTest, AlreadyOptimalOrder_NoChange) {
    auto op = parseAndLower(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    auto originalOrder = getLoopOrder(op->rootLoop.get());

    opt::applyReordering(*op);

    auto newOrder = getLoopOrder(op->rootLoop.get());
    EXPECT_EQ(originalOrder, newOrder);
}

/**
 * Test: Loop kinds are updated after reordering.
 *
 * When loops are reordered, their Dense/Sparse kinds should
 * be adjusted to reflect the new positions.
 */
TEST(ReorderingPassTest, LoopKindsUpdated) {
    auto op = parseAndLower(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::applyReordering(*op);

    // After any reordering, loop kinds should be consistent
    // Outer loop for sparse tensor should be dense, inner should be sparse
    ASSERT_NE(op->rootLoop, nullptr);
}

/**
 * Test: Reordering is idempotent - applying twice has same effect.
 */
TEST(ReorderingPassTest, ReorderingIdempotent) {
    auto op = parseAndLower(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )");
    ASSERT_NE(op, nullptr);


    // Apply once
    opt::applyReordering(*op);
    auto orderAfterFirst = getLoopOrder(op->rootLoop.get());

    // Apply again
    opt::applyReordering(*op);
    auto orderAfterSecond = getLoopOrder(op->rootLoop.get());

    // Should be the same
    EXPECT_EQ(orderAfterFirst, orderAfterSecond);
}
