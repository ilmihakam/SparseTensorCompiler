/**
 * Test Suite: Format Analyzer
 *
 * Tests the format analyzer that determines natural iteration order
 * for tensor formats and whether reordering is needed.
 */

#include <gtest/gtest.h>
#include "optimizations.h"
#include "ir.h"

// ============================================================================
// Natural Order Detection Tests
// ============================================================================

/**
 * Test: CSR natural order is row-outer, col-inner.
 *
 * CSR stores data row-by-row, so iterating row→col is efficient.
 * The outer loop (row) is dense, inner loop (col) iterates over non-zeros.
 */
TEST(FormatAnalyzerTest, CSR_NaturalOrder_RowOuter) {
    auto order = opt::getNaturalOrder(ir::Format::CSR);

    ASSERT_EQ(order.size(), 2);
    EXPECT_EQ(order[0], "row");  // Outer
    EXPECT_EQ(order[1], "col");  // Inner
}

/**
 * Test: CSC natural order is col-outer, row-inner.
 *
 * CSC stores data column-by-column, so iterating col→row is efficient.
 */
TEST(FormatAnalyzerTest, CSC_NaturalOrder_ColOuter) {
    auto order = opt::getNaturalOrder(ir::Format::CSC);

    ASSERT_EQ(order.size(), 2);
    EXPECT_EQ(order[0], "col");  // Outer
    EXPECT_EQ(order[1], "row");  // Inner
}

/**
 * Test: Dense defaults to row-major order.
 *
 * Dense matrices are typically stored in row-major order in C.
 */
TEST(FormatAnalyzerTest, Dense_NaturalOrder_RowMajor) {
    auto order = opt::getNaturalOrder(ir::Format::Dense);

    ASSERT_EQ(order.size(), 2);
    EXPECT_EQ(order[0], "row");
    EXPECT_EQ(order[1], "col");
}

/**
 * Test: For CSR, the outer (row) index iterates densely.
 *
 * In CSR, we iterate through all rows: for (i = 0; i < M; i++)
 */
TEST(FormatAnalyzerTest, CSR_OuterIndexIsDense) {
    EXPECT_TRUE(opt::isOuterIndexDense(ir::Format::CSR));
}

/**
 * Test: For CSR, the inner (col) index iterates sparsely.
 *
 * In CSR, for each row we iterate only non-zeros: for (p = ptr[i]; p < ptr[i+1]; p++)
 */
TEST(FormatAnalyzerTest, CSR_InnerIndexIsSparse) {
    EXPECT_TRUE(opt::isInnerIndexSparse(ir::Format::CSR));
}

/**
 * Test: For CSC, the outer (col) index iterates densely.
 */
TEST(FormatAnalyzerTest, CSC_OuterIndexIsDense) {
    EXPECT_TRUE(opt::isOuterIndexDense(ir::Format::CSC));
}

// ============================================================================
// Order Comparison Tests
// ============================================================================

/**
 * Test: CSR with matching access pattern needs no reorder.
 *
 * A[i,j] accessed with CSR format - row (i) is outer, col (j) is inner.
 * This matches CSR's natural order.
 */
TEST(FormatAnalyzerTest, CSR_MatchingOrder_NoReorderNeeded) {
    ir::Tensor tensor("A", ir::Format::CSR, {100, 100}, {"i", "j"});

    EXPECT_FALSE(opt::needsReordering(tensor));
}

/**
 * Test: CSR with mismatched access pattern needs reorder.
 *
 * A[j,i] accessed with CSR format - col (j) is outer, row (i) is inner.
 * This is opposite to CSR's natural order and will cause cache misses.
 */
TEST(FormatAnalyzerTest, CSR_MismatchedOrder_ReorderNeeded) {
    ir::Tensor tensor("A", ir::Format::CSR, {100, 100}, {"j", "i"});

    EXPECT_TRUE(opt::needsReordering(tensor));
}

/**
 * Test: CSC with matching access pattern needs no reorder.
 *
 * A[j,i] accessed with CSC format - col (j) is outer, row (i) is inner.
 * This matches CSC's natural order.
 */
TEST(FormatAnalyzerTest, CSC_MatchingOrder_NoReorderNeeded) {
    ir::Tensor tensor("A", ir::Format::CSC, {100, 100}, {"j", "i"});

    EXPECT_FALSE(opt::needsReordering(tensor));
}

/**
 * Test: CSC with mismatched access pattern needs reorder.
 *
 * A[i,j] accessed with CSC format - row (i) is outer, col (j) is inner.
 * This is opposite to CSC's natural order.
 */
TEST(FormatAnalyzerTest, CSC_MismatchedOrder_ReorderNeeded) {
    ir::Tensor tensor("A", ir::Format::CSC, {100, 100}, {"i", "j"});

    EXPECT_TRUE(opt::needsReordering(tensor));
}

// ============================================================================
// Format Utilities Tests
// ============================================================================

/**
 * Test: Dense tensors never need reordering.
 *
 * Dense tensors have uniform access patterns, so loop order doesn't
 * significantly affect cache performance.
 */
TEST(FormatAnalyzerTest, Dense_NeverNeedsReordering) {
    ir::Tensor tensorIJ("A", ir::Format::Dense, {100, 100}, {"i", "j"});
    ir::Tensor tensorJI("B", ir::Format::Dense, {100, 100}, {"j", "i"});

    EXPECT_FALSE(opt::needsReordering(tensorIJ));
    EXPECT_FALSE(opt::needsReordering(tensorJI));
}
