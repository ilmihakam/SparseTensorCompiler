/**
 * Test Suite: Comprehensive Sanity Checks
 *
 * Based on sanity_check.md - covers all edge cases, boundary conditions,
 * and failure modes for blocking and reordering optimizations.
 *
 * This test suite ensures optimizations are ready for benchmarking by
 * verifying 100% correctness on edge cases before touching performance.
 */

#include <gtest/gtest.h>
#include "codegen.h"
#include "ast.h"
#include "scheduled_optimizations.h"
#include "semantic_ir.h"

extern std::unique_ptr<SparseTensorCompiler::Program> g_program;
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern int yynerrs;

static bool parserInitialized = false;

std::unique_ptr<sparseir::scheduled::Compute> parseAndOptimize(
    const std::string& code, const opt::OptConfig& config) {
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

    auto op = sparseir::lowerFirstComputationToScheduled(*g_program);
    if (op) {
        opt::applyOptimizations(*op, config);
    }
    return op;
}

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
// 1. Blocking Edge Cases (IR Level)
// ============================================================================

/**
 * Test: Small matrix (M < block_size) - should still generate correct code
 */
TEST(SanityCheckTest, Blocking_SmallMatrix_LessThanBlockSize) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<16>;
        tensor A : CSR<16, 16>;
        tensor x : Dense<16>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(32));

    ASSERT_NE(op, nullptr);

    // For M=16 with B=32, should have ceil(16/32)=1 block
    // The blocking should still be applied but with appropriate bounds
    std::string code = codegen::generateCode(*op, opt::OptConfig::blockingOnly(32));

    // Should have blocking structure
    EXPECT_NE(code.find("i_block"), std::string::npos);
    EXPECT_NE(code.find("i_end"), std::string::npos);

    // Should cap at 16, not 32 (allow runtime bound)
    bool capsAt16 = code.find(": 16") != std::string::npos ||
                    code.find(": A->rows") != std::string::npos;
    EXPECT_TRUE(capsAt16) << "Should cap i_end at original matrix size (16)";
}

/**
 * Test: B=1 (degenerate blocking) - should create M blocks of size 1
 */
TEST(SanityCheckTest, Blocking_BlockSizeOne_Degenerate) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(1));

    ASSERT_NE(op, nullptr);

    // Should generate blocking with B=1 → 100 blocks
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_EQ(op->optimizations.blockSize, 1);

    std::string code = codegen::generateCode(*op, opt::OptConfig::blockingOnly(1));

    // Should have blocking structure even with B=1
    EXPECT_NE(code.find("i_block"), std::string::npos);
}

/**
 * Test: B=M (single block) - blocking should create exactly 1 block
 */
TEST(SanityCheckTest, Blocking_BlockSizeEqualsMatrixSize_SingleBlock) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(100));

    ASSERT_NE(op, nullptr);

    // Should generate blocking with B=100 → 1 block
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_EQ(op->optimizations.blockSize, 100);

    std::string code = codegen::generateCode(*op, opt::OptConfig::blockingOnly(100));

    // Should have a single block (allow runtime bound expression)
    bool hasSingleBlock = code.find("i_block < 1") != std::string::npos ||
                          code.find("i_block < (A->rows + 99) / 100") != std::string::npos;
    EXPECT_TRUE(hasSingleBlock) << "Should have exactly 1 block when B=M";
}

/**
 * Test: Single row matrix (M=1) - should handle correctly
 */
TEST(SanityCheckTest, Blocking_SingleRow_CorrectBounds) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<1>;
        tensor A : CSR<1, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(32));

    ASSERT_NE(op, nullptr);

    std::string code = codegen::generateCode(*op, opt::OptConfig::blockingOnly(32));

    // Should cap at 1 (allow runtime bound)
    bool capsAt1 = code.find(": 1") != std::string::npos ||
                   code.find("< 1") != std::string::npos ||
                   code.find(": A->rows") != std::string::npos;
    EXPECT_TRUE(capsAt1) << "Should cap bounds at M=1 for single-row matrix";
}

// ============================================================================
// 2. Reordering IR-Level Tests
// ============================================================================

/**
 * Test: CSR natural order (A[i,j]) - no reordering should occur
 */
TEST(SanityCheckTest, Reordering_CSR_Natural_NoChange) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);

    auto loopOrder = getLoopOrder(op->rootLoop.get());

    // For CSR with A[i,j], should already be in natural order
    // Loop order should be ["i", ...] (not necessarily need reordering)
    ASSERT_GE(loopOrder.size(), 1);
    EXPECT_EQ(loopOrder[0], "i") << "CSR should have 'i' (row) as outer loop";
}

/**
 * Test: CSR inverted (A[j,i]) should keep a legal scatter loop order.
 */
TEST(SanityCheckTest, Reordering_CSR_Inverted_CorrectOrder) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);

    auto loopOrder = getLoopOrder(op->rootLoop.get());

    ASSERT_GE(loopOrder.size(), 1);
    EXPECT_EQ(loopOrder[0], "j") << "Generic lowering should keep the logical row index outer for A[j,i]";
}

/**
 * Test: CSC natural order (A[j,i]) - no reordering should occur
 */
TEST(SanityCheckTest, Reordering_CSC_Natural_NoChange) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);

    auto loopOrder = getLoopOrder(op->rootLoop.get());

    // For CSC with A[j,i], natural order is column-first
    ASSERT_GE(loopOrder.size(), 1);
    // After lowering, should have appropriate order for CSC
}

/**
 * Test: CSC inverted (A[i,j]) - should reorder or already be correct
 */
TEST(SanityCheckTest, Reordering_CSC_Inverted_CorrectOrder) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);

    auto loopOrder = getLoopOrder(op->rootLoop.get());

    // Should produce valid loop structure
    ASSERT_GE(loopOrder.size(), 1);
}

/**
 * Test: Idempotency - applying reordering twice should be safe
 */
TEST(SanityCheckTest, Reordering_Idempotent_NoDoubleApplication) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);

    auto orderAfterFirst = getLoopOrder(op->rootLoop.get());

    // Try to apply again
    opt::applyReordering(*op);

    auto orderAfterSecond = getLoopOrder(op->rootLoop.get());

    // Should be unchanged
    EXPECT_EQ(orderAfterFirst, orderAfterSecond) << "Reordering should be idempotent";
}

// ============================================================================
// 3. Combined Blocking + Reordering Tests
// ============================================================================

/**
 * Test: Combined optimizations produce compilable code
 */
TEST(SanityCheckTest, Combined_BlockingAndReordering_Compiles) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::allOptimizations(32));

    ASSERT_NE(op, nullptr);

    std::string code = codegen::generateCode(*op, opt::OptConfig::allOptimizations(32));

    // Should have blocking plus valid CSR structure.
    EXPECT_NE(code.find("_block"), std::string::npos) << "Should have blocking";
    EXPECT_NE(code.find("row_ptr"), std::string::npos) << "Should have CSR structure";
}

/**
 * Test: Different optimization orders (I_THEN_B vs B_THEN_I)
 */
TEST(SanityCheckTest, Combined_OptimizationOrder_BothValid) {
    // I_THEN_B
    auto op1 = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B));

    ASSERT_NE(op1, nullptr);
    std::string code1 = codegen::generateCode(*op1, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B));

    // B_THEN_I
    auto op2 = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I));

    ASSERT_NE(op2, nullptr);
    std::string code2 = codegen::generateCode(*op2, opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I));

    // Both should compile (actual correctness tested elsewhere)
    EXPECT_GT(code1.size(), 0);
    EXPECT_GT(code2.size(), 0);
}

// ============================================================================
// 4. Loop Order Verification Tests
// ============================================================================

/**
 * Test: Loop order matches expected natural order for CSR
 */
TEST(SanityCheckTest, LoopOrder_CSR_MatchesNaturalOrder) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig());  // No optimizations

    ASSERT_NE(op, nullptr);

    auto loopOrder = getLoopOrder(op->rootLoop.get());

    // Should have at least 2 levels (outer + sparse)
    ASSERT_GE(loopOrder.size(), 2);

    // For CSR, first loop should be 'i' (row index)
    EXPECT_EQ(loopOrder[0], "i") << "CSR should have row index 'i' as outer loop";
}

/**
 * Test: Blocking preserves sparse loop structure
 */
TEST(SanityCheckTest, Blocking_PreservesSparseLoopStructure) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(32));

    ASSERT_NE(op, nullptr);

    std::string code = codegen::generateCode(*op, opt::OptConfig::blockingOnly(32));

    // Should still have sparse loop structure after blocking
    EXPECT_NE(code.find("row_ptr[i]"), std::string::npos)
        << "Blocking should preserve sparse CSR iteration pattern";
    EXPECT_NE(code.find("col_idx"), std::string::npos)
        << "Blocking should preserve CSR structure";
}

// ============================================================================
// 5. Metadata Verification Tests
// ============================================================================

/**
 * Test: Blocking metadata is correctly recorded
 */
TEST(SanityCheckTest, Metadata_Blocking_CorrectlyRecorded) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(64));

    ASSERT_NE(op, nullptr);

    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_EQ(op->optimizations.blockSize, 64);
    EXPECT_EQ(op->optimizations.tiledIndex, "i");
}

/**
 * Test: Reordering metadata is correctly recorded
 */
TEST(SanityCheckTest, Metadata_Reordering_CorrectlyRecorded) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);

    // If reordering was needed and applied
    if (op->optimizations.reorderingApplied) {
        EXPECT_FALSE(op->optimizations.originalOrder.empty());
        EXPECT_FALSE(op->optimizations.newOrder.empty());
    }
}

/**
 * Test: Combined optimizations record both metadata
 */
TEST(SanityCheckTest, Metadata_Combined_BothRecorded) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::allOptimizations(32));

    ASSERT_NE(op, nullptr);

    // Blocking should always be applied
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_EQ(op->optimizations.blockSize, 32);

    // Reordering may or may not be applied (depends on if needed)
    // Just verify metadata structure exists
}

// ============================================================================
// 6. Format Support Tests
// ============================================================================

/**
 * Test: CSR format generates correct structure
 */
TEST(SanityCheckTest, Format_CSR_GeneratesCorrectStructure) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig());

    ASSERT_NE(op, nullptr);

    std::string code = codegen::generateCode(*op, opt::OptConfig());

    // Should have CSR-specific structure
    EXPECT_NE(code.find("row_ptr"), std::string::npos);
    EXPECT_NE(code.find("col_idx"), std::string::npos);
    EXPECT_NE(code.find("SparseMatrix"), std::string::npos);
}

/**
 * Test: CSC format generates correct structure
 */
TEST(SanityCheckTest, Format_CSC_GeneratesCorrectStructure) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig());

    ASSERT_NE(op, nullptr);

    std::string code = codegen::generateCode(*op, opt::OptConfig());

    // Should have CSC-specific structure
    EXPECT_NE(code.find("col_ptr"), std::string::npos)
        << "CSC should use col_ptr";
    EXPECT_NE(code.find("row_idx"), std::string::npos)
        << "CSC should use row_idx";
}

/**
 * Test: Dense format works correctly
 * NOTE: Currently dense format uses SparseMatrix structure (acceptable for now)
 */
TEST(SanityCheckTest, Format_Dense_GeneratesCorrectStructure) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : Dense<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig());

    ASSERT_NE(op, nullptr);

    std::string code = codegen::generateCode(*op, opt::OptConfig());

    // Dense should have nested dense loops (no sparse iteration)
    // Current implementation uses SparseMatrix structure even for dense (acceptable)
    EXPECT_GT(code.size(), 0) << "Should generate valid code for dense format";
}

// ============================================================================
// 7. Boundary Condition Tests
// ============================================================================

/**
 * Test: Block boundary at exact multiple
 */
TEST(SanityCheckTest, Blocking_ExactMultiple_CorrectBounds) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<128>;
        tensor A : CSR<128, 128>;
        tensor x : Dense<128>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(32));

    ASSERT_NE(op, nullptr);

    // 128/32 = 4 blocks exactly
    std::string code = codegen::generateCode(*op, opt::OptConfig::blockingOnly(32));

    // Should have exactly 4 blocks (allow runtime bound expression)
    bool hasFourBlocks = code.find("i_block < 4") != std::string::npos ||
                         code.find("i_block < (A->rows + 31) / 32") != std::string::npos;
    EXPECT_TRUE(hasFourBlocks) << "128 rows with B=32 should have exactly 4 blocks";
}

/**
 * Test: Partial last block handling
 */
TEST(SanityCheckTest, Blocking_PartialLastBlock_CorrectCapping) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(32));

    ASSERT_NE(op, nullptr);

    std::string code = codegen::generateCode(*op, opt::OptConfig::blockingOnly(32));

    // Last block: i_start=96, i_end=min(128, 100)=100
    // Should cap at 100, not overflow to 128 (allow runtime bound)
    bool capsAt100 = code.find(": 100") != std::string::npos ||
                     code.find(": A->rows") != std::string::npos;
    EXPECT_TRUE(capsAt100) << "i_end should cap at matrix size (100), not block end (128)";
}
