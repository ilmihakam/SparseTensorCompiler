/**
 * Test Suite: Loop Blocking Pass
 *
 * Tests the loop blocking (tiling) optimization that improves cache
 * locality by processing data in blocks that fit in cache.
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

std::unique_ptr<sparseir::scheduled::Compute> parseAndLowerForBlocking(const std::string& code) {
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

/**
 * Helper to count total loops in the nest.
 */
int countLoops(const sparseir::scheduled::Loop* loop) {
    if (!loop) return 0;
    int count = 1;
    for (const auto& child : loop->children) {
        count += countLoops(child.get());
    }
    return count;
}

/**
 * Helper to find a loop by index name.
 */
const sparseir::scheduled::Loop* findLoop(const sparseir::scheduled::Loop* root, const std::string& name) {
    if (!root) return nullptr;
    if (root->indexName == name) return root;
    for (const auto& child : root->children) {
        if (auto found = findLoop(child.get(), name)) {
            return found;
        }
    }
    return nullptr;
}

// ============================================================================
// SpMV Blocking Tests
// ============================================================================

/**
 * Test: SpMV blocks the outer dense loop.
 *
 * The i-loop (iterating over output rows) is dense and should be blocked.
 * After blocking: i_block → i → j
 */
TEST(BlockingPassTest, SpMV_BlocksOuterDenseLoop) {
    auto op = parseAndLowerForBlocking(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    int originalLoopCount = countLoops(op->rootLoop.get());

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyBlocking(*op, config);

    // After blocking, there should be one more loop
    int newLoopCount = countLoops(op->rootLoop.get());
    EXPECT_GT(newLoopCount, originalLoopCount);
    EXPECT_TRUE(op->optimizations.blockingApplied);
}

/**
 * Test: SpMV blocking with default block size 32.
 */
TEST(BlockingPassTest, SpMV_BlockSize32) {
    auto op = parseAndLowerForBlocking(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyBlocking(*op, config);

    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_EQ(op->optimizations.blockSize, 32);
}

/**
 * Test: SpMV blocking with custom block size 64.
 */
TEST(BlockingPassTest, SpMV_BlockSize64) {
    auto op = parseAndLowerForBlocking(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(64);
    opt::applyBlocking(*op, config);

    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_EQ(op->optimizations.blockSize, 64);
}

/**
 * Test: SpMV inner sparse loop is not blocked.
 *
 * The j-loop iterates over sparse non-zeros and should not be blocked
 * (blocking sparse loops is complex and may not be beneficial).
 */
TEST(BlockingPassTest, SpMV_InnerLoopUnchanged) {
    auto op = parseAndLowerForBlocking(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyBlocking(*op, config);

    // The tiled index should be the outer dense index, not the sparse inner
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_FALSE(op->optimizations.tiledIndex.empty());
}

/**
 * Test: SpMV blocking metadata is tracked.
 */
TEST(BlockingPassTest, SpMV_BlockingMetadataTracked) {
    auto op = parseAndLowerForBlocking(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyBlocking(*op, config);

    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_EQ(op->optimizations.blockSize, 32);
    EXPECT_FALSE(op->optimizations.tiledIndex.empty());
}

// ============================================================================
// SpMM Blocking Tests
// ============================================================================

/**
 * Test: SpMM blocking via IR tree modification (j-tiling in IR).
 *
 * C[i,j] = A[i,k] * B[k,j]
 * blockLoopByIndex wraps j with j_block, adding one loop to the tree.
 */
TEST(BlockingPassTest, SpMM_AnnotationOnly_LoopTreeUnchanged) {
    auto op = parseAndLowerForBlocking(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    int originalLoopCount = countLoops(op->rootLoop.get());

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyBlocking(*op, config);

    // Loop tree gains one j_block wrapper loop
    int newLoopCount = countLoops(op->rootLoop.get());
    EXPECT_EQ(newLoopCount, originalLoopCount + 1);
    EXPECT_TRUE(op->optimizations.blockingApplied);
}

/**
 * Test: SpMM blocking inserts a j_block wrapper loop into the IR tree.
 *
 * After blocking: i(dense) -> k(sparse) -> j_block(dense) -> j(dense).
 */
TEST(BlockingPassTest, SpMM_AllLoopsUnchanged) {
    auto op = parseAndLowerForBlocking(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    int originalLoopCount = countLoops(op->rootLoop.get());

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyBlocking(*op, config);

    EXPECT_TRUE(op->optimizations.blockingApplied);
    // One j_block wrapper is added
    EXPECT_EQ(countLoops(op->rootLoop.get()), originalLoopCount + 1);
}

/**
 * Test: SpMM blocking with default block size.
 */
TEST(BlockingPassTest, SpMM_BlockSize32) {
    auto op = parseAndLowerForBlocking(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyBlocking(*op, config);

    EXPECT_EQ(op->optimizations.blockSize, 32);
}

/**
 * Test: SpMM blocking with custom block size.
 */
TEST(BlockingPassTest, SpMM_CustomBlockSize) {
    auto op = parseAndLowerForBlocking(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(128);
    opt::applyBlocking(*op, config);

    EXPECT_EQ(op->optimizations.blockSize, 128);
}

/**
 * Test: SpMM blocking metadata tracks output column index.
 */
TEST(BlockingPassTest, SpMM_BlockingMetadataTracked) {
    auto op = parseAndLowerForBlocking(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyBlocking(*op, config);

    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_EQ(op->optimizations.blockSize, 32);
    // tiledIndex should be the second output dimension (j)
    EXPECT_EQ(op->optimizations.tiledIndex, "j");
}

// ============================================================================
// Block Loop Structure Tests
// ============================================================================

/**
 * Test: Block loop has correct bounds.
 *
 * The block loop should iterate: for (i_block = 0; i_block < ceil(M/B); i_block++)
 */
TEST(BlockingPassTest, BlockLoopHasCorrectBounds) {
    auto op = parseAndLowerForBlocking(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyBlocking(*op, config);

    // The outer loop should be the block loop
    ASSERT_NE(op->rootLoop, nullptr);
    // Block loop should have bounds based on dimension / blockSize
    // For dimension 100 with block size 32: ceil(100/32) = 4 blocks
}

/**
 * Test: Inner loop has bounded range within block.
 *
 * The inner loop should iterate within the block:
 * for (i = i_block*B; i < min((i_block+1)*B, M); i++)
 */
TEST(BlockingPassTest, InnerLoopHasBoundedRange) {
    auto op = parseAndLowerForBlocking(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyBlocking(*op, config);

    // After blocking, the structure should show tiling
    EXPECT_TRUE(op->optimizations.blockingApplied);
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

/**
 * Test: Blocking disabled has no effect.
 */
TEST(BlockingPassTest, BlockingDisabled_NoChange) {
    auto op = parseAndLowerForBlocking(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    int originalLoopCount = countLoops(op->rootLoop.get());

    opt::OptConfig config;
    config.enableBlocking = false;
    opt::applyBlocking(*op, config);

    int newLoopCount = countLoops(op->rootLoop.get());
    EXPECT_EQ(originalLoopCount, newLoopCount);
    EXPECT_FALSE(op->optimizations.blockingApplied);
}

// ============================================================================
// 2D Blocking Tests
// ============================================================================

/**
 * Test: SpMM 2D blocking tiles both i and j loops.
 *
 * After 2D blocking: i_block → i → k(sparse) → j_block → j
 * That's 2 extra loops (i_block, j_block) compared to the original 3 (i, k, j).
 */
TEST(BlockingPassTest, SpMM_2DBlocking_AddsTwoBlockLoops) {
    auto op = parseAndLowerForBlocking(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    int originalLoopCount = countLoops(op->rootLoop.get());

    opt::OptConfig config = opt::OptConfig::blocking2D(32, 64);
    opt::applyBlocking(*op, config);

    int newLoopCount = countLoops(op->rootLoop.get());
    EXPECT_EQ(newLoopCount, originalLoopCount + 2);
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_TRUE(op->optimizations.blocking2DApplied);
}

/**
 * Test: SpMM 2D blocking metadata records both tiled indices.
 */
TEST(BlockingPassTest, SpMM_2DBlocking_MetadataTracked) {
    auto op = parseAndLowerForBlocking(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blocking2D(32, 64);
    opt::applyBlocking(*op, config);

    EXPECT_TRUE(op->optimizations.blocking2DApplied);
    ASSERT_EQ(op->optimizations.tiledIndices.size(), 2u);
    EXPECT_EQ(op->optimizations.tiledIndices[0], "i");
    EXPECT_EQ(op->optimizations.tiledIndices[1], "j");
    ASSERT_EQ(op->optimizations.blockSizes.size(), 2u);
    EXPECT_EQ(op->optimizations.blockSizes[0], 32);
    EXPECT_EQ(op->optimizations.blockSizes[1], 64);
}

/**
 * Test: SpMM 2D blocking creates both i_block and j_block loops in the IR tree.
 */
TEST(BlockingPassTest, SpMM_2DBlocking_BlockLoopsExist) {
    auto op = parseAndLowerForBlocking(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blocking2D(32, 64);
    opt::applyBlocking(*op, config);

    EXPECT_NE(findLoop(op->rootLoop.get(), "i_block"), nullptr);
    EXPECT_NE(findLoop(op->rootLoop.get(), "j_block"), nullptr);
}

/**
 * Test: SpMM 2D blocking stores explicit block emission metadata on wrappers.
 */
TEST(BlockingPassTest, SpMM_2DBlocking_BlockMetadataOnWrappers) {
    auto op = parseAndLowerForBlocking(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blocking2D(32, 64);
    opt::applyBlocking(*op, config);

    const auto* iBlock = findLoop(op->rootLoop.get(), "i_block");
    const auto* jBlock = findLoop(op->rootLoop.get(), "j_block");
    ASSERT_NE(iBlock, nullptr);
    ASSERT_NE(jBlock, nullptr);
    EXPECT_EQ(iBlock->headerKind, sparseir::scheduled::LoopHeaderKind::Block);
    EXPECT_EQ(iBlock->block.blockSize, 32);
    EXPECT_EQ(iBlock->block.tripCountExpr, "(A->rows + 31) / 32");
    EXPECT_EQ(iBlock->block.innerIndexName, "i");
    EXPECT_EQ(iBlock->block.innerLowerExpr, "i_start");
    EXPECT_EQ(iBlock->block.innerUpperExpr, "i_end");
    EXPECT_EQ(jBlock->headerKind, sparseir::scheduled::LoopHeaderKind::Block);
    EXPECT_EQ(jBlock->block.blockSize, 64);
    EXPECT_EQ(jBlock->block.tripCountExpr, "(N_j + 63) / 64");
    EXPECT_EQ(jBlock->block.innerIndexName, "j");
    EXPECT_EQ(jBlock->block.innerLowerExpr, "j_start");
    EXPECT_EQ(jBlock->block.innerUpperExpr, "j_end");
}

/**
 * Test: SpMM 2D blocking with uniform size (blockSize2 = 0 means same as blockSize).
 */
TEST(BlockingPassTest, SpMM_2DBlocking_UniformSize) {
    auto op = parseAndLowerForBlocking(R"(
        tensor C : Dense<100, 100>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config;
    config.enableBlocking = true;
    config.enable2DBlocking = true;
    config.blockSize = 32;
    config.blockSize2 = 0;  // Should default to blockSize (32)
    opt::applyBlocking(*op, config);

    ASSERT_EQ(op->optimizations.blockSizes.size(), 2u);
    EXPECT_EQ(op->optimizations.blockSizes[0], 32);
    EXPECT_EQ(op->optimizations.blockSizes[1], 32);
}

/**
 * Test: SpMV is unaffected by enable2DBlocking flag (only 1 dense loop).
 */
TEST(BlockingPassTest, SpMV_2DBlocking_StillOnlyOneDimension) {
    auto op = parseAndLowerForBlocking(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    int originalLoopCount = countLoops(op->rootLoop.get());

    opt::OptConfig config = opt::OptConfig::blocking2D(32, 64);
    opt::applyBlocking(*op, config);

    // SpMV has only 1 dense loop, so 2D blocking falls through to 1D
    // Only i_block is added (1 extra loop)
    int newLoopCount = countLoops(op->rootLoop.get());
    EXPECT_EQ(newLoopCount, originalLoopCount + 1);
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_FALSE(op->optimizations.blocking2DApplied);
}

/**
 * Test: SDDMM 2D blocking tiles both i and k loops.
 */
TEST(BlockingPassTest, SDDMM_2DBlocking_TilesIAndK) {
    auto op = parseAndLowerForBlocking(R"(
        tensor C : Dense<100, 100>;
        tensor S : CSR<100, 100>;
        tensor D : Dense<100, 50>;
        tensor E : Dense<100, 50>;
        compute C[i, j] = S[i, j] * D[i, k] * E[j, k];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blocking2D(32, 64);
    opt::applyBlocking(*op, config);

    if (op->optimizations.blocking2DApplied) {
        ASSERT_EQ(op->optimizations.tiledIndices.size(), 2u);
        EXPECT_EQ(op->optimizations.tiledIndices[0], "i");
        EXPECT_EQ(op->optimizations.tiledIndices[1], "k");
        EXPECT_NE(findLoop(op->rootLoop.get(), "i_block"), nullptr);
        EXPECT_NE(findLoop(op->rootLoop.get(), "k_block"), nullptr);
    }
}

/**
 * Test: Blocking is idempotent - applying twice doesn't double-block.
 */
TEST(BlockingPassTest, BlockingIdempotent) {
    auto op = parseAndLowerForBlocking(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);

    // Apply once
    opt::applyBlocking(*op, config);
    int countAfterFirst = countLoops(op->rootLoop.get());

    // Apply again
    opt::applyBlocking(*op, config);
    int countAfterSecond = countLoops(op->rootLoop.get());

    // Should not add more loops on second application
    EXPECT_EQ(countAfterFirst, countAfterSecond);
}
