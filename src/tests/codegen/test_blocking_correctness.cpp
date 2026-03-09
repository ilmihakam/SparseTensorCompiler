/**
 * Test Suite: Blocking Correctness
 *
 * Tests that verify the semantic correctness of blocking transformation.
 * These tests would have caught the three bugs fixed in the blocking implementation.
 */

#include <gtest/gtest.h>
#include <regex>
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

// ============================================================================
// Bug Regression Tests (verify the three bugs are fixed)
// ============================================================================

/**
 * Test: Block outer loop uses numBlocks, not (blockSize + blockSize - 1) / blockSize
 *
 * BUG 1 (Fixed): Previously used (32 + 31) / 32 = 1, causing only 1 iteration
 * CORRECT: For 100 rows with block size 32, should iterate 4 times
 */
TEST(BlockingCorrectnessTest, Bug1_OuterLoopUsesNumBlocks) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(32));

    ASSERT_NE(op, nullptr);
    std::string output = codegen::generateCode(*op, opt::OptConfig::blockingOnly(32));

    // Should have numBlocks = ceil(M/32)
    // Look for: i_block < 4, or i_block < (100 + 31) / 32, or runtime A->rows
    bool hasCorrectBound =
        output.find("i_block < 4") != std::string::npos ||
        output.find("i_block < (100 + 31) / 32") != std::string::npos ||
        output.find("i_block < (A->rows + 31) / 32") != std::string::npos;

    // Should NOT have the buggy pattern: (32 + 31) / 32
    bool hasBuggyBound = output.find("(32 + 31) / 32") != std::string::npos;

    EXPECT_TRUE(hasCorrectBound) << "Outer loop should iterate 4 times for 100 rows / 32 block size";
    EXPECT_FALSE(hasBuggyBound) << "Should not use (blockSize + blockSize - 1) / blockSize";
}

/**
 * Test: i_end calculation uses original row count (100), not blockSize (32)
 *
 * BUG 2 (Fixed): Previously used (i_start + 32 < 32) which is always false
 * CORRECT: Should use (i_start + 32 < 100) to properly cap the last block
 */
TEST(BlockingCorrectnessTest, Bug2_IEndUsesOriginalUpperBound) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(32));

    ASSERT_NE(op, nullptr);
    std::string output = codegen::generateCode(*op, opt::OptConfig::blockingOnly(32));

    // Should compare against original row count, not block size
    bool hasCorrectEndCalc =
        (output.find("< 100") != std::string::npos ||
         output.find("< A->rows") != std::string::npos) &&
        output.find("_end") != std::string::npos;

    // The ternary should cap at original row count, not block size
    bool capsAt100 =
        output.find(": 100") != std::string::npos ||
        output.find("; 100") != std::string::npos ||
        output.find(": A->rows") != std::string::npos;

    EXPECT_TRUE(hasCorrectEndCalc) << "i_end calculation should compare against original upper bound (100)";
    EXPECT_TRUE(capsAt100) << "i_end should be capped at 100, not at blockSize (32)";
}

/**
 * Test: No variable shadowing - should NOT have nested loops with same variable
 *
 * BUG 3 (Fixed): Previously had:
 *   for (int i = i_start; i < i_end; i++) {
 *       for (int i = 0; i < 100; i++) {  // SHADOWING!
 *
 * CORRECT: Inner loop should be the sparse loop, not another dense i loop
 */
TEST(BlockingCorrectnessTest, Bug3_NoVariableShadowing) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(32));

    ASSERT_NE(op, nullptr);
    std::string output = codegen::generateCode(*op, opt::OptConfig::blockingOnly(32));

    // Extract just the spmv kernel function
    size_t kernel_start = output.find("void compute(");
    ASSERT_NE(kernel_start, std::string::npos) << "Should have compute kernel function";

    size_t kernel_end = output.find("\nvoid reference(", kernel_start);
    ASSERT_NE(kernel_end, std::string::npos) << "Should have reference function after kernel";

    std::string kernel_only = output.substr(kernel_start, kernel_end - kernel_start);

    // Count occurrences of "for (int i" in the kernel only
    std::regex i_loop_pattern(R"(for\s*\(\s*int\s+i\s*=)");
    auto words_begin = std::sregex_iterator(kernel_only.begin(), kernel_only.end(), i_loop_pattern);
    auto words_end = std::sregex_iterator();
    int i_loop_count = std::distance(words_begin, words_end);

    // Should have exactly ONE "for (int i" loop (the bounded inner loop)
    // NOT two nested "for (int i" loops
    EXPECT_EQ(i_loop_count, 1)
        << "Should have exactly ONE 'for (int i' loop in kernel, not nested i loops (variable shadowing)";

    // The inner loop after the blocked i should be the sparse loop (for int p)
    bool hasSparseLoop = kernel_only.find("for (int p") != std::string::npos;
    EXPECT_TRUE(hasSparseLoop) << "Inner loop should be sparse loop (p), not another dense i loop";
}

// ============================================================================
// Semantic Correctness Tests
// ============================================================================

/**
 * Test: Every row from 0 to M-1 is visited exactly once
 *
 * For M=100, block_size=32, we should have:
 * - Block 0: i in [0, 32)
 * - Block 1: i in [32, 64)
 * - Block 2: i in [64, 96)
 * - Block 3: i in [96, 100)  ← Last block is partial
 */
TEST(BlockingCorrectnessTest, AllRowsVisitedExactlyOnce) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(32));

    ASSERT_NE(op, nullptr);
    std::string output = codegen::generateCode(*op, opt::OptConfig::blockingOnly(32));

    // Verify block loop structure
    EXPECT_NE(output.find("i_block"), std::string::npos) << "Should have block loop variable";
    EXPECT_NE(output.find("i_start"), std::string::npos) << "Should calculate i_start";
    EXPECT_NE(output.find("i_end"), std::string::npos) << "Should calculate i_end";

    // Verify bounded inner loop uses i_start and i_end
    bool hasBoundedLoop =
        output.find("for (int i = i_start") != std::string::npos &&
        output.find("i < i_end") != std::string::npos;

    EXPECT_TRUE(hasBoundedLoop) << "Inner loop should iterate from i_start to i_end";
}

/**
 * Test: Blocking preserves the sparse loop structure
 *
 * The sparse iteration (for p in row_ptr[i]..row_ptr[i+1]) should be intact
 */
TEST(BlockingCorrectnessTest, BlockingPreservesSparseIteration) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(32));

    ASSERT_NE(op, nullptr);
    std::string output = codegen::generateCode(*op, opt::OptConfig::blockingOnly(32));

    // Should still have CSR sparse iteration pattern
    bool hasCSRPattern =
        output.find("for (int p") != std::string::npos &&
        output.find("row_ptr[i]") != std::string::npos &&
        output.find("row_ptr[i + 1]") != std::string::npos;

    EXPECT_TRUE(hasCSRPattern) << "Blocking should preserve CSR sparse iteration";

    // Should access col_idx and vals
    EXPECT_NE(output.find("col_idx"), std::string::npos);
    EXPECT_NE(output.find("vals"), std::string::npos);
}

/**
 * Test: Blocking preserves accumulation semantics
 *
 * y[i] += ... should still be present and use the blocked i
 */
TEST(BlockingCorrectnessTest, BlockingPreservesAccumulation) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(32));

    ASSERT_NE(op, nullptr);
    std::string output = codegen::generateCode(*op, opt::OptConfig::blockingOnly(32));

    // Should have y[i] +=  (using the blocked i)
    EXPECT_NE(output.find("y[i] +="), std::string::npos)
        << "Should accumulate into y[i] using blocked i";
}

// ============================================================================
// Different Block Sizes
// ============================================================================

/**
 * Test: Block size 16 generates correct number of blocks
 */
TEST(BlockingCorrectnessTest, BlockSize16_CorrectBounds) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(16));

    ASSERT_NE(op, nullptr);
    std::string output = codegen::generateCode(*op, opt::OptConfig::blockingOnly(16));

    // For 100 rows with block size 16: ceil(100/16) = 7 blocks
    bool hasCorrectBound =
        output.find("i_block < 7") != std::string::npos ||
        output.find("i_block < (100 + 15) / 16") != std::string::npos ||
        output.find("i_block < (A->rows + 15) / 16") != std::string::npos;

    EXPECT_TRUE(hasCorrectBound) << "Should iterate 7 times for 100 rows / 16 block size";

    // Should use 16 as block size
    EXPECT_NE(output.find("16"), std::string::npos);
}

/**
 * Test: Block size 64 generates correct number of blocks
 */
TEST(BlockingCorrectnessTest, BlockSize64_CorrectBounds) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(64));

    ASSERT_NE(op, nullptr);
    std::string output = codegen::generateCode(*op, opt::OptConfig::blockingOnly(64));

    // For 100 rows with block size 64: ceil(100/64) = 2 blocks
    bool hasCorrectBound =
        output.find("i_block < 2") != std::string::npos ||
        output.find("i_block < (100 + 63) / 64") != std::string::npos ||
        output.find("i_block < (A->rows + 63) / 64") != std::string::npos;

    EXPECT_TRUE(hasCorrectBound) << "Should iterate 2 times for 100 rows / 64 block size";
}

// ============================================================================
// Combined Optimizations
// ============================================================================

/**
 * Test: Reordering + Blocking produces correct structure
 */
TEST(BlockingCorrectnessTest, ReorderingPlusBlocking_NoShadowing) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::allOptimizations(32));

    ASSERT_NE(op, nullptr);
    std::string output = codegen::generateCode(*op, opt::OptConfig::allOptimizations(32));

    // Extract just the spmv kernel function
    size_t kernel_start = output.find("void compute(");
    ASSERT_NE(kernel_start, std::string::npos) << "Should have compute kernel function";

    size_t kernel_end = output.find("\nvoid reference(", kernel_start);
    ASSERT_NE(kernel_end, std::string::npos) << "Should have reference function after kernel";

    std::string kernel_only = output.substr(kernel_start, kernel_end - kernel_start);

    // Even with reordering, should not have variable shadowing
    std::regex i_loop_pattern(R"(for\s*\(\s*int\s+i\s*=)");
    auto words_begin = std::sregex_iterator(kernel_only.begin(), kernel_only.end(), i_loop_pattern);
    auto words_end = std::sregex_iterator();
    int i_loop_count = std::distance(words_begin, words_end);

    EXPECT_LE(i_loop_count, 1)
        << "Even with reordering, should not have nested i loops in kernel";
}

/**
 * Test: Metadata correctly records blocking application
 */
TEST(BlockingCorrectnessTest, MetadataRecordsBlocking) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::blockingOnly(32));

    ASSERT_NE(op, nullptr);

    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_EQ(op->optimizations.blockSize, 32);
    EXPECT_EQ(op->optimizations.tiledIndex, "i");
}
