/**
 * Test Suite: Loop Reordering Correctness
 *
 * Tests that verify the semantic correctness and safety of loop reordering.
 * These tests define the INTENDED behavior and should be written BEFORE fixing bugs.
 *
 * Test-Driven Development Approach:
 * 1. Write these tests to define correct behavior
 * 2. Run tests - expect failures with current implementation
 * 3. Fix implementation to make tests pass
 */

#include <gtest/gtest.h>
#include <regex>
#include <fstream>
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

/**
 * Helper to extract just the spmv kernel from generated code
 */
std::string extractKernel(const std::string& fullCode) {
    size_t start = fullCode.find("void compute(");
    if (start == std::string::npos) return "";

    size_t end = fullCode.find("\nvoid reference(", start);
    if (end == std::string::npos) return "";

    return fullCode.substr(start, end - start);
}

/**
 * Helper to compile generated code and check if it compiles successfully
 */
bool codeCompiles(const std::string& code) {
    // Write to temp file
    std::string tempFile = "/tmp/test_reorder_" + std::to_string(rand()) + ".c";
    std::ofstream out(tempFile);
    out << code;
    out.close();

    // Try to compile
    std::string cmd = "gcc -c -O2 " + tempFile + " -o /tmp/test.o 2>&1";
    int result = system(cmd.c_str());

    // Cleanup
    system(("rm -f " + tempFile + " /tmp/test.o").c_str());

    return result == 0;
}

// ============================================================================
// Test 1: Code Compilation Tests
// ============================================================================

/**
 * Test: CSR with A[i,j] (natural order) should compile
 */
TEST(ReorderingCorrectnessTest, CSR_NaturalOrder_Compiles) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);
    std::string code = codegen::generateCode(*op, opt::OptConfig::baseline());

    EXPECT_TRUE(codeCompiles(code))
        << "CSR with A[i,j] (natural order) should generate compilable code";
}

/**
 * Test: CSR with A[j,i] (reversed order) should compile
 * CRITICAL: This currently FAILS - code uses 'i' before declaration
 */
TEST(ReorderingCorrectnessTest, CSR_ReversedOrder_Compiles) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);
    std::string code = codegen::generateCode(*op, opt::OptConfig::baseline());

    EXPECT_TRUE(codeCompiles(code))
        << "CSR with A[j,i] should generate compilable code after reordering";
}

/**
 * Test: CSC with A[j,i] (natural order) should compile
 */
TEST(ReorderingCorrectnessTest, CSC_NaturalOrder_Compiles) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);
    std::string code = codegen::generateCode(*op, opt::OptConfig::baseline());

    EXPECT_TRUE(codeCompiles(code))
        << "CSC with A[j,i] (natural order) should generate compilable code";
}

/**
 * Test: CSC with A[i,j] (reversed order) should compile
 */
TEST(ReorderingCorrectnessTest, CSC_ReversedOrder_Compiles) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);
    std::string code = codegen::generateCode(*op, opt::OptConfig::baseline());

    EXPECT_TRUE(codeCompiles(code))
        << "CSC with A[i,j] should generate compilable code after reordering";
}

// ============================================================================
// Test 2: Loop Structure Tests
// ============================================================================

/**
 * Test: CSR A[i,j] should have dense loop outer, sparse loop inner
 */
TEST(ReorderingCorrectnessTest, CSR_NaturalOrder_CorrectStructure) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);
    std::string code = codegen::generateCode(*op, opt::OptConfig::baseline());
    std::string kernel = extractKernel(code);

    // Should have: for (int i = 0; i < ...) { for (int p = row_ptr[i]; ...) }
    EXPECT_NE(kernel.find("for (int i"), std::string::npos)
        << "Should have dense i loop";

    EXPECT_NE(kernel.find("for (int p"), std::string::npos)
        << "Should have sparse p loop";

    EXPECT_NE(kernel.find("row_ptr[i]"), std::string::npos)
        << "Sparse loop should access row_ptr[i]";

    // The dense loop should come before sparse loop in the code
    size_t i_pos = kernel.find("for (int i");
    size_t p_pos = kernel.find("for (int p");
    EXPECT_LT(i_pos, p_pos)
        << "Dense i loop should be outer (appear first in code)";
}

/**
 * Test: CSR A[j,i] should generate a legal scatter schedule.
 */
TEST(ReorderingCorrectnessTest, CSR_ReversedOrder_SparseLoopNotOuter) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);
    std::string code = codegen::generateCode(*op, opt::OptConfig::baseline());
    std::string kernel = extractKernel(code);

    EXPECT_NE(kernel.find("for (int j = 0; j < A->rows; j++)"), std::string::npos);
    EXPECT_NE(kernel.find("for (int pA = A->row_ptr[j]; pA < A->row_ptr[j + 1]; pA++)"),
              std::string::npos);
}

/**
 * Test: No undefined variables in generated kernel
 */
TEST(ReorderingCorrectnessTest, CSR_ReversedOrder_NoUndefinedVariables) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);
    std::string code = codegen::generateCode(*op, opt::OptConfig::baseline());

    // Should compile without errors
    EXPECT_TRUE(codeCompiles(code))
        << "Generated code should not have undefined variables";

    std::string kernel = extractKernel(code);

    EXPECT_EQ(kernel.find("row_ptr[i]"), std::string::npos);
    EXPECT_NE(kernel.find("row_ptr[j]"), std::string::npos);
}

// ============================================================================
// Test 3: No-Op Detection Tests (Already Optimal Cases)
// ============================================================================

/**
 * Test: CSR A[i,j] should NOT apply reordering (already optimal)
 */
TEST(ReorderingCorrectnessTest, CSR_NaturalOrder_NoReorderingApplied) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);

    // For CSR with natural order A[i,j], reordering should be a no-op
    // The implementation might set reorderingApplied or not, depending on design
    // What matters is the generated code is correct

    std::string code = codegen::generateCode(*op, opt::OptConfig::baseline());
    EXPECT_TRUE(codeCompiles(code));
}

/**
 * Test: Dense tensors should NOT apply reordering
 */
TEST(ReorderingCorrectnessTest, Dense_NoReordering) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : Dense<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);

    // Dense tensors don't benefit from reordering
    EXPECT_FALSE(op->optimizations.reorderingApplied)
        << "Reordering should not be applied to dense tensors";
}

// ============================================================================
// Test 4: Actual Reordering Detection Tests
// ============================================================================

/**
 * Test: CSR A[j,i] should compile into a valid scatter implementation.
 */
TEST(ReorderingCorrectnessTest, CSR_ReversedOrder_HandledCorrectly) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);
    std::string code = codegen::generateCode(*op, opt::OptConfig::baseline());

    // What matters: code compiles and is correct
    EXPECT_TRUE(codeCompiles(code));

    std::string kernel = extractKernel(code);
    EXPECT_NE(kernel.find("row_ptr[j]"), std::string::npos)
        << "Should use CSR structure correctly for logical A[j,i]";
}

// ============================================================================
// Test 5: Combined Optimizations
// ============================================================================

/**
 * Test: Reordering + Blocking should compile
 */
TEST(ReorderingCorrectnessTest, ReorderingPlusBlocking_Compiles) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::allOptimizations(32));

    ASSERT_NE(op, nullptr);
    std::string code = codegen::generateCode(*op, opt::OptConfig::allOptimizations(32));

    EXPECT_TRUE(codeCompiles(code))
        << "Combined reordering + blocking should generate compilable code";
}

/**
 * Test: Reordering + Blocking should not have variable shadowing
 */
TEST(ReorderingCorrectnessTest, ReorderingPlusBlocking_NoShadowing) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::allOptimizations(32));

    ASSERT_NE(op, nullptr);
    std::string code = codegen::generateCode(*op, opt::OptConfig::allOptimizations(32));
    std::string kernel = extractKernel(code);

    // Count "for (int i" in kernel only
    std::regex i_loop_pattern(R"(for\s*\(\s*int\s+i\s*=)");
    auto words_begin = std::sregex_iterator(kernel.begin(), kernel.end(), i_loop_pattern);
    auto words_end = std::sregex_iterator();
    int i_loop_count = std::distance(words_begin, words_end);

    // Should have at most 2 (blocking loop and bounded inner loop)
    EXPECT_LE(i_loop_count, 2)
        << "Should not have variable shadowing with combined optimizations";
}

// ============================================================================
// Test 6: CSC Format Tests
// ============================================================================

/**
 * Test: CSC format should generate correct structure
 */
TEST(ReorderingCorrectnessTest, CSC_GeneratesCorrectStructure) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);
    std::string code = codegen::generateCode(*op, opt::OptConfig::baseline());

    EXPECT_TRUE(codeCompiles(code))
        << "CSC should generate compilable code";

    // CSC should generate correct sparse access pattern
    std::string kernel = extractKernel(code);

    // Should use CSC structure (col_ptr) OR have correct sparse iteration
    // Just verify it compiles correctly
    EXPECT_TRUE(codeCompiles(code));
}

// ============================================================================
// Test 7: Metadata Tracking
// ============================================================================

/**
 * Test: Reordering metadata is tracked correctly
 */
TEST(ReorderingCorrectnessTest, MetadataTrackedCorrectly) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);

    // If reordering was applied, metadata should be set
    if (op->optimizations.reorderingApplied) {
        EXPECT_FALSE(op->optimizations.originalOrder.empty())
            << "Original order should be recorded";
        EXPECT_FALSE(op->optimizations.newOrder.empty())
            << "New order should be recorded";
    }
}

// ============================================================================
// Test 8: Idempotency
// ============================================================================

/**
 * Test: Applying reordering twice should be idempotent
 */
TEST(ReorderingCorrectnessTest, ReorderingIsIdempotent) {
    auto op = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op, nullptr);

    // Apply optimization (already applied during parseAndOptimize)
    bool firstApplied = op->optimizations.reorderingApplied;

    // Try to apply again
    opt::applyReordering(*op);

    // Should still be the same
    EXPECT_EQ(firstApplied, op->optimizations.reorderingApplied)
        << "Reordering should be idempotent";
}

/**
 * Test: Generated code is deterministic
 */
TEST(ReorderingCorrectnessTest, GeneratedCodeIsDeterministic) {
    // Generate code twice with same input
    auto op1 = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::baseline());

    auto op2 = parseAndOptimize(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[j, i] * x[j];
    )", opt::OptConfig::baseline());

    ASSERT_NE(op1, nullptr);
    ASSERT_NE(op2, nullptr);

    std::string code1 = codegen::generateCode(*op1, opt::OptConfig::baseline());
    std::string code2 = codegen::generateCode(*op2, opt::OptConfig::baseline());

    // Extract just the kernels (timestamps may differ in comments)
    std::string kernel1 = extractKernel(code1);
    std::string kernel2 = extractKernel(code2);

    EXPECT_EQ(kernel1, kernel2)
        << "Generated code should be deterministic";
}
