/**
 * Test Suite: Dense Loop Code Generation
 *
 * Tests generation of standard dense loop constructs.
 */

#include <gtest/gtest.h>
#include <sstream>
#include "codegen.h"
#include "ir.h"
#include "ast.h"
#include "optimizations.h"
#include "semantic_ir.h"

extern std::unique_ptr<SparseTensorCompiler::Program> g_program;
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern int yynerrs;

static bool parserInitialized = false;

std::string generateForDenseLoops(const std::string& code) {
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
        return {};
    }

    opt::OptConfig config = opt::OptConfig::baseline();
    auto scheduled = sparseir::lowerFirstComputationToScheduled(*g_program);
    if (!scheduled) {
        return {};
    }
    return codegen::generateCode(*scheduled, config);
}

// ============================================================================
// Basic Dense Loop Tests
// ============================================================================

/**
 * Test: Simple dense loop generates for-loop syntax.
 */
TEST(DenseLoopsTest, SimpleDenseLoop) {
    auto output = generateForDenseLoops(R"(
        tensor y : Dense<100>;
        tensor A : Dense<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_FALSE(output.empty());

    // Should contain for-loop with proper syntax
    EXPECT_NE(output.find("for ("), std::string::npos);
    EXPECT_NE(output.find("int i"), std::string::npos);
}

/**
 * Test: Dense loop with bounds.
 */
TEST(DenseLoopsTest, DenseLoopWithBounds) {
    auto output = generateForDenseLoops(R"(
        tensor y : Dense<50>;
        tensor A : Dense<50, 50>;
        tensor x : Dense<50>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_FALSE(output.empty());

    // Should contain dimension in bounds
    EXPECT_NE(output.find("for ("), std::string::npos);
}

/**
 * Test: Nested dense loops for matrix-vector multiply.
 */
TEST(DenseLoopsTest, NestedDenseLoops) {
    auto output = generateForDenseLoops(R"(
        tensor y : Dense<100>;
        tensor A : Dense<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_FALSE(output.empty());

    // Should have nested for loops
    size_t first_for = output.find("for (");
    ASSERT_NE(first_for, std::string::npos);

    size_t second_for = output.find("for (", first_for + 1);
    EXPECT_NE(second_for, std::string::npos);
}

/**
 * Test: Triple nested dense loops for matrix-matrix multiply.
 */
TEST(DenseLoopsTest, TripleNestedDenseLoops) {
    auto output = generateForDenseLoops(R"(
        tensor C : Dense<100, 100>;
        tensor A : Dense<100, 50>;
        tensor B : Dense<50, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_FALSE(output.empty());

    // Should have three nested for loops
    size_t first_for = output.find("for (");
    ASSERT_NE(first_for, std::string::npos);

    size_t second_for = output.find("for (", first_for + 1);
    ASSERT_NE(second_for, std::string::npos);

    size_t third_for = output.find("for (", second_for + 1);
    EXPECT_NE(third_for, std::string::npos);
}

/**
 * Test: Dense loop uses proper indentation.
 */
TEST(DenseLoopsTest, DenseLoopIndentation) {
    auto output = generateForDenseLoops(R"(
        tensor y : Dense<100>;
        tensor A : Dense<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_FALSE(output.empty());

    // Should contain properly indented code (spaces at start of lines)
    EXPECT_NE(output.find("\n    "), std::string::npos);
}

/**
 * Test: Dense loop with body contains assignment.
 */
TEST(DenseLoopsTest, DenseLoopWithBody) {
    auto output = generateForDenseLoops(R"(
        tensor y : Dense<100>;
        tensor A : Dense<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_FALSE(output.empty());

    // Should contain accumulation pattern
    EXPECT_NE(output.find("+="), std::string::npos);
}

/**
 * Test: Dense loop variable naming uses IR index names.
 */
TEST(DenseLoopsTest, DenseLoopVariableNaming) {
    auto output = generateForDenseLoops(R"(
        tensor y : Dense<100>;
        tensor A : Dense<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_FALSE(output.empty());

    // Should use i and j as loop variable names
    EXPECT_NE(output.find("int i"), std::string::npos);
    EXPECT_NE(output.find("int j"), std::string::npos);
}

/**
 * Test: Dense loop uses consistent brace placement.
 */
TEST(DenseLoopsTest, DenseLoopBracePlacement) {
    auto output = generateForDenseLoops(R"(
        tensor y : Dense<100>;
        tensor A : Dense<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_FALSE(output.empty());

    // Should have opening braces after for statements
    EXPECT_NE(output.find(") {"), std::string::npos);

    // Should have closing braces
    EXPECT_NE(output.find("}"), std::string::npos);
}
