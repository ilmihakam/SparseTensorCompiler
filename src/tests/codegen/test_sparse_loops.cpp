/**
 * Test Suite: Sparse Loop Code Generation
 *
 * Tests generation of sparse loop constructs for CSR/CSC formats.
 */

#include <gtest/gtest.h>
#include <sstream>
#include "codegen.h"
#include "ir.h"
#include "optimizations.h"
#include "semantic_ir.h"
#include "ast.h"

extern std::unique_ptr<SparseTensorCompiler::Program> g_program;
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern int yynerrs;

static bool parserInitialized = false;

std::unique_ptr<sparseir::scheduled::Compute> parseForSparseLoops(const std::string& code) {
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
// CSR Sparse Loop Tests
// ============================================================================

/**
 * Test: CSR sparse loop generates pointer-based iteration.
 */
TEST(SparseLoopsTest, CSR_SparseLoop) {
    auto op = parseForSparseLoops(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should contain CSR-style iteration: for (p = row_ptr[i]; p < row_ptr[i+1]; p++)
    EXPECT_NE(output.find("row_ptr"), std::string::npos);
}

/**
 * Test: CSR loop extracts column index.
 */
TEST(SparseLoopsTest, CSR_ColumnIndexExtraction) {
    auto op = parseForSparseLoops(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should extract column index: j = col_idx[p]
    EXPECT_NE(output.find("col_idx"), std::string::npos);
}

/**
 * Test: CSR loop accesses values array.
 */
TEST(SparseLoopsTest, CSR_ValueAccess) {
    auto op = parseForSparseLoops(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should access values: vals[p] or values[p]
    bool hasVals = output.find("vals[") != std::string::npos ||
                   output.find("values[") != std::string::npos;
    EXPECT_TRUE(hasVals);
}

// ============================================================================
// CSC Sparse Loop Tests
// ============================================================================

/**
 * Test: CSC sparse loop generates column-based iteration.
 */
TEST(SparseLoopsTest, CSC_SparseLoop) {
    auto op = parseForSparseLoops(R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should contain CSC-style iteration: col_ptr
    EXPECT_NE(output.find("col_ptr"), std::string::npos);
}

/**
 * Test: CSC loop extracts row index.
 */
TEST(SparseLoopsTest, CSC_RowIndexExtraction) {
    auto op = parseForSparseLoops(R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should extract row index: i = row_idx[p]
    EXPECT_NE(output.find("row_idx"), std::string::npos);
}

/**
 * Test: CSC loop accesses values array.
 */
TEST(SparseLoopsTest, CSC_ValueAccess) {
    auto op = parseForSparseLoops(R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should access values
    bool hasVals = output.find("vals[") != std::string::npos ||
                   output.find("values[") != std::string::npos;
    EXPECT_TRUE(hasVals);
}

// ============================================================================
// Nested Sparse Loop Tests
// ============================================================================

/**
 * Test: Dense outer loop with sparse inner loop.
 */
TEST(SparseLoopsTest, SparseLoopNested) {
    auto op = parseForSparseLoops(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should have dense outer loop (for i) and sparse inner loop (using row_ptr)
    EXPECT_NE(output.find("for ("), std::string::npos);
    EXPECT_NE(output.find("row_ptr"), std::string::npos);
}

/**
 * Test: Sparse loops maintain proper indentation.
 */
TEST(SparseLoopsTest, SparseLoopIndentation) {
    auto op = parseForSparseLoops(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should have indented nested content
    EXPECT_NE(output.find("\n        "), std::string::npos);
}

/**
 * Test: Sparse loop uses pointer variable.
 */
TEST(SparseLoopsTest, SparsePointerVariable) {
    auto op = parseForSparseLoops(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should use p or ptr as pointer variable
    bool hasPointerVar = output.find("int p") != std::string::npos ||
                         output.find("int ptr") != std::string::npos;
    EXPECT_TRUE(hasPointerVar);
}

/**
 * Test: Sparse loop may include comment indicating sparse iteration.
 */
TEST(SparseLoopsTest, SparseIndexComment) {
    auto op = parseForSparseLoops(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should contain some form of comment (// or /* */)
    EXPECT_NE(output.find("//"), std::string::npos);
}

/**
 * Test: CSR accesses struct members properly.
 */
TEST(SparseLoopsTest, CSR_StructAccess) {
    auto op = parseForSparseLoops(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should access via struct pointer: A->row_ptr or A.row_ptr
    bool hasStructAccess = output.find("->row_ptr") != std::string::npos ||
                           output.find(".row_ptr") != std::string::npos;
    EXPECT_TRUE(hasStructAccess);
}

/**
 * Test: CSC accesses struct members properly.
 */
TEST(SparseLoopsTest, CSC_StructAccess) {
    auto op = parseForSparseLoops(R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should access via struct pointer: A->col_ptr or A.col_ptr
    bool hasStructAccess = output.find("->col_ptr") != std::string::npos ||
                           output.find(".col_ptr") != std::string::npos;
    EXPECT_TRUE(hasStructAccess);
}
