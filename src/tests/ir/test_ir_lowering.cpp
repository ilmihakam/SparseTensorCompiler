/**
 * Test Suite: AST to IR Lowering
 *
 * Tests the lowering of parsed AST nodes into IR Operations.
 * This is the core transformation that converts high-level DSL
 * computations into structured loop nests.
 */

#include <gtest/gtest.h>
#include "ast.h"
#include "ir.h"
#include "optimizations.h"
#include "semantic_ir.h"

using namespace SparseTensorCompiler;

// Forward declaration - we'll need the parser to create ASTs
extern std::unique_ptr<Program> g_program;
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern int yynerrs;

// Global flag to track if parser has been initialized
static bool parserInitialized = false;

// Helper to parse DSL code and return the AST
std::unique_ptr<Program> parseCode(const std::string& code) {
    // First-time initialization warmup
    if (!parserInitialized) {
        yynerrs = 0;
        g_program.reset();
        yy_scan_string("tensor x : Dense;");
        yyparse();
        yylex_destroy();
        g_program.reset();
        parserInitialized = true;
    }

    // Reset state before parsing
    yynerrs = 0;
    g_program.reset();

    // Parse the code
    yy_scan_string(code.c_str());
    int result = yyparse();
    yylex_destroy();

    // Return AST if successful
    if (result == 0 && yynerrs == 0) {
        return std::move(g_program);
    }
    return nullptr;
}

std::unique_ptr<sparseir::scheduled::Compute> lowerFirstScheduled(
    const std::string& code) {
    auto ast = parseCode(code);
    if (!ast) return nullptr;
    return sparseir::lowerFirstComputationToScheduled(*ast);
}

std::unique_ptr<sparseir::scheduled::Program> lowerScheduledProgram(
    const std::string& code) {
    auto ast = parseCode(code);
    if (!ast) return nullptr;
    auto semantic = sparseir::lowerToSemanticProgram(*ast);
    if (!semantic) return nullptr;
    return sparseir::scheduleProgram(*semantic);
}

void collectScheduledComputes(
    const std::vector<std::unique_ptr<sparseir::scheduled::Stmt>>& stmts,
    std::vector<const sparseir::scheduled::Compute*>& out) {
    for (const auto& stmt : stmts) {
        if (auto* compute = dynamic_cast<sparseir::scheduled::Compute*>(stmt.get())) {
            out.push_back(compute);
        } else if (auto* region = dynamic_cast<sparseir::scheduled::Region*>(stmt.get())) {
            collectScheduledComputes(region->body, out);
        }
    }
}

// Test fixture for IR lowering tests
class IRLoweringTest : public ::testing::Test {
protected:
    void TearDown() override {
        g_program.reset();
    }
};

// ============================================================================
// SpMV Lowering Tests
// ============================================================================

/**
 * Test: Lower basic SpMV computation.
 *
 * DSL Input:
 *   tensor y : Dense<100>;
 *   tensor A : CSR<100, 50>;
 *   tensor x : Dense<50>;
 *   compute y[i] = A[i, j] * x[j];
 *
 * Expected scheduled IR:
 *   output: y (Dense)
 *   inputs: [A (CSR), x (Dense)]
 *   rootLoop: i (dense) -> j (sparse) -> body
 */
TEST_F(IRLoweringTest, SpMV_Basic_CSR) {
    const char* code = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        compute y[i] = A[i, j] * x[j];
    )";

    auto ast = parseCode(code);
    ASSERT_NE(ast, nullptr);

    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);
    ASSERT_NE(compute, nullptr);

    // Verify output tensor
    EXPECT_EQ(compute->output.name, "y");
    EXPECT_EQ(compute->output.format, ir::Format::Dense);

    // Verify input tensors
    ASSERT_EQ(compute->inputs.size(), 2);
    EXPECT_EQ(compute->inputs[0].name, "A");
    EXPECT_EQ(compute->inputs[0].format, ir::Format::CSR);
    EXPECT_EQ(compute->inputs[1].name, "x");
    EXPECT_EQ(compute->inputs[1].format, ir::Format::Dense);

    // Verify loop structure
    ASSERT_NE(compute->rootLoop, nullptr);

    // Outer loop: i (dense, iterates over rows)
    EXPECT_EQ(compute->rootLoop->indexName, "i");
    EXPECT_EQ(compute->rootLoop->kind, sparseir::scheduled::LoopKind::Dense);

    // Inner loop: j (sparse, iterates over non-zeros in row)
    ASSERT_EQ(compute->rootLoop->children.size(), 1);
    EXPECT_EQ(compute->rootLoop->children[0]->indexName, "j");
    EXPECT_EQ(compute->rootLoop->children[0]->kind, sparseir::scheduled::LoopKind::Sparse);
}

/**
 * Test: SpMV with CSC matrix (different iteration pattern).
 *
 * With CSC, the natural iteration is column-outer, row-inner.
 * This tests that format affects loop structure.
 *
 * DSL Input:
 *   tensor y : Dense<100>;
 *   tensor A : CSC<100, 50>;
 *   tensor x : Dense<50>;
 *   compute y[i] = A[i, j] * x[j];
 */
TEST_F(IRLoweringTest, SpMV_CSC_Format) {
    const char* code = R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 50>;
        tensor x : Dense<50>;
        compute y[i] = A[i, j] * x[j];
    )";

    auto ast = parseCode(code);
    ASSERT_NE(ast, nullptr);

    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);
    ASSERT_NE(compute, nullptr);

    // With CSC format, the sparse tensor's format should be recorded
    EXPECT_EQ(compute->inputs[0].format, ir::Format::CSC);

    // Loop structure should still be built based on DSL index order
    // (optimization passes will reorder later if needed)
    ASSERT_NE(compute->rootLoop, nullptr);
}

/**
 * Test: SpMV with reversed index order in DSL.
 *
 * DSL Input:
 *   tensor y : Dense<100>;
 *   tensor A : CSR<100, 50>;
 *   tensor x : Dense<50>;
 *   compute y[i] = A[j, i] * x[j];
 *
 * Note: A[j, i] means the user wrote the indices in j, i order.
 * This affects how the loop nest is initially constructed.
 */
TEST_F(IRLoweringTest, SpMV_ReversedIndexOrder) {
    const char* code = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        compute y[i] = A[j, i] * x[j];
    )";

    auto ast = parseCode(code);
    ASSERT_NE(ast, nullptr);

    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);
    ASSERT_NE(compute, nullptr);

    // The tensor should record the index order as written
    EXPECT_EQ(compute->inputs[0].indices[0], "j");
    EXPECT_EQ(compute->inputs[0].indices[1], "i");
}

// ============================================================================
// SpMM Lowering Tests
// ============================================================================

/**
 * Test: Lower basic SpMM computation.
 *
 * DSL Input:
 *   tensor C : Dense<100, 20>;
 *   tensor A : CSR<100, 50>;
 *   tensor B : Dense<50, 20>;
 *   compute C[i, j] = A[i, k] * B[k, j];
 *
 * Expected scheduled IR:
 *   output: C (Dense 2D)
 *   inputs: [A (CSR), B (Dense)]
 *   rootLoop: i (dense) -> k (sparse) -> j (dense) -> body
 */
TEST_F(IRLoweringTest, SpMM_Basic_CSR) {
    const char* code = R"(
        tensor C : Dense<100, 20>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 20>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    auto ast = parseCode(code);
    ASSERT_NE(ast, nullptr);

    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);
    ASSERT_NE(compute, nullptr);

    // Verify output tensor
    EXPECT_EQ(compute->output.name, "C");
    EXPECT_EQ(compute->output.format, ir::Format::Dense);
    EXPECT_EQ(compute->output.dims.size(), 2);

    // Verify input tensors
    ASSERT_EQ(compute->inputs.size(), 2);
    EXPECT_EQ(compute->inputs[0].name, "A");
    EXPECT_EQ(compute->inputs[0].format, ir::Format::CSR);
    EXPECT_EQ(compute->inputs[1].name, "B");
    EXPECT_EQ(compute->inputs[1].format, ir::Format::Dense);

    // Verify loop structure: i -> k -> j
    ASSERT_NE(compute->rootLoop, nullptr);

    // Outer loop: i (dense)
    EXPECT_EQ(compute->rootLoop->indexName, "i");
    EXPECT_EQ(compute->rootLoop->kind, sparseir::scheduled::LoopKind::Dense);

    // Middle loop: k (sparse - from CSR iteration)
    ASSERT_EQ(compute->rootLoop->children.size(), 1);
    auto* k_loop = compute->rootLoop->children[0].get();
    EXPECT_EQ(k_loop->indexName, "k");
    EXPECT_EQ(k_loop->kind, sparseir::scheduled::LoopKind::Sparse);

    // Inner loop: j (dense - iterating over B's columns)
    ASSERT_EQ(k_loop->children.size(), 1);
    auto* j_loop = k_loop->children[0].get();
    EXPECT_EQ(j_loop->indexName, "j");
    EXPECT_EQ(j_loop->kind, sparseir::scheduled::LoopKind::Dense);
}

/**
 * Test: SpMM with CSC sparse matrix.
 */
TEST_F(IRLoweringTest, SpMM_CSC_Format) {
    const char* code = R"(
        tensor C : Dense<100, 20>;
        tensor A : CSC<100, 50>;
        tensor B : Dense<50, 20>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    auto ast = parseCode(code);
    ASSERT_NE(ast, nullptr);

    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);
    ASSERT_NE(compute, nullptr);

    EXPECT_EQ(compute->inputs[0].format, ir::Format::CSC);
}

// ============================================================================
// Index Classification Tests
// ============================================================================

/**
 * Test: Identify free indices (appear on LHS).
 *
 * In compute y[i] = A[i,j] * x[j]:
 *   - i is a FREE index (appears in output y[i])
 *   - j is a BOUND index (only on RHS, summed over)
 */
TEST_F(IRLoweringTest, IndexClassification_FreeIndices) {
    const char* code = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        compute y[i] = A[i, j] * x[j];
    )";

    auto ast = parseCode(code);
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    // Check that output tensor has the free index
    EXPECT_EQ(compute->output.indices.size(), 1);
    EXPECT_EQ(compute->output.indices[0], "i");
}

/**
 * Test: Identify bound indices (summation indices).
 *
 * In compute C[i,j] = A[i,k] * B[k,j]:
 *   - i, j are FREE indices (appear in output C[i,j])
 *   - k is a BOUND index (summed over, Einstein convention)
 */
TEST_F(IRLoweringTest, IndexClassification_BoundIndices_SpMM) {
    const char* code = R"(
        tensor C : Dense<100, 20>;
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 20>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    auto ast = parseCode(code);
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    // Output has free indices i, j
    EXPECT_EQ(compute->output.indices.size(), 2);
    EXPECT_EQ(compute->output.indices[0], "i");
    EXPECT_EQ(compute->output.indices[1], "j");

    // k appears in A and B but not in output - it's the summation index
    // The middle loop should be over k
    auto* k_loop = compute->rootLoop->children[0].get();
    EXPECT_EQ(k_loop->indexName, "k");
}

/**
 * Test: Sparse vs Dense index classification based on tensor format.
 *
 * For CSR matrix A[i,j]:
 *   - i (row index) iterates densely over all rows
 *   - j (column index) iterates sparsely over non-zeros
 */
TEST_F(IRLoweringTest, SparseVsDenseClassification_CSR) {
    const char* code = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        compute y[i] = A[i, j] * x[j];
    )";

    auto ast = parseCode(code);
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    // i loop is dense (row iteration)
    EXPECT_EQ(compute->rootLoop->kind, sparseir::scheduled::LoopKind::Dense);

    // j loop is sparse (column iteration through CSR)
    EXPECT_EQ(compute->rootLoop->children[0]->kind, sparseir::scheduled::LoopKind::Sparse);
}

// ============================================================================
// Dimension Extraction Tests
// ============================================================================

/**
 * Test: Extract dimensions from tensor declarations.
 */
TEST_F(IRLoweringTest, DimensionExtraction) {
    const char* code = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        compute y[i] = A[i, j] * x[j];
    )";

    auto ast = parseCode(code);
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    // Check dimensions were extracted correctly
    EXPECT_EQ(compute->output.dims.size(), 1);
    EXPECT_EQ(compute->output.dims[0], 100);

    EXPECT_EQ(compute->inputs[0].dims.size(), 2);
    EXPECT_EQ(compute->inputs[0].dims[0], 100);
    EXPECT_EQ(compute->inputs[0].dims[1], 50);

    EXPECT_EQ(compute->inputs[1].dims.size(), 1);
    EXPECT_EQ(compute->inputs[1].dims[0], 50);
}

/**
 * Test: Loop bounds derived from dimensions.
 */
TEST_F(IRLoweringTest, LoopBoundsFromDimensions) {
    const char* code = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        compute y[i] = A[i, j] * x[j];
    )";

    auto ast = parseCode(code);
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    // i loop: 0 to 100 (rows of A / size of y)
    EXPECT_EQ(compute->rootLoop->lower, 0);
    EXPECT_EQ(compute->rootLoop->upper, 100);

    // j loop: sparse, but logical upper bound is 50 (columns of A / size of x)
    EXPECT_EQ(compute->rootLoop->children[0]->upper, 50);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

/**
 * Test: Handle tensor with no shape specified.
 *
 * DSL allows: tensor A : CSR;  (shape inferred or dynamic)
 */
TEST_F(IRLoweringTest, TensorWithoutShape) {
    const char* code = R"(
        tensor y : Dense;
        tensor A : CSR;
        tensor x : Dense;
        compute y[i] = A[i, j] * x[j];
    )";

    auto ast = parseCode(code);
    ASSERT_NE(ast, nullptr) << "Failed to parse tensor declarations without shape";

    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    // Should still create operation, dims may be empty or have default
    EXPECT_NE(compute, nullptr);
    EXPECT_EQ(compute->output.name, "y");
}

/**
 * Test: Multiple computations - only lower the relevant one.
 *
 * For now, we focus on single compute statements.
 */
TEST_F(IRLoweringTest, SingleComputeStatement) {
    const char* code = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        compute y[i] = A[i, j] * x[j];
    )";

    auto ast = parseCode(code);

    // Should have 4 statements (3 declarations + 1 compute)
    ASSERT_NE(ast, nullptr);
    EXPECT_EQ(ast->statements.size(), 4);

    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);
    EXPECT_NE(compute, nullptr);
}

TEST_F(IRLoweringTest, ScheduleProgram_MultipleTopLevelComputes_Success) {
    const char* code = R"(
        tensor y : Dense<100>;
        tensor z : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        compute y[i] = A[i, j] * x[j];
        compute z[i] = A[i, j] * x[j];
    )";

    auto ast = parseCode(code);
    ASSERT_NE(ast, nullptr);

    auto prog = lowerScheduledProgram(code);
    ASSERT_NE(prog, nullptr);

    std::vector<const sparseir::scheduled::Compute*> computes;
    collectScheduledComputes(prog->statements, computes);
    ASSERT_EQ(computes.size(), 2);
    EXPECT_EQ(computes[0]->output.name, "y");
    EXPECT_EQ(computes[1]->output.name, "z");
    EXPECT_NE(computes[0]->rootLoop, nullptr);
    EXPECT_NE(computes[1]->rootLoop, nullptr);
}

TEST_F(IRLoweringTest, ScheduleProgram_NestedForComputes_Success) {
    const char* code = R"(
        tensor y : Dense<100>;
        tensor z : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        for [A, x] [i, j] {
            compute y[i] = A[i, j] * x[j];
            for [A, x] [i, j] {
                compute z[i] = A[i, j] * x[j];
            }
        }
    )";

    auto ast = parseCode(code);
    ASSERT_NE(ast, nullptr);

    auto prog = lowerScheduledProgram(code);
    ASSERT_NE(prog, nullptr);

    std::vector<const sparseir::scheduled::Compute*> computes;
    collectScheduledComputes(prog->statements, computes);
    ASSERT_EQ(computes.size(), 2);
    EXPECT_EQ(computes[0]->output.name, "y");
    EXPECT_EQ(computes[1]->output.name, "z");
}

TEST_F(IRLoweringTest, ScheduleProgram_PreviouslyUnknownNowCanonicalScheduled) {
    const char* code = R"(
        tensor y : Dense<100>;
        tensor z : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        compute y[i] = A[i, j] * x[j];
        compute z[i] = A[i, j] + x[j];
    )";

    auto ast = parseCode(code);
    ASSERT_NE(ast, nullptr);

    auto prog = lowerScheduledProgram(code);
    ASSERT_NE(prog, nullptr);

    std::vector<const sparseir::scheduled::Compute*> computes;
    collectScheduledComputes(prog->statements, computes);
    ASSERT_EQ(computes.size(), 2u);
    EXPECT_EQ(computes[0]->output.name, "y");
    EXPECT_EQ(computes[1]->output.name, "z");
    EXPECT_EQ(computes[1]->outputStrategy, ir::OutputStrategy::DenseArray);
    EXPECT_TRUE(computes[1]->fullyLowered);
}

// ============================================================================
// Format Detection from AST
// ============================================================================

/**
 * Test: Correctly map AST tensor types to IR formats.
 */
TEST_F(IRLoweringTest, FormatMapping_Dense) {
    const char* code = R"(
        tensor x : Dense<100>;
        tensor A : Dense<100, 50>;
        compute x[i] = A[i, j] * x[j];
    )";

    auto ast = parseCode(code);
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    EXPECT_EQ(compute->output.format, ir::Format::Dense);
    EXPECT_EQ(compute->inputs[0].format, ir::Format::Dense);
}

/**
 * Test: Map CSR type correctly.
 */
TEST_F(IRLoweringTest, FormatMapping_CSR) {
    const char* code = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        compute y[i] = A[i, j] * x[j];
    )";

    auto ast = parseCode(code);
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    EXPECT_EQ(compute->inputs[0].format, ir::Format::CSR);
}

/**
 * Test: Map CSC type correctly.
 */
TEST_F(IRLoweringTest, FormatMapping_CSC) {
    const char* code = R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 50>;
        tensor x : Dense<50>;
        compute y[i] = A[i, j] * x[j];
    )";

    auto ast = parseCode(code);
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    EXPECT_EQ(compute->inputs[0].format, ir::Format::CSC);
}
