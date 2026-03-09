/**
 * Test Suite: General Loop Nest Synthesis (Phase B)
 *
 * Tests the general fallback builder that handles arbitrary expressions
 * which don't match any recognized kernel pattern (SpMV, SpMM, etc.).
 *
 * Key insight: ExpressionAnalyzer counts TensorAccess nodes only (not Number).
 * So "2.0 * A[i,j] * x[j]" has numAccesses=2 and still matches SpMV.
 * To trigger "general", we need patterns that genuinely mismatch all detectors.
 */

#include <gtest/gtest.h>
#include "ir.h"
#include "semantic_ir.h"
#include "ast.h"
#include <memory>
#include <vector>
#include <string>

using namespace SparseTensorCompiler;

// ============================================================================
// Helpers
// ============================================================================

static int countLoopDepth(const sparseir::scheduled::Loop* loop) {
    if (!loop) return 0;
    int maxChild = 0;
    for (const auto& child : loop->children) {
        int d = countLoopDepth(child.get());
        if (d > maxChild) maxChild = d;
    }
    return 1 + maxChild;
}

static sparseir::scheduled::Loop* findLoopByIndex(
    sparseir::scheduled::Loop* root, const std::string& name) {
    if (!root) return nullptr;
    if (root->indexName == name) return root;
    for (auto& child : root->children) {
        auto* found = findLoopByIndex(child.get(), name);
        if (found) return found;
    }
    return nullptr;
}

static sparseir::scheduled::Loop* findLoopWithAccumulatorStmts(
    sparseir::scheduled::Loop* root) {
    if (!root) return nullptr;
    if (!root->preStmts.empty() && !root->postStmts.empty()) {
        return root;
    }
    for (auto& child : root->children) {
        if (auto* found = findLoopWithAccumulatorStmts(child.get())) {
            return found;
        }
    }
    return nullptr;
}

// ============================================================================
// Standard "general" kernel: y[i] = A[i,j] * x[j] * z[j]
// 3 tensor accesses, 1 free, 1 bound, 1 sparse + 2 dense, MULT
// Doesn't match any existing pattern (SpMV needs exactly 2 accesses)
// ============================================================================

static std::unique_ptr<Program> createTripleProductSpMVAst() {
    auto prog = std::make_unique<Program>();

    prog->addStatement(std::make_unique<Declaration>(
        "y", "Dense", std::vector<std::string>{"100"}));
    prog->addStatement(std::make_unique<Declaration>(
        "A", "CSR", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "x", "Dense", std::vector<std::string>{"80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "z", "Dense", std::vector<std::string>{"80"}));

    // y[i] = (A[i,j] * x[j]) * z[j]
    auto lhs = std::make_unique<TensorAccess>("y", std::vector<std::string>{"i"});
    auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
    auto xAccess = std::make_unique<TensorAccess>("x", std::vector<std::string>{"j"});
    auto zAccess = std::make_unique<TensorAccess>("z", std::vector<std::string>{"j"});

    auto mul1 = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(aAccess), std::move(xAccess));
    auto mul2 = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(mul1), std::move(zAccess));

    prog->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(mul2)));
    return prog;
}

// ============================================================================
// Test 1: TripleProductSpMV — detected as "general"
// ============================================================================

TEST(GeneralLoweringTest, ScalarTimesSpMV_UsesTaglessGeneralPath) {
    auto ast = createTripleProductSpMVAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    EXPECT_TRUE(compute->fullyLowered);
    EXPECT_EQ(compute->exprInfo.numTensorAccesses, 3);
}

TEST(GeneralLoweringTest, ScalarTimesSpMV_HasValidLoopNest) {
    auto ast = createTripleProductSpMVAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);

    // Should have 2 levels: dense i -> sparse j
    EXPECT_EQ(countLoopDepth(compute->rootLoop.get()), 2);

    auto* outerLoop = compute->rootLoop.get();
    EXPECT_EQ(outerLoop->indexName, "i");
    EXPECT_EQ(outerLoop->kind, sparseir::scheduled::LoopKind::Dense);

    ASSERT_EQ(outerLoop->children.size(), 1u);
    auto* innerLoop = outerLoop->children[0].get();
    EXPECT_EQ(innerLoop->indexName, "j");
    EXPECT_EQ(innerLoop->kind, sparseir::scheduled::LoopKind::Sparse);
}

// ============================================================================
// Test 2: ThreeTensorAdd — C[i,j] = A[i,j] + B[i,j] + D[i,j] (3 sparse)
// 3 accesses → not the 2-access spadd pattern → general with union merge
// ============================================================================

static std::unique_ptr<Program> createThreeTensorAddAst() {
    auto prog = std::make_unique<Program>();

    prog->addStatement(std::make_unique<Declaration>(
        "C", "Dense", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "A", "CSR", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "B", "CSR", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "D", "CSR", std::vector<std::string>{"100", "80"}));

    // C[i,j] = (A[i,j] + B[i,j]) + D[i,j]
    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});
    auto a = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
    auto b = std::make_unique<TensorAccess>("B", std::vector<std::string>{"i", "j"});
    auto d = std::make_unique<TensorAccess>("D", std::vector<std::string>{"i", "j"});

    auto add1 = std::make_unique<BinaryOp>(BinaryOp::ADD, std::move(a), std::move(b));
    auto add2 = std::make_unique<BinaryOp>(BinaryOp::ADD, std::move(add1), std::move(d));

    prog->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(add2)));
    return prog;
}

TEST(GeneralLoweringTest, ThreeTensorAdd_UsesGeneralScheduledShape) {
    auto ast = createThreeTensorAddAst();
    auto semanticProgram = sparseir::lowerToSemanticProgram(*ast);
    ASSERT_NE(semanticProgram, nullptr);
    auto* compute = dynamic_cast<sparseir::semantic::Compute*>(
        semanticProgram->statements.back().get());

    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->exprInfo.numSparseInputs, 3);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::DenseArray);
}

TEST(GeneralLoweringTest, ThreeTensorAdd_HasUnionMergeLoop) {
    auto ast = createThreeTensorAddAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);

    // Outer: dense loop with a generic union-merge sparse child
    auto* outerLoop = compute->rootLoop.get();
    EXPECT_EQ(outerLoop->kind, sparseir::scheduled::LoopKind::Dense);
    ASSERT_EQ(outerLoop->children.size(), 1u);
    EXPECT_EQ(outerLoop->children[0]->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_EQ(outerLoop->children[0]->headerKind, sparseir::scheduled::LoopHeaderKind::SparseMerge);
    EXPECT_EQ(outerLoop->children[0]->merge.strategy, ir::MergeStrategy::Union);
    EXPECT_EQ(outerLoop->children[0]->merge.terms.size(), 3u);
}

// ============================================================================
// Test 3: ScalarMatrix — B[i,j] = A[i,j] * D[i,j] * E[i,j] (3 sparse MUL, 0 bound)
// 3 accesses, not 2 → doesn't match spelmul
// ============================================================================

static std::unique_ptr<Program> createTripleElMulAst() {
    auto prog = std::make_unique<Program>();

    prog->addStatement(std::make_unique<Declaration>(
        "C", "Dense", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "A", "CSR", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "B", "CSR", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "D", "CSR", std::vector<std::string>{"100", "80"}));

    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});
    auto a = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
    auto b = std::make_unique<TensorAccess>("B", std::vector<std::string>{"i", "j"});
    auto d = std::make_unique<TensorAccess>("D", std::vector<std::string>{"i", "j"});

    auto mul1 = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(a), std::move(b));
    auto mul2 = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(mul1), std::move(d));

    prog->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(mul2)));
    return prog;
}

TEST(GeneralLoweringTest, TripleElMul_CorrectDepth) {
    auto ast = createTripleElMulAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);

    // 2 free indices, 0 bound → 2-level loop nest
    EXPECT_EQ(countLoopDepth(compute->rootLoop.get()), 2);
}

// ============================================================================
// Test 4: IntersectionMerge — C[i,j] = A[i,j] * B[i,j] * D[i,j] (3 sparse MUL)
// ============================================================================

TEST(GeneralLoweringTest, IntersectionMerge_HasMergeStrategy) {
    auto ast = createTripleElMulAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);

    // Should have dense outer loop
    EXPECT_EQ(compute->rootLoop->kind, sparseir::scheduled::LoopKind::Dense);
    ASSERT_GE(compute->rootLoop->children.size(), 1u);

    // The inner sparse loop should have Intersection merge strategy
    auto* innerLoop = compute->rootLoop->children[0].get();
    EXPECT_EQ(innerLoop->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_EQ(innerLoop->headerKind, sparseir::scheduled::LoopHeaderKind::SparseMerge);
    EXPECT_EQ(innerLoop->merge.strategy, ir::MergeStrategy::Intersection);
}

// ============================================================================
// Test 5: CSCFormatReversesOrder
// y[i] = A[i,j] * x[j] * z[j] with A:CSC → general, j should be sparse
// ============================================================================

static std::unique_ptr<Program> createCSCTripleProductAst() {
    auto prog = std::make_unique<Program>();

    prog->addStatement(std::make_unique<Declaration>(
        "y", "Dense", std::vector<std::string>{"80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "A", "CSC", std::vector<std::string>{"80", "100"}));
    prog->addStatement(std::make_unique<Declaration>(
        "x", "Dense", std::vector<std::string>{"100"}));
    prog->addStatement(std::make_unique<Declaration>(
        "z", "Dense", std::vector<std::string>{"100"}));

    // y[i] = (A[i,j] * x[j]) * z[j]
    auto lhs = std::make_unique<TensorAccess>("y", std::vector<std::string>{"i"});
    auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
    auto xAccess = std::make_unique<TensorAccess>("x", std::vector<std::string>{"j"});
    auto zAccess = std::make_unique<TensorAccess>("z", std::vector<std::string>{"j"});

    auto mul1 = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(aAccess), std::move(xAccess));
    auto mul2 = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(mul1), std::move(zAccess));

    prog->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(mul2)));
    return prog;
}

TEST(GeneralLoweringTest, CSCFormatReversesOrder) {
    auto ast = createCSCTripleProductAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);

    // CSC single-sparse contraction lowers as dense j -> sparse i driven by A.
    auto* outerLoop = compute->rootLoop.get();
    EXPECT_EQ(outerLoop->indexName, "j");
    EXPECT_EQ(outerLoop->kind, sparseir::scheduled::LoopKind::Dense);

    auto* sparseLoop = findLoopByIndex(compute->rootLoop.get(), "i");
    ASSERT_NE(sparseLoop, nullptr);
    EXPECT_EQ(sparseLoop->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_EQ(sparseLoop->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(sparseLoop->iterator.beginExpr, "A->col_ptr[j]");
    EXPECT_EQ(sparseLoop->iterator.endExpr, "A->col_ptr[j + 1]");
    EXPECT_EQ(sparseLoop->bindingExpr, "A->row_idx[pA]");
}

// ============================================================================
// Test 6: DimensionInference_SparsePreferred
// ============================================================================

TEST(GeneralLoweringTest, DimensionInference_SparsePreferred) {
    auto prog = std::make_unique<Program>();

    // Output has dim 50, sparse input A has dim 100 for index i
    prog->addStatement(std::make_unique<Declaration>(
        "y", "Dense", std::vector<std::string>{"50"}));
    prog->addStatement(std::make_unique<Declaration>(
        "A", "CSR", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "x", "Dense", std::vector<std::string>{"80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "z", "Dense", std::vector<std::string>{"80"}));

    // y[i] = (A[i,j] * x[j]) * z[j] → general (3 accesses)
    auto lhs = std::make_unique<TensorAccess>("y", std::vector<std::string>{"i"});
    auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
    auto xAccess = std::make_unique<TensorAccess>("x", std::vector<std::string>{"j"});
    auto zAccess = std::make_unique<TensorAccess>("z", std::vector<std::string>{"j"});

    auto mul1 = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(aAccess), std::move(xAccess));
    auto mul2 = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(mul1), std::move(zAccess));

    prog->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(mul2)));

    auto compute = sparseir::lowerFirstComputationToScheduled(*prog);
    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);

    // "i" dimension should be 100 (from sparse A), not 50 (from output y)
    auto* iLoop = findLoopByIndex(compute->rootLoop.get(), "i");
    ASSERT_NE(iLoop, nullptr);
    EXPECT_EQ(iLoop->upper, 100);
}

// ============================================================================
// Test 7: SparseIteratorsPopulated
// ============================================================================

TEST(GeneralLoweringTest, SparseIteratorsPopulated) {
    auto ast = createTripleProductSpMVAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);
    ASSERT_EQ(compute->rootLoop->children.size(), 1u);

    auto* sparseLoop = compute->rootLoop->children[0].get();
    EXPECT_EQ(sparseLoop->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_EQ(sparseLoop->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(sparseLoop->iterator.beginExpr, "A->row_ptr[i]");
    EXPECT_EQ(sparseLoop->iterator.endExpr, "A->row_ptr[i + 1]");
    EXPECT_EQ(sparseLoop->bindingExpr, "A->col_idx[pA]");
}

// ============================================================================
// Test 8: StructuredBodyPopulated
// ============================================================================

TEST(GeneralLoweringTest, StructuredBodyPopulated) {
    auto ast = createTripleProductSpMVAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);
    ASSERT_EQ(compute->rootLoop->children.size(), 1u);

    auto* innerLoop = compute->rootLoop->children[0].get();
    ASSERT_GE(innerLoop->postStmts.size(), 1u);

    auto* assign = dynamic_cast<ir::IRAssign*>(innerLoop->postStmts[0].get());
    ASSERT_NE(assign, nullptr);
    EXPECT_TRUE(assign->accumulate);  // +=
}

// ============================================================================
// Test 9: AccumulatorPattern (SDDMM-like with 4 tensor accesses → general)
// C[i,j] = S[i,j] * D[i,k] * E[k,j] * F[i,j]
// 4 accesses, 1 sparse + 3 dense → doesn't match SDDMM (3 accesses)
// ============================================================================

static std::unique_ptr<Program> createAccumulatorPatternAst() {
    auto prog = std::make_unique<Program>();

    prog->addStatement(std::make_unique<Declaration>(
        "C", "Dense", std::vector<std::string>{"100", "50"}));
    prog->addStatement(std::make_unique<Declaration>(
        "S", "CSR", std::vector<std::string>{"100", "50"}));
    prog->addStatement(std::make_unique<Declaration>(
        "D", "Dense", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "E", "Dense", std::vector<std::string>{"80", "50"}));
    prog->addStatement(std::make_unique<Declaration>(
        "F", "Dense", std::vector<std::string>{"100", "50"}));

    // C[i,j] = ((S[i,j] * D[i,k]) * E[k,j]) * F[i,j]
    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});
    auto s = std::make_unique<TensorAccess>("S", std::vector<std::string>{"i", "j"});
    auto d = std::make_unique<TensorAccess>("D", std::vector<std::string>{"i", "k"});
    auto e = std::make_unique<TensorAccess>("E", std::vector<std::string>{"k", "j"});
    auto f = std::make_unique<TensorAccess>("F", std::vector<std::string>{"i", "j"});

    auto mul1 = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(s), std::move(d));
    auto mul2 = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(mul1), std::move(e));
    auto mul3 = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(mul2), std::move(f));

    prog->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(mul3)));
    return prog;
}

TEST(GeneralLoweringTest, AccumulatorPattern_RemainsGeneralWithoutScalarAccumulator) {
    auto ast = createAccumulatorPatternAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);

    auto* accumulatorLoop = findLoopWithAccumulatorStmts(compute->rootLoop.get());
    EXPECT_EQ(accumulatorLoop, nullptr);
}

// ============================================================================
// Test 10: Scheduled fallback stays canonical for previously "unknown" patterns
// ============================================================================

TEST(GeneralLoweringTest, ScheduledFallbackProducesCanonicalLoopNest) {
    auto ast = createTripleProductSpMVAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    EXPECT_TRUE(compute->fullyLowered);
    ASSERT_NE(compute->rootLoop, nullptr);
    EXPECT_EQ(compute->rootLoop->indexName, "i");
    ASSERT_EQ(compute->rootLoop->children.size(), 1u);
    EXPECT_EQ(compute->rootLoop->children[0]->indexName, "j");
}

// ============================================================================
// Test 11: BoundIndicesInnermost
// y[i] = A[i,j] * B[j,k] * x[k] — 1 free, 2 bound → unknown → general
// ============================================================================

static std::unique_ptr<Program> createChainContractionAst() {
    auto prog = std::make_unique<Program>();

    prog->addStatement(std::make_unique<Declaration>(
        "y", "Dense", std::vector<std::string>{"100"}));
    prog->addStatement(std::make_unique<Declaration>(
        "A", "CSR", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "B", "CSR", std::vector<std::string>{"80", "50"}));
    prog->addStatement(std::make_unique<Declaration>(
        "x", "Dense", std::vector<std::string>{"50"}));

    // y[i] = (A[i,j] * B[j,k]) * x[k]
    auto lhs = std::make_unique<TensorAccess>("y", std::vector<std::string>{"i"});
    auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
    auto bAccess = std::make_unique<TensorAccess>("B", std::vector<std::string>{"j", "k"});
    auto xAccess = std::make_unique<TensorAccess>("x", std::vector<std::string>{"k"});

    auto mul1 = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(aAccess), std::move(bAccess));
    auto mul2 = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(mul1), std::move(xAccess));

    prog->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(mul2)));
    return prog;
}

TEST(GeneralLoweringTest, BoundIndicesInnermost) {
    auto ast = createChainContractionAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);

    // The outermost loop should be a free index (i)
    auto* root = compute->rootLoop.get();
    EXPECT_EQ(root->indexName, "i");

    // Bound indices (j, k) should be nested below i
    EXPECT_GE(countLoopDepth(compute->rootLoop.get()), 2);
}
