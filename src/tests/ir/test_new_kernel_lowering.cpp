/**
 * Test Suite: New Kernel IR Lowering
 *
 * Tests full AST -> IR lowering for new kernel types:
 * SpAdd, SpElMul, SpGEMM, SDDMM, and fused SpMV.
 * Also verifies backward compatibility for SpMV/SpMM.
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
// SpAdd Tests: C[i,j] = A[i,j] + B[i,j]
// ============================================================================

std::unique_ptr<Program> createSpAddAst() {
    auto prog = std::make_unique<Program>();

    prog->addStatement(std::make_unique<Declaration>(
        "C", "Dense", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "A", "CSR", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "B", "CSR", std::vector<std::string>{"100", "80"}));

    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});
    auto left = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
    auto right = std::make_unique<TensorAccess>("B", std::vector<std::string>{"i", "j"});
    auto binop = std::make_unique<BinaryOp>(BinaryOp::ADD, std::move(left), std::move(right));

    prog->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(binop)));
    return prog;
}

TEST(SpAddLoweringTest, ProducesDenseTwoSparseCompute) {
    auto ast = createSpAddAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    EXPECT_TRUE(compute->fullyLowered);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::DenseArray);
    EXPECT_EQ(compute->exprInfo.rootOp, ir::RootOpKind::ADD);
}

TEST(SpAddLoweringTest, OuterDenseWithUnionMergeChild) {
    auto ast = createSpAddAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);

    // Outer loop should be dense (row iteration)
    auto* outerLoop = compute->rootLoop.get();
    EXPECT_EQ(outerLoop->kind, sparseir::scheduled::LoopKind::Dense);
    EXPECT_EQ(outerLoop->indexName, "i");

    ASSERT_EQ(outerLoop->children.size(), 1);
    EXPECT_EQ(outerLoop->children[0]->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_EQ(outerLoop->children[0]->mergeStrategy, ir::MergeStrategy::Union);
}

TEST(SpAddLoweringTest, UnionMergeLoopTagsBothInputs) {
    auto ast = createSpAddAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);
    ASSERT_EQ(compute->rootLoop->children.size(), 1);

    auto* mergeLoop = compute->rootLoop->children[0].get();
    EXPECT_EQ(mergeLoop->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_EQ(mergeLoop->mergeStrategy, ir::MergeStrategy::Union);
    ASSERT_EQ(mergeLoop->mergedTensors.size(), 2);
    EXPECT_EQ(mergeLoop->mergedTensors[0], "A");
    EXPECT_EQ(mergeLoop->mergedTensors[1], "B");
}

TEST(SpAddLoweringTest, ExpressionInfoRootOpIsAdd) {
    auto ast = createSpAddAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->exprInfo.rootOp, ir::RootOpKind::ADD);
    EXPECT_EQ(compute->exprInfo.numSparseInputs, 2);
    EXPECT_EQ(compute->exprInfo.numDenseInputs, 0);
}

// ============================================================================
// SpElMul Tests: C[i,j] = A[i,j] * B[i,j]
// ============================================================================

std::unique_ptr<Program> createSpElMulAst() {
    auto prog = std::make_unique<Program>();

    prog->addStatement(std::make_unique<Declaration>(
        "C", "Dense", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "A", "CSR", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "B", "CSR", std::vector<std::string>{"100", "80"}));

    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});
    auto left = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
    auto right = std::make_unique<TensorAccess>("B", std::vector<std::string>{"i", "j"});
    auto binop = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(left), std::move(right));

    prog->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(binop)));
    return prog;
}

TEST(SpElMulLoweringTest, ProducesIntersectionTwoSparseCompute) {
    auto ast = createSpElMulAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    EXPECT_TRUE(compute->fullyLowered);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::DenseArray);
    EXPECT_EQ(compute->exprInfo.rootOp, ir::RootOpKind::MULT);
}

TEST(SpElMulLoweringTest, InnerLoopHasIntersectionMerge) {
    auto ast = createSpElMulAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);
    ASSERT_EQ(compute->rootLoop->children.size(), 1);

    auto* innerLoop = compute->rootLoop->children[0].get();
    EXPECT_EQ(innerLoop->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_EQ(innerLoop->mergeStrategy, ir::MergeStrategy::Intersection);
}

TEST(SpElMulLoweringTest, MergeLoopTagsBothTensors) {
    auto ast = createSpElMulAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);
    ASSERT_EQ(compute->rootLoop->children.size(), 1);

    auto* innerLoop = compute->rootLoop->children[0].get();
    EXPECT_EQ(innerLoop->mergedTensors.size(), 2u);
    EXPECT_EQ(innerLoop->mergedTensors[0], "A");
    EXPECT_EQ(innerLoop->mergedTensors[1], "B");
}

// ============================================================================
// SpGEMM Tests: C[i,j] = A[i,k] * B[k,j] (both sparse)
// ============================================================================

std::unique_ptr<Program> createSpGEMMAst() {
    auto prog = std::make_unique<Program>();

    prog->addStatement(std::make_unique<Declaration>(
        "C", "Dense", std::vector<std::string>{"100", "50"}));
    prog->addStatement(std::make_unique<Declaration>(
        "A", "CSR", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "B", "CSR", std::vector<std::string>{"80", "50"}));

    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});
    auto left = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "k"});
    auto right = std::make_unique<TensorAccess>("B", std::vector<std::string>{"k", "j"});
    auto binop = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(left), std::move(right));

    prog->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(binop)));
    return prog;
}

TEST(SpGEMMLoweringTest, ProducesDenseSparseProductCompute) {
    auto ast = createSpGEMMAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    EXPECT_TRUE(compute->fullyLowered);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::DenseArray);
    EXPECT_EQ(compute->exprInfo.numSparseInputs, 2);
}

TEST(SpGEMMLoweringTest, ThreeLevelNestDenseSparseASparseB) {
    auto ast = createSpGEMMAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);

    // Level 1: dense i
    auto* iLoop = compute->rootLoop.get();
    EXPECT_EQ(iLoop->indexName, "i");
    EXPECT_EQ(iLoop->kind, sparseir::scheduled::LoopKind::Dense);

    // Level 2: sparse k (A's iteration)
    ASSERT_EQ(iLoop->children.size(), 1);
    auto* kLoop = iLoop->children[0].get();
    EXPECT_EQ(kLoop->indexName, "k");
    EXPECT_EQ(kLoop->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_EQ(kLoop->driverTensor, "A");

    // Level 3: sparse j (B's iteration)
    ASSERT_EQ(kLoop->children.size(), 1);
    auto* jLoop = kLoop->children[0].get();
    EXPECT_EQ(jLoop->indexName, "j");
    EXPECT_EQ(jLoop->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_EQ(jLoop->driverTensor, "B");

    EXPECT_FALSE(jLoop->postStmts.empty());
}

// ============================================================================
// SDDMM Tests: C[i,j] = S[i,j] * D[i,k] * E[k,j]
// ============================================================================

std::unique_ptr<Program> createSDDMMAst() {
    auto prog = std::make_unique<Program>();

    prog->addStatement(std::make_unique<Declaration>(
        "C", "Dense", std::vector<std::string>{"100", "50"}));
    prog->addStatement(std::make_unique<Declaration>(
        "S", "CSR", std::vector<std::string>{"100", "50"}));
    prog->addStatement(std::make_unique<Declaration>(
        "D", "Dense", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "E", "Dense", std::vector<std::string>{"80", "50"}));

    // C[i,j] = S[i,j] * D[i,k] * E[k,j]
    // AST: (S[i,j] * D[i,k]) * E[k,j]
    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});

    auto sAccess = std::make_unique<TensorAccess>("S", std::vector<std::string>{"i", "j"});
    auto dAccess = std::make_unique<TensorAccess>("D", std::vector<std::string>{"i", "k"});
    auto eAccess = std::make_unique<TensorAccess>("E", std::vector<std::string>{"k", "j"});

    auto innerMul = std::make_unique<BinaryOp>(
        BinaryOp::MULT, std::move(sAccess), std::move(dAccess));
    auto outerMul = std::make_unique<BinaryOp>(
        BinaryOp::MULT, std::move(innerMul), std::move(eAccess));

    prog->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(outerMul)));
    return prog;
}

TEST(SDDMMLoweringTest, ProducesSampledContractionCompute) {
    auto ast = createSDDMMAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    EXPECT_TRUE(compute->fullyLowered);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::DenseArray);
}

TEST(SDDMMLoweringTest, ExprInfoHasThreeAccesses) {
    auto ast = createSDDMMAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    EXPECT_EQ(compute->exprInfo.numTensorAccesses, 3);
    EXPECT_EQ(compute->exprInfo.numSparseInputs, 1);
    EXPECT_EQ(compute->exprInfo.numDenseInputs, 2);
}

TEST(SDDMMLoweringTest, SparseSamplingLoopWithDenseChild) {
    auto ast = createSDDMMAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);

    // Level 1: dense i
    auto* iLoop = compute->rootLoop.get();
    EXPECT_EQ(iLoop->indexName, "i");
    EXPECT_EQ(iLoop->kind, sparseir::scheduled::LoopKind::Dense);

    // Level 2: sparse j (sampling loop)
    ASSERT_EQ(iLoop->children.size(), 1);
    auto* jLoop = iLoop->children[0].get();
    EXPECT_EQ(jLoop->indexName, "j");
    EXPECT_EQ(jLoop->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_EQ(jLoop->driverTensor, "S");

    // Level 3: dense k (contraction)
    ASSERT_EQ(jLoop->children.size(), 1);
    auto* kLoop = jLoop->children[0].get();
    EXPECT_EQ(kLoop->indexName, "k");
    EXPECT_EQ(kLoop->kind, sparseir::scheduled::LoopKind::Dense);

    EXPECT_FALSE(jLoop->postStmts.empty());
}

// ============================================================================
// Fused SpMV Tests: y[i] = relu(A[i,j] * x[j])
// ============================================================================

std::unique_ptr<Program> createFusedSpMVAst() {
    auto prog = std::make_unique<Program>();

    prog->addStatement(std::make_unique<Declaration>(
        "y", "Dense", std::vector<std::string>{"100"}));
    prog->addStatement(std::make_unique<Declaration>(
        "A", "CSR", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "x", "Dense", std::vector<std::string>{"80"}));

    // y[i] = relu(A[i,j] * x[j])
    auto lhs = std::make_unique<TensorAccess>("y", std::vector<std::string>{"i"});

    auto aAccess = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
    auto xAccess = std::make_unique<TensorAccess>("x", std::vector<std::string>{"j"});
    auto mul = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(aAccess), std::move(xAccess));

    auto reluCall = std::make_unique<FunctionCall>("relu");
    reluCall->addArgument(std::move(mul));

    prog->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(reluCall)));
    return prog;
}

TEST(FusedSpMVLoweringTest, IsFusedWithReluFunction) {
    auto ast = createFusedSpMVAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    EXPECT_TRUE(compute->exprInfo.isFused);
    EXPECT_EQ(compute->exprInfo.fusionFunction, "relu");
}

TEST(FusedSpMVLoweringTest, ProducesSingleSparseContraction) {
    auto ast = createFusedSpMVAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    EXPECT_TRUE(compute->fullyLowered);
    EXPECT_EQ(compute->exprInfo.numSparseInputs, 1);
    EXPECT_EQ(compute->outputStrategy, ir::OutputStrategy::DenseArray);
}

TEST(FusedSpMVLoweringTest, LoopNestIdenticalToNonFused) {
    auto ast = createFusedSpMVAst();
    auto compute = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);

    // Same structure as regular SpMV: dense i -> sparse j
    auto* outerLoop = compute->rootLoop.get();
    EXPECT_EQ(outerLoop->indexName, "i");
    EXPECT_EQ(outerLoop->kind, sparseir::scheduled::LoopKind::Dense);

    ASSERT_EQ(outerLoop->children.size(), 1);
    auto* innerLoop = outerLoop->children[0].get();
    EXPECT_EQ(innerLoop->indexName, "j");
    EXPECT_EQ(innerLoop->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_FALSE(innerLoop->postStmts.empty());
}

// ============================================================================
// Backward Compatibility Tests
// ============================================================================

TEST(BackwardCompatTest, SpMVStillLowersThroughScheduledPath) {
    auto prog = std::make_unique<Program>();

    prog->addStatement(std::make_unique<Declaration>(
        "y", "Dense", std::vector<std::string>{"100"}));
    prog->addStatement(std::make_unique<Declaration>(
        "A", "CSR", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "x", "Dense", std::vector<std::string>{"80"}));

    auto lhs = std::make_unique<TensorAccess>("y", std::vector<std::string>{"i"});
    auto left = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
    auto right = std::make_unique<TensorAccess>("x", std::vector<std::string>{"j"});
    auto binop = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(left), std::move(right));

    prog->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(binop)));

    auto compute = sparseir::lowerFirstComputationToScheduled(*prog);
    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);
    ASSERT_EQ(compute->rootLoop->children.size(), 1);
    EXPECT_EQ(compute->rootLoop->children[0]->driverTensor, "A");
}

TEST(BackwardCompatTest, SpMMStillLowersThroughScheduledPath) {
    auto prog = std::make_unique<Program>();

    prog->addStatement(std::make_unique<Declaration>(
        "C", "Dense", std::vector<std::string>{"100", "50"}));
    prog->addStatement(std::make_unique<Declaration>(
        "A", "CSR", std::vector<std::string>{"100", "80"}));
    prog->addStatement(std::make_unique<Declaration>(
        "B", "Dense", std::vector<std::string>{"80", "50"}));

    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});
    auto left = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "k"});
    auto right = std::make_unique<TensorAccess>("B", std::vector<std::string>{"k", "j"});
    auto binop = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(left), std::move(right));

    prog->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(binop)));

    auto compute = sparseir::lowerFirstComputationToScheduled(*prog);
    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);
    ASSERT_EQ(compute->rootLoop->children.size(), 1);
    auto* sparseLoop = compute->rootLoop->children[0].get();
    EXPECT_EQ(sparseLoop->driverTensor, "A");
}
