#include <gtest/gtest.h>
#include "ir.h"
#include "semantic_ir.h"
#include "ast.h"
#include <memory>
#include <vector>
#include <string>

using namespace SparseTensorCompiler;

// Helper to create a SpMM AST for testing
std::unique_ptr<Program> createSpMMAst(const std::string& sparseFormat) {
    auto prog = std::make_unique<Program>();

    // tensor C : Dense<100, 50>;
    prog->addStatement(std::make_unique<Declaration>(
        "C", "Dense", std::vector<std::string>{"100", "50"}
    ));

    // tensor A : CSR<100, 80>;  or CSC
    prog->addStatement(std::make_unique<Declaration>(
        "A", sparseFormat, std::vector<std::string>{"100", "80"}
    ));

    // tensor B : Dense<80, 50>;
    prog->addStatement(std::make_unique<Declaration>(
        "B", "Dense", std::vector<std::string>{"80", "50"}
    ));

    // compute C[i, j] = A[i, k] * B[k, j];
    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});

    auto leftOperand = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "k"});
    auto rightOperand = std::make_unique<TensorAccess>("B", std::vector<std::string>{"k", "j"});

    auto binop = std::make_unique<BinaryOp>(
        BinaryOp::MULT,
        std::move(leftOperand),
        std::move(rightOperand)
    );

    prog->addStatement(std::make_unique<Computation>(
        std::move(lhs),
        std::move(binop)
    ));

    return prog;
}

class IRSpMMTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(IRSpMMTest, LowerSpMMToIR_CSR) {
    // Create AST for: C[i,j] = A[i,k] * B[k,j] with A in CSR
    auto ast = createSpMMAst("CSR");

    // Lower to IR
    auto scheduled = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(scheduled, nullptr);
    ASSERT_TRUE(scheduled->fullyLowered);

    ASSERT_EQ(scheduled->inputs.size(), 2);
    EXPECT_EQ(scheduled->inputs[0].name, "A");
    EXPECT_EQ(scheduled->inputs[0].format, ir::Format::CSR);
    EXPECT_EQ(scheduled->inputs[1].name, "B");
    EXPECT_EQ(scheduled->inputs[1].format, ir::Format::Dense);
    EXPECT_EQ(scheduled->output.name, "C");
    EXPECT_EQ(scheduled->output.format, ir::Format::Dense);
    EXPECT_EQ(scheduled->outputStrategy, ir::OutputStrategy::DenseArray);
}

TEST_F(IRSpMMTest, BuildSpMMLoopNest_CSR) {
    // C[i,j] = A[i,k] * B[k,j], A is CSR
    // Expected structure (CSR natural order):
    // for i (dense, 0 to 100):       <- rows
    //   for k (sparse, row_ptr[i]):   <- sparse on row i
    //     for j (dense, 0 to 50):     <- columns of B and C
    //       C[i][j] += vals[p] * B[k][j]

    auto ast = createSpMMAst("CSR");
    auto scheduled = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(scheduled, nullptr);
    ASSERT_NE(scheduled->rootLoop, nullptr);

    // Verify outer loop (i)
    auto* outerLoop = scheduled->rootLoop.get();
    EXPECT_EQ(outerLoop->indexName, "i");
    EXPECT_EQ(outerLoop->kind, sparseir::scheduled::LoopKind::Dense);
    EXPECT_EQ(outerLoop->lower, 0);
    EXPECT_EQ(outerLoop->upper, 100);

    // Verify middle loop (k - sparse)
    ASSERT_FALSE(outerLoop->children.empty());
    auto* middleLoop = outerLoop->children[0].get();
    EXPECT_EQ(middleLoop->indexName, "k");
    EXPECT_EQ(middleLoop->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_EQ(middleLoop->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(middleLoop->iterator.beginExpr, "A->row_ptr[i]");
    EXPECT_EQ(middleLoop->iterator.endExpr, "A->row_ptr[i + 1]");

    // Verify inner loop (j)
    ASSERT_FALSE(middleLoop->children.empty());
    auto* innerLoop = middleLoop->children[0].get();
    EXPECT_EQ(innerLoop->indexName, "j");
    EXPECT_EQ(innerLoop->kind, sparseir::scheduled::LoopKind::Dense);
    EXPECT_EQ(innerLoop->lower, 0);
    EXPECT_EQ(innerLoop->upper, 50);

    EXPECT_FALSE(innerLoop->postStmts.empty());
}

TEST_F(IRSpMMTest, BuildSpMMLoopNest_CSC) {
    // C[i,j] = A[i,k] * B[k,j], A is CSC
    // Expected structure (CSC natural order):
    // for k (dense, 0 to 80):         <- columns of A
    //   for i (sparse, col_ptr[k]):    <- sparse on column k
    //     for j (dense, 0 to 50):      <- columns of B and C
    //       C[i][j] += vals[p] * B[k][j]

    auto ast = createSpMMAst("CSC");
    auto scheduled = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(scheduled, nullptr);
    ASSERT_NE(scheduled->rootLoop, nullptr);

    // Verify outer loop (k - CSC must iterate columns first)
    auto* outerLoop = scheduled->rootLoop.get();
    EXPECT_EQ(outerLoop->indexName, "k");
    EXPECT_EQ(outerLoop->kind, sparseir::scheduled::LoopKind::Dense);
    EXPECT_EQ(outerLoop->lower, 0);
    EXPECT_EQ(outerLoop->upper, 80);

    // Verify middle loop (i - sparse)
    ASSERT_FALSE(outerLoop->children.empty());
    auto* middleLoop = outerLoop->children[0].get();
    EXPECT_EQ(middleLoop->indexName, "i");
    EXPECT_EQ(middleLoop->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_EQ(middleLoop->headerKind, sparseir::scheduled::LoopHeaderKind::SparseIterator);
    EXPECT_EQ(middleLoop->iterator.beginExpr, "A->col_ptr[k]");
    EXPECT_EQ(middleLoop->iterator.endExpr, "A->col_ptr[k + 1]");

    // Verify inner loop (j)
    ASSERT_FALSE(middleLoop->children.empty());
    auto* innerLoop = middleLoop->children[0].get();
    EXPECT_EQ(innerLoop->indexName, "j");
    EXPECT_EQ(innerLoop->kind, sparseir::scheduled::LoopKind::Dense);
    EXPECT_EQ(innerLoop->lower, 0);
    EXPECT_EQ(innerLoop->upper, 50);
}

TEST_F(IRSpMMTest, IdentifyFreeAndBoundIndices) {
    // For C[i,j] = A[i,k] * B[k,j]
    // Free indices: {i, j} - appear in output C[i,j]
    // Bound indices: {k} - summation index (appears in RHS but not LHS)

    auto ast = createSpMMAst("CSR");
    auto scheduled = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(scheduled, nullptr);

    // Collect all indices from loop nest
    std::vector<std::string> loopIndices;

    auto* loop1 = scheduled->rootLoop.get();
    loopIndices.push_back(loop1->indexName);

    if (!loop1->children.empty()) {
        auto* loop2 = loop1->children[0].get();
        loopIndices.push_back(loop2->indexName);

        if (!loop2->children.empty()) {
            auto* loop3 = loop2->children[0].get();
            loopIndices.push_back(loop3->indexName);
        }
    }

    // Should have exactly 3 loop indices
    EXPECT_EQ(loopIndices.size(), 3);

    // For CSR: order should be i, k, j
    EXPECT_EQ(loopIndices[0], "i");
    EXPECT_EQ(loopIndices[1], "k");
    EXPECT_EQ(loopIndices[2], "j");

    // Check that output has 2 dimensions (i, j)
    EXPECT_EQ(scheduled->output.dims.size(), 2);
}

TEST_F(IRSpMMTest, CorrectDimensionBounds) {
    // C[100, 50] = A[100, 80] * B[80, 50]
    // Loop bounds should match tensor dimensions

    auto ast = createSpMMAst("CSR");
    auto scheduled = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(scheduled, nullptr);
    ASSERT_NE(scheduled->rootLoop, nullptr);

    // For CSR:
    // i: 0 to 100 (rows of A, rows of C)
    // k: sparse (columns of A, rows of B)
    // j: 0 to 50 (columns of B, columns of C)

    auto* iLoop = scheduled->rootLoop.get();
    EXPECT_EQ(iLoop->upper, 100);

    auto* kLoop = iLoop->children[0].get();
    // k is sparse, so upper bound is implicit from row_ptr

    auto* jLoop = kLoop->children[0].get();
    EXPECT_EQ(jLoop->upper, 50);
}

TEST_F(IRSpMMTest, VerifyDenseOutputStrategy) {
    auto ast = createSpMMAst("CSR");
    auto scheduled = sparseir::lowerFirstComputationToScheduled(*ast);

    ASSERT_NE(scheduled, nullptr);
    EXPECT_EQ(scheduled->outputStrategy, ir::OutputStrategy::DenseArray);
    EXPECT_EQ(scheduled->exprInfo.numSparseInputs, 1);
    EXPECT_EQ(scheduled->exprInfo.numDenseInputs, 1);
}

TEST_F(IRSpMMTest, DifferentMatrixSizes) {
    // Test with different dimensions to ensure generality
    auto prog = std::make_unique<Program>();

    // tensor C : Dense<50, 30>;
    prog->addStatement(std::make_unique<Declaration>(
        "C", "Dense", std::vector<std::string>{"50", "30"}
    ));

    // tensor A : CSR<50, 40>;
    prog->addStatement(std::make_unique<Declaration>(
        "A", "CSR", std::vector<std::string>{"50", "40"}
    ));

    // tensor B : Dense<40, 30>;
    prog->addStatement(std::make_unique<Declaration>(
        "B", "Dense", std::vector<std::string>{"40", "30"}
    ));

    // compute C[i, j] = A[i, k] * B[k, j];
    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});
    auto leftOp = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "k"});
    auto rightOp = std::make_unique<TensorAccess>("B", std::vector<std::string>{"k", "j"});
    auto binop = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(leftOp), std::move(rightOp));
    prog->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(binop)));

    auto scheduled = sparseir::lowerFirstComputationToScheduled(*prog);

    ASSERT_NE(scheduled, nullptr);
    ASSERT_NE(scheduled->rootLoop, nullptr);

    // Verify dimensions
    auto* iLoop = scheduled->rootLoop.get();
    EXPECT_EQ(iLoop->upper, 50);

    auto* jLoop = iLoop->children[0]->children[0].get();
    EXPECT_EQ(jLoop->upper, 30);
}
