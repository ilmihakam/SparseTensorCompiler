#include <gtest/gtest.h>
#include "ast.h"
#include <sstream>
#include <memory>
#include <set>

using namespace SparseTensorCompiler;

// Forward declarations for parser interface
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern std::unique_ptr<Program> g_program;
extern int yynerrs;

// Helper function to parse a string and return the AST
std::unique_ptr<Program> parse_to_ast(const std::string& input) {
    yynerrs = 0;
    g_program = nullptr;
    yy_scan_string(input.c_str());
    int result = yyparse();
    yylex_destroy();

    if (result == 0 && yynerrs == 0) {
        return std::move(g_program);
    }
    return nullptr;
}

class ParserSpMMTest : public ::testing::Test {
protected:
    void TearDown() override {
        g_program = nullptr;
    }

    std::unique_ptr<Program> parseInput(const char* input) {
        return parse_to_ast(input);
    }
};

TEST_F(ParserSpMMTest, BasicSpMMDeclaration) {
    const char* input = R"(
        tensor C : Dense<100, 50>;
        tensor A : CSR<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    auto ast = parseInput(input);
    ASSERT_NE(ast, nullptr);

    // Verify 4 statements (3 tensors + 1 compute)
    ASSERT_EQ(ast->statements.size(), 4);

    // Verify first three are tensor declarations
    auto* tensorC = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(tensorC, nullptr);
    EXPECT_EQ(tensorC->tensorName, "C");
    EXPECT_EQ(tensorC->tensorType, "Dense");
    EXPECT_EQ(tensorC->shape.size(), 2);
    EXPECT_EQ(tensorC->shape[0], "100");
    EXPECT_EQ(tensorC->shape[1], "50");

    auto* tensorA = dynamic_cast<Declaration*>(ast->statements[1].get());
    ASSERT_NE(tensorA, nullptr);
    EXPECT_EQ(tensorA->tensorName, "A");
    EXPECT_EQ(tensorA->tensorType, "CSR");
    EXPECT_EQ(tensorA->shape.size(), 2);
    EXPECT_EQ(tensorA->shape[0], "100");
    EXPECT_EQ(tensorA->shape[1], "80");

    auto* tensorB = dynamic_cast<Declaration*>(ast->statements[2].get());
    ASSERT_NE(tensorB, nullptr);
    EXPECT_EQ(tensorB->tensorName, "B");
    EXPECT_EQ(tensorB->tensorType, "Dense");

    // Verify compute statement
    auto* compute = dynamic_cast<Computation*>(ast->statements[3].get());
    ASSERT_NE(compute, nullptr);

    // Verify LHS is C[i,j]
    ASSERT_NE(compute->lhs, nullptr);
    EXPECT_EQ(compute->lhs->tensorName, "C");
    EXPECT_EQ(compute->lhs->indices.size(), 2);
    EXPECT_EQ(compute->lhs->indices[0], "i");
    EXPECT_EQ(compute->lhs->indices[1], "j");

    // Verify RHS is A[i,k] * B[k,j]
    auto* binop = dynamic_cast<BinaryOp*>(compute->rhs.get());
    ASSERT_NE(binop, nullptr);
    EXPECT_EQ(binop->op, BinaryOp::MULT);

    // Verify left operand is A[i,k]
    auto* leftAccess = dynamic_cast<TensorAccess*>(binop->left.get());
    ASSERT_NE(leftAccess, nullptr);
    EXPECT_EQ(leftAccess->tensorName, "A");
    EXPECT_EQ(leftAccess->indices.size(), 2);
    EXPECT_EQ(leftAccess->indices[0], "i");
    EXPECT_EQ(leftAccess->indices[1], "k");

    // Verify right operand is B[k,j]
    auto* rightAccess = dynamic_cast<TensorAccess*>(binop->right.get());
    ASSERT_NE(rightAccess, nullptr);
    EXPECT_EQ(rightAccess->tensorName, "B");
    EXPECT_EQ(rightAccess->indices.size(), 2);
    EXPECT_EQ(rightAccess->indices[0], "k");
    EXPECT_EQ(rightAccess->indices[1], "j");
}

TEST_F(ParserSpMMTest, SpMMWithCSCFormat) {
    const char* input = R"(
        tensor C : Dense<100, 50>;
        tensor A : CSC<100, 80>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    auto ast = parseInput(input);
    ASSERT_NE(ast, nullptr);
    ASSERT_EQ(ast->statements.size(), 4);

    // Verify A is CSC format
    auto* tensorA = dynamic_cast<Declaration*>(ast->statements[1].get());
    ASSERT_NE(tensorA, nullptr);
    EXPECT_EQ(tensorA->tensorName, "A");
    EXPECT_EQ(tensorA->tensorType, "CSC");
    EXPECT_EQ(tensorA->shape.size(), 2);
}

TEST_F(ParserSpMMTest, SpMMWithDifferentIndexNames) {
    const char* input = R"(
        tensor Y : Dense<50, 30>;
        tensor X : CSR<50, 40>;
        tensor Z : Dense<40, 30>;
        compute Y[m, n] = X[m, p] * Z[p, n];
    )";

    auto ast = parseInput(input);
    ASSERT_NE(ast, nullptr);

    auto* compute = dynamic_cast<Computation*>(ast->statements[3].get());
    ASSERT_NE(compute, nullptr);

    // Verify indices
    EXPECT_EQ(compute->lhs->indices[0], "m");
    EXPECT_EQ(compute->lhs->indices[1], "n");

    auto* binop = dynamic_cast<BinaryOp*>(compute->rhs.get());
    auto* leftAccess = dynamic_cast<TensorAccess*>(binop->left.get());
    auto* rightAccess = dynamic_cast<TensorAccess*>(binop->right.get());

    EXPECT_EQ(leftAccess->indices[1], "p");  // summation index
    EXPECT_EQ(rightAccess->indices[0], "p"); // summation index
}

TEST_F(ParserSpMMTest, ParsesThreeLoopIndices) {
    const char* input = R"(
        tensor C : Dense<10, 20>;
        tensor A : CSR<10, 15>;
        tensor B : Dense<15, 20>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    auto ast = parseInput(input);
    ASSERT_NE(ast, nullptr);

    auto* compute = dynamic_cast<Computation*>(ast->statements[3].get());
    ASSERT_NE(compute, nullptr);

    // Collect all unique indices: i, j (free), k (bound)
    std::set<std::string> allIndices;

    // From LHS
    for (const auto& idx : compute->lhs->indices) {
        allIndices.insert(idx);
    }

    // From RHS
    auto* binop = dynamic_cast<BinaryOp*>(compute->rhs.get());
    auto* leftAccess = dynamic_cast<TensorAccess*>(binop->left.get());
    auto* rightAccess = dynamic_cast<TensorAccess*>(binop->right.get());

    for (const auto& idx : leftAccess->indices) {
        allIndices.insert(idx);
    }
    for (const auto& idx : rightAccess->indices) {
        allIndices.insert(idx);
    }

    // Should have 3 unique indices: i, j, k
    EXPECT_EQ(allIndices.size(), 3);
    EXPECT_TRUE(allIndices.count("i") > 0);
    EXPECT_TRUE(allIndices.count("j") > 0);
    EXPECT_TRUE(allIndices.count("k") > 0);
}

TEST_F(ParserSpMMTest, RejectsMismatchedTensorDimensions) {
    // This test checks if parser/AST can catch dimension mismatches
    // Note: Full dimension checking might be in semantic analysis, not parser
    const char* input = R"(
        tensor C : Dense<100, 50>;
        tensor A : CSR<100, 70>;
        tensor B : Dense<80, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    // Parser should accept this (syntax is valid)
    // Semantic checker should reject (dimensions don't match)
    auto ast = parseInput(input);
    ASSERT_NE(ast, nullptr);

    // TODO: Add semantic validation that checks:
    // - A's second dimension (70) should match B's first dimension (80) - MISMATCH!
    // - A's first dimension (100) should match C's first dimension (100) - OK
    // - B's second dimension (50) should match C's second dimension (50) - OK
}
