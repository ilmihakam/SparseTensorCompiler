#include <gtest/gtest.h>
#include <memory>
#include "ast.h"

using namespace SparseTensorCompiler;

// ============================================
// Helper Classes for AST Validation
// ============================================

// Simple visitor to count nodes and validate structure
class ValidationVisitor : public ASTVisitor {
public:
    int declarationCount = 0;
    int computationCount = 0;
    int callCount = 0;
    int forCount = 0;
    int tensorAccessCount = 0;
    int functionCallCount = 0;
    int binaryOpCount = 0;
    int numberCount = 0;
    int identifierCount = 0;

    void visit(Program& node) override {
        for (auto& stmt : node.statements) {
            stmt->accept(*this);
        }
    }

    void visit(Declaration& node) override {
        declarationCount++;
    }

    void visit(Computation& node) override {
        computationCount++;
        if (node.lhs) node.lhs->accept(*this);
        if (node.rhs) node.rhs->accept(*this);
    }

    void visit(CallStatement& node) override {
        callCount++;
        for (auto& arg : node.arguments) {
            arg->accept(*this);
        }
    }

    void visit(ForStatement& node) override {
        forCount++;
        for (auto& stmt : node.body) {
            stmt->accept(*this);
        }
    }

    void visit(TensorAccess& node) override {
        tensorAccessCount++;
    }

    void visit(FunctionCall& node) override {
        functionCallCount++;
        for (auto& arg : node.arguments) {
            arg->accept(*this);
        }
    }

    void visit(BinaryOp& node) override {
        binaryOpCount++;
        if (node.left) node.left->accept(*this);
        if (node.right) node.right->accept(*this);
    }

    void visit(Number& node) override {
        numberCount++;
    }

    void visit(Identifier& node) override {
        identifierCount++;
    }
};

// ============================================
// Test Fixture
// ============================================

class ASTGenerationTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================
// DECLARATION NODE TESTS
// ============================================

TEST_F(ASTGenerationTest, DeclarationNodeSimple) {
    // Test: tensor A : Dense;
    auto decl = std::make_unique<Declaration>("A", "Dense");

    EXPECT_EQ(decl->tensorName, "A");
    EXPECT_EQ(decl->tensorType, "Dense");
    EXPECT_TRUE(decl->shape.empty());
}

TEST_F(ASTGenerationTest, DeclarationNodeWithShape) {
    // Test: tensor A : CSR<100, 50>;
    auto decl = std::make_unique<Declaration>("A", "CSR", std::vector<std::string>{"100", "50"});

    EXPECT_EQ(decl->tensorName, "A");
    EXPECT_EQ(decl->tensorType, "CSR");
    ASSERT_EQ(decl->shape.size(), 2);
    EXPECT_EQ(decl->shape[0], "100");
    EXPECT_EQ(decl->shape[1], "50");
}

// ============================================
// TENSOR ACCESS NODE TESTS
// ============================================

TEST_F(ASTGenerationTest, TensorAccessSimple) {
    // Test: A[i]
    auto access = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i"});

    EXPECT_EQ(access->tensorName, "A");
    EXPECT_EQ(access->indices.size(), 1);
    EXPECT_EQ(access->indices[0], "i");
}

TEST_F(ASTGenerationTest, TensorAccessMultipleIndices) {
    // Test: A[i, j]
    auto access = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});

    EXPECT_EQ(access->tensorName, "A");
    EXPECT_EQ(access->indices.size(), 2);
    EXPECT_EQ(access->indices[0], "i");
    EXPECT_EQ(access->indices[1], "j");
}

// ============================================
// NUMBER AND IDENTIFIER NODE TESTS
// ============================================

TEST_F(ASTGenerationTest, NumberNodeInteger) {
    // Test: 42
    auto num = std::make_unique<Number>("42");
    EXPECT_EQ(num->value, "42");
}

TEST_F(ASTGenerationTest, NumberNodeDecimal) {
    // Test: 3.14
    auto num = std::make_unique<Number>("3.14");
    EXPECT_EQ(num->value, "3.14");
}

TEST_F(ASTGenerationTest, IdentifierNode) {
    // Test: A
    auto id = std::make_unique<Identifier>("A");
    EXPECT_EQ(id->name, "A");
}

// ============================================
// BINARY OPERATION NODE TESTS
// ============================================

TEST_F(ASTGenerationTest, BinaryOpAddition) {
    // Test: A[i] + B[i]
    auto left = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i"});
    auto right = std::make_unique<TensorAccess>("B", std::vector<std::string>{"i"});
    auto binOp = std::make_unique<BinaryOp>(BinaryOp::ADD, std::move(left), std::move(right));

    EXPECT_EQ(binOp->op, BinaryOp::ADD);
    EXPECT_NE(binOp->left, nullptr);
    EXPECT_NE(binOp->right, nullptr);
}

TEST_F(ASTGenerationTest, BinaryOpMultiplication) {
    // Test: A[i] * B[i]
    auto left = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i"});
    auto right = std::make_unique<TensorAccess>("B", std::vector<std::string>{"i"});
    auto binOp = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(left), std::move(right));

    EXPECT_EQ(binOp->op, BinaryOp::MULT);
    EXPECT_NE(binOp->left, nullptr);
    EXPECT_NE(binOp->right, nullptr);
}

TEST_F(ASTGenerationTest, BinaryOpChained) {
    // Test: A[i] * B[i] + C[i]
    // Structure: (A[i] * B[i]) + C[i]
    auto a = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i"});
    auto b = std::make_unique<TensorAccess>("B", std::vector<std::string>{"i"});
    auto mult = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(a), std::move(b));

    auto c = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i"});
    auto add = std::make_unique<BinaryOp>(BinaryOp::ADD, std::move(mult), std::move(c));

    EXPECT_EQ(add->op, BinaryOp::ADD);
    EXPECT_NE(add->left, nullptr);
    EXPECT_NE(add->right, nullptr);
}

// ============================================
// FUNCTION CALL NODE TESTS
// ============================================

TEST_F(ASTGenerationTest, FunctionCallNoArguments) {
    // Test: f()
    auto call = std::make_unique<FunctionCall>("f");

    EXPECT_EQ(call->functionName, "f");
    EXPECT_EQ(call->arguments.size(), 0);
}

TEST_F(ASTGenerationTest, FunctionCallSingleArgument) {
    // Test: relu(A[i])
    auto call = std::make_unique<FunctionCall>("relu");
    call->addArgument(std::make_unique<TensorAccess>("A", std::vector<std::string>{"i"}));

    EXPECT_EQ(call->functionName, "relu");
    EXPECT_EQ(call->arguments.size(), 1);
}

TEST_F(ASTGenerationTest, FunctionCallMultipleArguments) {
    // Test: f(A[i], B[i], 42)
    auto call = std::make_unique<FunctionCall>("f");
    call->addArgument(std::make_unique<TensorAccess>("A", std::vector<std::string>{"i"}));
    call->addArgument(std::make_unique<TensorAccess>("B", std::vector<std::string>{"i"}));
    call->addArgument(std::make_unique<Number>("42"));

    EXPECT_EQ(call->functionName, "f");
    EXPECT_EQ(call->arguments.size(), 3);
}

// ============================================
// COMPUTATION STATEMENT TESTS
// ============================================

TEST_F(ASTGenerationTest, ComputationSimple) {
    // Test: compute C[i] = A[i];
    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i"});
    auto rhs = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i"});
    auto comp = std::make_unique<Computation>(std::move(lhs), std::move(rhs));

    EXPECT_NE(comp->lhs, nullptr);
    EXPECT_NE(comp->rhs, nullptr);
    EXPECT_EQ(comp->lhs->tensorName, "C");
}

TEST_F(ASTGenerationTest, ComputationWithBinaryOp) {
    // Test: compute C[i] = A[i] + B[i];
    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i"});
    auto a = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i"});
    auto b = std::make_unique<TensorAccess>("B", std::vector<std::string>{"i"});
    auto binOp = std::make_unique<BinaryOp>(BinaryOp::ADD, std::move(a), std::move(b));
    auto comp = std::make_unique<Computation>(std::move(lhs), std::move(binOp));

    EXPECT_NE(comp->lhs, nullptr);
    EXPECT_NE(comp->rhs, nullptr);
}

TEST_F(ASTGenerationTest, ComputationWithFunctionCall) {
    // Test: compute C[i] = relu(A[i]);
    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i"});
    auto call = std::make_unique<FunctionCall>("relu");
    call->addArgument(std::make_unique<TensorAccess>("A", std::vector<std::string>{"i"}));
    auto comp = std::make_unique<Computation>(std::move(lhs), std::move(call));

    EXPECT_NE(comp->lhs, nullptr);
    EXPECT_NE(comp->rhs, nullptr);
}

// ============================================
// CALL STATEMENT TESTS
// ============================================

TEST_F(ASTGenerationTest, CallStatementNoArguments) {
    // Test: call print();
    auto call = std::make_unique<CallStatement>("print");

    EXPECT_EQ(call->functionName, "print");
    EXPECT_EQ(call->arguments.size(), 0);
}

TEST_F(ASTGenerationTest, CallStatementSingleArgument) {
    // Test: call optimize(A);
    auto call = std::make_unique<CallStatement>("optimize");
    call->addArgument(std::make_unique<Identifier>("A"));

    EXPECT_EQ(call->functionName, "optimize");
    EXPECT_EQ(call->arguments.size(), 1);
}

TEST_F(ASTGenerationTest, CallStatementMultipleArguments) {
    // Test: call function(A, B, 42);
    auto call = std::make_unique<CallStatement>("function");
    call->addArgument(std::make_unique<Identifier>("A"));
    call->addArgument(std::make_unique<Identifier>("B"));
    call->addArgument(std::make_unique<Number>("42"));

    EXPECT_EQ(call->functionName, "function");
    EXPECT_EQ(call->arguments.size(), 3);
}

// ============================================
// FOR STATEMENT TESTS
// ============================================

TEST_F(ASTGenerationTest, ForStatementEmpty) {
    // Test: for [A] [i] { }
    auto forStmt = std::make_unique<ForStatement>(
        std::vector<std::string>{"A"},
        std::vector<std::string>{"i"}
    );

    EXPECT_EQ(forStmt->tensors.size(), 1);
    EXPECT_EQ(forStmt->tensors[0], "A");
    EXPECT_EQ(forStmt->indices.size(), 1);
    EXPECT_EQ(forStmt->indices[0], "i");
    EXPECT_EQ(forStmt->body.size(), 0);
}

TEST_F(ASTGenerationTest, ForStatementMultipleTensorsAndIndices) {
    // Test: for [A, B, C] [i, j, k] { }
    auto forStmt = std::make_unique<ForStatement>(
        std::vector<std::string>{"A", "B", "C"},
        std::vector<std::string>{"i", "j", "k"}
    );

    EXPECT_EQ(forStmt->tensors.size(), 3);
    EXPECT_EQ(forStmt->indices.size(), 3);
}

TEST_F(ASTGenerationTest, ForStatementWithBody) {
    // Test: for [A] [i] { compute A[i] = 0; }
    auto forStmt = std::make_unique<ForStatement>(
        std::vector<std::string>{"A"},
        std::vector<std::string>{"i"}
    );

    auto lhs = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i"});
    auto rhs = std::make_unique<Number>("0");
    auto comp = std::make_unique<Computation>(std::move(lhs), std::move(rhs));
    forStmt->addStatement(std::move(comp));

    EXPECT_EQ(forStmt->body.size(), 1);
}

// ============================================
// PROGRAM NODE TESTS
// ============================================

TEST_F(ASTGenerationTest, ProgramEmpty) {
    auto program = std::make_unique<Program>();
    EXPECT_EQ(program->statements.size(), 0);
}

TEST_F(ASTGenerationTest, ProgramSingleStatement) {
    // Test: tensor A : Dense;
    auto program = std::make_unique<Program>();
    program->addStatement(std::make_unique<Declaration>("A", "Dense"));

    EXPECT_EQ(program->statements.size(), 1);
}

TEST_F(ASTGenerationTest, ProgramMultipleStatements) {
    // Test:
    // tensor A : Dense<10>;
    // tensor B : Dense<10>;
    // compute C[i] = A[i] + B[i];
    auto program = std::make_unique<Program>();

    program->addStatement(std::make_unique<Declaration>("A", "Dense", std::vector<std::string>{"10"}));
    program->addStatement(std::make_unique<Declaration>("B", "Dense", std::vector<std::string>{"10"}));

    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i"});
    auto a = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i"});
    auto b = std::make_unique<TensorAccess>("B", std::vector<std::string>{"i"});
    auto binOp = std::make_unique<BinaryOp>(BinaryOp::ADD, std::move(a), std::move(b));
    program->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(binOp)));

    EXPECT_EQ(program->statements.size(), 3);
}

// ============================================
// COMPLEX AST TESTS (IR-Compatible)
// ============================================

TEST_F(ASTGenerationTest, CompleteMatrixMultiplication) {
    // Test: Full matrix multiplication program that maps to IR
    // tensor A : CSR<100, 50>;
    // tensor B : Dense<50, 20>;
    // tensor C : Dense<100, 20>;
    // for [A, B, C] [i, j, k] {
    //     compute C[i, j] = A[i, k] * B[k, j];
    // }

    auto program = std::make_unique<Program>();

    // Declarations
    program->addStatement(std::make_unique<Declaration>("A", "CSR", std::vector<std::string>{"100", "50"}));
    program->addStatement(std::make_unique<Declaration>("B", "Dense", std::vector<std::string>{"50", "20"}));
    program->addStatement(std::make_unique<Declaration>("C", "Dense", std::vector<std::string>{"100", "20"}));

    // For loop
    auto forStmt = std::make_unique<ForStatement>(
        std::vector<std::string>{"A", "B", "C"},
        std::vector<std::string>{"i", "j", "k"}
    );

    // Compute statement inside for loop
    auto lhs = std::make_unique<TensorAccess>("C", std::vector<std::string>{"i", "j"});
    auto a = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "k"});
    auto b = std::make_unique<TensorAccess>("B", std::vector<std::string>{"k", "j"});
    auto mult = std::make_unique<BinaryOp>(BinaryOp::MULT, std::move(a), std::move(b));
    forStmt->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(mult)));

    program->addStatement(std::move(forStmt));

    // Validate structure using visitor
    ValidationVisitor visitor;
    program->accept(visitor);

    EXPECT_EQ(visitor.declarationCount, 3);
    EXPECT_EQ(visitor.forCount, 1);
    EXPECT_EQ(visitor.computationCount, 1);
    EXPECT_EQ(visitor.binaryOpCount, 1);
    EXPECT_EQ(visitor.tensorAccessCount, 3);  // C[i,j], A[i,k], B[k,j]
}

TEST_F(ASTGenerationTest, CompleteWithFunctionCall) {
    // Test: Program with function call
    // tensor A : Dense<10, 20>;
    // for [A] [i, j] {
    //     compute A[i, j] = relu(A[i, j]);
    //     call optimize(A);
    // }

    auto program = std::make_unique<Program>();

    program->addStatement(std::make_unique<Declaration>("A", "Dense", std::vector<std::string>{"10", "20"}));

    auto forStmt = std::make_unique<ForStatement>(
        std::vector<std::string>{"A"},
        std::vector<std::string>{"i", "j"}
    );

    // Compute statement
    auto lhs = std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"});
    auto funcCall = std::make_unique<FunctionCall>("relu");
    funcCall->addArgument(std::make_unique<TensorAccess>("A", std::vector<std::string>{"i", "j"}));
    forStmt->addStatement(std::make_unique<Computation>(std::move(lhs), std::move(funcCall)));

    // Call statement
    auto callStmt = std::make_unique<CallStatement>("optimize");
    callStmt->addArgument(std::make_unique<Identifier>("A"));
    forStmt->addStatement(std::move(callStmt));

    program->addStatement(std::move(forStmt));

    // Validate
    ValidationVisitor visitor;
    program->accept(visitor);

    EXPECT_EQ(visitor.declarationCount, 1);
    EXPECT_EQ(visitor.forCount, 1);
    EXPECT_EQ(visitor.computationCount, 1);
    EXPECT_EQ(visitor.callCount, 1);
    EXPECT_EQ(visitor.functionCallCount, 1);
    EXPECT_EQ(visitor.identifierCount, 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
