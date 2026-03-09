#include <gtest/gtest.h>
#include <string>
#include "ast.h"

using namespace SparseTensorCompiler;

// Forward declarations for parser interface
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern std::unique_ptr<Program> g_program;

// Global flag to track parse errors
extern int yynerrs;

// Helper function to parse a string and return the AST
std::unique_ptr<Program> parse_to_ast(const std::string& input) {
    // Reset error count
    yynerrs = 0;

    // Clear previous AST
    g_program = nullptr;

    // Set input string for lexer
    yy_scan_string(input.c_str());

    // Run parser
    int result = yyparse();

    // Cleanup lexer
    yylex_destroy();

    // Return the AST if parsing succeeded
    if (result == 0 && yynerrs == 0) {
        return std::move(g_program);
    }

    return nullptr;
}

// ============================================
// Test Fixture
// ============================================

class ParserASTIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {
        g_program = nullptr;
    }
};

// ============================================
// DECLARATION TESTS
// ============================================

TEST_F(ParserASTIntegrationTest, SimpleDeclaration) {
    auto ast = parse_to_ast("tensor A : Dense;");

    ASSERT_NE(ast, nullptr);
    ASSERT_EQ(ast->statements.size(), 1);

    auto* decl = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(decl, nullptr);
    EXPECT_EQ(decl->tensorName, "A");
    EXPECT_EQ(decl->tensorType, "Dense");
    EXPECT_TRUE(decl->shape.empty());
}

TEST_F(ParserASTIntegrationTest, DeclarationWithShape) {
    auto ast = parse_to_ast("tensor A : CSR<100, 50, 20>;");

    ASSERT_NE(ast, nullptr);
    ASSERT_EQ(ast->statements.size(), 1);

    auto* decl = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(decl, nullptr);
    EXPECT_EQ(decl->tensorName, "A");
    EXPECT_EQ(decl->tensorType, "CSR");
    ASSERT_EQ(decl->shape.size(), 3);
    EXPECT_EQ(decl->shape[0], "100");
    EXPECT_EQ(decl->shape[1], "50");
    EXPECT_EQ(decl->shape[2], "20");
}

// ============================================
// COMPUTATION TESTS
// ============================================

TEST_F(ParserASTIntegrationTest, ComputeSimple) {
    auto ast = parse_to_ast("compute C[i] = A[i];");

    ASSERT_NE(ast, nullptr);
    ASSERT_EQ(ast->statements.size(), 1);

    auto* comp = dynamic_cast<Computation*>(ast->statements[0].get());
    ASSERT_NE(comp, nullptr);
    EXPECT_EQ(comp->lhs->tensorName, "C");
    EXPECT_EQ(comp->lhs->indices.size(), 1);
    EXPECT_EQ(comp->lhs->indices[0], "i");

    auto* rhs = dynamic_cast<TensorAccess*>(comp->rhs.get());
    ASSERT_NE(rhs, nullptr);
    EXPECT_EQ(rhs->tensorName, "A");
}

TEST_F(ParserASTIntegrationTest, ComputeWithAddition) {
    auto ast = parse_to_ast("compute C[i] = A[i] + B[i];");

    ASSERT_NE(ast, nullptr);
    ASSERT_EQ(ast->statements.size(), 1);

    auto* comp = dynamic_cast<Computation*>(ast->statements[0].get());
    ASSERT_NE(comp, nullptr);

    auto* binOp = dynamic_cast<BinaryOp*>(comp->rhs.get());
    ASSERT_NE(binOp, nullptr);
    EXPECT_EQ(binOp->op, BinaryOp::ADD);

    auto* left = dynamic_cast<TensorAccess*>(binOp->left.get());
    auto* right = dynamic_cast<TensorAccess*>(binOp->right.get());
    ASSERT_NE(left, nullptr);
    ASSERT_NE(right, nullptr);
    EXPECT_EQ(left->tensorName, "A");
    EXPECT_EQ(right->tensorName, "B");
}

TEST_F(ParserASTIntegrationTest, ComputeWithMultiplication) {
    auto ast = parse_to_ast("compute C[i] = A[i] * B[i];");

    ASSERT_NE(ast, nullptr);
    auto* comp = dynamic_cast<Computation*>(ast->statements[0].get());
    auto* binOp = dynamic_cast<BinaryOp*>(comp->rhs.get());

    ASSERT_NE(binOp, nullptr);
    EXPECT_EQ(binOp->op, BinaryOp::MULT);
}

TEST_F(ParserASTIntegrationTest, ComputeWithNumber) {
    auto ast = parse_to_ast("compute A[i] = 42;");

    ASSERT_NE(ast, nullptr);
    auto* comp = dynamic_cast<Computation*>(ast->statements[0].get());
    auto* num = dynamic_cast<Number*>(comp->rhs.get());

    ASSERT_NE(num, nullptr);
    EXPECT_EQ(num->value, "42");
}

TEST_F(ParserASTIntegrationTest, ComputeWithFunctionCall) {
    auto ast = parse_to_ast("compute result[i] = relu(A[i]);");

    ASSERT_NE(ast, nullptr);
    auto* comp = dynamic_cast<Computation*>(ast->statements[0].get());
    auto* funcCall = dynamic_cast<FunctionCall*>(comp->rhs.get());

    ASSERT_NE(funcCall, nullptr);
    EXPECT_EQ(funcCall->functionName, "relu");
    EXPECT_EQ(funcCall->arguments.size(), 1);
}

// ============================================
// CALL STATEMENT TESTS
// ============================================

TEST_F(ParserASTIntegrationTest, CallStatementNoArguments) {
    auto ast = parse_to_ast("call print();");

    ASSERT_NE(ast, nullptr);
    ASSERT_EQ(ast->statements.size(), 1);

    auto* call = dynamic_cast<CallStatement*>(ast->statements[0].get());
    ASSERT_NE(call, nullptr);
    EXPECT_EQ(call->functionName, "print");
    EXPECT_EQ(call->arguments.size(), 0);
}

TEST_F(ParserASTIntegrationTest, CallStatementSingleArgument) {
    auto ast = parse_to_ast("call optimize(A);");

    ASSERT_NE(ast, nullptr);
    auto* call = dynamic_cast<CallStatement*>(ast->statements[0].get());

    ASSERT_NE(call, nullptr);
    EXPECT_EQ(call->functionName, "optimize");
    EXPECT_EQ(call->arguments.size(), 1);

    auto* arg = dynamic_cast<Identifier*>(call->arguments[0].get());
    ASSERT_NE(arg, nullptr);
    EXPECT_EQ(arg->name, "A");
}

TEST_F(ParserASTIntegrationTest, CallStatementMultipleArguments) {
    auto ast = parse_to_ast("call function(A, B, 42);");

    ASSERT_NE(ast, nullptr);
    auto* call = dynamic_cast<CallStatement*>(ast->statements[0].get());

    ASSERT_NE(call, nullptr);
    EXPECT_EQ(call->arguments.size(), 3);
}

// ============================================
// FOR STATEMENT TESTS
// ============================================

TEST_F(ParserASTIntegrationTest, ForStatementEmpty) {
    auto ast = parse_to_ast("for [A] [i] { }");

    ASSERT_NE(ast, nullptr);
    ASSERT_EQ(ast->statements.size(), 1);

    auto* forStmt = dynamic_cast<ForStatement*>(ast->statements[0].get());
    ASSERT_NE(forStmt, nullptr);
    EXPECT_EQ(forStmt->tensors.size(), 1);
    EXPECT_EQ(forStmt->tensors[0], "A");
    EXPECT_EQ(forStmt->indices.size(), 1);
    EXPECT_EQ(forStmt->indices[0], "i");
    EXPECT_EQ(forStmt->body.size(), 0);
}

TEST_F(ParserASTIntegrationTest, ForStatementWithBody) {
    auto ast = parse_to_ast("for [A] [i] { compute A[i] = 0; }");

    ASSERT_NE(ast, nullptr);
    auto* forStmt = dynamic_cast<ForStatement*>(ast->statements[0].get());

    ASSERT_NE(forStmt, nullptr);
    EXPECT_EQ(forStmt->body.size(), 1);

    auto* comp = dynamic_cast<Computation*>(forStmt->body[0].get());
    ASSERT_NE(comp, nullptr);
}

TEST_F(ParserASTIntegrationTest, ForStatementMultipleTensorsAndIndices) {
    auto ast = parse_to_ast("for [A, B, C] [i, j, k] { }");

    ASSERT_NE(ast, nullptr);
    auto* forStmt = dynamic_cast<ForStatement*>(ast->statements[0].get());

    ASSERT_NE(forStmt, nullptr);
    EXPECT_EQ(forStmt->tensors.size(), 3);
    EXPECT_EQ(forStmt->tensors[0], "A");
    EXPECT_EQ(forStmt->tensors[1], "B");
    EXPECT_EQ(forStmt->tensors[2], "C");
    EXPECT_EQ(forStmt->indices.size(), 3);
}

// ============================================
// COMPLEX PROGRAM TESTS (IR-Compatible)
// ============================================

TEST_F(ParserASTIntegrationTest, CompleteMatrixMultiplication) {
    std::string program = R"(
        tensor A : CSR<100, 50>;
        tensor B : Dense<50, 20>;
        tensor C : Dense<100, 20>;
        for [A, B, C] [i, j, k] {
            compute C[i, j] = A[i, k] * B[k, j];
        }
    )";

    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    ASSERT_EQ(ast->statements.size(), 4);

    // Check declarations
    auto* declA = dynamic_cast<Declaration*>(ast->statements[0].get());
    auto* declB = dynamic_cast<Declaration*>(ast->statements[1].get());
    auto* declC = dynamic_cast<Declaration*>(ast->statements[2].get());

    ASSERT_NE(declA, nullptr);
    ASSERT_NE(declB, nullptr);
    ASSERT_NE(declC, nullptr);

    EXPECT_EQ(declA->tensorName, "A");
    EXPECT_EQ(declA->tensorType, "CSR");
    EXPECT_EQ(declB->tensorName, "B");
    EXPECT_EQ(declB->tensorType, "Dense");
    EXPECT_EQ(declC->tensorName, "C");
    EXPECT_EQ(declC->tensorType, "Dense");

    // Check for statement
    auto* forStmt = dynamic_cast<ForStatement*>(ast->statements[3].get());
    ASSERT_NE(forStmt, nullptr);
    EXPECT_EQ(forStmt->tensors.size(), 3);
    EXPECT_EQ(forStmt->indices.size(), 3);
    EXPECT_EQ(forStmt->body.size(), 1);

    // Check computation inside for loop
    auto* comp = dynamic_cast<Computation*>(forStmt->body[0].get());
    ASSERT_NE(comp, nullptr);
    EXPECT_EQ(comp->lhs->tensorName, "C");

    auto* binOp = dynamic_cast<BinaryOp*>(comp->rhs.get());
    ASSERT_NE(binOp, nullptr);
    EXPECT_EQ(binOp->op, BinaryOp::MULT);
}

TEST_F(ParserASTIntegrationTest, CompleteWithFunctionCallAndOptimize) {
    std::string program = R"(
        tensor A : Dense<10, 20>;
        for [A] [i, j] {
            compute A[i, j] = relu(A[i, j]);
            call optimize(A);
        }
    )";

    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    ASSERT_EQ(ast->statements.size(), 2);

    // Check declaration
    auto* decl = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(decl, nullptr);
    EXPECT_EQ(decl->tensorName, "A");
    EXPECT_EQ(decl->tensorType, "Dense");

    // Check for statement
    auto* forStmt = dynamic_cast<ForStatement*>(ast->statements[1].get());
    ASSERT_NE(forStmt, nullptr);
    EXPECT_EQ(forStmt->body.size(), 2);

    // Check compute with function call
    auto* comp = dynamic_cast<Computation*>(forStmt->body[0].get());
    ASSERT_NE(comp, nullptr);

    auto* funcCall = dynamic_cast<FunctionCall*>(comp->rhs.get());
    ASSERT_NE(funcCall, nullptr);
    EXPECT_EQ(funcCall->functionName, "relu");

    // Check call statement
    auto* call = dynamic_cast<CallStatement*>(forStmt->body[1].get());
    ASSERT_NE(call, nullptr);
    EXPECT_EQ(call->functionName, "optimize");
}

TEST_F(ParserASTIntegrationTest, MultipleDeclarations) {
    std::string program = R"(
        tensor A : Dense<10>;
        tensor B : CSR<10>;
        tensor C : COO<10>;
    )";

    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    ASSERT_EQ(ast->statements.size(), 3);

    for (int i = 0; i < 3; i++) {
        auto* decl = dynamic_cast<Declaration*>(ast->statements[i].get());
        ASSERT_NE(decl, nullptr);
    }
}

TEST_F(ParserASTIntegrationTest, EmptyProgram) {
    auto ast = parse_to_ast("");

    ASSERT_NE(ast, nullptr);
    EXPECT_EQ(ast->statements.size(), 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
