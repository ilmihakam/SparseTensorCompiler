#include <gtest/gtest.h>
#include <string>
#include "ast.h"
#include "parser_generated.h"

using namespace SparseTensorCompiler;

// Forward declarations for parser interface
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern int yynerrs;
extern std::unique_ptr<Program> g_program;

// Helper function to parse a string and return the AST
std::unique_ptr<Program> parse_to_ast(const std::string& input) {
    // Reset error count and global program
    yynerrs = 0;
    g_program.reset();

    // Set input string for lexer
    yy_scan_string(input.c_str());

    // Run parser
    int result = yyparse();

    // Cleanup
    yylex_destroy();

    // Return the AST (nullptr if parse failed)
    if (result == 0 && yynerrs == 0) {
        return std::move(g_program);
    }
    return nullptr;
}

class TypedIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {
        g_program.reset();
    }
};

// ============================================
// SINGLE DECLARATION TESTS
// ============================================

TEST_F(TypedIntegrationTest, ParseDenseDeclaration) {
    std::string program = "tensor A : Dense;";
    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    ASSERT_EQ(ast->statements.size(), 1);

    auto* decl = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(decl, nullptr);
    EXPECT_EQ(decl->tensorName, "A");
    EXPECT_EQ(decl->tensorType, "Dense");
    EXPECT_TRUE(decl->shape.empty());
}

TEST_F(TypedIntegrationTest, ParseCSRDeclaration) {
    std::string program = "tensor B : CSR;";
    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    ASSERT_EQ(ast->statements.size(), 1);

    auto* decl = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(decl, nullptr);
    EXPECT_EQ(decl->tensorName, "B");
    EXPECT_EQ(decl->tensorType, "CSR");
}

TEST_F(TypedIntegrationTest, ParseCOODeclaration) {
    std::string program = "tensor C : COO;";
    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    auto* decl = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(decl, nullptr);
    EXPECT_EQ(decl->tensorType, "COO");
}

TEST_F(TypedIntegrationTest, ParseCSCDeclaration) {
    std::string program = "tensor D : CSC;";
    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    auto* decl = dynamic_cast<Declaration*>(ast->statements[0].get());
    EXPECT_EQ(decl->tensorType, "CSC");
}

TEST_F(TypedIntegrationTest, ParseELLPACKDeclaration) {
    std::string program = "tensor E : ELLPACK;";
    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    auto* decl = dynamic_cast<Declaration*>(ast->statements[0].get());
    EXPECT_EQ(decl->tensorType, "ELLPACK");
}

TEST_F(TypedIntegrationTest, ParseDIADeclaration) {
    std::string program = "tensor F : DIA;";
    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    auto* decl = dynamic_cast<Declaration*>(ast->statements[0].get());
    EXPECT_EQ(decl->tensorType, "DIA");
}

// ============================================
// DECLARATIONS WITH SHAPE TESTS
// ============================================

TEST_F(TypedIntegrationTest, ParseDenseWithShape1D) {
    std::string program = "tensor A : Dense<10>;";
    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    ASSERT_EQ(ast->statements.size(), 1);

    auto* decl = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(decl, nullptr);
    EXPECT_EQ(decl->tensorName, "A");
    EXPECT_EQ(decl->tensorType, "Dense");
    ASSERT_EQ(decl->shape.size(), 1);
    EXPECT_EQ(decl->shape[0], "10");
}

TEST_F(TypedIntegrationTest, ParseDenseWithShape2D) {
    std::string program = "tensor A : Dense<10, 20>;";
    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    auto* decl = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(decl, nullptr);
    EXPECT_EQ(decl->tensorType, "Dense");
    ASSERT_EQ(decl->shape.size(), 2);
    EXPECT_EQ(decl->shape[0], "10");
    EXPECT_EQ(decl->shape[1], "20");
}

TEST_F(TypedIntegrationTest, ParseCSRWithShape2D) {
    std::string program = "tensor A : CSR<100, 50>;";
    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    auto* decl = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(decl, nullptr);
    EXPECT_EQ(decl->tensorType, "CSR");
    ASSERT_EQ(decl->shape.size(), 2);
    EXPECT_EQ(decl->shape[0], "100");
    EXPECT_EQ(decl->shape[1], "50");
}

TEST_F(TypedIntegrationTest, ParseCOOWithShape3D) {
    std::string program = "tensor B : COO<10, 20, 30>;";
    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    auto* decl = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(decl, nullptr);
    EXPECT_EQ(decl->tensorType, "COO");
    ASSERT_EQ(decl->shape.size(), 3);
    EXPECT_EQ(decl->shape[0], "10");
    EXPECT_EQ(decl->shape[1], "20");
    EXPECT_EQ(decl->shape[2], "30");
}

// ============================================
// WHITESPACE VARIATIONS
// ============================================

TEST_F(TypedIntegrationTest, ParseNoWhitespace) {
    std::string program = "tensor A:CSR<3,4>;";
    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    auto* decl = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(decl, nullptr);
    EXPECT_EQ(decl->tensorName, "A");
    EXPECT_EQ(decl->tensorType, "CSR");
    EXPECT_EQ(decl->shape.size(), 2);
}

TEST_F(TypedIntegrationTest, ParseExtraWhitespace) {
    std::string program = "tensor  A  :  Dense  <  10  ,  20  >  ;";
    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    auto* decl = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(decl, nullptr);
    EXPECT_EQ(decl->tensorType, "Dense");
    EXPECT_EQ(decl->shape.size(), 2);
}

// ============================================
// MULTIPLE DECLARATIONS
// ============================================

TEST_F(TypedIntegrationTest, ParseMultipleTypedDeclarations) {
    std::string program = R"(
        tensor A : Dense<10>;
        tensor B : CSR<10, 20>;
        tensor C : COO<5, 5, 5>;
    )";
    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    ASSERT_EQ(ast->statements.size(), 3);

    // Check first declaration
    auto* decl1 = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(decl1, nullptr);
    EXPECT_EQ(decl1->tensorName, "A");
    EXPECT_EQ(decl1->tensorType, "Dense");
    EXPECT_EQ(decl1->shape.size(), 1);

    // Check second declaration
    auto* decl2 = dynamic_cast<Declaration*>(ast->statements[1].get());
    ASSERT_NE(decl2, nullptr);
    EXPECT_EQ(decl2->tensorName, "B");
    EXPECT_EQ(decl2->tensorType, "CSR");
    EXPECT_EQ(decl2->shape.size(), 2);

    // Check third declaration
    auto* decl3 = dynamic_cast<Declaration*>(ast->statements[2].get());
    ASSERT_NE(decl3, nullptr);
    EXPECT_EQ(decl3->tensorName, "C");
    EXPECT_EQ(decl3->tensorType, "COO");
    EXPECT_EQ(decl3->shape.size(), 3);
}

// ============================================
// COMPLETE PROGRAMS WITH COMPUTATION
// ============================================

TEST_F(TypedIntegrationTest, ParseTypedProgramWithCompute) {
    std::string program = R"(
        tensor A : Dense<10>;
        tensor B : Dense<10>;
        compute B[i] = A[i];
    )";
    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    ASSERT_EQ(ast->statements.size(), 3);

    // Verify declarations
    auto* decl1 = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(decl1, nullptr);
    EXPECT_EQ(decl1->tensorType, "Dense");

    auto* decl2 = dynamic_cast<Declaration*>(ast->statements[1].get());
    ASSERT_NE(decl2, nullptr);
    EXPECT_EQ(decl2->tensorType, "Dense");

    // Verify computation
    auto* comp = dynamic_cast<Computation*>(ast->statements[2].get());
    ASSERT_NE(comp, nullptr);
}

TEST_F(TypedIntegrationTest, ParseTypedMatrixMultiplication) {
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

    // Verify first declaration - CSR
    auto* decl1 = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(decl1, nullptr);
    EXPECT_EQ(decl1->tensorName, "A");
    EXPECT_EQ(decl1->tensorType, "CSR");
    ASSERT_EQ(decl1->shape.size(), 2);
    EXPECT_EQ(decl1->shape[0], "100");
    EXPECT_EQ(decl1->shape[1], "50");

    // Verify second declaration - Dense
    auto* decl2 = dynamic_cast<Declaration*>(ast->statements[1].get());
    ASSERT_NE(decl2, nullptr);
    EXPECT_EQ(decl2->tensorName, "B");
    EXPECT_EQ(decl2->tensorType, "Dense");
    ASSERT_EQ(decl2->shape.size(), 2);
    EXPECT_EQ(decl2->shape[0], "50");
    EXPECT_EQ(decl2->shape[1], "20");

    // Verify third declaration - Dense
    auto* decl3 = dynamic_cast<Declaration*>(ast->statements[2].get());
    ASSERT_NE(decl3, nullptr);
    EXPECT_EQ(decl3->tensorName, "C");
    EXPECT_EQ(decl3->tensorType, "Dense");
    ASSERT_EQ(decl3->shape.size(), 2);
    EXPECT_EQ(decl3->shape[0], "100");
    EXPECT_EQ(decl3->shape[1], "20");

    // Verify for loop
    auto* forStmt = dynamic_cast<ForStatement*>(ast->statements[3].get());
    ASSERT_NE(forStmt, nullptr);
    EXPECT_EQ(forStmt->tensors.size(), 3);
    EXPECT_EQ(forStmt->indices.size(), 3);
    EXPECT_EQ(forStmt->body.size(), 1);
}

TEST_F(TypedIntegrationTest, ParseTypedProgramWithCalls) {
    std::string program = R"(
        tensor A : CSR<1000, 1000>;
        tensor B : Dense<1000>;
        for [A, B] [i, j] {
            compute B[i] = A[i, j] * B[j];
            call optimize(B);
        }
    )";
    auto ast = parse_to_ast(program);

    ASSERT_NE(ast, nullptr);
    ASSERT_EQ(ast->statements.size(), 3);

    // Verify CSR declaration
    auto* decl1 = dynamic_cast<Declaration*>(ast->statements[0].get());
    ASSERT_NE(decl1, nullptr);
    EXPECT_EQ(decl1->tensorType, "CSR");
    EXPECT_EQ(decl1->shape.size(), 2);

    // Verify Dense declaration
    auto* decl2 = dynamic_cast<Declaration*>(ast->statements[1].get());
    ASSERT_NE(decl2, nullptr);
    EXPECT_EQ(decl2->tensorType, "Dense");
    EXPECT_EQ(decl2->shape.size(), 1);

    // Verify for loop with call statement
    auto* forStmt = dynamic_cast<ForStatement*>(ast->statements[2].get());
    ASSERT_NE(forStmt, nullptr);
    EXPECT_EQ(forStmt->body.size(), 2);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
