#include <gtest/gtest.h>
#include <string>
#include "parser_generated.h"

// Forward declarations for parser interface
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();

// Global flag to track parse errors
extern int yynerrs;

// Helper function to parse a string and return success/failure
bool parse_program(const std::string& input) {
    // Reset error count
    yynerrs = 0;

    // Set input string for lexer
    yy_scan_string(input.c_str());

    // Run parser
    int result = yyparse();

    // Cleanup
    yylex_destroy();

    // Return true if parse succeeded (result == 0 and no errors)
    return (result == 0 && yynerrs == 0);
}

class TensorDeclarationTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================
// NEW TENSOR TYPE DECLARATION SYNTAX TESTS
// ============================================

// Test 1: Simple tensor type declarations without shapes
TEST_F(TensorDeclarationTest, DenseDeclaration) {
    EXPECT_TRUE(parse_program("tensor A : Dense;"));
}

TEST_F(TensorDeclarationTest, CSRDeclaration) {
    EXPECT_TRUE(parse_program("tensor A : CSR;"));
}

TEST_F(TensorDeclarationTest, COODeclaration) {
    EXPECT_TRUE(parse_program("tensor B : COO;"));
}

TEST_F(TensorDeclarationTest, CSCDeclaration) {
    EXPECT_TRUE(parse_program("tensor C : CSC;"));
}

TEST_F(TensorDeclarationTest, ELLPACKDeclaration) {
    EXPECT_TRUE(parse_program("tensor D : ELLPACK;"));
}

TEST_F(TensorDeclarationTest, DIADeclaration) {
    EXPECT_TRUE(parse_program("tensor E : DIA;"));
}

// Test 2: Tensor type declarations with shapes
TEST_F(TensorDeclarationTest, DenseWithShape1D) {
    EXPECT_TRUE(parse_program("tensor A : Dense<10>;"));
}

TEST_F(TensorDeclarationTest, DenseWithShape2D) {
    EXPECT_TRUE(parse_program("tensor A : Dense<10, 20>;"));
}

TEST_F(TensorDeclarationTest, CSRWithShape2D) {
    EXPECT_TRUE(parse_program("tensor A : CSR<3, 4>;"));
}

TEST_F(TensorDeclarationTest, COOWithShape3D) {
    EXPECT_TRUE(parse_program("tensor B : COO<10, 20, 30>;"));
}

TEST_F(TensorDeclarationTest, CSCWithShape2D) {
    EXPECT_TRUE(parse_program("tensor C : CSC<100, 200>;"));
}

TEST_F(TensorDeclarationTest, ELLPACKWithShape) {
    EXPECT_TRUE(parse_program("tensor D : ELLPACK<5, 8>;"));
}

TEST_F(TensorDeclarationTest, DIAWithShape) {
    EXPECT_TRUE(parse_program("tensor E : DIA<7, 7>;"));
}

// Test 3: Whitespace variations
TEST_F(TensorDeclarationTest, NoWhitespace) {
    EXPECT_TRUE(parse_program("tensor A:CSR<3,4>;"));
}

TEST_F(TensorDeclarationTest, ExtraWhitespace) {
    EXPECT_TRUE(parse_program("tensor  A  :  Dense  <  10  ,  20  >  ;"));
}

// Test 4: Multiple declarations in a program
TEST_F(TensorDeclarationTest, MultipleDeclarations) {
    EXPECT_TRUE(parse_program(
        "tensor A : Dense<10>;\n"
        "tensor B : CSR<10, 20>;\n"
        "tensor C : COO<5, 5, 5>;"
    ));
}

// Test 5: Declarations with compute statements
TEST_F(TensorDeclarationTest, DeclarationWithCompute) {
    EXPECT_TRUE(parse_program(
        "tensor A : Dense<10>;\n"
        "tensor B : Dense<10>;\n"
        "compute B[i] = A[i];"
    ));
}

// Test 6: Complete program with new syntax
TEST_F(TensorDeclarationTest, CompleteProgram) {
    EXPECT_TRUE(parse_program(
        "tensor A : CSR<100, 50>;\n"
        "tensor B : Dense<50, 20>;\n"
        "tensor C : Dense<100, 20>;\n"
        "for [A, B, C] [i, j, k] {\n"
        "    compute C[i, j] = A[i, k] * B[k, j];\n"
        "}"
    ));
}

// ============================================
// INVALID SYNTAX TESTS
// ============================================

TEST_F(TensorDeclarationTest, MissingTensorKeyword) {
    EXPECT_FALSE(parse_program("A : Dense;"));
}

TEST_F(TensorDeclarationTest, MissingColon) {
    EXPECT_FALSE(parse_program("tensor A Dense;"));
}

TEST_F(TensorDeclarationTest, MissingType) {
    EXPECT_FALSE(parse_program("tensor A : ;"));
}

TEST_F(TensorDeclarationTest, InvalidTypeLowercase) {
    EXPECT_FALSE(parse_program("tensor A : dense;"));
}

TEST_F(TensorDeclarationTest, MissingSemicolon) {
    EXPECT_FALSE(parse_program("tensor A : Dense"));
}

TEST_F(TensorDeclarationTest, MismatchedAngleBrackets) {
    EXPECT_FALSE(parse_program("tensor A : Dense<10;"));
}

TEST_F(TensorDeclarationTest, MissingClosingAngleBracket) {
    EXPECT_FALSE(parse_program("tensor A : Dense<10, 20;"));
}

TEST_F(TensorDeclarationTest, EmptyShapeList) {
    EXPECT_FALSE(parse_program("tensor A : Dense<>;"));
}

TEST_F(TensorDeclarationTest, TrailingCommaInShape) {
    EXPECT_FALSE(parse_program("tensor A : Dense<10, 20,>;"));
}

TEST_F(TensorDeclarationTest, MissingShapeDimension) {
    EXPECT_FALSE(parse_program("tensor A : Dense<10, >;"));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
