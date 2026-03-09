#include <gtest/gtest.h>
#include <string>
#include "tokens.h"

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

class ParserBasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup before each test
    }

    void TearDown() override {
        // Cleanup after each test
    }
};

// ============================================
// VALID SYNTAX TESTS - These should parse successfully
// ============================================

// Test 1: Tensor Declarations (Type-Enforced)
TEST_F(ParserBasicTest, SimpleTensorDeclaration) {
    EXPECT_TRUE(parse_program("tensor A : Dense;"));
}

TEST_F(ParserBasicTest, TensorDeclarationWithShape1D) {
    EXPECT_TRUE(parse_program("tensor A : Dense<10>;"));
}

TEST_F(ParserBasicTest, TensorDeclarationWithShape2D) {
    EXPECT_TRUE(parse_program("tensor A : CSR<10, 20>;"));
}

TEST_F(ParserBasicTest, TensorDeclarationWithShape3D) {
    EXPECT_TRUE(parse_program("tensor A : COO<10, 20, 30>;"));
}

// Test 2: Compute Statements
TEST_F(ParserBasicTest, SimpleComputeStatement) {
    EXPECT_TRUE(parse_program("compute result[i] = A[i];"));
}

TEST_F(ParserBasicTest, ComputeWithAddition) {
    EXPECT_TRUE(parse_program("compute C[i] = A[i] + B[i];"));
}

TEST_F(ParserBasicTest, ComputeWithMultiplication) {
    EXPECT_TRUE(parse_program("compute C[i] = A[i] * B[i];"));
}

TEST_F(ParserBasicTest, ComputeWithMixedOperators) {
    EXPECT_TRUE(parse_program("compute C[i, j] = A[i, k] * B[k, j] + bias[j];"));
}

TEST_F(ParserBasicTest, ComputeWithNumber) {
    EXPECT_TRUE(parse_program("compute A[i] = 42;"));
}

TEST_F(ParserBasicTest, ComputeWithDecimal) {
    EXPECT_TRUE(parse_program("compute A[i] = 3.14;"));
}

TEST_F(ParserBasicTest, ComputeWithFunctionCall) {
    EXPECT_TRUE(parse_program("compute result[i] = relu(A[i]);"));
}

TEST_F(ParserBasicTest, ComputeWithFunctionCallMultipleArgs) {
    EXPECT_TRUE(parse_program("compute result[i] = f(A[i], B[i]);"));
}

TEST_F(ParserBasicTest, ComputeWithFunctionCallAndNumber) {
    EXPECT_TRUE(parse_program("compute result[i] = f(A[i], 2.5);"));
}

// Test 3: Call Statements
TEST_F(ParserBasicTest, SimpleCallStatement) {
    EXPECT_TRUE(parse_program("call optimize(A);"));
}

TEST_F(ParserBasicTest, CallStatementNoArguments) {
    EXPECT_TRUE(parse_program("call print();"));
}

TEST_F(ParserBasicTest, CallStatementMultipleArguments) {
    EXPECT_TRUE(parse_program("call function(A, B, C);"));
}

TEST_F(ParserBasicTest, CallStatementWithNumbers) {
    EXPECT_TRUE(parse_program("call function(A, 42, 3.14);"));
}

// Test 4: For Statements
TEST_F(ParserBasicTest, SimpleForStatement) {
    EXPECT_TRUE(parse_program("for [A] [i] { compute A[i] = A[i] + 1; }"));
}

TEST_F(ParserBasicTest, ForStatementMultipleTensors) {
    EXPECT_TRUE(parse_program("for [A, B] [i] { compute C[i] = A[i] + B[i]; }"));
}

TEST_F(ParserBasicTest, ForStatementMultipleIndices) {
    EXPECT_TRUE(parse_program("for [A] [i, j] { compute A[i, j] = 0; }"));
}

TEST_F(ParserBasicTest, ForStatementMultipleStatements) {
    EXPECT_TRUE(parse_program(
        "for [A] [i] { "
        "compute A[i] = relu(A[i]); "
        "call normalize(A); "
        "}"
    ));
}

TEST_F(ParserBasicTest, ForStatementEmpty) {
    EXPECT_TRUE(parse_program("for [A] [i] { }"));
}

// Test 5: Complete Programs (Multiple Statements)
TEST_F(ParserBasicTest, ProgramWithDeclarationAndCompute) {
    EXPECT_TRUE(parse_program(
        "tensor A : Dense<10>;\n"
        "compute A[i] = 0;"
    ));
}

TEST_F(ParserBasicTest, ProgramWithMultipleDeclarations) {
    EXPECT_TRUE(parse_program(
        "tensor A : Dense<10>;\n"
        "tensor B : CSR<10>;\n"
        "tensor C : COO<10>;"
    ));
}

TEST_F(ParserBasicTest, ProgramComplete) {
    EXPECT_TRUE(parse_program(
        "tensor A : Dense<10, 20>;\n"
        "for [A] [i, j] {\n"
        "    compute A[i, j] = relu(A[i, j]);\n"
        "    call optimize(A);\n"
        "}"
    ));
}

TEST_F(ParserBasicTest, ProgramMatrixMultiplication) {
    EXPECT_TRUE(parse_program(
        "tensor A : CSR<100, 50>;\n"
        "tensor B : Dense<50, 20>;\n"
        "tensor C : Dense<100, 20>;\n"
        "for [A, B, C] [i, j, k] {\n"
        "    compute C[i, j] = A[i, k] * B[k, j];\n"
        "}"
    ));
}

// Test 6: Empty Program
TEST_F(ParserBasicTest, EmptyProgram) {
    EXPECT_TRUE(parse_program(""));
}

// ============================================
// INVALID SYNTAX TESTS - These should fail to parse
// ============================================

TEST_F(ParserBasicTest, MissingSemicolonInDeclaration) {
    EXPECT_FALSE(parse_program("tensor A : Dense"));
}

TEST_F(ParserBasicTest, MissingSemicolonInCompute) {
    EXPECT_FALSE(parse_program("compute A[i] = B[i]"));
}

TEST_F(ParserBasicTest, MissingSemicolonInCall) {
    EXPECT_FALSE(parse_program("call optimize(A)"));
}

TEST_F(ParserBasicTest, MismatchedBracketsInTensorAccess) {
    EXPECT_FALSE(parse_program("compute A[i = B[i];"));
}

TEST_F(ParserBasicTest, MismatchedBracesInFor) {
    EXPECT_FALSE(parse_program("for [A] [i] { compute A[i] = 0;"));
}

TEST_F(ParserBasicTest, InvalidComputeMissingAssign) {
    EXPECT_FALSE(parse_program("compute A[i] B[i];"));
}

TEST_F(ParserBasicTest, InvalidForMissingTensorList) {
    EXPECT_FALSE(parse_program("for [i] { compute A[i] = 0; }"));
}

TEST_F(ParserBasicTest, InvalidExpressionMissingOperand) {
    EXPECT_FALSE(parse_program("compute A[i] = B[i] +;"));
}

TEST_F(ParserBasicTest, InvalidCallMissingParenthesis) {
    EXPECT_FALSE(parse_program("call optimize A;"));
}

TEST_F(ParserBasicTest, InvalidDeclarationMissingColon) {
    EXPECT_FALSE(parse_program("tensor A Dense;"));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
