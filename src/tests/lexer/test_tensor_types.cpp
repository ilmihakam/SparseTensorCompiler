#include <gtest/gtest.h>
#include "parser_generated.h"  // Use Bison-generated tokens

// Flex/Bison interface
extern int yylex();
extern void yy_scan_string(const char*);
extern void yylex_destroy();
extern char* yytext;

class TensorTypeTokenTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {
        yylex_destroy();
    }

    int scanToken(const char* input) {
        yy_scan_string(input);
        return yylex();
    }
};

// ============================================
// TENSOR TYPE KEYWORD TESTS
// ============================================

TEST_F(TensorTypeTokenTest, DenseKeyword) {
    EXPECT_EQ(scanToken("Dense"), TOKEN_DENSE);
}

TEST_F(TensorTypeTokenTest, CSRKeyword) {
    EXPECT_EQ(scanToken("CSR"), TOKEN_CSR);
}

TEST_F(TensorTypeTokenTest, COOKeyword) {
    EXPECT_EQ(scanToken("COO"), TOKEN_COO);
}

TEST_F(TensorTypeTokenTest, CSCKeyword) {
    EXPECT_EQ(scanToken("CSC"), TOKEN_CSC);
}

TEST_F(TensorTypeTokenTest, ELLPACKKeyword) {
    EXPECT_EQ(scanToken("ELLPACK"), TOKEN_ELLPACK);
}

TEST_F(TensorTypeTokenTest, DIAKeyword) {
    EXPECT_EQ(scanToken("DIA"), TOKEN_DIA);
}

// ============================================
// ANGLE BRACKET TESTS
// ============================================

TEST_F(TensorTypeTokenTest, LeftAngleBracket) {
    EXPECT_EQ(scanToken("<"), TOKEN_LANGLE);
}

TEST_F(TensorTypeTokenTest, RightAngleBracket) {
    EXPECT_EQ(scanToken(">"), TOKEN_RANGLE);
}

// ============================================
// COMBINED TESTS - New Declaration Syntax
// ============================================

TEST_F(TensorTypeTokenTest, DenseDeclarationSequence) {
    yy_scan_string("tensor A : Dense ;");

    EXPECT_EQ(yylex(), TOKEN_TENSOR);
    EXPECT_EQ(yylex(), TOKEN_IDENTIFIER);
    EXPECT_EQ(yylex(), TOKEN_COLON);
    EXPECT_EQ(yylex(), TOKEN_DENSE);
    EXPECT_EQ(yylex(), TOKEN_SEMICOLON);
}

TEST_F(TensorTypeTokenTest, CSRWithShapeSequence) {
    yy_scan_string("tensor A : CSR < 3 , 4 > ;");

    EXPECT_EQ(yylex(), TOKEN_TENSOR);
    EXPECT_EQ(yylex(), TOKEN_IDENTIFIER);
    EXPECT_EQ(yylex(), TOKEN_COLON);
    EXPECT_EQ(yylex(), TOKEN_CSR);
    EXPECT_EQ(yylex(), TOKEN_LANGLE);
    EXPECT_EQ(yylex(), TOKEN_NUMBER);
    EXPECT_EQ(yylex(), TOKEN_COMMA);
    EXPECT_EQ(yylex(), TOKEN_NUMBER);
    EXPECT_EQ(yylex(), TOKEN_RANGLE);
    EXPECT_EQ(yylex(), TOKEN_SEMICOLON);
}

TEST_F(TensorTypeTokenTest, COODeclarationCompact) {
    yy_scan_string("tensor B:COO<10,20,30>;");

    EXPECT_EQ(yylex(), TOKEN_TENSOR);
    EXPECT_EQ(yylex(), TOKEN_IDENTIFIER);
    EXPECT_EQ(yylex(), TOKEN_COLON);
    EXPECT_EQ(yylex(), TOKEN_COO);
    EXPECT_EQ(yylex(), TOKEN_LANGLE);
    EXPECT_EQ(yylex(), TOKEN_NUMBER);
    EXPECT_EQ(yylex(), TOKEN_COMMA);
    EXPECT_EQ(yylex(), TOKEN_NUMBER);
    EXPECT_EQ(yylex(), TOKEN_COMMA);
    EXPECT_EQ(yylex(), TOKEN_NUMBER);
    EXPECT_EQ(yylex(), TOKEN_RANGLE);
    EXPECT_EQ(yylex(), TOKEN_SEMICOLON);
}

// ============================================
// CASE SENSITIVITY TESTS
// ============================================

TEST_F(TensorTypeTokenTest, TypeKeywordsAreCaseSensitive) {
    // Lowercase should be treated as identifiers
    EXPECT_EQ(scanToken("csr"), TOKEN_IDENTIFIER);
    EXPECT_EQ(scanToken("dense"), TOKEN_IDENTIFIER);
    EXPECT_EQ(scanToken("coo"), TOKEN_IDENTIFIER);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
