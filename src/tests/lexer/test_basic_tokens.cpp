#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <sstream>
#include "parser_generated.h"

// Forward declarations for lexer interface
extern int yylex();
extern char* yytext;
extern void yy_scan_string(const char* str);
extern void yylex_destroy();

// Helper function to tokenize a string and return token sequence
std::vector<int> tokenize_string(const std::string& input) {
    std::vector<int> tokens;

    yy_scan_string(input.c_str());

    int token;
    while ((token = yylex()) != 0) {  // 0 = EOF
        tokens.push_back(token);
    }

    yylex_destroy();
    return tokens;
}

class BasicTokenTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup before each test
    }

    void TearDown() override {
        // Cleanup after each test
    }
};

// Test basic keywords
TEST_F(BasicTokenTest, KeywordCompute) {
    auto tokens = tokenize_string("compute");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_COMPUTE);
}

TEST_F(BasicTokenTest, KeywordCall) {
    auto tokens = tokenize_string("call");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_CALL);
}

TEST_F(BasicTokenTest, KeywordTensor) {
    auto tokens = tokenize_string("tensor");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_TENSOR);
}

TEST_F(BasicTokenTest, KeywordFor) {
    auto tokens = tokenize_string("for");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_FOR);
}

// Test operators
TEST_F(BasicTokenTest, OperatorPlus) {
    auto tokens = tokenize_string("+");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_PLUS);
}

TEST_F(BasicTokenTest, OperatorMult) {
    auto tokens = tokenize_string("*");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_MULT);
}

TEST_F(BasicTokenTest, OperatorAssign) {
    auto tokens = tokenize_string("=");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_ASSIGN);
}

// Test delimiters
TEST_F(BasicTokenTest, DelimiterLBracket) {
    auto tokens = tokenize_string("[");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_LBRACKET);
}

TEST_F(BasicTokenTest, DelimiterRBracket) {
    auto tokens = tokenize_string("]");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_RBRACKET);
}

TEST_F(BasicTokenTest, DelimiterLParen) {
    auto tokens = tokenize_string("(");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_LPAREN);
}

TEST_F(BasicTokenTest, DelimiterRParen) {
    auto tokens = tokenize_string(")");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_RPAREN);
}

TEST_F(BasicTokenTest, DelimiterLBrace) {
    auto tokens = tokenize_string("{");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_LBRACE);
}

TEST_F(BasicTokenTest, DelimiterRBrace) {
    auto tokens = tokenize_string("}");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_RBRACE);
}

TEST_F(BasicTokenTest, DelimiterComma) {
    auto tokens = tokenize_string(",");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_COMMA);
}

TEST_F(BasicTokenTest, DelimiterSemicolon) {
    auto tokens = tokenize_string(";");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_SEMICOLON);
}

TEST_F(BasicTokenTest, DelimiterColon) {
    auto tokens = tokenize_string(":");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_COLON);
}

// Test multiple tokens together
TEST_F(BasicTokenTest, MultipleOperators) {
    auto tokens = tokenize_string("+ * =");
    ASSERT_EQ(tokens.size(), 3);
    EXPECT_EQ(tokens[0], TOKEN_PLUS);
    EXPECT_EQ(tokens[1], TOKEN_MULT);
    EXPECT_EQ(tokens[2], TOKEN_ASSIGN);
}

TEST_F(BasicTokenTest, AllDelimiters) {
    auto tokens = tokenize_string("[ ] ( ) { } , ; :");
    ASSERT_EQ(tokens.size(), 9);
    EXPECT_EQ(tokens[0], TOKEN_LBRACKET);
    EXPECT_EQ(tokens[1], TOKEN_RBRACKET);
    EXPECT_EQ(tokens[2], TOKEN_LPAREN);
    EXPECT_EQ(tokens[3], TOKEN_RPAREN);
    EXPECT_EQ(tokens[4], TOKEN_LBRACE);
    EXPECT_EQ(tokens[5], TOKEN_RBRACE);
    EXPECT_EQ(tokens[6], TOKEN_COMMA);
    EXPECT_EQ(tokens[7], TOKEN_SEMICOLON);
    EXPECT_EQ(tokens[8], TOKEN_COLON);
}

// Test whitespace handling
TEST_F(BasicTokenTest, WhitespaceIgnored) {
    auto tokens = tokenize_string("  \t\n  compute  \t  +  \n  ");
    ASSERT_EQ(tokens.size(), 2);
    EXPECT_EQ(tokens[0], TOKEN_COMPUTE);
    EXPECT_EQ(tokens[1], TOKEN_PLUS);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}