#include <gtest/gtest.h>
#include <vector>
#include <string>
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

// Helper function to get lexeme value
std::string get_lexeme_value(const std::string& input) {
    yy_scan_string(input.c_str());
    yylex();
    std::string result(yytext);
    yylex_destroy();
    return result;
}

class LiteralTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup before each test
    }

    void TearDown() override {
        // Cleanup after each test
    }
};

// Test identifiers following [a-zA-Z_][a-zA-Z0-9_]* pattern
TEST_F(LiteralTest, SimpleIdentifiers) {
    auto tokens = tokenize_string("A");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_IDENTIFIER);

    tokens = tokenize_string("result");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_IDENTIFIER);

    tokens = tokenize_string("f");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_IDENTIFIER);
}

TEST_F(LiteralTest, IdentifiersWithUnderscore) {
    auto tokens = tokenize_string("_variable");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_IDENTIFIER);

    tokens = tokenize_string("tensor_name");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_IDENTIFIER);

    tokens = tokenize_string("_");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_IDENTIFIER);
}

TEST_F(LiteralTest, IdentifiersWithNumbers) {
    auto tokens = tokenize_string("variable123");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_IDENTIFIER);

    tokens = tokenize_string("A1");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_IDENTIFIER);

    tokens = tokenize_string("tensor_2d");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_IDENTIFIER);
}

TEST_F(LiteralTest, MLFunctionNames) {
    // Common ML function names should be recognized as identifiers
    auto tokens = tokenize_string("relu");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_IDENTIFIER);

    tokens = tokenize_string("softmax");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_IDENTIFIER);

    tokens = tokenize_string("sigmoid");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_IDENTIFIER);
}

// Test numbers following [0-9]+(\.[0-9]+)? pattern
TEST_F(LiteralTest, IntegerNumbers) {
    auto tokens = tokenize_string("42");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_NUMBER);

    tokens = tokenize_string("0");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_NUMBER);

    tokens = tokenize_string("123");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_NUMBER);
}

TEST_F(LiteralTest, DecimalNumbers) {
    auto tokens = tokenize_string("3.14");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_NUMBER);

    tokens = tokenize_string("0.0");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_NUMBER);

    tokens = tokenize_string("2.5");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_NUMBER);

    tokens = tokenize_string("1.0");
    ASSERT_EQ(tokens.size(), 1);
    EXPECT_EQ(tokens[0], TOKEN_NUMBER);
}

// Test lexeme value extraction
TEST_F(LiteralTest, IdentifierLexemeValues) {
    EXPECT_EQ(get_lexeme_value("tensor_name"), "tensor_name");
    EXPECT_EQ(get_lexeme_value("A"), "A");
    EXPECT_EQ(get_lexeme_value("relu"), "relu");
    EXPECT_EQ(get_lexeme_value("variable123"), "variable123");
}

TEST_F(LiteralTest, NumberLexemeValues) {
    EXPECT_EQ(get_lexeme_value("42"), "42");
    EXPECT_EQ(get_lexeme_value("3.14"), "3.14");
    EXPECT_EQ(get_lexeme_value("0.0"), "0.0");
    EXPECT_EQ(get_lexeme_value("123"), "123");
}

// Test mixed identifier and number sequences
TEST_F(LiteralTest, MixedLiterals) {
    auto tokens = tokenize_string("A 42 tensor_name 3.14");
    ASSERT_EQ(tokens.size(), 4);
    EXPECT_EQ(tokens[0], TOKEN_IDENTIFIER);
    EXPECT_EQ(tokens[1], TOKEN_NUMBER);
    EXPECT_EQ(tokens[2], TOKEN_IDENTIFIER);
    EXPECT_EQ(tokens[3], TOKEN_NUMBER);
}

// Test that keywords are not recognized as identifiers
TEST_F(LiteralTest, KeywordsNotIdentifiers) {
    auto tokens = tokenize_string("compute call tensor for");
    ASSERT_EQ(tokens.size(), 4);
    EXPECT_EQ(tokens[0], TOKEN_COMPUTE);
    EXPECT_EQ(tokens[1], TOKEN_CALL);
    EXPECT_EQ(tokens[2], TOKEN_TENSOR);
    EXPECT_EQ(tokens[3], TOKEN_FOR);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}