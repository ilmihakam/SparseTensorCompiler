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

class SequenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup before each test
    }

    void TearDown() override {
        // Cleanup after each test
    }
};

// Test simple compute statement tokenization
TEST_F(SequenceTest, SimpleComputeStatement) {
    std::string input = "compute result[i] = A[i];";
    auto tokens = tokenize_string(input);

    std::vector<int> expected = {
        TOKEN_COMPUTE,
        TOKEN_IDENTIFIER,    // result
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // i
        TOKEN_RBRACKET,
        TOKEN_ASSIGN,
        TOKEN_IDENTIFIER,    // A
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // i
        TOKEN_RBRACKET,
        TOKEN_SEMICOLON
    };

    ASSERT_EQ(tokens.size(), expected.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i], expected[i]) << "Token mismatch at position " << i;
    }
}

// Test binary operation tokenization
TEST_F(SequenceTest, BinaryOperationStatement) {
    std::string input = "compute C[i, j] = A[i, j] + B[i, j];";
    auto tokens = tokenize_string(input);

    std::vector<int> expected = {
        TOKEN_COMPUTE,
        TOKEN_IDENTIFIER,    // C
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // i
        TOKEN_COMMA,
        TOKEN_IDENTIFIER,    // j
        TOKEN_RBRACKET,
        TOKEN_ASSIGN,
        TOKEN_IDENTIFIER,    // A
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // i
        TOKEN_COMMA,
        TOKEN_IDENTIFIER,    // j
        TOKEN_RBRACKET,
        TOKEN_PLUS,
        TOKEN_IDENTIFIER,    // B
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // i
        TOKEN_COMMA,
        TOKEN_IDENTIFIER,    // j
        TOKEN_RBRACKET,
        TOKEN_SEMICOLON
    };

    ASSERT_EQ(tokens.size(), expected.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i], expected[i]) << "Token mismatch at position " << i;
    }
}

// Test function call tokenization
TEST_F(SequenceTest, FunctionCallStatement) {
    std::string input = "compute result[i] = f(A[i], 2.5);";
    auto tokens = tokenize_string(input);

    std::vector<int> expected = {
        TOKEN_COMPUTE,
        TOKEN_IDENTIFIER,    // result
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // i
        TOKEN_RBRACKET,
        TOKEN_ASSIGN,
        TOKEN_IDENTIFIER,    // f
        TOKEN_LPAREN,
        TOKEN_IDENTIFIER,    // A
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // i
        TOKEN_RBRACKET,
        TOKEN_COMMA,
        TOKEN_NUMBER,        // 2.5
        TOKEN_RPAREN,
        TOKEN_SEMICOLON
    };

    ASSERT_EQ(tokens.size(), expected.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i], expected[i]) << "Token mismatch at position " << i;
    }
}

// Test call statement tokenization
TEST_F(SequenceTest, CallStatement) {
    std::string input = "call optimize(A);";
    auto tokens = tokenize_string(input);

    std::vector<int> expected = {
        TOKEN_CALL,
        TOKEN_IDENTIFIER,    // optimize
        TOKEN_LPAREN,
        TOKEN_IDENTIFIER,    // A
        TOKEN_RPAREN,
        TOKEN_SEMICOLON
    };

    ASSERT_EQ(tokens.size(), expected.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i], expected[i]) << "Token mismatch at position " << i;
    }
}

// Test tensor declaration tokenization
TEST_F(SequenceTest, TensorDeclaration) {
    std::string input = "A: tensor[i, j];";
    auto tokens = tokenize_string(input);

    std::vector<int> expected = {
        TOKEN_IDENTIFIER,    // A
        TOKEN_COLON,
        TOKEN_TENSOR,
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // i
        TOKEN_COMMA,
        TOKEN_IDENTIFIER,    // j
        TOKEN_RBRACKET,
        TOKEN_SEMICOLON
    };

    ASSERT_EQ(tokens.size(), expected.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i], expected[i]) << "Token mismatch at position " << i;
    }
}

// Test for statement tokenization
TEST_F(SequenceTest, ForStatement) {
    std::string input = "for [A, B] [i, j] { compute C[i] = A[i] + B[i]; }";
    auto tokens = tokenize_string(input);

    std::vector<int> expected = {
        TOKEN_FOR,
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // A
        TOKEN_COMMA,
        TOKEN_IDENTIFIER,    // B
        TOKEN_RBRACKET,
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // i
        TOKEN_COMMA,
        TOKEN_IDENTIFIER,    // j
        TOKEN_RBRACKET,
        TOKEN_LBRACE,
        TOKEN_COMPUTE,
        TOKEN_IDENTIFIER,    // C
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // i
        TOKEN_RBRACKET,
        TOKEN_ASSIGN,
        TOKEN_IDENTIFIER,    // A
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // i
        TOKEN_RBRACKET,
        TOKEN_PLUS,
        TOKEN_IDENTIFIER,    // B
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // i
        TOKEN_RBRACKET,
        TOKEN_SEMICOLON,
        TOKEN_RBRACE
    };

    ASSERT_EQ(tokens.size(), expected.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i], expected[i]) << "Token mismatch at position " << i;
    }
}

// Test complex expression with multiple operators
TEST_F(SequenceTest, ComplexExpression) {
    std::string input = "compute result[i, j] = A[i, k] * B[k, j] + bias[j];";
    auto tokens = tokenize_string(input);

    std::vector<int> expected = {
        TOKEN_COMPUTE,
        TOKEN_IDENTIFIER,    // result
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // i
        TOKEN_COMMA,
        TOKEN_IDENTIFIER,    // j
        TOKEN_RBRACKET,
        TOKEN_ASSIGN,
        TOKEN_IDENTIFIER,    // A
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // i
        TOKEN_COMMA,
        TOKEN_IDENTIFIER,    // k
        TOKEN_RBRACKET,
        TOKEN_MULT,
        TOKEN_IDENTIFIER,    // B
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // k
        TOKEN_COMMA,
        TOKEN_IDENTIFIER,    // j
        TOKEN_RBRACKET,
        TOKEN_PLUS,
        TOKEN_IDENTIFIER,    // bias
        TOKEN_LBRACKET,
        TOKEN_IDENTIFIER,    // j
        TOKEN_RBRACKET,
        TOKEN_SEMICOLON
    };

    ASSERT_EQ(tokens.size(), expected.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i], expected[i]) << "Token mismatch at position " << i;
    }
}

// Test complete IR-compatible program
TEST_F(SequenceTest, CompleteProgram) {
    std::string input =
        "A: tensor[i, j];\n"
        "for [A] [i, j] {\n"
        "    compute A[i, j] = relu(A[i, j]);\n"
        "    call optimize(A);\n"
        "}";

    auto tokens = tokenize_string(input);

    // Just verify we get the right number of tokens and key tokens are present
    EXPECT_GT(tokens.size(), 20);

    // Check that all required token types appear
    bool has_tensor = false, has_for = false, has_compute = false, has_call = false;
    for (int token : tokens) {
        if (token == TOKEN_TENSOR) has_tensor = true;
        if (token == TOKEN_FOR) has_for = true;
        if (token == TOKEN_COMPUTE) has_compute = true;
        if (token == TOKEN_CALL) has_call = true;
    }

    EXPECT_TRUE(has_tensor);
    EXPECT_TRUE(has_for);
    EXPECT_TRUE(has_compute);
    EXPECT_TRUE(has_call);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}