#pragma once

// Token definitions for our DSL lexer
// Based on extended grammar for IR compatibility

// Keywords
#define TOKEN_COMPUTE       1000
#define TOKEN_CALL          1001
#define TOKEN_TENSOR        1002
#define TOKEN_FOR           1003

// Tensor type keywords
#define TOKEN_DENSE         1004
#define TOKEN_CSR           1005
#define TOKEN_COO           1006
#define TOKEN_CSC           1007
#define TOKEN_ELLPACK       1008
#define TOKEN_DIA           1009

// Operators
#define TOKEN_PLUS          1010
#define TOKEN_MULT          1011
#define TOKEN_ASSIGN        1012

// Delimiters
#define TOKEN_LBRACKET      1020
#define TOKEN_RBRACKET      1021
#define TOKEN_LPAREN        1022
#define TOKEN_RPAREN        1023
#define TOKEN_LBRACE        1024
#define TOKEN_RBRACE        1025
#define TOKEN_COMMA         1026
#define TOKEN_SEMICOLON     1027
#define TOKEN_COLON         1028
#define TOKEN_LANGLE        1029
#define TOKEN_RANGLE        1030

// Literals
#define TOKEN_IDENTIFIER    1040
#define TOKEN_NUMBER        1041

// End of file
#define TOKEN_EOF           0