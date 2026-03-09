%{
/* Sparse Tensor Compiler - Parser Definition
 * Grammar rules for DSL syntax validation and AST generation
 */

#include "tokens.h"
#include "ast.h"
#include <stdio.h>
#include <string.h>
#include <vector>
#include <memory>
#include <string>

using namespace SparseTensorCompiler;

/* Type aliases for union compatibility */
typedef Program ASTProgram;
typedef Statement ASTStatement;
typedef Expression ASTExpression;
typedef Declaration ASTDeclaration;
typedef Computation ASTComputation;
typedef CallStatement ASTCallStatement;
typedef ForStatement ASTForStatement;
typedef TensorAccess ASTTensorAccess;
typedef FunctionCall ASTFunctionCall;
typedef std::vector<std::unique_ptr<Statement>> StatementList;
typedef std::vector<std::unique_ptr<Expression>> ExpressionList;
typedef std::vector<std::string> StringList;

/* Forward declarations */
int yyerror(const char *s);
extern int yylex();
extern int yylineno;
extern char* yytext;

/* Global AST root - accessible after parsing */
std::unique_ptr<Program> g_program = nullptr;

/* Helper to create a string from yytext */
std::string makeString(const char* text) {
    return std::string(text);
}

%}

/* ============================================
 * BISON UNION - Semantic Values
 * Using void* for C++ compatibility
 * ============================================ */

%union {
    char* str;
    void* ptr;  /* Generic pointer for all AST nodes */
}

/* ============================================
 * TOKEN DECLARATIONS
 * ============================================ */

/* Keywords */
%token TOKEN_COMPUTE
%token TOKEN_CALL
%token TOKEN_TENSOR
%token TOKEN_FOR

/* Tensor type keywords */
%token TOKEN_DENSE
%token TOKEN_CSR
%token TOKEN_COO
%token TOKEN_CSC
%token TOKEN_ELLPACK
%token TOKEN_DIA

/* Operators */
%token TOKEN_PLUS
%token TOKEN_MULT
%token TOKEN_ASSIGN

/* Delimiters */
%token TOKEN_LBRACKET
%token TOKEN_RBRACKET
%token TOKEN_LPAREN
%token TOKEN_RPAREN
%token TOKEN_LBRACE
%token TOKEN_RBRACE
%token TOKEN_COMMA
%token TOKEN_SEMICOLON
%token TOKEN_COLON
%token TOKEN_LANGLE
%token TOKEN_RANGLE

/* Literals */
%token<str> TOKEN_IDENTIFIER
%token<str> TOKEN_NUMBER

/* ============================================
 * NON-TERMINAL TYPES
 * All use ptr type with appropriate casts in actions
 * ============================================ */

%type<ptr> program
%type<ptr> statement_list
%type<ptr> statement
%type<ptr> declaration
%type<ptr> computation
%type<ptr> call_statement
%type<ptr> for_statement
%type<ptr> tensor_access
%type<ptr> expression
%type<ptr> function_call
%type<ptr> argument_list
%type<ptr> index_list identifier_list
%type<ptr> tensor_type
%type<ptr> shape_list

/* ============================================
 * OPERATOR PRECEDENCE AND ASSOCIATIVITY
 * ============================================ */

%left TOKEN_PLUS          /* + is left-associative, lower precedence */
%left TOKEN_MULT          /* * is left-associative, higher precedence */

%%

/* ============================================
 * GRAMMAR RULES WITH SEMANTIC ACTIONS
 * ============================================ */

/* Entry point: program is a sequence of statements */
program:
      /* empty program */
      {
          auto* prog = new Program();
          g_program = std::unique_ptr<Program>(prog);
          $$ = prog;
      }
    | statement_list
      {
          auto* prog = new Program();
          auto* stmts = static_cast<StatementList*>($1);
          for (auto& stmt : *stmts) {
              prog->addStatement(std::move(stmt));
          }
          delete stmts;
          g_program = std::unique_ptr<Program>(prog);
          $$ = prog;
      }
    ;

/* Statement list: one or more statements */
statement_list:
      statement
      {
          auto* list = new StatementList();
          list->push_back(std::unique_ptr<Statement>(static_cast<Statement*>($1)));
          $$ = list;
      }
    | statement_list statement
      {
          auto* list = static_cast<StatementList*>($1);
          list->push_back(std::unique_ptr<Statement>(static_cast<Statement*>($2)));
          $$ = list;
      }
    ;

/* Statement: can be declaration, compute, call, or for loop */
statement:
      declaration TOKEN_SEMICOLON
      {
          $$ = $1;
      }
    | computation TOKEN_SEMICOLON
      {
          $$ = $1;
      }
    | call_statement TOKEN_SEMICOLON
      {
          $$ = $1;
      }
    | for_statement
      {
          $$ = $1;
      }
    ;

/* Tensor declaration: tensor A : Dense<10, 20>; */
declaration:
    /* Typed declaration without shape: tensor A : Dense; */
    TOKEN_TENSOR TOKEN_IDENTIFIER TOKEN_COLON tensor_type
    {
        auto* type = static_cast<std::string*>($4);
        $$ = new Declaration(std::string($2), *type);
        delete type;
        free($2);
    }
    /* Typed declaration with shape: tensor A : Dense<10, 20>; */
    | TOKEN_TENSOR TOKEN_IDENTIFIER TOKEN_COLON tensor_type TOKEN_LANGLE shape_list TOKEN_RANGLE
    {
        auto* type = static_cast<std::string*>($4);
        auto* shapeList = static_cast<StringList*>($6);
        $$ = new Declaration(std::string($2), *type, *shapeList);
        delete type;
        delete shapeList;
        free($2);
    }
    ;

/* Compute statement: compute C[i] = A[i] + B[i]; */
computation:
    TOKEN_COMPUTE tensor_access TOKEN_ASSIGN expression
    {
        $$ = new Computation(
            std::unique_ptr<TensorAccess>(static_cast<TensorAccess*>($2)),
            std::unique_ptr<Expression>(static_cast<Expression*>($4))
        );
    }
    ;

/* Call statement: call optimize(A); */
call_statement:
    TOKEN_CALL TOKEN_IDENTIFIER TOKEN_LPAREN argument_list TOKEN_RPAREN
    {
        auto* call = new CallStatement(std::string($2));
        auto* args = static_cast<ExpressionList*>($4);
        for (auto& arg : *args) {
            call->addArgument(std::move(arg));
        }
        delete args;
        free($2);
        $$ = call;
    }
    | TOKEN_CALL TOKEN_IDENTIFIER TOKEN_LPAREN TOKEN_RPAREN  /* no arguments */
    {
        $$ = new CallStatement(std::string($2));
        free($2);
    }
    ;

/* For statement: for [A, B] [i, j] { statements } */
for_statement:
    TOKEN_FOR TOKEN_LBRACKET identifier_list TOKEN_RBRACKET
              TOKEN_LBRACKET index_list TOKEN_RBRACKET
              TOKEN_LBRACE statement_list TOKEN_RBRACE
    {
        auto* tensors = static_cast<StringList*>($3);
        auto* indices = static_cast<StringList*>($6);
        auto* stmts = static_cast<StatementList*>($9);
        auto* forStmt = new ForStatement(*tensors, *indices);
        for (auto& stmt : *stmts) {
            forStmt->addStatement(std::move(stmt));
        }
        delete tensors;
        delete indices;
        delete stmts;
        $$ = forStmt;
    }
    | TOKEN_FOR TOKEN_LBRACKET identifier_list TOKEN_RBRACKET
              TOKEN_LBRACKET index_list TOKEN_RBRACKET
              TOKEN_LBRACE TOKEN_RBRACE  /* empty body */
    {
        auto* tensors = static_cast<StringList*>($3);
        auto* indices = static_cast<StringList*>($6);
        $$ = new ForStatement(*tensors, *indices);
        delete tensors;
        delete indices;
    }
    ;

/* Tensor access: A[i, j] */
tensor_access:
    TOKEN_IDENTIFIER TOKEN_LBRACKET index_list TOKEN_RBRACKET
    {
        auto* indices = static_cast<StringList*>($3);
        $$ = new TensorAccess(std::string($1), *indices);
        delete indices;
        free($1);
    }
    ;

/* Expression: tensor access, function call, number, identifier, or binary operations */
expression:
      tensor_access
      {
          $$ = $1;
      }
    | function_call
      {
          $$ = $1;
      }
    | TOKEN_NUMBER
      {
          $$ = new Number(std::string($1));
          free($1);
      }
    | TOKEN_IDENTIFIER
      {
          $$ = new Identifier(std::string($1));
          free($1);
      }
    | expression TOKEN_PLUS expression
      {
          $$ = new BinaryOp(
              BinaryOp::ADD,
              std::unique_ptr<Expression>(static_cast<Expression*>($1)),
              std::unique_ptr<Expression>(static_cast<Expression*>($3))
          );
      }
    | expression TOKEN_MULT expression
      {
          $$ = new BinaryOp(
              BinaryOp::MULT,
              std::unique_ptr<Expression>(static_cast<Expression*>($1)),
              std::unique_ptr<Expression>(static_cast<Expression*>($3))
          );
      }
    ;

/* Function call: relu(A[i]) or f(A, B, 2.5) */
function_call:
    TOKEN_IDENTIFIER TOKEN_LPAREN argument_list TOKEN_RPAREN
    {
        auto* func = new FunctionCall(std::string($1));
        auto* args = static_cast<ExpressionList*>($3);
        for (auto& arg : *args) {
            func->addArgument(std::move(arg));
        }
        delete args;
        free($1);
        $$ = func;
    }
    | TOKEN_IDENTIFIER TOKEN_LPAREN TOKEN_RPAREN  /* no arguments */
    {
        $$ = new FunctionCall(std::string($1));
        free($1);
    }
    ;

/* Argument list: comma-separated expressions */
argument_list:
      expression
      {
          auto* list = new ExpressionList();
          list->push_back(std::unique_ptr<Expression>(static_cast<Expression*>($1)));
          $$ = list;
      }
    | argument_list TOKEN_COMMA expression
      {
          auto* list = static_cast<ExpressionList*>($1);
          list->push_back(std::unique_ptr<Expression>(static_cast<Expression*>($3)));
          $$ = list;
      }
    ;

/* Index list: comma-separated identifiers (i, j, k) */
index_list:
      TOKEN_IDENTIFIER
      {
          auto* list = new StringList();
          list->push_back(std::string($1));
          free($1);
          $$ = list;
      }
    | index_list TOKEN_COMMA TOKEN_IDENTIFIER
      {
          auto* list = static_cast<StringList*>($1);
          list->push_back(std::string($3));
          free($3);
          $$ = list;
      }
    ;

/* Identifier list: comma-separated identifiers (A, B, C) */
identifier_list:
      TOKEN_IDENTIFIER
      {
          auto* list = new StringList();
          list->push_back(std::string($1));
          free($1);
          $$ = list;
      }
    | identifier_list TOKEN_COMMA TOKEN_IDENTIFIER
      {
          auto* list = static_cast<StringList*>($1);
          list->push_back(std::string($3));
          free($3);
          $$ = list;
      }
    ;

/* Tensor type: Dense | CSR | COO | CSC | ELLPACK | DIA */
tensor_type:
      TOKEN_DENSE
      {
          /* Store as string for now - will be used when extending AST */
          auto* type = new std::string("Dense");
          $$ = type;
      }
    | TOKEN_CSR
      {
          auto* type = new std::string("CSR");
          $$ = type;
      }
    | TOKEN_COO
      {
          auto* type = new std::string("COO");
          $$ = type;
      }
    | TOKEN_CSC
      {
          auto* type = new std::string("CSC");
          $$ = type;
      }
    | TOKEN_ELLPACK
      {
          auto* type = new std::string("ELLPACK");
          $$ = type;
      }
    | TOKEN_DIA
      {
          auto* type = new std::string("DIA");
          $$ = type;
      }
    ;

/* Shape list: comma-separated numbers (10, 20, 30) */
shape_list:
      TOKEN_NUMBER
      {
          auto* list = new StringList();
          list->push_back(std::string($1));
          free($1);
          $$ = list;
      }
    | shape_list TOKEN_COMMA TOKEN_NUMBER
      {
          auto* list = static_cast<StringList*>($1);
          list->push_back(std::string($3));
          free($3);
          $$ = list;
      }
    ;

%%

/* ============================================
 * ERROR HANDLING
 * ============================================ */

int yyerror(const char *s) {
    /* Track errors for test validation */
    yynerrs++;

    /* Print error message (can be suppressed in tests) */
    #ifdef PARSER_DEBUG
    fprintf(stderr, "Parser error at line %d: %s\n", yylineno, s);
    #endif

    return 0;
}
