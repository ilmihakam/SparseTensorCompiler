/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     TOKEN_COMPUTE = 258,
     TOKEN_CALL = 259,
     TOKEN_TENSOR = 260,
     TOKEN_FOR = 261,
     TOKEN_DENSE = 262,
     TOKEN_CSR = 263,
     TOKEN_COO = 264,
     TOKEN_CSC = 265,
     TOKEN_ELLPACK = 266,
     TOKEN_DIA = 267,
     TOKEN_PLUS = 268,
     TOKEN_MULT = 269,
     TOKEN_ASSIGN = 270,
     TOKEN_LBRACKET = 271,
     TOKEN_RBRACKET = 272,
     TOKEN_LPAREN = 273,
     TOKEN_RPAREN = 274,
     TOKEN_LBRACE = 275,
     TOKEN_RBRACE = 276,
     TOKEN_COMMA = 277,
     TOKEN_SEMICOLON = 278,
     TOKEN_COLON = 279,
     TOKEN_LANGLE = 280,
     TOKEN_RANGLE = 281,
     TOKEN_IDENTIFIER = 282,
     TOKEN_NUMBER = 283
   };
#endif
/* Tokens.  */
#define TOKEN_COMPUTE 258
#define TOKEN_CALL 259
#define TOKEN_TENSOR 260
#define TOKEN_FOR 261
#define TOKEN_DENSE 262
#define TOKEN_CSR 263
#define TOKEN_COO 264
#define TOKEN_CSC 265
#define TOKEN_ELLPACK 266
#define TOKEN_DIA 267
#define TOKEN_PLUS 268
#define TOKEN_MULT 269
#define TOKEN_ASSIGN 270
#define TOKEN_LBRACKET 271
#define TOKEN_RBRACKET 272
#define TOKEN_LPAREN 273
#define TOKEN_RPAREN 274
#define TOKEN_LBRACE 275
#define TOKEN_RBRACE 276
#define TOKEN_COMMA 277
#define TOKEN_SEMICOLON 278
#define TOKEN_COLON 279
#define TOKEN_LANGLE 280
#define TOKEN_RANGLE 281
#define TOKEN_IDENTIFIER 282
#define TOKEN_NUMBER 283




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 51 "src/parser.y"
{
    char* str;
    void* ptr;  /* Generic pointer for all AST nodes */
}
/* Line 1529 of yacc.c.  */
#line 110 "/Users/ilmihakam/co/backend/SparseTensorCompiler/build/parser_generated.h"
	YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE yylval;

