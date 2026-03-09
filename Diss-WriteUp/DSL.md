# SparseTensorCompiler DSL — Comprehensive Reference

This document covers the SparseTensorCompiler Domain-Specific Language (DSL) from two angles: **design** (what the language is, why it was built this way, its theoretical foundations) and **implementation** (how it was actually built using Flex, Bison, and a C++ AST/IR pipeline).

---

## Part 1: Design

### 1. Overview and Motivation

#### Why a DSL?

Sparse linear algebra is notoriously hard to write efficiently by hand. The access patterns of CSR and CSC formats require boilerplate pointer arithmetic that obscures intent, and the choice of loop order has non-obvious performance consequences that depend on the input matrix structure.

A DSL lets the programmer express *what* to compute (the mathematical equation) and leaves *how* to compute it (loop order, tiling, format-specific iteration) to the compiler. This separation is the core thesis of compilers like TACO; our DSL adopts the same philosophy but restricts scope to the two kernels where we want to study optimization behaviour: SpMV and SpMM.

#### Design Goals

The DSL was designed around five goals, traceable through `docs/grammar.md`:

| Goal | How it is achieved |
|------|--------------------|
| **Type-enforced** | Every tensor declaration carries an explicit storage format (`Dense`, `CSR`, `CSC`, ...) |
| **Shape-aware** | Optional dimension list `<M, N>` in angle brackets; used for static validation and code generation |
| **Einstein notation** | Repeated index on the RHS triggers implicit summation; no explicit `sum` keyword needed |
| **Clean syntax** | Statements end with `;`; index notation mirrors mathematical convention |
| **Forward compatible** | Parser accepts call statements and for-statements; IR lowering ignores what it does not yet handle |

#### Target Domain

The compiler targets single-threaded sparse linear algebra on CSR and CSC matrices. Out of scope by design: parallelism, COO/ELLPACK/DIA codegen, data-dependent control flow, format inference, and schedule search. These constraints keep the research question focused: *when do loop interchange and loop blocking help, and on which input characteristics?*

---

### 2. Syntax and Grammar

#### 2.1 Complete EBNF Grammar

The authoritative grammar is defined in `src/parser.y`. The canonical EBNF form is:

```ebnf
program         ::= statement*

statement       ::= declaration ';'
                  | computation ';'
                  | call_statement ';'
                  | for_statement

declaration     ::= 'tensor' IDENTIFIER ':' tensor_type
                  | 'tensor' IDENTIFIER ':' tensor_type '<' shape_list '>'

tensor_type     ::= 'Dense' | 'CSR' | 'CSC' | 'COO' | 'ELLPACK' | 'DIA'

shape_list      ::= NUMBER (',' NUMBER)*

computation     ::= 'compute' tensor_access '=' expression

call_statement  ::= 'call' IDENTIFIER '(' argument_list ')'
                  | 'call' IDENTIFIER '(' ')'

for_statement   ::= 'for' '[' identifier_list ']' '[' index_list ']'
                        '{' statement_list '}'
                  | 'for' '[' identifier_list ']' '[' index_list ']'
                        '{' '}'

expression      ::= tensor_access
                  | function_call
                  | NUMBER
                  | IDENTIFIER
                  | expression '+' expression
                  | expression '*' expression

tensor_access   ::= IDENTIFIER '[' index_list ']'

function_call   ::= IDENTIFIER '(' argument_list ')'
                  | IDENTIFIER '(' ')'

argument_list   ::= expression (',' expression)*

index_list      ::= IDENTIFIER (',' IDENTIFIER)*

identifier_list ::= IDENTIFIER (',' IDENTIFIER)*
```

Note: parenthesised expressions `'(' expression ')'` appear in the grammar documentation but are not present as an explicit production in `src/parser.y`; operator precedence declarations handle associativity directly.

#### 2.2 Operator Precedence

Declared in `src/parser.y` using Bison `%left` directives (lower declaration = lower precedence):

```
%left TOKEN_PLUS    /* lower precedence, left-associative */
%left TOKEN_MULT    /* higher precedence, left-associative */
```

This means `a + b * c` parses as `a + (b * c)` and `a * b * c` as `(a * b) * c`.

#### 2.3 Lexical Rules

From `src/lexer.l`:

| Category | Pattern | Notes |
|----------|---------|-------|
| Keywords | exact string match | Must appear before `IDENTIFIER` rule in Flex |
| Format keywords | `Dense`, `CSR`, `CSC`, `COO`, `ELLPACK`, `DIA` | Matched as keywords, not identifiers |
| `IDENTIFIER` | `[a-zA-Z_][a-zA-Z0-9_]*` | Any name not matched by a keyword rule |
| `NUMBER` | `[0-9]+(\.[0-9]+)?` | Integer or decimal; stored as string via `strdup(yytext)` |
| Whitespace | `[ \t\n\r]+` | Silently discarded |
| Unrecognised chars | `.` | Silently discarded (no error token) |

#### 2.4 Token Inventory

Full token list from `include/tokens.h`:

| Token | Value | Lexeme |
|-------|-------|--------|
| `TOKEN_COMPUTE` | 1000 | `compute` |
| `TOKEN_CALL` | 1001 | `call` |
| `TOKEN_TENSOR` | 1002 | `tensor` |
| `TOKEN_FOR` | 1003 | `for` |
| `TOKEN_DENSE` | 1004 | `Dense` |
| `TOKEN_CSR` | 1005 | `CSR` |
| `TOKEN_COO` | 1006 | `COO` |
| `TOKEN_CSC` | 1007 | `CSC` |
| `TOKEN_ELLPACK` | 1008 | `ELLPACK` |
| `TOKEN_DIA` | 1009 | `DIA` |
| `TOKEN_PLUS` | 1010 | `+` |
| `TOKEN_MULT` | 1011 | `*` |
| `TOKEN_ASSIGN` | 1012 | `=` |
| `TOKEN_LBRACKET` | 1020 | `[` |
| `TOKEN_RBRACKET` | 1021 | `]` |
| `TOKEN_LPAREN` | 1022 | `(` |
| `TOKEN_RPAREN` | 1023 | `)` |
| `TOKEN_LBRACE` | 1024 | `{` |
| `TOKEN_RBRACE` | 1025 | `}` |
| `TOKEN_COMMA` | 1026 | `,` |
| `TOKEN_SEMICOLON` | 1027 | `;` |
| `TOKEN_COLON` | 1028 | `:` |
| `TOKEN_LANGLE` | 1029 | `<` |
| `TOKEN_RANGLE` | 1030 | `>` |
| `TOKEN_IDENTIFIER` | 1040 | name |
| `TOKEN_NUMBER` | 1041 | numeric literal |
| `TOKEN_EOF` | 0 | end of input |

---

### 3. Language Constructs

#### 3.1 Tensor Declarations

```
tensor A : CSR<M, K>;
tensor B : Dense<K, N>;
tensor x : Dense<K>;
```

A declaration binds a name to a format and an optional shape:

- **Name**: any valid identifier.
- **Format**: one of `Dense`, `CSR`, `CSC`, `COO`, `ELLPACK`, `DIA`. Only `Dense`, `CSR`, and `CSC` produce specialised code generation; the rest are stored in the AST and treated as `Dense` in IR lowering.
- **Shape** (optional): comma-separated integer dimensions in angle brackets. When omitted, dimensions are inferred at runtime from the loaded matrix.

#### 3.2 Compute Statements

```
compute y[i] = A[i, j] * x[j];
compute C[i, j] = A[i, k] * B[k, j];
```

The `compute` keyword introduces a tensor assignment in Einstein notation. The left-hand side is a tensor access (output tensor and free indices); the right-hand side is an expression over tensor accesses, numbers, binary operators, and function calls.

#### 3.3 Call Statements

```
call optimize(A);
call checkpoint();
```

A `call` statement invokes a named function with an argument list. In the standard single-kernel compilation path (`lowerToIR`), call statements are silently ignored. In the Phase D multi-statement program path (`lowerToIRProgram`), call statements are lowered to `IRCallStmt` nodes that produce C function calls in the output.

#### 3.4 For Statements

```
for [A, B] [i, j] {
    compute C[i, j] = A[i, j] + B[i, j];
}
```

The `for` statement is a Phase D construct for expressing explicit loop bodies over named tensors and index sets. It takes a bracket-delimited tensor list and a bracket-delimited index list, followed by a brace-enclosed body of statements. In the standard path, for-statements are not lowered.

#### 3.5 Expressions

Expressions are built from:

| Form | Example | Notes |
|------|---------|-------|
| Tensor access | `A[i, j]` | Name + bracketed index list |
| Binary add | `A[i] + B[i]` | Left-associative |
| Binary multiply | `A[i] * x[j]` | Left-associative, higher precedence than `+` |
| Number literal | `2.0`, `42` | Integer or decimal |
| Bare identifier | `scale` | Scalar variable (no indices) |
| Function call | `relu(A[i, j])` | Name + parenthesised argument list |

---

### 4. Notation — Indexing and Index Algebra

#### 4.1 Index Kinds

Every index variable in a compute statement is classified as either **free** or **bound**:

- **Free indices** appear on the left-hand side (the output tensor). They define the iteration space of the result.
- **Bound indices** appear on the right-hand side only. They do not appear on the left-hand side. By the Einstein summation convention, they are implicitly summed over.

Formally, for a statement `compute T[f₁, ..., fₙ] = E`:

```
F = indices appearing in the LHS tensor access
B = indices appearing in E but not in F
```

Every distinct combination of values of indices in F produces one output element. For each such combination, the compiler reduces over all values of indices in B (the bound set) by accumulating the RHS expression.

#### 4.2 Einstein Summation Convention

The convention used here is that a **repeated index on the RHS, absent from the LHS, triggers implicit summation**. This is a restriction of the full Einstein convention (which also sums over index pairs where one is raised and one lowered), but sufficient for tensor contractions of the form used in SpMV and SpMM.

**SpMV example**:
```
compute y[i] = A[i, j] * x[j];
```
- Free: `{i}` — iterates over rows
- Bound: `{j}` — summed over columns for each row

Mathematical equivalent: y_i = Σ_j A_{ij} · x_j

**SpMM example**:
```
compute C[i, j] = A[i, k] * B[k, j];
```
- Free: `{i, j}` — iterates over all output positions
- Bound: `{k}` — summed over the shared dimension

Mathematical equivalent: C_{ij} = Σ_k A_{ik} · B_{kj}

#### 4.3 Index Order Semantics

The DSL index order in `A[i, j]` specifies which index maps to which **tensor coordinate**, not which loop comes first. Loop order is determined later, during IR lowering, based on the tensor's storage format.

The **format-correctness pass** in the IR compares the DSL index order against the natural iteration order of the format:

| Format | Natural outer index | Natural inner index |
|--------|---------------------|---------------------|
| CSR | row (`i`) | column (`j`) |
| CSC | column (`j`) | row (`i`) |
| Dense | either (row-major default) | — |

If the DSL order conflicts with the natural format order, the pass reorders the loop nest before any optimization passes run. This is a correctness requirement, not a performance toggle — iterating a CSR matrix column-first would produce wrong results or severe cache penalties.

**Example of a mismatch**:
```
tensor A : CSR<M, K>;
compute y[i] = A[j, i] * x[j];   -- column-first access on row-stored matrix
```
The pass detects that `j` is the outer index but CSR's natural outer is `i`. It reorders to produce the correct `i → j` nesting.

---

### 5. Type System

The DSL has a shallow, first-order type system: no polymorphism, no generics, no higher-kinded types.

#### 5.1 Format Types

| Format | IR treatment | Codegen support |
|--------|-------------|-----------------|
| `Dense` | `Format::Dense` | Full |
| `CSR` | `Format::CSR` | Full |
| `CSC` | `Format::CSC` | Full |
| `COO` | `Format::Dense` (fallback) | None (treated as Dense) |
| `ELLPACK` | `Format::Dense` (fallback) | None |
| `DIA` | `Format::Dense` (fallback) | None |

The format is stored in the `Declaration` AST node as a plain string and converted to the `ir::Format` enum by `stringToFormat()` in `src/ir.cpp`.

#### 5.2 Shape

Shape dimensions are declared as a comma-separated list of integers in angle brackets: `tensor A : CSR<1000, 500>`. They are stored as `vector<string>` in the AST and as `vector<int>` in the IR `Tensor` struct after `parseShape()` converts them. A shape of `0` in the IR means "infer from the loaded matrix at runtime."

When shape is omitted from the declaration, `dims` is empty and the generated code uses matrix dimensions read from the Matrix Market file header.

#### 5.3 Static Constraints (Enforced by Convention, Not by Checker)

The DSL design assumes:
- All tensors used in a compute statement have been declared earlier in the program.
- Tensor accesses have the correct arity (e.g., a 2-D tensor is accessed with two indices).
- Indices shared between two tensor accesses are semantically compatible (same extent).

The current implementation does not have a dedicated type-checker pass. Violations are caught implicitly: if a tensor is not in the declaration map, IR lowering falls back to default dimensions; arity mismatches produce unexpected loop structures. A future type-checker could enforce these statically.

#### 5.4 Expressiveness and Scope

**In scope** (parser + IR + codegen):
- SpMV: `compute y[i] = A[i, j] * x[j]` with A sparse
- SpMM: `compute C[i, j] = A[i, k] * B[k, j]` with A sparse, B dense
- Element-wise: `compute C[i, j] = A[i, j] + B[i, j]` (dense)
- Scalar multiply: `compute C[i, j] = 2.0 * A[i, j]`
- Phase D multi-statement programs with for-loops and call statements

**Out of scope by design**:
- Parallelism
- Format inference (format must be explicit)
- Schedule search (optimizations are selected by flags, not auto-tuned)
- COO/ELLPACK/DIA-specific iteration patterns

---

### 6. Formal Semantics

#### 6.1 Denotational Semantics

Each `compute` statement denotes a mathematical tensor equation. Let σ be the store (a mapping from tensor names to their values).

**Judgment**: `[[compute T[f₁,...,fₙ] = E]]_σ`

**Rule** (single compute):
```
[[compute T[f₁,...,fₙ] = E]]_σ =
    σ[T ↦ λ(v₁,...,vₙ). Σ_{b∈B} [[E]]_{σ, f₁=v₁,...,fₙ=vₙ, b=...}]
```

Where:
- F = {f₁, ..., fₙ} is the free index set (LHS indices)
- B = indices(E) \ F is the bound index set
- The reduction is additive accumulation over all values of bound indices

**Expression denotation**:

| Form | Denotation |
|------|-----------|
| `A[i, j]` (Dense) | σ(A)[i][j] — direct 2-D array lookup |
| `A[i, j]` (CSR) | σ(A).vals[p] where p is the CSR pointer for (i, j) |
| `e₁ + e₂` | [[e₁]] + [[e₂]] |
| `e₁ * e₂` | [[e₁]] × [[e₂]] |
| `n` (number) | the numeric value of n |
| `f(e₁,...,eₖ)` | the result of applying function f to its arguments |

#### 6.2 Program Denotation

A program is a sequence of statements. Declarations initialise the store with zero-valued tensors of the declared format and shape. Compute statements update the store destructively:

```
[[decl; P]]_σ = [[P]]_{σ[T ↦ zero_tensor(format, shape)]}
[[compute stmt; P]]_σ = [[P]]_{[[compute stmt]]_σ}
```

Statements are evaluated left to right; later statements see the results of earlier ones. This enables multi-statement programs where intermediate tensors are computed and then consumed.

#### 6.3 Operational Semantics (Small-Step Sketch)

The operational semantics describe how the DSL is *executed*, not just what it denotes.

**Compute reduction rule**:
```
⟨compute T[i] = A[i,j] * x[j], σ⟩
    →
⟨for i in [0, M): T[i] := 0; for j in [0, N): T[i] += A[i,j] * x[j], σ⟩
```

Key points:
- The DSL specifies the mathematical equation; the compiler produces a specific loop nest.
- Loop order is **not** determined by the DSL. The format-correctness pass and optimization scheduler determine the final loop ordering.
- The semantic content (the mathematical result) is the same regardless of loop order, as long as the loops cover the same index space.

**Loop ordering is semantics-preserving**: interchange of independent loops does not change the result for associative, commutative reduction operators (floating-point rounding differences aside).

---

### 7. Comparison to TACO

TACO (Tensor Algebra Compiler) is the closest related system. Differences:

| Dimension | TACO | This DSL |
|-----------|------|----------|
| Format support | All sparse formats + format inference | Dense, CSR, CSC only; format must be explicit |
| Schedule search | Full schedule space via `Schedule` API | Fixed optimizations selected by CLI flags |
| Parallelism | OpenMP, CUDA | None (single-threaded) |
| Target kernels | Any tensor algebra expression | SpMV, SpMM (primary) |
| Code output | Optimised C/C++ | Standalone C with benchmark harness |
| Research focus | Compiler completeness | Empirical characterisation of when optimizations help |

The DSL is intentionally less expressive than TACO. The constraint lets us focus the research question and keep the compiler small enough to understand and modify in a dissertation context.

---

## Part 2: Implementation

### 8. Lexer (`src/lexer.l`)

The lexer is a Flex specification that tokenises the DSL source.

**Key design decisions**:

1. **Keywords before identifiers**: Flex matches rules in order. All keyword rules (`compute`, `call`, `tensor`, `for`, `Dense`, `CSR`, etc.) appear before the `IDENTIFIER` rule. This ensures that `compute` is never matched as an identifier.

2. **Format keywords as tokens**: `Dense`, `CSR`, `CSC`, `COO`, `ELLPACK`, `DIA` are separate token types, not just identifiers that are checked later. This makes the grammar unambiguous and avoids string comparisons in the parser actions.

3. **`yylval.str` for literals**: For `TOKEN_IDENTIFIER` and `TOKEN_NUMBER`, the lexer calls `strdup(yytext)` and stores the result in `yylval.str`. The parser actions are responsible for `free()`-ing this memory after use.

4. **Silent discard of unknown characters**: The catch-all rule `.` discards unrecognised characters without emitting an error. This was a deliberate choice to accommodate test expectations; a production compiler would emit a lexical error here.

5. **Options used**: `%option noyywrap` (no multi-file input), `%option nounput` and `%option noinput` (suppress unused-function warnings).

**Generated interface**: Flex produces `yylex()`, `yytext`, `yy_scan_string()` (for scanning strings in tests), and `yylex_destroy()` (for cleanup).

---

### 9. Parser (`src/parser.y`)

The parser is a Bison LALR(1) grammar that builds a heap-allocated AST.

**Key design decisions**:

1. **`void*` union field for AST nodes**: Bison's `%union` uses a `void* ptr` field for all AST node types. This avoids C++'s inability to put non-POD types in unions and allows the parser to return any AST subtype. Casts are done explicitly in semantic actions using `static_cast`.

2. **Global `g_program` root**: The parser stores the completed AST in `std::unique_ptr<Program> g_program`. The caller in `src/main.cpp` retrieves this after `yyparse()` returns.

3. **Heap allocation with `new`**: All AST nodes are allocated with `new` inside parser actions and wrapped in `unique_ptr` where collected into lists. Intermediate `StatementList*`, `ExpressionList*`, and `StringList*` are also heap-allocated and `delete`-d after being transferred to their parent node.

4. **Operator precedence via `%left`**: The expression grammar is written with direct left-recursion (`expression '+' expression`) rather than a separate precedence-climbing non-terminal hierarchy. Bison resolves shift/reduce conflicts using the `%left` declarations.

5. **Error handling**: `yyerror` increments `yynerrs` and conditionally prints to stderr (only when `PARSER_DEBUG` is defined). The parser does not attempt error recovery; on a syntax error, parsing terminates.

---

### 10. AST Design (`include/ast.h`, `src/ast.cpp`)

The AST is defined in the `SparseTensorCompiler` namespace.

#### 10.1 Node Hierarchy

```
ASTNode
├── Statement
│   ├── Declaration       tensor A : CSR<M, K>;
│   ├── Computation       compute y[i] = A[i, j] * x[j];
│   ├── CallStatement     call optimize(A);
│   └── ForStatement      for [A, B] [i, j] { ... }
└── Expression
    ├── TensorAccess      A[i, j]
    ├── FunctionCall      relu(A[i, j])
    ├── BinaryOp          expr + expr  |  expr * expr
    ├── Number            42  |  3.14
    └── Identifier        scale  (bare name, no indices)

Program                   root; holds vector<Statement*>
```

#### 10.2 Key Fields

| Node | Key fields |
|------|-----------|
| `Program` | `vector<unique_ptr<Statement>> statements` |
| `Declaration` | `string tensorName`, `string tensorType`, `vector<string> shape` |
| `Computation` | `unique_ptr<TensorAccess> lhs`, `unique_ptr<Expression> rhs` |
| `CallStatement` | `string functionName`, `vector<unique_ptr<Expression>> arguments` |
| `ForStatement` | `vector<string> tensors`, `vector<string> indices`, `vector<unique_ptr<Statement>> body` |
| `TensorAccess` | `string tensorName`, `vector<string> indices` |
| `FunctionCall` | `string functionName`, `vector<unique_ptr<Expression>> arguments` |
| `BinaryOp` | `Operator op` (ADD or MULT), `unique_ptr<Expression> left`, `unique_ptr<Expression> right` |
| `Number` | `string value` |
| `Identifier` | `string name` |

#### 10.3 Visitor Pattern

Every `ASTNode` subclass implements `accept(ASTVisitor&)`. The `ASTVisitor` interface declares a `visit()` overload for every concrete node type. This allows traversals (IR lowering, pretty-printing, declaration collection) to be written as separate visitor classes without downcasting.

IR lowering uses three visitors:
- `DeclarationCollector`: walks the AST and builds a map of tensor name → `ir::Tensor`
- `TensorAccessExtractor`: collects all tensor accesses from an expression subtree
- `ASTToIRLowering`: the main visitor that lowers a `Computation` node to an `ir::Operation`

---

### 11. IR Lowering (`src/ir.cpp`)

Lowering translates the AST representation into the `ir` namespace data structures used by optimization passes and code generation.

#### 11.1 Entry Point

```cpp
ir::Operation lowerToIR(const SparseTensorCompiler::Program& program, const opt::OptConfig& cfg);
```

This function:
1. Runs `DeclarationCollector` to build the tensor map.
2. Finds the first `Computation` statement.
3. Classifies the kernel as `"spmv"` or `"spmm"` based on the number of distinct index dimensions and tensor formats involved.
4. Constructs the loop nest: outer dense loop(s) + inner sparse loop from CSR/CSC pointers.
5. Classifies each index as free or bound by set difference.
6. Returns a fully populated `ir::Operation` with the root `Loop` tree and `LoopOptimizations` metadata.

#### 11.2 Expression Lowering

`lowerExpression()` maps AST expression nodes to `IRExpr` nodes:

| AST node | IR node | Notes |
|----------|---------|-------|
| `TensorAccess` | `IRTensorAccess` | If sparse, sets `isSparseVals=true` and `pointerVar` |
| `Number` | `IRConstant` | `std::stod(value)` |
| `Identifier` | `IRScalarVar` | Bare scalar variable |
| `BinaryOp(ADD)` | `IRBinaryOp(ADD)` | Recursive |
| `BinaryOp(MULT)` | `IRBinaryOp(MUL)` | Recursive |
| `FunctionCall` | `IRFuncCall` | Arguments recursively lowered |

#### 11.3 Format Conversion

`stringToFormat()` maps the string stored in `Declaration::tensorType` to the `ir::Format` enum:

```cpp
"Dense"   → Format::Dense
"CSR"     → Format::CSR
"CSC"     → Format::CSC
anything else → Format::Dense   // COO, ELLPACK, DIA fallback
```

#### 11.4 Known Limitations

- **Format-correctness reordering**: The `needsReordering()` heuristic uses the first character of index names to guess row vs. column (`'i'` = row, `'j'` = column). This works for the conventional SpMV/SpMM index names but fails for arbitrary names like `a`, `b`, `r`, `c`.
- **SpMM 3-deep reordering**: Reordering is fully implemented for SpMV (2-loop nest) but has known edge cases for SpMM (3-loop nest) as noted in CLAUDE.md.
- **Single computation**: The standard path only lowers the first `Computation` statement. Multi-statement programs use the Phase D `lowerToIRProgram()` path.

---

### 12. What Is Parsed But Not Lowered

The parser is intentionally more expressive than the IR handles. This is by design for forward compatibility:

| Construct | Parsed? | Lowered (standard path)? | Lowered (Phase D path)? |
|-----------|---------|--------------------------|-------------------------|
| `Dense`, `CSR`, `CSC` tensor declarations | Yes | Yes | Yes |
| `COO`, `ELLPACK`, `DIA` declarations | Yes | As Dense | As Dense |
| `compute` (SpMV/SpMM) | Yes | Yes | Yes |
| `compute` (element-wise, scalar) | Yes | Yes (loop nest) | Yes |
| `call` statement | Yes | No (ignored) | Yes (IRCallStmt) |
| `for` statement | Yes | No (ignored) | Yes (IRForLoop) |
| `function_call` in expression | Yes | Partial (IRFuncCall) | Yes |
| Parenthesised sub-expressions | No (not in parser.y) | — | — |

The gap between what the parser accepts and what the IR handles is a deliberate design choice: the parser defines the full intended language surface; the IR handles what is needed for the dissertation research.

---

### 13. Phase D — Multi-Statement Program Path

Phase D (`lowerToIRProgram()` in `src/ir.cpp`, `generateProgramToFile()` in `src/codegen.cpp`) extends the compiler to handle programs with multiple statements, explicit for-loops, and call statements. The dispatch in `src/main.cpp` activates this path when the AST contains `ForStatement` or `CallStatement` nodes, or more than one `Computation`.

The Phase D IR uses a parallel node hierarchy in the `SparseTensorCompiler` namespace (not the `ir` namespace):

- `IRProgram` — root of a multi-statement program
- `IRDeclaration` — tensor allocation
- `IRForLoop` — explicit loop over index sets
- `IRCompute` — a single assignment statement
- `IRCall` — external function call

For sparse tensors in for-loop bodies, Phase D uses `sp_csr_get()` / `sp_csc_get()` accessor functions for random access rather than the streaming pointer iteration used in SpMV/SpMM kernels. Dense tensor accesses use flat array indexing: `name[i * name_ncols + j]` for 2-D tensors.

---

## Appendix: Quick Reference

### Canonical SpMV Program

```
tensor A : CSR<1000, 1000>;
tensor x : Dense<1000>;
tensor y : Dense<1000>;

compute y[i] = A[i, j] * x[j];
```

### Canonical SpMM Program

```
tensor A : CSR<100, 50>;
tensor B : Dense<50, 20>;
tensor C : Dense<100, 20>;

compute C[i, j] = A[i, k] * B[k, j];
```

### Format-Correctness Example

```
tensor A : CSC<100, 100>;
tensor x : Dense<100>;
tensor y : Dense<100>;

-- CSC natural order: outer=j, inner=i
compute y[i] = A[i, j] * x[j];
-- IR reorders to: for j: for p in col_ptr[j]..col_ptr[j+1]: i=row_idx[p]; y[i] += vals[p]*x[j]
```

### Compilation Pipeline

```
input.tc
   │
   ├─[Flex lexer]────→ token stream
   │
   ├─[Bison parser]──→ Program AST (SparseTensorCompiler namespace)
   │
   ├─[DeclarationCollector]──→ tensor map (name → ir::Tensor)
   │
   ├─[ASTToIRLowering]──→ ir::Operation (loop nest + optimizations metadata)
   │
   ├─[Format-correctness pass]──→ loop nest reordered for CSR/CSC alignment
   │
   ├─[Optimization passes (opt:: namespace)]
   │      applyLoopInterchange()
   │      applyBlocking()
   │
   └─[CodeGenerator]──→ standalone C file with kernel + benchmark harness
```
