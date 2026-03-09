/**
 * Test Suite: SpGEMM Dense Output
 *
 * Verifies that SpGEMM codegen emits dense double** C output through
 * the generic visitor path (no hash-map structures).
 */

#include <gtest/gtest.h>
#include <algorithm>
#include "codegen.h"
#include "ast.h"
#include "scheduled_optimizations.h"
#include "semantic_ir.h"

extern std::unique_ptr<SparseTensorCompiler::Program> g_program;
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern int yynerrs;

static bool parserInit = false;

static std::string generateSpGEMM(const std::string& code) {
    if (!parserInit) {
        yynerrs = 0;
        g_program.reset();
        yy_scan_string("tensor x : Dense;");
        yyparse();
        yylex_destroy();
        g_program.reset();
        parserInit = true;
    }

    yynerrs = 0;
    g_program.reset();
    yy_scan_string(code.c_str());
    int result = yyparse();
    yylex_destroy();

    if (result != 0 || !g_program) return "";

    auto op = sparseir::lowerFirstComputationToScheduled(*g_program);
    g_program.reset();
    if (!op) return "";

    opt::OptConfig config = opt::OptConfig::baseline();
    opt::applyOptimizations(*op, config);

    return codegen::generateCode(*op, config);
}

static std::string spgemmCode() {
    return
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = A[i, k] * B[k, j];\n";
}

TEST(SpGEMMDenseOutput, SignatureTakesDenseC) {
    std::string output = generateSpGEMM(spgemmCode());
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("double** C"), std::string::npos) << "Kernel signature should use double** C";
}

TEST(SpGEMMDenseOutput, EmitsDenseAccumulation) {
    std::string output = generateSpGEMM(spgemmCode());
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("C[i][j]"), std::string::npos) << "Should emit C[i][j] dense accumulation";
}

TEST(SpGEMMDenseOutput, HasSparseLoopOverA) {
    std::string output = generateSpGEMM(spgemmCode());
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("row_ptr"), std::string::npos) << "Should have CSR row_ptr for A";
    EXPECT_NE(output.find("col_idx"), std::string::npos) << "Should have CSR col_idx for A";
}

TEST(SpGEMMDenseOutput, HasSparseLoopOverB) {
    std::string output = generateSpGEMM(spgemmCode());
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("col_ptr"), std::string::npos) << "Should have CSC col_ptr for B";
}

TEST(SpGEMMDenseOutput, ReferenceKernelPresent) {
    std::string output = generateSpGEMM(spgemmCode());
    ASSERT_FALSE(output.empty());
    EXPECT_NE(output.find("void reference("), std::string::npos) << "Should emit reference kernel";
}

TEST(SpGEMMDenseOutput, BalancedBraces) {
    std::string output = generateSpGEMM(spgemmCode());
    ASSERT_FALSE(output.empty());
    int open = std::count(output.begin(), output.end(), '{');
    int close = std::count(output.begin(), output.end(), '}');
    EXPECT_EQ(open, close) << "Braces should be balanced";
}
