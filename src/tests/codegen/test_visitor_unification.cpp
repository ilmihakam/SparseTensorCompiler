/**
 * Test Suite: Visitor Unification
 *
 * Verifies that all 6 kernel types (SpMV, SpMM, SpAdd, SpElMul, SpGEMM, SDDMM)
 * go through the generic IR visitor codegen path and produce correct C code.
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

static bool parserInitialized = false;

static std::string generateForKernel(const std::string& code) {
    if (!parserInitialized) {
        yynerrs = 0;
        g_program.reset();
        yy_scan_string("tensor x : Dense;");
        yyparse();
        yylex_destroy();
        g_program.reset();
        parserInitialized = true;
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

// ============================================================================
// SpAdd CSR: union merge
// ============================================================================

TEST(VisitorUnification, SpAdd_CSR_UnionMerge) {
    std::string code =
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = A[i, j] + B[i, j];\n";

    std::string output = generateForKernel(code);
    ASSERT_FALSE(output.empty());

    // Should have tensor-aware sparse iteration plus generic union handling.
    EXPECT_NE(output.find("A->row_ptr"), std::string::npos) << "Should iterate A's rows";
    EXPECT_NE(output.find("B->row_ptr"), std::string::npos) << "Should iterate B's rows";
    EXPECT_NE(output.find("A->vals[pA]"), std::string::npos) << "Should use A's values";
    EXPECT_NE(output.find("sp_csr_get(B, i, j)"), std::string::npos) << "Should use generic random access for B";
}

TEST(VisitorUnification, SpAdd_CSC_ColumnMajor) {
    std::string code =
        "tensor A : CSC<100, 100>;\n"
        "tensor B : CSC<100, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = A[i, j] + B[i, j];\n";

    std::string output = generateForKernel(code);
    ASSERT_FALSE(output.empty());

    EXPECT_NE(output.find("A->col_ptr"), std::string::npos) << "Should iterate A's columns";
    EXPECT_NE(output.find("B->col_ptr"), std::string::npos) << "Should iterate B's columns";
}

// ============================================================================
// SpElMul CSR: Merge-intersection
// ============================================================================

TEST(VisitorUnification, SpElMul_CSR_MergeIntersection) {
    std::string code =
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = A[i, j] * B[i, j];\n";

    std::string output = generateForKernel(code);
    ASSERT_FALSE(output.empty());

    // Merge-intersection pattern
    EXPECT_NE(output.find("while (pA <"), std::string::npos) << "Should have merge while-loop";
    EXPECT_NE(output.find("if (min_idx == max_idx)"), std::string::npos) << "Should check merged intersection";
    EXPECT_NE(output.find("A->vals[pA]"), std::string::npos);
    EXPECT_NE(output.find("B->vals[pB]"), std::string::npos);
}

// ============================================================================
// SpGEMM CSR: Nested sparse loops
// ============================================================================

TEST(VisitorUnification, SpGEMM_CSR_NestedSparse) {
    std::string code =
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = A[i, k] * B[k, j];\n";

    std::string output = generateForKernel(code);
    ASSERT_FALSE(output.empty());

    // Should have nested sparse loops using different tensors
    EXPECT_NE(output.find("A->row_ptr[i]"), std::string::npos) << "Should iterate A by row i";
    EXPECT_NE(output.find("B->row_ptr[k]"), std::string::npos) << "Should iterate B by row k";
    EXPECT_NE(output.find("A->vals[pA]"), std::string::npos) << "Should use A's values";
    EXPECT_NE(output.find("B->vals[pB]"), std::string::npos) << "Should use B's values";
}

// ============================================================================
// SDDMM CSR: preBody and postBody
// ============================================================================

TEST(VisitorUnification, SDDMM_CSR_PreAndPostBody) {
    std::string code =
        "tensor S : CSR<100, 100>;\n"
        "tensor D : Dense<100, 64>;\n"
        "tensor E : Dense<64, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = S[i, j] * D[i, k] * E[k, j];\n";

    std::string output = generateForKernel(code);
    ASSERT_FALSE(output.empty());

    EXPECT_NE(output.find("double sum = 0.0;"), std::string::npos) << "Should have preBody sum init";
    EXPECT_NE(output.find("sum +="), std::string::npos) << "Should accumulate into sum";
    EXPECT_NE(output.find("S->vals[pS]"), std::string::npos) << "Should use sampling matrix values";
}

// ============================================================================
// SpMV regression
// ============================================================================

TEST(VisitorUnification, SpMV_CSR_Regression) {
    std::string code =
        "tensor A : CSR<100, 50>;\n"
        "tensor x : Dense<50>;\n"
        "tensor y : Dense<100>;\n"
        "compute y[i] = A[i, j] * x[j];\n";

    std::string output = generateForKernel(code);
    ASSERT_FALSE(output.empty());

    EXPECT_NE(output.find("A->row_ptr[i]"), std::string::npos);
    EXPECT_NE(output.find("A->col_idx[pA]"), std::string::npos);
    EXPECT_NE(output.find("A->vals[pA]"), std::string::npos);
    EXPECT_NE(output.find("y[i]"), std::string::npos);
}

// ============================================================================
// SpMM regression
// ============================================================================

TEST(VisitorUnification, SpMM_CSR_Regression) {
    std::string code =
        "tensor A : CSR<100, 50>;\n"
        "tensor B : Dense<50, 30>;\n"
        "tensor C : Dense<100, 30>;\n"
        "compute C[i, j] = A[i, k] * B[k, j];\n";

    std::string output = generateForKernel(code);
    ASSERT_FALSE(output.empty());

    EXPECT_NE(output.find("A->row_ptr[i]"), std::string::npos);
    EXPECT_NE(output.find("A->col_idx[pA]"), std::string::npos);
    EXPECT_NE(output.find("A->vals[pA]"), std::string::npos);
    EXPECT_NE(output.find("C[i][j]"), std::string::npos);
    EXPECT_NE(output.find("B[k][j]"), std::string::npos);
}

// ============================================================================
// Balanced braces for all kernels
// ============================================================================

TEST(VisitorUnification, AllKernels_BalancedBraces) {
    std::vector<std::pair<std::string, std::string>> kernels = {
        {"spmv", "tensor A : CSR<100, 50>;\ntensor x : Dense<50>;\ntensor y : Dense<100>;\ncompute y[i] = A[i, j] * x[j];\n"},
        {"spmm", "tensor A : CSR<100, 50>;\ntensor B : Dense<50, 30>;\ntensor C : Dense<100, 30>;\ncompute C[i, j] = A[i, k] * B[k, j];\n"},
        {"spadd", "tensor A : CSR<100, 100>;\ntensor B : CSR<100, 100>;\ntensor C : Dense<100, 100>;\ncompute C[i, j] = A[i, j] + B[i, j];\n"},
        {"spelmul", "tensor A : CSR<100, 100>;\ntensor B : CSR<100, 100>;\ntensor C : Dense<100, 100>;\ncompute C[i, j] = A[i, j] * B[i, j];\n"},
        {"spgemm", "tensor A : CSR<100, 100>;\ntensor B : CSR<100, 100>;\ntensor C : Dense<100, 100>;\ncompute C[i, j] = A[i, k] * B[k, j];\n"},
        {"sddmm", "tensor S : CSR<100, 100>;\ntensor D : Dense<100, 64>;\ntensor E : Dense<64, 100>;\ntensor C : Dense<100, 100>;\ncompute C[i, j] = S[i, j] * D[i, k] * E[k, j];\n"},
    };

    for (const auto& [name, code] : kernels) {
        std::string output = generateForKernel(code);
        ASSERT_FALSE(output.empty()) << "Failed to generate " << name;

        int openBraces = std::count(output.begin(), output.end(), '{');
        int closeBraces = std::count(output.begin(), output.end(), '}');
        EXPECT_EQ(openBraces, closeBraces)
            << name << ": unbalanced braces (open=" << openBraces << " close=" << closeBraces << ")";
    }
}
