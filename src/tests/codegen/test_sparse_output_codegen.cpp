/**
 * Test Suite: Sparse-Output Codegen Shape
 *
 * Verifies that sparse-output kernels generate correct code via the visitor
 * path (not hardcoded emitters), and that optimizations produce expected
 * code patterns in the output.
 */

#include <gtest/gtest.h>
#include "codegen.h"
#include "ir.h"
#include "optimizations.h"
#include "scheduled_optimizations.h"
#include "semantic_ir.h"
#include "ast.h"

extern std::unique_ptr<SparseTensorCompiler::Program> g_program;
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern int yynerrs;

static bool parserInit = false;

static sparseir::scheduled::Compute* findFirstScheduledCompute(sparseir::scheduled::Program& program) {
    for (auto& stmt : program.statements) {
        if (auto* compute = dynamic_cast<sparseir::scheduled::Compute*>(stmt.get())) {
            return compute;
        }
        if (auto* region = dynamic_cast<sparseir::scheduled::Region*>(stmt.get())) {
            for (auto& inner : region->body) {
                if (auto* compute = dynamic_cast<sparseir::scheduled::Compute*>(inner.get())) {
                    return compute;
                }
            }
        }
    }
    return nullptr;
}

static std::unique_ptr<sparseir::scheduled::Compute> parse(const std::string& code) {
    if (!parserInit) {
        yynerrs = 0; g_program.reset();
        yy_scan_string("tensor x : Dense;"); yyparse(); yylex_destroy();
        g_program.reset(); parserInit = true;
    }
    yynerrs = 0; g_program.reset();
    yy_scan_string(code.c_str());
    int r = yyparse(); yylex_destroy();
    if (r != 0 || yynerrs != 0 || !g_program) return nullptr;

    auto semantic = sparseir::lowerToSemanticProgram(*g_program);
    if (!semantic) return nullptr;
    auto scheduled = sparseir::scheduleProgram(*semantic);
    if (!scheduled) return nullptr;

    auto* compute = findFirstScheduledCompute(*scheduled);
    if (!compute) return nullptr;

    auto cloned = compute->clone();
    return std::unique_ptr<sparseir::scheduled::Compute>(
        dynamic_cast<sparseir::scheduled::Compute*>(cloned.release()));
}

static std::string generate(sparseir::scheduled::Compute& compute, const opt::OptConfig& cfg) {
    opt::applyOptimizations(compute, cfg);
    return codegen::generateCode(compute, cfg);
}

static std::string extractComputeFunction(const std::string& code) {
    size_t start = code.find("void compute(");
    if (start == std::string::npos) return "";
    size_t end = code.find("\nvoid reference(", start);
    if (end == std::string::npos) return "";
    return code.substr(start, end - start);
}

// ============================================================================
// DSL sources
// ============================================================================

static const char* SPADD = R"(
    tensor A : CSR<100, 100>;
    tensor B : CSR<100, 100>;
    tensor C : CSR<100, 100>;
    compute C[i, j] = A[i, j] + B[i, j];
)";

static const char* SPELMUL = R"(
    tensor A : CSR<100, 100>;
    tensor B : CSR<100, 100>;
    tensor C : CSR<100, 100>;
    compute C[i, j] = A[i, j] * B[i, j];
)";

static const char* SPADD_CSC = R"(
    tensor A : CSC<100, 100>;
    tensor B : CSC<100, 100>;
    tensor C : CSC<100, 100>;
    compute C[i, j] = A[i, j] + B[i, j];
)";

static const char* SPELMUL_CSC = R"(
    tensor A : CSC<100, 100>;
    tensor B : CSC<100, 100>;
    tensor C : CSC<100, 100>;
    compute C[i, j] = A[i, j] * B[i, j];
)";

static const char* SDDMM = R"(
    tensor S : CSR<100, 100>;
    tensor D : Dense<100, 50>;
    tensor E : Dense<50, 100>;
    tensor C : CSR<100, 100>;
    compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
)";

static const char* SPGEMM = R"(
    tensor A : CSR<100, 50>;
    tensor B : CSR<50, 100>;
    tensor C : CSR<100, 100>;
    compute C[i, j] = A[i, k] * B[k, j];
)";

// ============================================================================
// SpAdd codegen shape
// ============================================================================

TEST(SparseOutputCodegen, SpAdd_Baseline_HasComputeFunction) {
    auto op = parse(SPADD);
    ASSERT_NE(op, nullptr);
    auto cfg = opt::OptConfig::baseline();
    std::string code = generate(*op, cfg);

    EXPECT_NE(code.find("void compute("), std::string::npos);
    EXPECT_NE(code.find("void assemble_output("), std::string::npos);
    EXPECT_EQ(code.find("void spadd("), std::string::npos);
    EXPECT_EQ(code.find("spadd_assemble"), std::string::npos);
    // Iterates over C's structure
    EXPECT_NE(code.find("C->row_ptr[i]"), std::string::npos);
    EXPECT_NE(code.find("C->col_idx[pC]"), std::string::npos);
}

TEST(SparseOutputCodegen, SpAdd_Blocked_HasBlockLoop) {
    auto op = parse(SPADD);
    ASSERT_NE(op, nullptr);
    auto cfg = opt::OptConfig::blockingOnly(32);
    std::string code = generate(*op, cfg);

    EXPECT_NE(code.find("i_block"), std::string::npos);
    EXPECT_NE(code.find("i_start"), std::string::npos);
    EXPECT_NE(code.find("i_end"), std::string::npos);
    EXPECT_NE(code.find("assemble_output"), std::string::npos);
}

TEST(SparseOutputCodegen, SpAdd_InterchangeOnly_SameAsBaseline) {
    auto op1 = parse(SPADD);
    auto op2 = parse(SPADD);
    ASSERT_NE(op1, nullptr);
    ASSERT_NE(op2, nullptr);

    auto cfg_base = opt::OptConfig::baseline();
    auto cfg_int = opt::OptConfig::interchangeOnly();
    std::string code_base = generate(*op1, cfg_base);
    std::string code_int = generate(*op2, cfg_int);

    // Interchange is a no-op for spadd; compute function should be same structure
    // Both should have the same loop pattern (no block loops)
    bool base_has_block = code_base.find("i_block") != std::string::npos;
    bool int_has_block = code_int.find("i_block") != std::string::npos;
    EXPECT_EQ(base_has_block, int_has_block);
}

// ============================================================================
// SpElMul codegen shape
// ============================================================================

TEST(SparseOutputCodegen, SpElMul_Baseline_HasComputeFunction) {
    auto op = parse(SPELMUL);
    ASSERT_NE(op, nullptr);
    auto cfg = opt::OptConfig::baseline();
    std::string code = generate(*op, cfg);

    EXPECT_NE(code.find("void compute("), std::string::npos);
    EXPECT_NE(code.find("void assemble_output("), std::string::npos);
    EXPECT_EQ(code.find("void spelmul("), std::string::npos);
    EXPECT_EQ(code.find("spelmul_assemble"), std::string::npos);
    EXPECT_NE(code.find("C->row_ptr[i]"), std::string::npos);
}

TEST(SparseOutputCodegen, SpElMul_Blocked_HasBlockLoop) {
    auto op = parse(SPELMUL);
    ASSERT_NE(op, nullptr);
    auto cfg = opt::OptConfig::blockingOnly(32);
    std::string code = generate(*op, cfg);

    EXPECT_NE(code.find("i_block"), std::string::npos);
}

TEST(SparseOutputCodegen, SpAdd_CSC_UsesTensorScopedLoopBounds) {
    auto op = parse(SPADD_CSC);
    ASSERT_NE(op, nullptr);
    auto cfg = opt::OptConfig::baseline();
    std::string code = generate(*op, cfg);
    std::string computeFn = extractComputeFunction(code);

    ASSERT_FALSE(computeFn.empty());
    EXPECT_NE(computeFn.find("for (int j = 0; j < C->cols; j++)"), std::string::npos);
    EXPECT_EQ(computeFn.find("for (int j = 0; j < cols; j++)"), std::string::npos);
}

TEST(SparseOutputCodegen, SpElMul_CSC_UsesTensorScopedLoopBounds) {
    auto op = parse(SPELMUL_CSC);
    ASSERT_NE(op, nullptr);
    auto cfg = opt::OptConfig::baseline();
    std::string code = generate(*op, cfg);
    std::string computeFn = extractComputeFunction(code);

    ASSERT_FALSE(computeFn.empty());
    EXPECT_NE(computeFn.find("for (int j = 0; j < C->cols; j++)"), std::string::npos);
    EXPECT_EQ(computeFn.find("for (int j = 0; j < cols; j++)"), std::string::npos);
}

// ============================================================================
// SDDMM codegen shape
// ============================================================================

TEST(SparseOutputCodegen, SDDMM_Baseline_HasComputeFunction) {
    auto op = parse(SDDMM);
    ASSERT_NE(op, nullptr);
    auto cfg = opt::OptConfig::baseline();
    std::string code = generate(*op, cfg);

    EXPECT_NE(code.find("void compute("), std::string::npos);
    EXPECT_NE(code.find("void assemble_output("), std::string::npos);
    EXPECT_EQ(code.find("void sddmm("), std::string::npos);
    EXPECT_EQ(code.find("sddmm_assemble"), std::string::npos);
    // Iterates C's structure for sampling
    EXPECT_NE(code.find("C->row_ptr"), std::string::npos);
}

TEST(SparseOutputCodegen, SDDMM_Blocked_HasBlockLoop) {
    auto op = parse(SDDMM);
    ASSERT_NE(op, nullptr);
    auto cfg = opt::OptConfig::blockingOnly(32);
    std::string code = generate(*op, cfg);

    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_NE(code.find("_block"), std::string::npos);
}

TEST(SparseOutputCodegen, SDDMM_Interchange_Applied) {
    auto op = parse(SDDMM);
    ASSERT_NE(op, nullptr);
    auto cfg = opt::OptConfig::interchangeOnly();
    std::string code = generate(*op, cfg);

    EXPECT_TRUE(op->optimizations.interchangeApplied);
    EXPECT_NE(code.find("void compute("), std::string::npos);
}

// ============================================================================
// SpGEMM codegen shape
// ============================================================================

TEST(SparseOutputCodegen, SpGEMM_Baseline_HasComputeFunction) {
    auto op = parse(SPGEMM);
    ASSERT_NE(op, nullptr);
    auto cfg = opt::OptConfig::baseline();
    std::string code = generate(*op, cfg);

    EXPECT_NE(code.find("void compute("), std::string::npos);
    EXPECT_NE(code.find("void assemble_output("), std::string::npos);
    EXPECT_EQ(code.find("void spgemm"), std::string::npos);
    EXPECT_EQ(code.find("spgemm_assemble"), std::string::npos);
    // Workspace: acc, marked, touched
    EXPECT_NE(code.find("acc"), std::string::npos);
    EXPECT_NE(code.find("marked"), std::string::npos);
    EXPECT_NE(code.find("touched"), std::string::npos);
}

TEST(SparseOutputCodegen, SpGEMM_Blocked_HasBlockLoop) {
    auto op = parse(SPGEMM);
    ASSERT_NE(op, nullptr);
    auto cfg = opt::OptConfig::blockingOnly(32);
    std::string code = generate(*op, cfg);

    EXPECT_NE(code.find("i_block"), std::string::npos);
}

TEST(SparseOutputCodegen, SpGEMM_InterchangeOnly_SameAsBaseline) {
    auto op1 = parse(SPGEMM);
    auto op2 = parse(SPGEMM);
    ASSERT_NE(op1, nullptr);
    ASSERT_NE(op2, nullptr);

    auto cfg_base = opt::OptConfig::baseline();
    auto cfg_int = opt::OptConfig::interchangeOnly();
    std::string code_base = generate(*op1, cfg_base);
    std::string code_int = generate(*op2, cfg_int);

    bool base_has_block = code_base.find("i_block") != std::string::npos;
    bool int_has_block = code_int.find("i_block") != std::string::npos;
    EXPECT_EQ(base_has_block, int_has_block);
}

// ============================================================================
// Fairness contract: assembly before timing, compute in timed region
// ============================================================================

TEST(SparseOutputCodegen, SpAdd_FairnessContract) {
    auto op = parse(SPADD);
    ASSERT_NE(op, nullptr);
    auto cfg = opt::OptConfig::baseline();
    std::string code = generate(*op, cfg);

    // Assembly function and compute function should both exist
    EXPECT_NE(code.find("assemble_output"), std::string::npos);
    EXPECT_NE(code.find("void compute("), std::string::npos);

    // In main(): assembly call should come before timing loop
    // Find the main function first, then check ordering within it
    auto main_pos = code.find("int main(");
    ASSERT_NE(main_pos, std::string::npos);
    std::string main_code = code.substr(main_pos);
    auto assemble_call = main_code.find("assemble_output");
    auto timing_call = main_code.find("get_time_ms");
    EXPECT_NE(assemble_call, std::string::npos);
    EXPECT_NE(timing_call, std::string::npos);
    EXPECT_LT(assemble_call, timing_call);
}

TEST(SparseOutputCodegen, SpGEMM_FairnessContract) {
    auto op = parse(SPGEMM);
    ASSERT_NE(op, nullptr);
    auto cfg = opt::OptConfig::baseline();
    std::string code = generate(*op, cfg);

    EXPECT_NE(code.find("assemble_output"), std::string::npos);
    EXPECT_NE(code.find("void compute("), std::string::npos);

    auto main_pos = code.find("int main(");
    ASSERT_NE(main_pos, std::string::npos);
    std::string main_code = code.substr(main_pos);
    auto assemble_call = main_code.find("assemble_output");
    auto timing_call = main_code.find("get_time_ms");
    EXPECT_NE(assemble_call, std::string::npos);
    EXPECT_NE(timing_call, std::string::npos);
    EXPECT_LT(assemble_call, timing_call);
}
