/**
 * Test Suite: Sparse-Output Config Effects
 *
 * Verifies that optimization passes produce correct flags and loop tree shapes
 * when applied to sparse-output kernels (spadd, spelmul, spgemm, sddmm).
 *
 * Expectations per the plan:
 *   | Kernel  | Blocking | Interchange | Notes                         |
 *   |---------|----------|-------------|-------------------------------|
 *   | spadd   | Applies  | No-op       | body opaque, blocking valid   |
 *   | spelmul | Applies  | No-op       | same as spadd                 |
 *   | spgemm  | Applies  | No-op       | workspace in prologue/epilogue|
 *   | sddmm   | Applies  | Applies     | accumulator fusion works      |
 */

#include <gtest/gtest.h>
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
    return sparseir::lowerFirstComputationToScheduled(*g_program);
}

// Helper: check if any loop in the tree has "_block" in its index name
static bool hasBlockLoop(const sparseir::scheduled::Loop* loop) {
    if (!loop) return false;
    if (loop->indexName.find("_block") != std::string::npos) return true;
    for (const auto& c : loop->children)
        if (hasBlockLoop(c.get())) return true;
    return false;
}

// Helper: check if a loop with sparseTensorName == name iterating output exists
static bool hasSparseOutputLoop(const sparseir::scheduled::Loop* loop, const std::string& name) {
    if (!loop) return false;
    if (loop->headerKind == sparseir::scheduled::LoopHeaderKind::SparseIterator &&
        loop->iterator.beginExpr.rfind(name + "->", 0) == 0) {
        return true;
    }
    for (const auto& c : loop->children)
        if (hasSparseOutputLoop(c.get(), name)) return true;
    return false;
}

// ============================================================================
// Sparse-output IR tree structure tests
// ============================================================================

static const char* SPADD_CSR = R"(
    tensor A : CSR<100, 100>;
    tensor B : CSR<100, 100>;
    tensor C : CSR<100, 100>;
    compute C[i, j] = A[i, j] + B[i, j];
)";

static const char* SPELMUL_CSR = R"(
    tensor A : CSR<100, 100>;
    tensor B : CSR<100, 100>;
    tensor C : CSR<100, 100>;
    compute C[i, j] = A[i, j] * B[i, j];
)";

static const char* SDDMM_CSR = R"(
    tensor S : CSR<100, 100>;
    tensor D : Dense<100, 50>;
    tensor E : Dense<50, 100>;
    tensor C : CSR<100, 100>;
    compute C[i, j] = S[i, j] * D[i, k] * E[k, j];
)";

static const char* SPGEMM_CSR = R"(
    tensor A : CSR<100, 50>;
    tensor B : CSR<50, 100>;
    tensor C : CSR<100, 100>;
    compute C[i, j] = A[i, k] * B[k, j];
)";

TEST(SparseOutputConfigEffects, SpAdd_HasSparseOutputTree) {
    auto op = parse(SPADD_CSR);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->outputPattern, sparseir::OutputPatternKind::Union);
    ASSERT_NE(op->rootLoop, nullptr);
    // Outer dense, inner sparse iterating C
    EXPECT_EQ(op->rootLoop->kind, sparseir::scheduled::LoopKind::Dense);
    EXPECT_TRUE(hasSparseOutputLoop(op->rootLoop.get(), "C"));
}

TEST(SparseOutputConfigEffects, SpElMul_HasSparseOutputTree) {
    auto op = parse(SPELMUL_CSR);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->outputPattern, sparseir::OutputPatternKind::Intersection);
    ASSERT_NE(op->rootLoop, nullptr);
    EXPECT_EQ(op->rootLoop->kind, sparseir::scheduled::LoopKind::Dense);
    EXPECT_TRUE(hasSparseOutputLoop(op->rootLoop.get(), "C"));
}

TEST(SparseOutputConfigEffects, SpGEMM_HasSparseOutputTree) {
    auto op = parse(SPGEMM_CSR);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->outputPattern, sparseir::OutputPatternKind::DynamicRowAccumulator);
    ASSERT_NE(op->rootLoop, nullptr);
    EXPECT_EQ(op->rootLoop->kind, sparseir::scheduled::LoopKind::Dense);
    // SpGEMM has A and B sparse loops, plus prologue/epilogue
    EXPECT_FALSE(op->prologueStmts.empty());
    EXPECT_FALSE(op->epilogueStmts.empty());
}

TEST(SparseOutputConfigEffects, SDDMM_HasSparseOutputTree) {
    auto op = parse(SDDMM_CSR);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->outputPattern, sparseir::OutputPatternKind::Sampled);
    ASSERT_NE(op->rootLoop, nullptr);
    EXPECT_EQ(op->rootLoop->kind, sparseir::scheduled::LoopKind::Dense);
    EXPECT_TRUE(hasSparseOutputLoop(op->rootLoop.get(), "C"));
}

// ============================================================================
// Blocking tests: all kernels should accept blocking
// ============================================================================

TEST(SparseOutputConfigEffects, SpAdd_BlockingApplied) {
    auto op = parse(SPADD_CSR);
    ASSERT_NE(op, nullptr);
    opt::OptConfig cfg = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, cfg);
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_TRUE(hasBlockLoop(op->rootLoop.get()));
}

TEST(SparseOutputConfigEffects, SpElMul_BlockingApplied) {
    auto op = parse(SPELMUL_CSR);
    ASSERT_NE(op, nullptr);
    opt::OptConfig cfg = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, cfg);
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_TRUE(hasBlockLoop(op->rootLoop.get()));
}

TEST(SparseOutputConfigEffects, SpGEMM_BlockingApplied) {
    auto op = parse(SPGEMM_CSR);
    ASSERT_NE(op, nullptr);
    opt::OptConfig cfg = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, cfg);
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_TRUE(hasBlockLoop(op->rootLoop.get()));
}

TEST(SparseOutputConfigEffects, SDDMM_BlockingApplied) {
    auto op = parse(SDDMM_CSR);
    ASSERT_NE(op, nullptr);
    opt::OptConfig cfg = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, cfg);
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_TRUE(hasBlockLoop(op->rootLoop.get()));
}

// ============================================================================
// Interchange tests: only SDDMM should apply
// ============================================================================

TEST(SparseOutputConfigEffects, SpAdd_InterchangeNoop) {
    auto op = parse(SPADD_CSR);
    ASSERT_NE(op, nullptr);
    opt::OptConfig cfg = opt::OptConfig::interchangeOnly();
    opt::applyOptimizations(*op, cfg);
    EXPECT_FALSE(op->optimizations.interchangeApplied);
}

TEST(SparseOutputConfigEffects, SpElMul_InterchangeNoop) {
    auto op = parse(SPELMUL_CSR);
    ASSERT_NE(op, nullptr);
    opt::OptConfig cfg = opt::OptConfig::interchangeOnly();
    opt::applyOptimizations(*op, cfg);
    EXPECT_FALSE(op->optimizations.interchangeApplied);
}

TEST(SparseOutputConfigEffects, SpGEMM_InterchangeNoop) {
    auto op = parse(SPGEMM_CSR);
    ASSERT_NE(op, nullptr);
    opt::OptConfig cfg = opt::OptConfig::interchangeOnly();
    opt::applyOptimizations(*op, cfg);
    EXPECT_FALSE(op->optimizations.interchangeApplied);
}

TEST(SparseOutputConfigEffects, SDDMM_InterchangeApplied) {
    auto op = parse(SDDMM_CSR);
    ASSERT_NE(op, nullptr);
    opt::OptConfig cfg = opt::OptConfig::interchangeOnly();
    opt::applyOptimizations(*op, cfg);
    EXPECT_TRUE(op->optimizations.interchangeApplied);
}

// ============================================================================
// Combined optimization tests
// ============================================================================

TEST(SparseOutputConfigEffects, SpAdd_AllOpts_OnlyBlockingApplies) {
    auto op = parse(SPADD_CSR);
    ASSERT_NE(op, nullptr);
    opt::OptConfig cfg = opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B);
    opt::applyOptimizations(*op, cfg);
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_FALSE(op->optimizations.interchangeApplied);
    EXPECT_TRUE(hasBlockLoop(op->rootLoop.get()));
}

TEST(SparseOutputConfigEffects, SDDMM_AllOpts_BothApply) {
    auto op = parse(SDDMM_CSR);
    ASSERT_NE(op, nullptr);
    opt::OptConfig cfg = opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B);
    opt::applyOptimizations(*op, cfg);
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_TRUE(op->optimizations.interchangeApplied);
}

// ============================================================================
// CSC format tests: same expectations, just different format
// ============================================================================

static const char* SPADD_CSC = R"(
    tensor A : CSC<100, 100>;
    tensor B : CSC<100, 100>;
    tensor C : CSC<100, 100>;
    compute C[i, j] = A[i, j] + B[i, j];
)";

TEST(SparseOutputConfigEffects, SpAdd_CSC_HasSparseOutputTree) {
    auto op = parse(SPADD_CSC);
    ASSERT_NE(op, nullptr);
    EXPECT_EQ(op->outputPattern, sparseir::OutputPatternKind::Union);
    ASSERT_NE(op->rootLoop, nullptr);
    EXPECT_EQ(op->rootLoop->kind, sparseir::scheduled::LoopKind::Dense);
    EXPECT_TRUE(hasSparseOutputLoop(op->rootLoop.get(), "C"));
}

TEST(SparseOutputConfigEffects, SpAdd_CSC_BlockingApplied) {
    auto op = parse(SPADD_CSC);
    ASSERT_NE(op, nullptr);
    opt::OptConfig cfg = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, cfg);
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_TRUE(hasBlockLoop(op->rootLoop.get()));
}
