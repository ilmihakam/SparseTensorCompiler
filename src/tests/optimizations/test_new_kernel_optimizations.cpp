#include <gtest/gtest.h>

#include "ast.h"
#include "ir.h"
#include "scheduled_optimizations.h"
#include "semantic_ir.h"

extern std::unique_ptr<SparseTensorCompiler::Program> g_program;
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern int yynerrs;

namespace {

bool parserInitialized = false;

void ensureParserReady() {
    if (parserInitialized) return;
    yynerrs = 0;
    g_program.reset();
    yy_scan_string("tensor x : Dense;");
    yyparse();
    yylex_destroy();
    g_program.reset();
    parserInitialized = true;
}

std::unique_ptr<sparseir::scheduled::Compute> parseScheduled(const std::string& code) {
    ensureParserReady();

    yynerrs = 0;
    g_program.reset();
    yy_scan_string(code.c_str());
    int result = yyparse();
    yylex_destroy();

    if (result != 0 || yynerrs != 0 || !g_program) {
        return nullptr;
    }

    auto compute = sparseir::lowerFirstComputationToScheduled(*g_program);
    g_program.reset();
    return compute;
}

std::vector<std::string> collectOrder(const sparseir::scheduled::Loop* loop) {
    std::vector<std::string> order;
    for (auto* current = loop; current; ) {
        order.push_back(current->indexName);
        current = current->children.empty() ? nullptr : current->children[0].get();
    }
    return order;
}

const sparseir::scheduled::Loop* childAt(const sparseir::scheduled::Loop* loop, size_t index) {
    if (!loop || loop->children.size() <= index) return nullptr;
    return loop->children[index].get();
}

std::string renderFirstPostStmt(const sparseir::scheduled::Loop* loop) {
    if (!loop || loop->postStmts.empty()) return "";
    return ir::renderStmt(*loop->postStmts.front());
}

std::string spaddCode() {
    return
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = A[i, j] + B[i, j];\n";
}

std::string spelmulCode() {
    return
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = A[i, j] * B[i, j];\n";
}

std::string spgemmCode() {
    return
        "tensor A : CSR<100, 100>;\n"
        "tensor B : CSR<100, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = A[i, k] * B[k, j];\n";
}

std::string sddmmCode() {
    return
        "tensor S : CSR<100, 100>;\n"
        "tensor D : Dense<100, 64>;\n"
        "tensor E : Dense<64, 100>;\n"
        "tensor C : Dense<100, 100>;\n"
        "compute C[i, j] = S[i, j] * D[i, k] * E[k, j];\n";
}

std::string spmmCode() {
    return
        "tensor A : CSR<100, 80>;\n"
        "tensor B : Dense<80, 50>;\n"
        "tensor C : Dense<100, 50>;\n"
        "compute C[i, j] = A[i, k] * B[k, j];\n";
}

}  // namespace

TEST(NewKernelOpts, Blocking_SpAdd_CreatesBlockLoop) {
    auto op = parseScheduled(spaddCode());
    ASSERT_NE(op, nullptr);
    opt::applyBlocking(*op, opt::OptConfig::blockingOnly(32));
    EXPECT_TRUE(op->optimizations.blockingApplied);
    ASSERT_NE(op->rootLoop, nullptr);
    EXPECT_EQ(op->rootLoop->indexName, "i_block");
}

TEST(NewKernelOpts, Blocking_SpElMul_CreatesBlockLoop) {
    auto op = parseScheduled(spelmulCode());
    ASSERT_NE(op, nullptr);
    opt::applyBlocking(*op, opt::OptConfig::blockingOnly(32));
    EXPECT_TRUE(op->optimizations.blockingApplied);
    ASSERT_NE(op->rootLoop, nullptr);
    EXPECT_EQ(op->rootLoop->indexName, "i_block");
}

TEST(NewKernelOpts, Blocking_SpGEMM_CreatesBlockLoop) {
    auto op = parseScheduled(spgemmCode());
    ASSERT_NE(op, nullptr);
    opt::applyBlocking(*op, opt::OptConfig::blockingOnly(32));
    EXPECT_TRUE(op->optimizations.blockingApplied);
    ASSERT_NE(op->rootLoop, nullptr);
    EXPECT_EQ(op->rootLoop->indexName, "i_block");
}

TEST(NewKernelOpts, Blocking_SDDMM_CreatesBlockLoop) {
    auto op = parseScheduled(sddmmCode());
    ASSERT_NE(op, nullptr);
    opt::applyBlocking(*op, opt::OptConfig::blockingOnly(32));
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_EQ(op->optimizations.tiledIndex, "k");
    ASSERT_NE(op->rootLoop, nullptr);
    EXPECT_EQ(op->rootLoop->indexName, "i");
    auto order = collectOrder(op->rootLoop.get());
    EXPECT_NE(std::find(order.begin(), order.end(), "k_block"), order.end());
}

TEST(NewKernelOpts, Interchange_SpAdd_NoOp) {
    auto op = parseScheduled(spaddCode());
    ASSERT_NE(op, nullptr);
    opt::applyLoopInterchange(*op, opt::OptConfig::interchangeOnly());
    EXPECT_FALSE(op->optimizations.interchangeApplied);
}

TEST(NewKernelOpts, Interchange_SpElMul_NoOp) {
    auto op = parseScheduled(spelmulCode());
    ASSERT_NE(op, nullptr);
    opt::applyLoopInterchange(*op, opt::OptConfig::interchangeOnly());
    EXPECT_FALSE(op->optimizations.interchangeApplied);
}

TEST(NewKernelOpts, Interchange_SpGEMM_NoOp) {
    auto op = parseScheduled(spgemmCode());
    ASSERT_NE(op, nullptr);
    opt::applyLoopInterchange(*op, opt::OptConfig::interchangeOnly());
    EXPECT_FALSE(op->optimizations.interchangeApplied);
}

TEST(NewKernelOpts, Interchange_SDDMM_Applies) {
    auto op = parseScheduled(sddmmCode());
    ASSERT_NE(op, nullptr);
    opt::applyLoopInterchange(*op, opt::OptConfig::interchangeOnly());
    EXPECT_TRUE(op->optimizations.interchangeApplied);

    auto order = collectOrder(op->rootLoop.get());
    ASSERT_EQ(order.size(), 3u);
    EXPECT_EQ(order[0], "i");
    EXPECT_EQ(order[1], "k");
    EXPECT_EQ(order[2], "j");
}

TEST(NewKernelOpts, Interchange_SDDMM_PreBodyCleared) {
    auto op = parseScheduled(sddmmCode());
    ASSERT_NE(op, nullptr);
    opt::applyLoopInterchange(*op, opt::OptConfig::interchangeOnly());
    ASSERT_TRUE(op->optimizations.interchangeApplied);

    const auto* kLoop = childAt(op->rootLoop.get(), 0);
    const auto* jLoop = childAt(kLoop, 0);
    ASSERT_NE(jLoop, nullptr);
    EXPECT_TRUE(jLoop->preStmts.empty());
}

TEST(NewKernelOpts, Interchange_SDDMM_BodyFused) {
    auto op = parseScheduled(sddmmCode());
    ASSERT_NE(op, nullptr);
    opt::applyLoopInterchange(*op, opt::OptConfig::interchangeOnly());
    ASSERT_TRUE(op->optimizations.interchangeApplied);

    const auto* kLoop = childAt(op->rootLoop.get(), 0);
    const auto* jLoop = childAt(kLoop, 0);
    ASSERT_NE(jLoop, nullptr);
    std::string stmt = renderFirstPostStmt(jLoop);
    EXPECT_NE(stmt.find("+="), std::string::npos);
    EXPECT_NE(stmt.find("C[i][j]"), std::string::npos);
    EXPECT_NE(stmt.find("D[i][k]"), std::string::npos);
    EXPECT_EQ(stmt.find("sum"), std::string::npos);
}

TEST(NewKernelOpts, Interchange_SDDMM_SparseTensorPreserved) {
    auto op = parseScheduled(sddmmCode());
    ASSERT_NE(op, nullptr);
    opt::applyLoopInterchange(*op, opt::OptConfig::interchangeOnly());
    ASSERT_TRUE(op->optimizations.interchangeApplied);

    const auto* kLoop = childAt(op->rootLoop.get(), 0);
    const auto* jLoop = childAt(kLoop, 0);
    ASSERT_NE(jLoop, nullptr);
    EXPECT_EQ(jLoop->driverTensor, "S");
}

TEST(NewKernelOpts, Interchange_SpMM_StillWorks) {
    auto op = parseScheduled(spmmCode());
    ASSERT_NE(op, nullptr);
    opt::applyLoopInterchange(*op, opt::OptConfig::interchangeOnly());
    EXPECT_TRUE(op->optimizations.interchangeApplied);

    auto order = collectOrder(op->rootLoop.get());
    ASSERT_EQ(order.size(), 3u);
    EXPECT_EQ(order[0], "i");
    EXPECT_EQ(order[1], "j");
    EXPECT_EQ(order[2], "k");
}

TEST(NewKernelOpts, Interchange_SpMM_SparseTensorPreserved) {
    auto op = parseScheduled(spmmCode());
    ASSERT_NE(op, nullptr);
    opt::applyLoopInterchange(*op, opt::OptConfig::interchangeOnly());
    ASSERT_TRUE(op->optimizations.interchangeApplied);

    const auto* jLoop = childAt(op->rootLoop.get(), 0);
    const auto* kLoop = childAt(jLoop, 0);
    ASSERT_NE(kLoop, nullptr);
    EXPECT_EQ(kLoop->driverTensor, "A");
}

TEST(NewKernelOpts, AllOpts_SpAdd_ITHENB_BlockOnly) {
    auto op = parseScheduled(spaddCode());
    ASSERT_NE(op, nullptr);
    opt::applyOptimizations(*op, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B));
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_FALSE(op->optimizations.interchangeApplied);
}

TEST(NewKernelOpts, AllOpts_SpGEMM_AllOrders) {
    for (auto order : {opt::OptOrder::I_THEN_B, opt::OptOrder::B_THEN_I, opt::OptOrder::I_B_I}) {
        auto op = parseScheduled(spgemmCode());
        ASSERT_NE(op, nullptr);
        opt::applyOptimizations(*op, opt::OptConfig::allOptimizations(32, order));
        EXPECT_TRUE(op->optimizations.blockingApplied);
        EXPECT_FALSE(op->optimizations.interchangeApplied);
    }
}

TEST(NewKernelOpts, AllOpts_SDDMM_ITHENB) {
    auto op = parseScheduled(sddmmCode());
    ASSERT_NE(op, nullptr);
    opt::applyOptimizations(*op, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B));
    EXPECT_TRUE(op->optimizations.interchangeApplied);
    EXPECT_TRUE(op->optimizations.blockingApplied);
}

TEST(NewKernelOpts, Reordering_SpAdd_CSR_NoOp) {
    auto op = parseScheduled(spaddCode());
    ASSERT_NE(op, nullptr);
    opt::applyReordering(*op);
    EXPECT_FALSE(op->optimizations.reorderingApplied);
}

TEST(NewKernelOpts, Reordering_SpElMul_CSR_NoOp) {
    auto op = parseScheduled(spelmulCode());
    ASSERT_NE(op, nullptr);
    opt::applyReordering(*op);
    EXPECT_FALSE(op->optimizations.reorderingApplied);
}
