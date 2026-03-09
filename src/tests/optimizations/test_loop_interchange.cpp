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

std::unique_ptr<sparseir::scheduled::Compute> createSpMMOperation() {
    ensureParserReady();

    const char* code = R"(
        tensor A : CSR<100, 80>;
        tensor B : Dense<80, 50>;
        tensor C : Dense<100, 50>;
        compute C[i, j] = A[i, k] * B[k, j];
    )";

    yynerrs = 0;
    g_program.reset();
    yy_scan_string(code);
    int result = yyparse();
    yylex_destroy();
    if (result != 0 || yynerrs != 0 || !g_program) {
        return nullptr;
    }

    auto compute = sparseir::lowerFirstComputationToScheduled(*g_program);
    g_program.reset();
    return compute;
}

std::string renderFirstPostStmt(const sparseir::scheduled::Loop* loop) {
    if (!loop || loop->postStmts.empty()) return "";
    return ir::renderStmt(*loop->postStmts.front());
}

}  // namespace

class LoopInterchangeTest : public ::testing::Test {};

TEST_F(LoopInterchangeTest, OriginalLoopOrder_CSR) {
    auto op = createSpMMOperation();
    ASSERT_NE(op, nullptr);
    ASSERT_NE(op->rootLoop, nullptr);

    auto* iLoop = op->rootLoop.get();
    EXPECT_EQ(iLoop->indexName, "i");
    EXPECT_EQ(iLoop->kind, sparseir::scheduled::LoopKind::Dense);

    ASSERT_FALSE(iLoop->children.empty());
    auto* kLoop = iLoop->children[0].get();
    EXPECT_EQ(kLoop->indexName, "k");
    EXPECT_EQ(kLoop->kind, sparseir::scheduled::LoopKind::Sparse);

    ASSERT_FALSE(kLoop->children.empty());
    auto* jLoop = kLoop->children[0].get();
    EXPECT_EQ(jLoop->indexName, "j");
    EXPECT_EQ(jLoop->kind, sparseir::scheduled::LoopKind::Dense);
}

TEST_F(LoopInterchangeTest, OptConfigHasInterchangeFlag) {
    opt::OptConfig config;
    EXPECT_FALSE(config.enableBlocking);
    EXPECT_FALSE(config.enableInterchange);

    auto interchangeConfig = opt::OptConfig::interchangeOnly();
    EXPECT_FALSE(interchangeConfig.enableBlocking);
    EXPECT_TRUE(interchangeConfig.enableInterchange);
}

TEST_F(LoopInterchangeTest, Baseline_NoInterchange) {
    auto op = createSpMMOperation();
    ASSERT_NE(op, nullptr);

    opt::applyOptimizations(*op, opt::OptConfig::baseline());

    ASSERT_NE(op->rootLoop, nullptr);
    EXPECT_EQ(op->rootLoop->indexName, "i");
    EXPECT_EQ(op->rootLoop->children[0]->indexName, "k");
    EXPECT_EQ(op->rootLoop->children[0]->children[0]->indexName, "j");
    EXPECT_FALSE(op->optimizations.interchangeApplied);
}

TEST_F(LoopInterchangeTest, IndividualOptimizations) {
    auto blockConfig = opt::OptConfig::blockingOnly();
    EXPECT_TRUE(blockConfig.enableBlocking);
    EXPECT_FALSE(blockConfig.enableInterchange);

    auto interchangeConfig = opt::OptConfig::interchangeOnly();
    EXPECT_FALSE(interchangeConfig.enableBlocking);
    EXPECT_TRUE(interchangeConfig.enableInterchange);

    auto bothConfig = opt::OptConfig::allOptimizations();
    EXPECT_TRUE(bothConfig.enableBlocking);
    EXPECT_TRUE(bothConfig.enableInterchange);
}

TEST_F(LoopInterchangeTest, InterchangeJK_CSR) {
    auto op = createSpMMOperation();
    ASSERT_NE(op, nullptr);

    opt::applyOptimizations(*op, opt::OptConfig::interchangeOnly());

    auto* iLoop = op->rootLoop.get();
    ASSERT_NE(iLoop, nullptr);
    EXPECT_EQ(iLoop->indexName, "i");

    auto* jLoop = iLoop->children[0].get();
    ASSERT_NE(jLoop, nullptr);
    EXPECT_EQ(jLoop->indexName, "j");
    EXPECT_EQ(jLoop->kind, sparseir::scheduled::LoopKind::Dense);

    auto* kLoop = jLoop->children[0].get();
    ASSERT_NE(kLoop, nullptr);
    EXPECT_EQ(kLoop->indexName, "k");
    EXPECT_EQ(kLoop->kind, sparseir::scheduled::LoopKind::Sparse);
    EXPECT_TRUE(op->optimizations.interchangeApplied);
}

TEST_F(LoopInterchangeTest, InterchangePreservesBody) {
    auto op = createSpMMOperation();
    ASSERT_NE(op, nullptr);

    const std::string originalBody = renderFirstPostStmt(op->rootLoop->children[0]->children[0].get());
    ASSERT_FALSE(originalBody.empty());

    opt::applyOptimizations(*op, opt::OptConfig::interchangeOnly());

    auto* iLoop = op->rootLoop.get();
    auto* jLoop = iLoop->children[0].get();
    auto* kLoop = jLoop->children[0].get();
    ASSERT_NE(kLoop, nullptr);
    EXPECT_EQ(renderFirstPostStmt(kLoop), originalBody);
}

TEST_F(LoopInterchangeTest, CannotMoveSparseToOutermost) {
    GTEST_SKIP() << "Interchange validation not yet implemented";
}

TEST_F(LoopInterchangeTest, InterchangeIsLocal) {
    GTEST_SKIP() << "Interchange optimization locality test not yet implemented";
}

TEST_F(LoopInterchangeTest, InterchangePreservesCorrectness) {
    GTEST_SKIP() << "End-to-end correctness test requires codegen";
}

TEST_F(LoopInterchangeTest, InterchangeWithBlocking) {
    GTEST_SKIP() << "Combined optimization integration is covered elsewhere";
}
