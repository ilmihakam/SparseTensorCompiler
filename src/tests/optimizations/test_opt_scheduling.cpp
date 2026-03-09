#include <gtest/gtest.h>

#include "ast.h"
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

std::vector<std::string> getLoopOrder(const sparseir::scheduled::Loop* loop) {
    std::vector<std::string> order;
    const auto* current = loop;
    while (current) {
        order.push_back(current->indexName);
        current = current->children.empty() ? nullptr : current->children[0].get();
    }
    return order;
}

std::string spmmCSR() {
    return
        "tensor A : CSR<100, 80>;\n"
        "tensor B : Dense<80, 50>;\n"
        "tensor C : Dense<100, 50>;\n"
        "compute C[i, j] = A[i, k] * B[k, j];\n";
}

std::string spmmCSC() {
    return
        "tensor A : CSC<100, 80>;\n"
        "tensor B : Dense<80, 50>;\n"
        "tensor C : Dense<100, 50>;\n"
        "compute C[i, j] = A[i, k] * B[k, j];\n";
}

}  // namespace

TEST(OptSchedulingTest, OptOrderEnumExists) {
    EXPECT_EQ(static_cast<int>(opt::OptOrder::I_THEN_B), 0);
    EXPECT_EQ(static_cast<int>(opt::OptOrder::B_THEN_I), 1);
    EXPECT_EQ(static_cast<int>(opt::OptOrder::I_B_I), 2);
}

TEST(OptSchedulingTest, OptConfigHasOrderField) {
    opt::OptConfig config;
    EXPECT_EQ(config.order, opt::OptOrder::I_THEN_B);
    config.order = opt::OptOrder::B_THEN_I;
    EXPECT_EQ(config.order, opt::OptOrder::B_THEN_I);
    config.order = opt::OptOrder::I_B_I;
    EXPECT_EQ(config.order, opt::OptOrder::I_B_I);
}

TEST(OptSchedulingTest, I_THEN_B_Order) {
    auto op = parseScheduled(spmmCSR());
    ASSERT_NE(op, nullptr);
    opt::applyOptimizations(*op, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B));

    auto order = getLoopOrder(op->rootLoop.get());
    ASSERT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], "i");
    EXPECT_EQ(order[1], "j_block");
    EXPECT_EQ(order[2], "j");
    EXPECT_EQ(order[3], "k");
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_TRUE(op->optimizations.interchangeApplied);
    EXPECT_EQ(op->optimizations.tiledIndex, "j");
}

TEST(OptSchedulingTest, B_THEN_I_Order) {
    auto op = parseScheduled(spmmCSR());
    ASSERT_NE(op, nullptr);
    opt::applyOptimizations(*op, opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I));

    auto order = getLoopOrder(op->rootLoop.get());
    ASSERT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], "i");
    EXPECT_EQ(order[1], "j_block");
    EXPECT_EQ(order[2], "k");
    EXPECT_EQ(order[3], "j");
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_TRUE(op->optimizations.interchangeApplied);
    EXPECT_EQ(op->optimizations.tiledIndex, "j");
}

TEST(OptSchedulingTest, I_B_I_Order) {
    auto op = parseScheduled(spmmCSR());
    ASSERT_NE(op, nullptr);
    opt::applyOptimizations(*op, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_B_I));

    auto order = getLoopOrder(op->rootLoop.get());
    ASSERT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], "i");
    EXPECT_EQ(order[1], "j_block");
    EXPECT_EQ(order[2], "j");
    EXPECT_EQ(order[3], "k");
    EXPECT_TRUE(op->optimizations.blockingApplied);
    EXPECT_TRUE(op->optimizations.interchangeApplied);
}

TEST(OptSchedulingTest, OnlyBlocking_OrderIrrelevant) {
    auto op1 = parseScheduled(spmmCSR());
    auto op2 = parseScheduled(spmmCSR());
    ASSERT_NE(op1, nullptr);
    ASSERT_NE(op2, nullptr);

    opt::applyOptimizations(*op1, opt::OptConfig::blockingOnly(32));
    auto config = opt::OptConfig::blockingOnly(32);
    config.order = opt::OptOrder::B_THEN_I;
    opt::applyOptimizations(*op2, config);

    EXPECT_EQ(getLoopOrder(op1->rootLoop.get()), getLoopOrder(op2->rootLoop.get()));
    EXPECT_TRUE(op1->optimizations.blockingApplied);
    EXPECT_TRUE(op2->optimizations.blockingApplied);
}

TEST(OptSchedulingTest, NoOptimizations_OrderIrrelevant) {
    auto op = parseScheduled(spmmCSR());
    ASSERT_NE(op, nullptr);

    opt::OptConfig config;
    config.enableInterchange = false;
    config.enableBlocking = false;
    config.order = opt::OptOrder::I_B_I;
    opt::applyOptimizations(*op, config);

    EXPECT_FALSE(op->optimizations.blockingApplied);
    EXPECT_FALSE(op->optimizations.interchangeApplied);
}

TEST(OptSchedulingTest, CSR_ScheduleFingerprintsDiffer) {
    auto opITB = parseScheduled(spmmCSR());
    auto opBTI = parseScheduled(spmmCSR());
    ASSERT_NE(opITB, nullptr);
    ASSERT_NE(opBTI, nullptr);

    opt::applyOptimizations(*opITB, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B));
    opt::applyOptimizations(*opBTI, opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I));
    EXPECT_NE(getLoopOrder(opITB->rootLoop.get()), getLoopOrder(opBTI->rootLoop.get()));
}

TEST(OptSchedulingTest, CSC_I_THEN_B_Order) {
    auto op = parseScheduled(spmmCSC());
    ASSERT_NE(op, nullptr);
    opt::applyOptimizations(*op, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B));

    auto order = getLoopOrder(op->rootLoop.get());
    ASSERT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], "k");
    EXPECT_EQ(order[1], "j_block");
    EXPECT_EQ(order[2], "j");
    EXPECT_EQ(order[3], "i");
}

TEST(OptSchedulingTest, CSC_I_B_I_Order) {
    auto op = parseScheduled(spmmCSC());
    ASSERT_NE(op, nullptr);
    opt::applyOptimizations(*op, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_B_I));

    auto order = getLoopOrder(op->rootLoop.get());
    ASSERT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], "k");
    EXPECT_EQ(order[1], "j_block");
    EXPECT_EQ(order[2], "j");
    EXPECT_EQ(order[3], "i");
}

TEST(OptSchedulingTest, CSC_B_THEN_I_Order) {
    auto op = parseScheduled(spmmCSC());
    ASSERT_NE(op, nullptr);
    opt::applyOptimizations(*op, opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I));

    auto order = getLoopOrder(op->rootLoop.get());
    ASSERT_EQ(order.size(), 4u);
    EXPECT_EQ(order[0], "k");
    EXPECT_EQ(order[1], "j_block");
    EXPECT_EQ(order[2], "i");
    EXPECT_EQ(order[3], "j");
    EXPECT_EQ(op->optimizations.tiledIndex, "j");
}

TEST(OptSchedulingTest, CSR_BThenI_DistinctFrom_IBI) {
    auto opBThenI = parseScheduled(spmmCSR());
    auto opIBI = parseScheduled(spmmCSR());
    ASSERT_NE(opBThenI, nullptr);
    ASSERT_NE(opIBI, nullptr);

    opt::applyOptimizations(*opBThenI, opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I));
    opt::applyOptimizations(*opIBI, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_B_I));
    EXPECT_NE(getLoopOrder(opBThenI->rootLoop.get()), getLoopOrder(opIBI->rootLoop.get()));
}

TEST(OptSchedulingTest, CSC_BThenI_DistinctFrom_IBI) {
    auto opBThenI = parseScheduled(spmmCSC());
    auto opIBI = parseScheduled(spmmCSC());
    ASSERT_NE(opBThenI, nullptr);
    ASSERT_NE(opIBI, nullptr);

    opt::applyOptimizations(*opBThenI, opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I));
    opt::applyOptimizations(*opIBI, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_B_I));
    EXPECT_NE(getLoopOrder(opBThenI->rootLoop.get()), getLoopOrder(opIBI->rootLoop.get()));
}

TEST(OptSchedulingTest, CSC_ScheduleFingerprintsDiffer) {
    auto opITB = parseScheduled(spmmCSC());
    auto opBTI = parseScheduled(spmmCSC());
    ASSERT_NE(opITB, nullptr);
    ASSERT_NE(opBTI, nullptr);

    opt::applyOptimizations(*opITB, opt::OptConfig::allOptimizations(32, opt::OptOrder::I_THEN_B));
    opt::applyOptimizations(*opBTI, opt::OptConfig::allOptimizations(32, opt::OptOrder::B_THEN_I));
    EXPECT_NE(getLoopOrder(opITB->rootLoop.get()), getLoopOrder(opBTI->rootLoop.get()));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
