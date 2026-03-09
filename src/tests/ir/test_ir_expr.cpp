/**
 * Test Suite: IR Expression Tree (Phase A)
 *
 * Tests the IRExpr/IRStmt hierarchy, renderExpr/renderStmt,
 * renderStmtsToStrings, and clone operations.
 */

#include <gtest/gtest.h>
#include "ir.h"

using namespace ir;

// ============================================================================
// IRExpr Construction & Clone Tests
// ============================================================================

TEST(IRExprTest, TensorAccessClone) {
    IRTensorAccess ta("A", {"i", "j"});
    auto cloned = ta.clone();
    auto* c = dynamic_cast<IRTensorAccess*>(cloned.get());
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->tensorName, "A");
    EXPECT_EQ(c->indices.size(), 2);
    EXPECT_EQ(c->indices[0], "i");
    EXPECT_EQ(c->indices[1], "j");
    EXPECT_FALSE(c->isSparseVals);
}

TEST(IRExprTest, TensorAccessSparseVals) {
    IRTensorAccess ta;
    ta.tensorName = "A";
    ta.isSparseVals = true;
    ta.pointerVar = "pA";
    auto cloned = ta.clone();
    auto* c = dynamic_cast<IRTensorAccess*>(cloned.get());
    ASSERT_NE(c, nullptr);
    EXPECT_TRUE(c->isSparseVals);
    EXPECT_EQ(c->pointerVar, "pA");
}

TEST(IRExprTest, ConstantClone) {
    IRConstant k(3.14);
    auto cloned = k.clone();
    auto* c = dynamic_cast<IRConstant*>(cloned.get());
    ASSERT_NE(c, nullptr);
    EXPECT_DOUBLE_EQ(c->value, 3.14);
}

TEST(IRExprTest, BinaryOpClone) {
    auto lhs = std::make_unique<IRConstant>(1.0);
    auto rhs = std::make_unique<IRConstant>(2.0);
    IRBinaryOp bin(IRBinaryOp::ADD, std::move(lhs), std::move(rhs));
    auto cloned = bin.clone();
    auto* c = dynamic_cast<IRBinaryOp*>(cloned.get());
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->op, IRBinaryOp::ADD);
    ASSERT_NE(c->lhs, nullptr);
    ASSERT_NE(c->rhs, nullptr);
}

TEST(IRExprTest, ScalarVarClone) {
    IRScalarVar sv("sum");
    auto cloned = sv.clone();
    auto* c = dynamic_cast<IRScalarVar*>(cloned.get());
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->name, "sum");
}

TEST(IRExprTest, FuncCallClone) {
    IRFuncCall fc("relu");
    fc.args.push_back(std::make_unique<IRConstant>(0.5));
    auto cloned = fc.clone();
    auto* c = dynamic_cast<IRFuncCall*>(cloned.get());
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->name, "relu");
    EXPECT_EQ(c->args.size(), 1);
}

// ============================================================================
// IRStmt Clone Tests
// ============================================================================

TEST(IRStmtTest, ScalarDeclClone) {
    IRScalarDecl decl("sum", 0.0);
    auto cloned = decl.clone();
    auto* c = dynamic_cast<IRScalarDecl*>(cloned.get());
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->varName, "sum");
    EXPECT_DOUBLE_EQ(c->initValue, 0.0);
}

TEST(IRStmtTest, AssignClone) {
    auto lhs = std::make_unique<IRTensorAccess>("y", std::vector<std::string>{"i"});
    auto rhs = std::make_unique<IRConstant>(1.0);
    IRAssign assign(std::move(lhs), std::move(rhs), true);
    auto cloned = assign.clone();
    auto* c = dynamic_cast<IRAssign*>(cloned.get());
    ASSERT_NE(c, nullptr);
    EXPECT_TRUE(c->accumulate);
    ASSERT_NE(c->lhs, nullptr);
    ASSERT_NE(c->rhs, nullptr);
}

TEST(IRStmtTest, CallStmtClone) {
    IRCallStmt call("hash_accumulate");
    call.args.push_back(std::make_unique<IRConstant>(42.0));
    auto cloned = call.clone();
    auto* c = dynamic_cast<IRCallStmt*>(cloned.get());
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->functionName, "hash_accumulate");
    EXPECT_EQ(c->args.size(), 1);
}

// ============================================================================
// renderExpr Tests
// ============================================================================

TEST(RenderExprTest, DenseTensorAccess) {
    IRTensorAccess ta("B", {"k", "j"});
    std::string result = renderExpr(ta);
    EXPECT_EQ(result, "B[k][j]");
}

TEST(RenderExprTest, SparseTensorAccess) {
    IRTensorAccess ta;
    ta.tensorName = "A";
    ta.isSparseVals = true;
    ta.pointerVar = "pA";
    std::string result = renderExpr(ta);
    EXPECT_EQ(result, "A_vals[pA]");
}

TEST(RenderExprTest, Constant) {
    IRConstant k(2.5);
    std::string result = renderExpr(k);
    EXPECT_EQ(result, "2.5");
}

TEST(RenderExprTest, ConstantInteger) {
    IRConstant k(3.0);
    std::string result = renderExpr(k);
    EXPECT_EQ(result, "3");
}

TEST(RenderExprTest, BinaryMul) {
    auto lhs = std::make_unique<IRTensorAccess>("A", std::vector<std::string>{});
    lhs->isSparseVals = true;
    lhs->pointerVar = "pA";
    auto rhs = std::make_unique<IRTensorAccess>("x", std::vector<std::string>{"j"});
    IRBinaryOp bin(IRBinaryOp::MUL, std::move(lhs), std::move(rhs));
    std::string result = renderExpr(bin);
    EXPECT_EQ(result, "A_vals[pA] * x[j]");
}

TEST(RenderExprTest, BinaryAdd) {
    auto lhs = std::make_unique<IRConstant>(1.0);
    auto rhs = std::make_unique<IRConstant>(2.0);
    IRBinaryOp bin(IRBinaryOp::ADD, std::move(lhs), std::move(rhs));
    std::string result = renderExpr(bin);
    EXPECT_EQ(result, "1 + 2");
}

TEST(RenderExprTest, ScalarVar) {
    IRScalarVar sv("sum");
    std::string result = renderExpr(sv);
    EXPECT_EQ(result, "sum");
}

TEST(RenderExprTest, FuncCall) {
    IRFuncCall fc("relu");
    fc.args.push_back(std::make_unique<IRScalarVar>("x"));
    std::string result = renderExpr(fc);
    EXPECT_EQ(result, "relu(x)");
}

// ============================================================================
// renderStmt Tests
// ============================================================================

TEST(RenderStmtTest, ScalarDecl) {
    IRScalarDecl decl("sum", 0.0);
    std::string result = renderStmt(decl);
    EXPECT_EQ(result, "double sum = 0.0;");
}

TEST(RenderStmtTest, AssignAccumulate) {
    auto lhs = std::make_unique<IRTensorAccess>("y", std::vector<std::string>{"i"});
    auto rhs = std::make_unique<IRConstant>(1.0);
    IRAssign assign(std::move(lhs), std::move(rhs), true);
    std::string result = renderStmt(assign);
    EXPECT_EQ(result, "y[i] += 1;");
}

TEST(RenderStmtTest, AssignDirect) {
    auto lhs = std::make_unique<IRTensorAccess>("y", std::vector<std::string>{"i"});
    auto rhs = std::make_unique<IRConstant>(0.0);
    IRAssign assign(std::move(lhs), std::move(rhs), false);
    std::string result = renderStmt(assign);
    EXPECT_EQ(result, "y[i] = 0;");
}

// ============================================================================
// renderStmtsToStrings Tests
// ============================================================================

TEST(RenderStmtsToStringsTest, PreAndPostStmts) {
    std::vector<std::unique_ptr<IRStmt>> preStmts;
    preStmts.push_back(std::make_unique<IRScalarDecl>("sum", 0.0));

    std::vector<std::unique_ptr<IRStmt>> postStmts;
    auto lhs = std::make_unique<IRTensorAccess>("y", std::vector<std::string>{"i"});
    auto rhs = std::make_unique<IRScalarVar>("sum");
    postStmts.push_back(std::make_unique<IRAssign>(std::move(lhs), std::move(rhs), false));

    std::string preBody, body;
    renderStmtsToStrings(preStmts, postStmts, preBody, body);

    EXPECT_EQ(preBody, "double sum = 0.0;");
    EXPECT_EQ(body, "y[i] = sum;");
}

// ============================================================================
// IRExprVisitor Tests
// ============================================================================

namespace {
class CountingVisitor : public IRExprVisitor {
public:
    int tensorAccesses = 0;
    int constants = 0;
    int binaryOps = 0;
    int scalarVars = 0;
    int funcCalls = 0;

    void visit(const IRTensorAccess&) override { tensorAccesses++; }
    void visit(const IRConstant&) override { constants++; }
    void visit(const IRBinaryOp& b) override {
        binaryOps++;
        b.lhs->accept(*this);
        b.rhs->accept(*this);
    }
    void visit(const IRScalarVar&) override { scalarVars++; }
    void visit(const IRFuncCall& f) override {
        funcCalls++;
        for (auto& a : f.args) a->accept(*this);
    }
    void visit(const IRIndexedAccess& a) override {
        tensorAccesses++;
        for (auto& idx : a.indices) idx->accept(*this);
    }
    void visit(const IRCompareExpr& c) override {
        binaryOps++;
        c.lhs->accept(*this);
        c.rhs->accept(*this);
    }
};
}  // namespace

TEST(IRExprVisitorTest, CountNodes) {
    // Build: A->vals[pA] * x[j]
    auto lhs = std::make_unique<IRTensorAccess>();
    lhs->tensorName = "A";
    lhs->isSparseVals = true;
    lhs->pointerVar = "pA";
    auto rhs = std::make_unique<IRTensorAccess>("x", std::vector<std::string>{"j"});
    auto mul = std::make_unique<IRBinaryOp>(IRBinaryOp::MUL, std::move(lhs), std::move(rhs));

    CountingVisitor cv;
    mul->accept(cv);
    EXPECT_EQ(cv.binaryOps, 1);
    EXPECT_EQ(cv.tensorAccesses, 2);
    EXPECT_EQ(cv.constants, 0);
}
