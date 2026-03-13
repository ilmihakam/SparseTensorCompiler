// ir.h — Shared IR types and rendering utilities
//
// This file defines the base types (Tensor, Format, IRExpr, IRStmt) and
// rendering functions used by the IR pipeline. The actual lowering logic
// lives in semantic_ir.h/cpp.
//
// Architecture:
//   ir.h/cpp          — Base types, expression/statement IR, rendering
//   semantic_ir.h/cpp — Two-stage lowering: AST → semantic IR → scheduled IR
//   codegen.h/cpp     — C code generation from scheduled IR

#ifndef IR_H
#define IR_H

#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ast.h"

namespace ir {

enum class Format {
    Dense,
    CSR,
    CSC
};

enum class RootOpKind { ADD, MULT };

enum class MergeStrategy {
    None,
    Union,
    Intersection
};

enum class OutputStrategy {
    DenseArray,
    SparseFixedPattern,
    HashPerRow
};

struct Tensor {
    std::string name;
    Format format = Format::Dense;
    std::vector<int> dims;
    std::vector<std::string> indices;

    Tensor() = default;
    Tensor(const std::string& n,
           Format f,
           const std::vector<int>& d,
           const std::vector<std::string>& idx)
        : name(n), format(f), dims(d), indices(idx) {}
};

struct ExpressionInfo {
    RootOpKind rootOp = RootOpKind::MULT;
    bool isFused = false;
    std::string fusionFunction;
    int numTensorAccesses = 0;
    int numSparseInputs = 0;
    int numDenseInputs = 0;
    std::vector<std::string> sparseInputNames;
    std::vector<std::string> denseInputNames;
};

struct IndexVar {
    std::string name;
    int lower = 0;
    int upper = 0;
    bool isSparse = false;

    IndexVar() = default;
    IndexVar(const std::string& n, int lo, int hi, bool sparse)
        : name(n), lower(lo), upper(hi), isSparse(sparse) {}
};

struct LoopOptimizations {
    bool reorderingApplied = false;
    std::vector<std::string> originalOrder;
    std::vector<std::string> newOrder;

    bool interchangeApplied = false;
    std::vector<std::string> interchangeOriginalOrder;
    std::vector<std::string> interchangeRequestedOrder;
    std::vector<std::string> interchangeFinalOrder;

    bool blockingApplied = false;
    int blockSize = 32;
    std::string tiledIndex;

    bool blocking2DApplied = false;
    std::vector<std::string> tiledIndices;
    std::vector<int> blockSizes;

    bool positionBlockingApplied = false;
    int positionBlockSize = 32;
    std::vector<std::string> positionTiledIndices;
};

class IRExprVisitor;
struct IRExpr;
struct IRStmt;

struct IRExpr {
    virtual ~IRExpr() = default;
    virtual void accept(IRExprVisitor& v) const = 0;
    virtual std::unique_ptr<IRExpr> clone() const = 0;
};

struct IRTensorAccess : IRExpr {
    std::string tensorName;
    std::vector<std::string> indices;
    bool isSparseVals = false;
    std::string pointerVar;
    bool useRandomAccess = false;
    std::string randomAccessFunc;

    IRTensorAccess() = default;
    IRTensorAccess(const std::string& name, const std::vector<std::string>& idx)
        : tensorName(name), indices(idx) {}

    void accept(IRExprVisitor& v) const override;
    std::unique_ptr<IRExpr> clone() const override;
};

struct IRConstant : IRExpr {
    double value = 0.0;

    IRConstant() = default;
    explicit IRConstant(double v) : value(v) {}

    void accept(IRExprVisitor& v) const override;
    std::unique_ptr<IRExpr> clone() const override;
};

struct IRBinaryOp : IRExpr {
    enum Op { ADD, MUL };
    Op op = MUL;
    std::unique_ptr<IRExpr> lhs;
    std::unique_ptr<IRExpr> rhs;

    IRBinaryOp() = default;
    IRBinaryOp(Op o, std::unique_ptr<IRExpr> l, std::unique_ptr<IRExpr> r)
        : op(o), lhs(std::move(l)), rhs(std::move(r)) {}

    void accept(IRExprVisitor& v) const override;
    std::unique_ptr<IRExpr> clone() const override;
};

struct IRScalarVar : IRExpr {
    std::string name;

    IRScalarVar() = default;
    explicit IRScalarVar(const std::string& n) : name(n) {}

    void accept(IRExprVisitor& v) const override;
    std::unique_ptr<IRExpr> clone() const override;
};

struct IRFuncCall : IRExpr {
    std::string name;
    std::vector<std::unique_ptr<IRExpr>> args;

    IRFuncCall() = default;
    explicit IRFuncCall(const std::string& n) : name(n) {}

    void accept(IRExprVisitor& v) const override;
    std::unique_ptr<IRExpr> clone() const override;
};

struct IRIndexedAccess : IRExpr {
    std::string baseName;
    std::vector<std::unique_ptr<IRExpr>> indices;

    IRIndexedAccess() = default;
    explicit IRIndexedAccess(const std::string& name) : baseName(name) {}

    void accept(IRExprVisitor& v) const override;
    std::unique_ptr<IRExpr> clone() const override;
};

struct IRCompareExpr : IRExpr {
    enum Op { EQ, LT };
    Op op = EQ;
    std::unique_ptr<IRExpr> lhs;
    std::unique_ptr<IRExpr> rhs;

    IRCompareExpr() = default;
    IRCompareExpr(Op compareOp, std::unique_ptr<IRExpr> left, std::unique_ptr<IRExpr> right)
        : op(compareOp), lhs(std::move(left)), rhs(std::move(right)) {}

    void accept(IRExprVisitor& v) const override;
    std::unique_ptr<IRExpr> clone() const override;
};

struct IRAccumulatorRef : IRExpr {
    std::string name;

    IRAccumulatorRef() = default;
    explicit IRAccumulatorRef(const std::string& n) : name(n) {}

    void accept(IRExprVisitor& v) const override;
    std::unique_ptr<IRExpr> clone() const override;
};

class IRExprVisitor {
public:
    virtual ~IRExprVisitor() = default;

    virtual void visit(const IRTensorAccess&) = 0;
    virtual void visit(const IRConstant&) = 0;
    virtual void visit(const IRBinaryOp&) = 0;
    virtual void visit(const IRScalarVar&) = 0;
    virtual void visit(const IRFuncCall&) = 0;
    virtual void visit(const IRIndexedAccess&) = 0;
    virtual void visit(const IRCompareExpr&) = 0;
    virtual void visit(const IRAccumulatorRef&) = 0;
};

struct IRStmt {
    virtual ~IRStmt() = default;
    virtual std::unique_ptr<IRStmt> clone() const = 0;
};

struct IRScalarDecl : IRStmt {
    std::string varName;
    double initValue = 0.0;

    IRScalarDecl() = default;
    IRScalarDecl(const std::string& name, double init)
        : varName(name), initValue(init) {}

    std::unique_ptr<IRStmt> clone() const override;
};

struct IRAccumulatorInit : IRStmt {
    std::string accumulatorName;
    double initValue = 0.0;

    IRAccumulatorInit() = default;
    IRAccumulatorInit(const std::string& name, double init)
        : accumulatorName(name), initValue(init) {}

    std::unique_ptr<IRStmt> clone() const override;
};

struct IRAssign : IRStmt {
    std::unique_ptr<IRExpr> lhs;
    std::unique_ptr<IRExpr> rhs;
    bool accumulate = true;

    IRAssign() = default;
    IRAssign(std::unique_ptr<IRExpr> l, std::unique_ptr<IRExpr> r, bool accum = true)
        : lhs(std::move(l)), rhs(std::move(r)), accumulate(accum) {}

    std::unique_ptr<IRStmt> clone() const override;
};

struct IRAccumulatorUpdate : IRStmt {
    std::string accumulatorName;
    std::unique_ptr<IRExpr> rhs;

    IRAccumulatorUpdate() = default;
    IRAccumulatorUpdate(const std::string& name, std::unique_ptr<IRExpr> expr)
        : accumulatorName(name), rhs(std::move(expr)) {}

    std::unique_ptr<IRStmt> clone() const override;
};

struct IRCallStmt : IRStmt {
    std::string functionName;
    std::vector<std::unique_ptr<IRExpr>> args;

    IRCallStmt() = default;
    explicit IRCallStmt(const std::string& name) : functionName(name) {}

    std::unique_ptr<IRStmt> clone() const override;
};

struct IRAccumulatorFinalize : IRStmt {
    std::unique_ptr<IRExpr> lhs;
    std::unique_ptr<IRExpr> rhs;

    IRAccumulatorFinalize() = default;
    IRAccumulatorFinalize(std::unique_ptr<IRExpr> out, std::unique_ptr<IRExpr> expr)
        : lhs(std::move(out)), rhs(std::move(expr)) {}

    std::unique_ptr<IRStmt> clone() const override;
};

struct IRRawStmt : IRStmt {
    std::string code;

    IRRawStmt() = default;
    explicit IRRawStmt(const std::string& text) : code(text) {}

    std::unique_ptr<IRStmt> clone() const override;
};

struct IRVarDecl : IRStmt {
    std::string varName;
    std::string type;       // "int", "double*", "unsigned char*"
    std::string initExpr;   // "0", "calloc((size_t)N, sizeof(double))", etc.

    IRVarDecl() = default;
    IRVarDecl(const std::string& name, const std::string& t, const std::string& init)
        : varName(name), type(t), initExpr(init) {}

    std::unique_ptr<IRStmt> clone() const override;
};

struct IRFreeStmt : IRStmt {
    std::string varName;

    IRFreeStmt() = default;
    explicit IRFreeStmt(const std::string& name) : varName(name) {}

    std::unique_ptr<IRStmt> clone() const override;
};

struct IRIfStmt : IRStmt {
    std::unique_ptr<IRExpr> condition;
    std::vector<std::unique_ptr<IRStmt>> thenBody;

    IRIfStmt() = default;
    explicit IRIfStmt(std::unique_ptr<IRExpr> cond) : condition(std::move(cond)) {}

    std::unique_ptr<IRStmt> clone() const override;
};

struct IRForStmt : IRStmt {
    std::string loopVar;
    std::unique_ptr<IRExpr> lower;
    std::unique_ptr<IRExpr> upper;
    std::vector<std::unique_ptr<IRStmt>> body;

    IRForStmt() = default;
    IRForStmt(const std::string& var,
              std::unique_ptr<IRExpr> lo,
              std::unique_ptr<IRExpr> hi)
        : loopVar(var), lower(std::move(lo)), upper(std::move(hi)) {}

    std::unique_ptr<IRStmt> clone() const override;
};

void renderStmtsToStrings(
    const std::vector<std::unique_ptr<IRStmt>>& preStmts,
    const std::vector<std::unique_ptr<IRStmt>>& postStmts,
    std::string& preBody,
    std::string& body);

std::string renderExpr(const IRExpr& expr);
std::string renderStmt(const IRStmt& stmt);

std::string formatToString(Format f);
std::string rootOpKindToString(RootOpKind k);
std::string mergeStrategyToString(MergeStrategy m);
void printExpressionInfo(std::ostream& out, const ExpressionInfo& info);

} // namespace ir

#endif // IR_H
