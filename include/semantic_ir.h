#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ast.h"
#include "ir.h"

namespace opt { struct OptConfig; }

namespace sparseir {

enum class ExprBinaryOp { Add, Mul };

struct Expr {
    virtual ~Expr() = default;
    virtual std::unique_ptr<Expr> clone() const = 0;
};

struct TensorRead : Expr {
    std::string tensorName;
    std::vector<std::string> indices;

    TensorRead() = default;
    TensorRead(const std::string& name, const std::vector<std::string>& idx)
        : tensorName(name), indices(idx) {}

    std::unique_ptr<Expr> clone() const override;
};

struct Constant : Expr {
    double value = 0.0;

    Constant() = default;
    explicit Constant(double v) : value(v) {}

    std::unique_ptr<Expr> clone() const override;
};

struct ScalarRef : Expr {
    std::string name;

    ScalarRef() = default;
    explicit ScalarRef(const std::string& n) : name(n) {}

    std::unique_ptr<Expr> clone() const override;
};

struct BinaryExpr : Expr {
    ExprBinaryOp op = ExprBinaryOp::Mul;
    std::unique_ptr<Expr> lhs;
    std::unique_ptr<Expr> rhs;

    BinaryExpr() = default;
    BinaryExpr(ExprBinaryOp o, std::unique_ptr<Expr> left, std::unique_ptr<Expr> right)
        : op(o), lhs(std::move(left)), rhs(std::move(right)) {}

    std::unique_ptr<Expr> clone() const override;
};

struct CallExpr : Expr {
    std::string functionName;
    std::vector<std::unique_ptr<Expr>> args;

    CallExpr() = default;
    explicit CallExpr(const std::string& name) : functionName(name) {}

    std::unique_ptr<Expr> clone() const override;
};

struct TensorUse {
    std::string tensorName;
    std::vector<std::string> indices;
};

enum class IteratorKind { Dense, Sparse };
enum class MergeKind { None, Union, Intersection };
enum class OutputPatternKind {
    None,
    Union,
    Intersection,
    Sampled,
    DynamicRowAccumulator
};

struct IteratorSource {
    std::string tensorName;
    ir::Format format = ir::Format::Dense;
    int accessPosition = -1;
    bool sparse = false;
};

struct IteratorNode {
    std::string indexName;
    int lower = 0;
    int upper = 0;
    bool isReduction = false;
    IteratorKind kind = IteratorKind::Dense;
    MergeKind merge = MergeKind::None;
    std::vector<IteratorSource> sources;
};

struct IteratorGraph {
    std::vector<IteratorNode> iterators;
};

namespace semantic {

struct Stmt {
    virtual ~Stmt() = default;
    virtual std::unique_ptr<Stmt> clone() const = 0;
};

struct Declaration : Stmt {
    ir::Tensor tensor;

    Declaration() = default;
    explicit Declaration(const ir::Tensor& t) : tensor(t) {}

    std::unique_ptr<Stmt> clone() const override;
};

struct Call : Stmt {
    std::string functionName;
    std::vector<std::unique_ptr<Expr>> args;

    Call() = default;
    explicit Call(const std::string& name) : functionName(name) {}

    std::unique_ptr<Stmt> clone() const override;
};

struct Compute : Stmt {
    TensorUse lhs;
    ir::Tensor output;
    std::vector<ir::Tensor> inputs;
    std::unique_ptr<Expr> rhs;
    IteratorGraph iteratorGraph;
    std::vector<std::string> freeIndices;
    std::vector<std::string> reductionIndices;
    ir::ExpressionInfo exprInfo;
    ir::OutputStrategy outputStrategy = ir::OutputStrategy::DenseArray;

    std::unique_ptr<Stmt> clone() const override;
};

struct Region : Stmt {
    std::vector<std::string> tensors;
    std::vector<std::string> indices;
    std::vector<std::string> runtimeBounds;
    std::vector<std::unique_ptr<Stmt>> body;

    Region() = default;
    Region(const std::vector<std::string>& tensorList,
           const std::vector<std::string>& indexList)
        : tensors(tensorList), indices(indexList) {}

    void addStatement(std::unique_ptr<Stmt> stmt) {
        body.push_back(std::move(stmt));
    }

    std::unique_ptr<Stmt> clone() const override;
};

struct Program {
    std::vector<std::unique_ptr<Stmt>> statements;

    void addStatement(std::unique_ptr<Stmt> stmt) {
        statements.push_back(std::move(stmt));
    }
};

} // namespace semantic

namespace scheduled {

enum class LoopKind { Dense, Sparse, Block };
enum class LoopHeaderKind { DenseFor, SparseIterator, SparseMerge, Block };

struct SparseIteratorEmission {
    std::string pointerVar;
    std::string beginExpr;
    std::string endExpr;
    std::string indexExpr;
};

struct MergeTermEmission {
    std::string tensorName;
    std::string pointerVar;
    std::string endVar;
    std::string beginExpr;
    std::string endExpr;
    std::string candidateExpr;
    std::string boundIndexVar;
    std::string matchExpr;
    std::string advanceOnMatchStmt;
    std::string advanceIfLessThanMaxStmt;
};

struct MergeEmission {
    ir::MergeStrategy strategy = ir::MergeStrategy::None;
    std::vector<MergeTermEmission> terms;
};

struct BlockEmission {
    std::string blockVar;
    int blockSize = 0;
    std::string tripCountExpr;
    std::string startVar;
    std::string endVar;
    std::string innerIndexName;
    std::string innerLowerExpr;
    std::string innerUpperExpr;
};

struct Loop {
    std::string indexName;
    int lower = 0;
    int upper = 0;
    std::string runtimeBound;  // symbolic upper bound string, e.g. "A->rows", "N_j"
    LoopKind kind = LoopKind::Dense;
    LoopHeaderKind headerKind = LoopHeaderKind::DenseFor;
    std::string lowerExpr;
    std::string upperExpr;
    std::string bindingVarName;
    std::string bindingExpr;
    SparseIteratorEmission iterator;
    MergeEmission merge;
    BlockEmission block;
    std::vector<std::unique_ptr<Loop>> children;
    std::vector<std::unique_ptr<ir::IRStmt>> preStmts;
    std::vector<std::unique_ptr<ir::IRStmt>> postStmts;
    bool isExternallyBound = false;

    std::unique_ptr<Loop> clone() const;
};

struct Stmt {
    virtual ~Stmt() = default;
    virtual std::unique_ptr<Stmt> clone() const = 0;
};

struct Declaration : Stmt {
    ir::Tensor tensor;

    Declaration() = default;
    explicit Declaration(const ir::Tensor& t) : tensor(t) {}

    std::unique_ptr<Stmt> clone() const override;
};

struct Call : Stmt {
    std::string functionName;
    std::vector<std::unique_ptr<Expr>> args;

    Call() = default;
    explicit Call(const std::string& name) : functionName(name) {}

    std::unique_ptr<Stmt> clone() const override;
};

struct Compute : Stmt {
    TensorUse lhs;
    ir::Tensor output;
    std::vector<ir::Tensor> inputs;
    std::unique_ptr<Expr> rhs;
    ir::ExpressionInfo exprInfo;
    ir::OutputStrategy outputStrategy = ir::OutputStrategy::DenseArray;
    OutputPatternKind outputPattern = OutputPatternKind::None;
    std::vector<std::string> patternSources;
    std::vector<std::unique_ptr<ir::IRStmt>> prologueStmts;
    std::vector<std::unique_ptr<ir::IRStmt>> epilogueStmts;
    ir::LoopOptimizations optimizations;
    std::unique_ptr<Loop> rootLoop;
    bool fullyLowered = false;

    std::unique_ptr<Stmt> clone() const override;
};

struct Region : Stmt {
    std::vector<std::string> tensors;
    std::vector<std::string> indices;
    std::vector<std::string> runtimeBounds;
    std::vector<std::unique_ptr<Stmt>> body;

    Region() = default;
    Region(const std::vector<std::string>& tensorList,
           const std::vector<std::string>& indexList)
        : tensors(tensorList), indices(indexList) {}

    void addStatement(std::unique_ptr<Stmt> stmt) {
        body.push_back(std::move(stmt));
    }

    std::unique_ptr<Stmt> clone() const override;
};

struct Program {
    std::vector<std::unique_ptr<Stmt>> statements;

    void addStatement(std::unique_ptr<Stmt> stmt) {
        statements.push_back(std::move(stmt));
    }
};

} // namespace scheduled

std::unique_ptr<semantic::Program> lowerToSemanticProgram(
    const SparseTensorCompiler::Program& ast);

std::unique_ptr<semantic::Compute> lowerComputationToSemantic(
    const SparseTensorCompiler::Computation& computation,
    const std::unordered_map<std::string, ir::Tensor>& declarations);

std::unique_ptr<scheduled::Program> scheduleProgram(
    const semantic::Program& prog);

std::unique_ptr<scheduled::Compute> scheduleCompute(
    const semantic::Compute& compute);

std::unique_ptr<scheduled::Compute> lowerFirstComputationToScheduled(
    const SparseTensorCompiler::Program& ast);

std::unique_ptr<scheduled::Compute> lowerFirstComputationToScheduledOptimized(
    const SparseTensorCompiler::Program& ast,
    const opt::OptConfig& cfg);

std::string renderScheduledCompute(const scheduled::Compute& compute);

} // namespace sparseir
