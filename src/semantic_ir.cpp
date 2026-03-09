#include "semantic_ir.h"
#include "scheduled_optimizations.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <sstream>
#include <unordered_set>

namespace sparseir {

namespace {

scheduled::Compute* findFirstScheduledCompute(
    const std::vector<std::unique_ptr<scheduled::Stmt>>& statements) {
    for (const auto& stmt : statements) {
        if (auto* compute = dynamic_cast<scheduled::Compute*>(stmt.get())) {
            return compute;
        }
        if (auto* region = dynamic_cast<scheduled::Region*>(stmt.get())) {
            if (auto* nested = findFirstScheduledCompute(region->body)) {
                return nested;
            }
        }
    }
    return nullptr;
}

struct AccessInfo {
    std::string tensorName;
    std::vector<std::string> indices;
};

ir::Format stringToFormat(const std::string& typeStr) {
    if (typeStr == "CSR") return ir::Format::CSR;
    if (typeStr == "CSC") return ir::Format::CSC;
    return ir::Format::Dense;
}

std::vector<int> parseShape(const std::vector<std::string>& shapeStrs) {
    std::vector<int> dims;
    dims.reserve(shapeStrs.size());
    for (const auto& dim : shapeStrs) {
        try {
            dims.push_back(std::stoi(dim));
        } catch (...) {
            dims.push_back(0);
        }
    }
    return dims;
}

bool isSparseAccessPosition(ir::Format format, int position) {
    if (format == ir::Format::CSR) {
        return position == 1;
    }
    if (format == ir::Format::CSC) {
        return position == 0;
    }
    return false;
}

double parseNumberValue(const std::string& raw) {
    try {
        return std::stod(raw);
    } catch (...) {
        return 0.0;
    }
}

std::unique_ptr<Expr> lowerSemanticExpr(const SparseTensorCompiler::Expression& expr) {
    using namespace SparseTensorCompiler;

    if (auto* ta = dynamic_cast<const TensorAccess*>(&expr)) {
        return std::make_unique<TensorRead>(ta->tensorName, ta->indices);
    }
    if (auto* num = dynamic_cast<const Number*>(&expr)) {
        return std::make_unique<Constant>(parseNumberValue(num->value));
    }
    if (auto* id = dynamic_cast<const Identifier*>(&expr)) {
        return std::make_unique<ScalarRef>(id->name);
    }
    if (auto* bin = dynamic_cast<const BinaryOp*>(&expr)) {
        auto op = (bin->op == BinaryOp::ADD) ? ExprBinaryOp::Add : ExprBinaryOp::Mul;
        return std::make_unique<BinaryExpr>(
            op, lowerSemanticExpr(*bin->left), lowerSemanticExpr(*bin->right));
    }
    if (auto* call = dynamic_cast<const FunctionCall*>(&expr)) {
        auto lowered = std::make_unique<CallExpr>(call->functionName);
        for (const auto& arg : call->arguments) {
            lowered->args.push_back(lowerSemanticExpr(*arg));
        }
        return lowered;
    }
    return std::make_unique<ScalarRef>("0");
}

std::unique_ptr<ir::IRExpr> lowerToLegacyExpr(
    const Expr& expr,
    const std::unordered_map<std::string, std::string>& sparsePointerMap) {
    if (auto* read = dynamic_cast<const TensorRead*>(&expr)) {
        auto lowered = std::make_unique<ir::IRTensorAccess>(read->tensorName, read->indices);
        auto it = sparsePointerMap.find(read->tensorName);
        if (it != sparsePointerMap.end()) {
            lowered->isSparseVals = true;
            lowered->pointerVar = it->second;
            lowered->indices.clear();
        }
        return lowered;
    }
    if (auto* constant = dynamic_cast<const Constant*>(&expr)) {
        return std::make_unique<ir::IRConstant>(constant->value);
    }
    if (auto* scalar = dynamic_cast<const ScalarRef*>(&expr)) {
        return std::make_unique<ir::IRScalarVar>(scalar->name);
    }
    if (auto* binary = dynamic_cast<const BinaryExpr*>(&expr)) {
        auto op = (binary->op == ExprBinaryOp::Add) ? ir::IRBinaryOp::ADD : ir::IRBinaryOp::MUL;
        return std::make_unique<ir::IRBinaryOp>(
            op,
            lowerToLegacyExpr(*binary->lhs, sparsePointerMap),
            lowerToLegacyExpr(*binary->rhs, sparsePointerMap));
    }
    if (auto* call = dynamic_cast<const CallExpr*>(&expr)) {
        auto lowered = std::make_unique<ir::IRFuncCall>(call->functionName);
        for (const auto& arg : call->args) {
            lowered->args.push_back(lowerToLegacyExpr(*arg, sparsePointerMap));
        }
        return lowered;
    }
    return std::make_unique<ir::IRScalarVar>("0");
}

std::unordered_map<std::string, ir::Tensor> buildTensorMap(
    const ir::Tensor& output,
    const std::vector<ir::Tensor>& inputs) {
    std::unordered_map<std::string, ir::Tensor> tensors;
    tensors[output.name] = output;
    for (const auto& input : inputs) {
        tensors[input.name] = input;
    }
    return tensors;
}

std::unique_ptr<ir::IRExpr> lowerToOutputFillExpr(
    const Expr& expr,
    const std::unordered_map<std::string, ir::Tensor>& tensors,
    const std::unordered_map<std::string, std::string>& sparsePointerMap = {}) {
    if (auto* read = dynamic_cast<const TensorRead*>(&expr)) {
        auto lowered = std::make_unique<ir::IRTensorAccess>(read->tensorName, read->indices);
        auto ptrIt = sparsePointerMap.find(read->tensorName);
        if (ptrIt != sparsePointerMap.end()) {
            lowered->isSparseVals = true;
            lowered->pointerVar = ptrIt->second;
            lowered->indices.clear();
            return lowered;
        }

        auto tensorIt = tensors.find(read->tensorName);
        if (tensorIt != tensors.end() &&
            (tensorIt->second.format == ir::Format::CSR ||
             tensorIt->second.format == ir::Format::CSC) &&
            read->indices.size() == 2) {
            lowered->useRandomAccess = true;
            lowered->randomAccessFunc = (tensorIt->second.format == ir::Format::CSR)
                ? "sp_csr_get"
                : "sp_csc_get";
        }
        return lowered;
    }
    if (auto* constant = dynamic_cast<const Constant*>(&expr)) {
        return std::make_unique<ir::IRConstant>(constant->value);
    }
    if (auto* scalar = dynamic_cast<const ScalarRef*>(&expr)) {
        return std::make_unique<ir::IRScalarVar>(scalar->name);
    }
    if (auto* binary = dynamic_cast<const BinaryExpr*>(&expr)) {
        auto op = (binary->op == ExprBinaryOp::Add) ? ir::IRBinaryOp::ADD : ir::IRBinaryOp::MUL;
        return std::make_unique<ir::IRBinaryOp>(
            op,
            lowerToOutputFillExpr(*binary->lhs, tensors, sparsePointerMap),
            lowerToOutputFillExpr(*binary->rhs, tensors, sparsePointerMap));
    }
    if (auto* call = dynamic_cast<const CallExpr*>(&expr)) {
        auto lowered = std::make_unique<ir::IRFuncCall>(call->functionName);
        for (const auto& arg : call->args) {
            lowered->args.push_back(lowerToOutputFillExpr(*arg, tensors, sparsePointerMap));
        }
        return lowered;
    }
    return std::make_unique<ir::IRScalarVar>("0");
}

std::unique_ptr<Expr> stripSparseFactor(const Expr& expr, const std::string& sparseTensorName) {
    if (auto* read = dynamic_cast<const TensorRead*>(&expr)) {
        if (read->tensorName == sparseTensorName) {
            return nullptr;
        }
        return read->clone();
    }
    if (auto* constant = dynamic_cast<const Constant*>(&expr)) {
        return constant->clone();
    }
    if (auto* scalar = dynamic_cast<const ScalarRef*>(&expr)) {
        return scalar->clone();
    }
    if (auto* call = dynamic_cast<const CallExpr*>(&expr)) {
        auto lowered = std::make_unique<CallExpr>(call->functionName);
        for (const auto& arg : call->args) {
            auto strippedArg = stripSparseFactor(*arg, sparseTensorName);
            if (!strippedArg) {
                return nullptr;
            }
            lowered->args.push_back(std::move(strippedArg));
        }
        return lowered;
    }
    if (auto* binary = dynamic_cast<const BinaryExpr*>(&expr)) {
        auto lhs = stripSparseFactor(*binary->lhs, sparseTensorName);
        auto rhs = stripSparseFactor(*binary->rhs, sparseTensorName);
        if (binary->op == ExprBinaryOp::Mul) {
            if (!lhs) return rhs;
            if (!rhs) return lhs;
            return std::make_unique<BinaryExpr>(ExprBinaryOp::Mul, std::move(lhs), std::move(rhs));
        }
        if (!lhs || !rhs) {
            return nullptr;
        }
        return std::make_unique<BinaryExpr>(ExprBinaryOp::Add, std::move(lhs), std::move(rhs));
    }
    return nullptr;
}

void collectTensorAccesses(const SparseTensorCompiler::Expression& expr,
                          std::vector<AccessInfo>& accesses) {
    using namespace SparseTensorCompiler;

    if (auto* ta = dynamic_cast<const TensorAccess*>(&expr)) {
        accesses.push_back({ta->tensorName, ta->indices});
        return;
    }
    if (auto* bin = dynamic_cast<const BinaryOp*>(&expr)) {
        collectTensorAccesses(*bin->left, accesses);
        collectTensorAccesses(*bin->right, accesses);
        return;
    }
    if (auto* call = dynamic_cast<const FunctionCall*>(&expr)) {
        for (const auto& arg : call->arguments) {
            collectTensorAccesses(*arg, accesses);
        }
    }
}

void appendUnique(std::vector<std::string>& values, const std::string& value) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

std::vector<std::string> collectRhsIndices(const std::vector<AccessInfo>& accesses) {
    std::vector<std::string> ordered;
    for (const auto& access : accesses) {
        for (const auto& index : access.indices) {
            appendUnique(ordered, index);
        }
    }
    return ordered;
}

int inferDimension(const std::string& indexName,
                   const std::unordered_map<std::string, ir::Tensor>& declarations,
                   const ir::Tensor& output,
                   const std::vector<AccessInfo>& accesses) {
    for (const auto& access : accesses) {
        auto tensorIt = declarations.find(access.tensorName);
        if (tensorIt == declarations.end()) {
            continue;
        }
        if (tensorIt->second.format != ir::Format::CSR &&
            tensorIt->second.format != ir::Format::CSC) {
            continue;
        }
        for (size_t i = 0; i < access.indices.size(); i++) {
            if (access.indices[i] == indexName && i < tensorIt->second.dims.size()) {
                return tensorIt->second.dims[i];
            }
        }
    }

    for (size_t i = 0; i < output.indices.size(); i++) {
        if (output.indices[i] == indexName && i < output.dims.size()) {
            return output.dims[i];
        }
    }

    for (const auto& access : accesses) {
        auto tensorIt = declarations.find(access.tensorName);
        if (tensorIt == declarations.end()) {
            continue;
        }
        for (size_t i = 0; i < access.indices.size(); i++) {
            if (access.indices[i] == indexName && i < tensorIt->second.dims.size()) {
                return tensorIt->second.dims[i];
            }
        }
    }

    return 0;
}

std::unordered_map<std::string, ir::Tensor> collectDeclarations(
    const std::vector<std::unique_ptr<SparseTensorCompiler::Statement>>& statements) {
    using namespace SparseTensorCompiler;

    std::unordered_map<std::string, ir::Tensor> declarations;
    std::function<void(const std::vector<std::unique_ptr<Statement>>&)> visitStatements;
    visitStatements = [&](const std::vector<std::unique_ptr<Statement>>& stmts) {
        for (const auto& stmt : stmts) {
            if (auto* decl = dynamic_cast<const Declaration*>(stmt.get())) {
                ir::Tensor tensor;
                tensor.name = decl->tensorName;
                tensor.format = stringToFormat(decl->tensorType);
                tensor.dims = parseShape(decl->shape);
                declarations[tensor.name] = tensor;
            } else if (auto* forStmt = dynamic_cast<const ForStatement*>(stmt.get())) {
                visitStatements(forStmt->body);
            }
        }
    };
    visitStatements(statements);
    return declarations;
}

std::string computeRuntimeBound(int position,
                                const std::vector<std::string>& tensorNames,
                                const std::unordered_map<std::string, ir::Tensor>& declarations) {
    for (const auto& name : tensorNames) {
        auto it = declarations.find(name);
        if (it == declarations.end()) {
            continue;
        }
        const auto& tensor = it->second;
        if (position < static_cast<int>(tensor.dims.size()) && tensor.dims[position] > 0) {
            return std::to_string(tensor.dims[position]);
        }
        if (tensor.format == ir::Format::CSR || tensor.format == ir::Format::CSC) {
            if (position == 0) return name + "->rows";
            if (position == 1) return name + "->cols";
        }
    }
    return "0 /* unknown bound */";
}

ir::Tensor buildTensorUse(const std::string& name,
                         const std::vector<std::string>& indices,
                         const std::unordered_map<std::string, ir::Tensor>& declarations) {
    ir::Tensor tensor;
    tensor.name = name;
    tensor.indices = indices;
    auto it = declarations.find(name);
    if (it != declarations.end()) {
        tensor.format = it->second.format;
        tensor.dims = it->second.dims;
    }
    return tensor;
}

std::unique_ptr<semantic::Stmt> lowerSemanticStmt(
    const SparseTensorCompiler::Statement& stmt,
    const std::unordered_map<std::string, ir::Tensor>& declarations);

std::unique_ptr<semantic::Region> lowerSemanticRegion(
    const SparseTensorCompiler::ForStatement& forStmt,
    const std::unordered_map<std::string, ir::Tensor>& declarations) {
    auto region = std::make_unique<semantic::Region>(forStmt.tensors, forStmt.indices);
    for (int i = 0; i < static_cast<int>(forStmt.indices.size()); i++) {
        region->runtimeBounds.push_back(computeRuntimeBound(i, forStmt.tensors, declarations));
    }
    for (const auto& bodyStmt : forStmt.body) {
        region->addStatement(lowerSemanticStmt(*bodyStmt, declarations));
    }
    return region;
}

std::unique_ptr<semantic::Call> lowerSemanticCall(
    const SparseTensorCompiler::CallStatement& call) {
    auto lowered = std::make_unique<semantic::Call>(call.functionName);
    for (const auto& arg : call.arguments) {
        lowered->args.push_back(lowerSemanticExpr(*arg));
    }
    return lowered;
}

std::unique_ptr<semantic::Stmt> lowerSemanticStmt(
    const SparseTensorCompiler::Statement& stmt,
    const std::unordered_map<std::string, ir::Tensor>& declarations) {
    using namespace SparseTensorCompiler;

    if (auto* decl = dynamic_cast<const Declaration*>(&stmt)) {
        return std::make_unique<semantic::Declaration>(
            buildTensorUse(decl->tensorName, {}, declarations));
    }
    if (auto* comp = dynamic_cast<const Computation*>(&stmt)) {
        return lowerComputationToSemantic(*comp, declarations);
    }
    if (auto* call = dynamic_cast<const CallStatement*>(&stmt)) {
        return lowerSemanticCall(*call);
    }
    if (auto* forStmt = dynamic_cast<const ForStatement*>(&stmt)) {
        return lowerSemanticRegion(*forStmt, declarations);
    }
    return std::make_unique<semantic::Call>("noop");
}

std::unique_ptr<scheduled::Stmt> scheduleStmt(
    const semantic::Stmt& stmt,
    const std::vector<std::string>& enclosingIndices);

const IteratorNode* findIteratorNode(const semantic::Compute& compute,
                                     const std::string& indexName) {
    for (const auto& node : compute.iteratorGraph.iterators) {
        if (node.indexName == indexName) {
            return &node;
        }
    }
    return nullptr;
}

const ir::Tensor* findSparseInput(const semantic::Compute& compute) {
    auto it = std::find_if(
        compute.inputs.begin(),
        compute.inputs.end(),
        [](const ir::Tensor& tensor) {
            return tensor.format == ir::Format::CSR || tensor.format == ir::Format::CSC;
        });
    return (it != compute.inputs.end()) ? &(*it) : nullptr;
}

std::unique_ptr<scheduled::Loop> makeDenseLoop(const std::string& indexName, int upper) {
    auto loop = std::make_unique<scheduled::Loop>();
    loop->indexName = indexName;
    loop->kind = scheduled::LoopKind::Dense;
    loop->upper = upper;
    return loop;
}

std::unique_ptr<scheduled::Loop> makeSparseLoop(const std::string& indexName,
                                                int upper) {
    auto loop = std::make_unique<scheduled::Loop>();
    loop->indexName = indexName;
    loop->kind = scheduled::LoopKind::Sparse;
    loop->upper = upper;
    return loop;
}

ir::Format findScheduledTensorFormat(const semantic::Compute& compute,
                                     const std::string& tensorName) {
    if (compute.output.name == tensorName) {
        return compute.output.format;
    }
    auto it = std::find_if(
        compute.inputs.begin(),
        compute.inputs.end(),
        [&](const ir::Tensor& tensor) { return tensor.name == tensorName; });
    return (it != compute.inputs.end()) ? it->format : ir::Format::CSR;
}

void configureDenseLoopEmission(scheduled::Loop& loop) {
    loop.headerKind = scheduled::LoopHeaderKind::DenseFor;
    loop.lowerExpr = std::to_string(loop.lower);
    loop.upperExpr = loop.runtimeBound.empty()
        ? std::to_string(loop.upper)
        : loop.runtimeBound;
    loop.bindingVarName.clear();
    loop.bindingExpr.clear();
    loop.iterator = {};
    loop.merge = {};
    loop.block = {};
}

void configureSparseIteratorEmission(scheduled::Loop& loop,
                                     const std::string& tensorName,
                                     const std::string& parentIndex,
                                     ir::Format format) {
    const std::string ptrVar = "p" + tensorName;
    const std::string ptrArray = (format == ir::Format::CSC) ? "col_ptr" : "row_ptr";
    const std::string indexArray = (format == ir::Format::CSC) ? "row_idx" : "col_idx";

    loop.headerKind = scheduled::LoopHeaderKind::SparseIterator;
    loop.lowerExpr.clear();
    loop.upperExpr.clear();
    loop.bindingVarName = loop.indexName;
    loop.bindingExpr = tensorName + "->" + indexArray + "[" + ptrVar + "]";
    loop.iterator.pointerVar = ptrVar;
    loop.iterator.beginExpr = tensorName + "->" + ptrArray + "[" + parentIndex + "]";
    loop.iterator.endExpr = tensorName + "->" + ptrArray + "[" + parentIndex + " + 1]";
    loop.iterator.indexExpr = loop.bindingExpr;
    loop.merge = {};
    loop.block = {};
}

void configureSparseMergeEmission(scheduled::Loop& loop,
                                  ir::MergeStrategy strategy,
                                  const std::vector<std::string>& mergedTensors,
                                  const std::string& parentIndex,
                                  ir::Format format) {
    const std::string ptrArray = (format == ir::Format::CSC) ? "col_ptr" : "row_ptr";
    const std::string indexArray = (format == ir::Format::CSC) ? "row_idx" : "col_idx";

    loop.headerKind = scheduled::LoopHeaderKind::SparseMerge;
    loop.lowerExpr.clear();
    loop.upperExpr.clear();
    loop.bindingVarName.clear();
    loop.bindingExpr.clear();
    loop.iterator = {};
    loop.block = {};
    loop.merge.strategy = strategy;
    loop.merge.terms.clear();

    for (const auto& tensorName : mergedTensors) {
        scheduled::MergeTermEmission term;
        term.tensorName = tensorName;
        term.pointerVar = "p" + tensorName;
        term.endVar = "end" + tensorName;
        term.beginExpr = tensorName + "->" + ptrArray + "[" + parentIndex + "]";
        term.endExpr = tensorName + "->" + ptrArray + "[" + parentIndex + " + 1]";
        term.candidateExpr = tensorName + "->" + indexArray + "[" + term.pointerVar + "]";
        term.boundIndexVar = loop.indexName + tensorName;
        term.matchExpr = term.candidateExpr + " == " + loop.indexName;
        term.advanceOnMatchStmt = term.pointerVar + "++;";
        term.advanceIfLessThanMaxStmt =
            "if (" + term.boundIndexVar + " < max_idx) " + term.pointerVar + "++;";
        loop.merge.terms.push_back(std::move(term));
    }
}

const ir::Tensor* findInputByName(const semantic::Compute& compute,
                                  const std::string& tensorName) {
    auto it = std::find_if(
        compute.inputs.begin(),
        compute.inputs.end(),
        [&](const ir::Tensor& tensor) { return tensor.name == tensorName; });
    return (it != compute.inputs.end()) ? &(*it) : nullptr;
}

std::unique_ptr<Expr> multiplyExprs(std::unique_ptr<Expr> lhs,
                                    std::unique_ptr<Expr> rhs) {
    if (!lhs) return rhs;
    if (!rhs) return lhs;
    return std::make_unique<BinaryExpr>(ExprBinaryOp::Mul, std::move(lhs), std::move(rhs));
}

bool exprDependsOnIndex(const Expr& expr, const std::string& indexName) {
    if (auto* read = dynamic_cast<const TensorRead*>(&expr)) {
        return std::find(read->indices.begin(), read->indices.end(), indexName) !=
               read->indices.end();
    }
    if (dynamic_cast<const Constant*>(&expr) || dynamic_cast<const ScalarRef*>(&expr)) {
        return false;
    }
    if (auto* call = dynamic_cast<const CallExpr*>(&expr)) {
        return std::any_of(
            call->args.begin(), call->args.end(),
            [&](const std::unique_ptr<Expr>& arg) { return exprDependsOnIndex(*arg, indexName); });
    }
    if (auto* binary = dynamic_cast<const BinaryExpr*>(&expr)) {
        return exprDependsOnIndex(*binary->lhs, indexName) ||
               exprDependsOnIndex(*binary->rhs, indexName);
    }
    return false;
}

std::pair<std::unique_ptr<Expr>, std::unique_ptr<Expr>> splitExprByIndexDependency(
    const Expr& expr,
    const std::string& indexName) {
    if (auto* binary = dynamic_cast<const BinaryExpr*>(&expr)) {
        if (binary->op == ExprBinaryOp::Mul) {
            auto [lhsDependent, lhsInvariant] =
                splitExprByIndexDependency(*binary->lhs, indexName);
            auto [rhsDependent, rhsInvariant] =
                splitExprByIndexDependency(*binary->rhs, indexName);
            return {
                multiplyExprs(std::move(lhsDependent), std::move(rhsDependent)),
                multiplyExprs(std::move(lhsInvariant), std::move(rhsInvariant))
            };
        }
    }

    if (exprDependsOnIndex(expr, indexName)) {
        return {expr.clone(), nullptr};
    }
    return {nullptr, expr.clone()};
}

bool isSparseFormat(ir::Format f) {
    return f == ir::Format::CSR || f == ir::Format::CSC;
}

// Two sparse inputs with a reduction (e.g. SpGEMM: C[i,j] = A[i,k] * B[k,j]).
bool hasDynamicSparseOutput(const semantic::Compute& c) {
    return isSparseFormat(c.output.format)
        && c.exprInfo.numSparseInputs == 2
        && c.reductionIndices.size() == 1
        && c.freeIndices.size() == 2;
}

// Two sparse inputs, no reduction (pointwise: SpAdd or SpElMul).
bool isPointwiseTwoSparse(const semantic::Compute& c) {
    return c.exprInfo.numSparseInputs == 2
        && c.reductionIndices.empty()
        && c.freeIndices.size() == 2;
}

// One sparse input sampling a dense contraction (e.g. SDDMM).
bool isSampledDenseContraction(const semantic::Compute& c) {
    return c.exprInfo.numSparseInputs == 1
        && c.exprInfo.numDenseInputs == 2
        && c.reductionIndices.size() == 1
        && c.freeIndices.size() == 2;
}

bool hasFixedPatternOutput(const semantic::Compute& c) {
    if (!isSparseFormat(c.output.format)) return false;
    return isPointwiseTwoSparse(c) || isSampledDenseContraction(c);
}

ir::OutputStrategy deriveOutputStrategy(const semantic::Compute& compute) {
    if (!isSparseFormat(compute.output.format)) return ir::OutputStrategy::DenseArray;
    if (hasDynamicSparseOutput(compute))        return ir::OutputStrategy::HashPerRow;
    if (hasFixedPatternOutput(compute))          return ir::OutputStrategy::SparseFixedPattern;
    return ir::OutputStrategy::DenseArray;
}

sparseir::OutputPatternKind deriveOutputPattern(const semantic::Compute& compute) {
    if (compute.outputStrategy == ir::OutputStrategy::HashPerRow) {
        return sparseir::OutputPatternKind::DynamicRowAccumulator;
    }
    if (compute.outputStrategy != ir::OutputStrategy::SparseFixedPattern) {
        return sparseir::OutputPatternKind::None;
    }
    if (isPointwiseTwoSparse(compute)) {
        return (compute.exprInfo.rootOp == ir::RootOpKind::ADD)
            ? sparseir::OutputPatternKind::Union
            : sparseir::OutputPatternKind::Intersection;
    }
    if (isSampledDenseContraction(compute)) {
        return sparseir::OutputPatternKind::Sampled;
    }
    return sparseir::OutputPatternKind::None;
}

std::vector<std::string> derivePatternSources(const semantic::Compute& compute) {
    std::vector<std::string> sources;
    for (const auto& input : compute.inputs) {
        if (input.format == ir::Format::CSR || input.format == ir::Format::CSC) {
            sources.push_back(input.name);
        }
    }
    return sources;
}

enum class AccumulationStrategy { None, ScalarAccumulator, WorkspaceAccumulator };

struct LoopLoweringSpec {
    std::string indexName;
    int upper = 0;
    bool isReduction = false;
    scheduled::LoopKind kind = scheduled::LoopKind::Dense;
    ir::MergeStrategy mergeStrategy = ir::MergeStrategy::None;
    std::vector<std::string> mergedTensors;
    std::string driverTensor;
    std::string parentIndex;
    AccumulationStrategy accumStrategy = AccumulationStrategy::None;
    // Workspace ownership: which iterator owns workspace setup/teardown.
    bool ownsWorkspaceSetup = false;
    // Structured workspace sizing: which tensor + which axis provides the dimension.
    std::string workspaceSizeTensor;     // e.g. "A" or "B"
    std::string workspaceSizeAxis;       // "rows" or "cols"
};

const ir::Tensor* findTensorByName(const semantic::Compute& compute,
                                   const std::string& tensorName) {
    if (compute.output.name == tensorName) {
        return &compute.output;
    }
    return findInputByName(compute, tensorName);
}

std::vector<IteratorSource> collectEffectiveSources(const semantic::Compute& compute,
                                                    const IteratorNode& node) {
    std::vector<IteratorSource> sources = node.sources;
    if (isSparseFormat(compute.output.format)) {
        for (size_t position = 0; position < compute.output.indices.size(); position++) {
            if (compute.output.indices[position] != node.indexName) {
                continue;
            }
            IteratorSource outputSource;
            outputSource.tensorName = compute.output.name;
            outputSource.format = compute.output.format;
            outputSource.accessPosition = static_cast<int>(position);
            outputSource.sparse = isSparseAccessPosition(
                compute.output.format, static_cast<int>(position));
            sources.push_back(outputSource);
        }
    }
    return sources;
}

std::string getParentIndexForSource(const semantic::Compute& compute,
                                    const IteratorSource& source) {
    const ir::Tensor* tensor = findTensorByName(compute, source.tensorName);
    if (!tensor) {
        return "";
    }
    if (source.sparse) {
        if (tensor->format == ir::Format::CSR && source.accessPosition == 1 &&
            !tensor->indices.empty()) {
            return tensor->indices[0];
        }
        if (tensor->format == ir::Format::CSC && source.accessPosition == 0 &&
            tensor->indices.size() > 1) {
            return tensor->indices[1];
        }
    }
    if (source.accessPosition <= 0) {
        return "";
    }
    size_t parentPos = static_cast<size_t>(source.accessPosition - 1);
    if (parentPos < tensor->indices.size()) {
        return tensor->indices[parentPos];
    }
    return "";
}

bool isPointwiseUnion(const semantic::Compute& compute) {
    return compute.reductionIndices.empty() &&
           compute.exprInfo.rootOp == ir::RootOpKind::ADD &&
           compute.exprInfo.numSparseInputs >= 2;
}

bool hasSparseDependentIterator(const semantic::Compute& compute,
                                const std::string& indexName) {
    for (const auto& node : compute.iteratorGraph.iterators) {
        auto sources = collectEffectiveSources(compute, node);
        for (const auto& source : sources) {
            if (!source.sparse) {
                continue;
            }
            if (getParentIndexForSource(compute, source) == indexName) {
                return true;
            }
        }
    }
    return false;
}

bool iteratorHasEffectiveSparseSource(const semantic::Compute& compute,
                                      const std::string& indexName) {
    const IteratorNode* node = findIteratorNode(compute, indexName);
    if (!node) {
        return false;
    }
    auto sources = collectEffectiveSources(compute, *node);
    return std::any_of(
        sources.begin(), sources.end(),
        [](const IteratorSource& source) { return source.sparse; });
}

std::string findPreferredDenseParentIndex(const semantic::Compute& compute,
                                          const IteratorNode& node,
                                          const std::vector<IteratorSource>& sources) {
    if (hasSparseDependentIterator(compute, node.indexName)) {
        return "";
    }

    std::string fallbackParent;
    bool hasRootSource = false;
    for (const auto& source : sources) {
        std::string parentIndex = getParentIndexForSource(compute, source);
        if (parentIndex.empty()) {
            hasRootSource = true;
            continue;
        }
        if (fallbackParent.empty()) {
            fallbackParent = parentIndex;
        }
    }

    if (!fallbackParent.empty()) {
        for (const auto& candidate : compute.iteratorGraph.iterators) {
            if (candidate.indexName == node.indexName ||
                !iteratorHasEffectiveSparseSource(compute, candidate.indexName)) {
                continue;
            }
            auto candidateSources = collectEffectiveSources(compute, candidate);
            for (const auto& candidateSource : candidateSources) {
                if (!candidateSource.sparse) {
                    continue;
                }
                if (getParentIndexForSource(compute, candidateSource) == fallbackParent) {
                    return candidate.indexName;
                }
            }
        }
    }

    for (const auto& source : sources) {
        const ir::Tensor* tensor = findTensorByName(compute, source.tensorName);
        if (!tensor) {
            continue;
        }
        for (const auto& candidateIndex : tensor->indices) {
            if (candidateIndex == node.indexName) {
                continue;
            }
            if (iteratorHasEffectiveSparseSource(compute, candidateIndex)) {
                return candidateIndex;
            }
        }
    }

    if (hasRootSource) {
        return "";
    }
    return fallbackParent;
}

bool nodeOwnsWorkspaceAccumulator(const semantic::Compute& compute,
                                  const std::string& indexName,
                                  bool isReduction) {
    if (compute.outputStrategy != ir::OutputStrategy::HashPerRow ||
        compute.reductionIndices.empty() || isReduction ||
        compute.freeIndices.size() < 2) {
        return false;
    }
    const std::string& outerFreeIdx =
        (compute.output.format == ir::Format::CSC)
            ? compute.freeIndices[1]
            : compute.freeIndices[0];
    return indexName == outerFreeIdx;
}

void populateWorkspaceOwnership(const semantic::Compute& compute,
                                LoopLoweringSpec& spec) {
    if (!nodeOwnsWorkspaceAccumulator(compute, spec.indexName, spec.isReduction)) {
        return;
    }
    spec.accumStrategy = AccumulationStrategy::WorkspaceAccumulator;
    spec.ownsWorkspaceSetup = true;
    if (compute.output.format == ir::Format::CSC) {
        spec.workspaceSizeTensor = compute.inputs[0].name;
        spec.workspaceSizeAxis = "rows";
    } else {
        spec.workspaceSizeTensor = compute.inputs[1].name;
        spec.workspaceSizeAxis = "cols";
    }
}

LoopLoweringSpec deriveLoopLoweringSpec(const semantic::Compute& compute,
                                        const IteratorNode& node) {
    LoopLoweringSpec spec;
    spec.indexName = node.indexName;
    spec.upper = node.upper;
    spec.isReduction = node.isReduction;

    auto sources = collectEffectiveSources(compute, node);
    std::vector<IteratorSource> sparseSources;
    sparseSources.reserve(sources.size());
    for (const auto& source : sources) {
        if (source.sparse) {
            sparseSources.push_back(source);
        }
    }

    if (compute.outputStrategy == ir::OutputStrategy::SparseFixedPattern) {
        auto outputIt = std::find_if(
            sparseSources.begin(), sparseSources.end(),
            [&](const IteratorSource& source) { return source.tensorName == compute.output.name; });
        if (outputIt != sparseSources.end()) {
            spec.kind = scheduled::LoopKind::Sparse;
            spec.driverTensor = compute.output.name;
            spec.parentIndex = getParentIndexForSource(compute, *outputIt);
            if (!node.isReduction && isSampledDenseContraction(compute)) {
                spec.accumStrategy = AccumulationStrategy::ScalarAccumulator;
            }
            return spec;
        }
    }

    if (node.merge == MergeKind::Union || node.merge == MergeKind::Intersection) {
        spec.kind = scheduled::LoopKind::Sparse;
        spec.mergeStrategy = (node.merge == MergeKind::Union)
            ? ir::MergeStrategy::Union
            : ir::MergeStrategy::Intersection;
        std::unordered_set<std::string> seen;
        for (const auto& source : sparseSources) {
            if (source.tensorName == compute.output.name) {
                continue;
            }
            if (seen.insert(source.tensorName).second) {
                spec.mergedTensors.push_back(source.tensorName);
            }
        }
        if (!spec.mergedTensors.empty()) {
            spec.driverTensor = spec.mergedTensors.front();
            for (const auto& source : sparseSources) {
                if (source.tensorName == spec.driverTensor) {
                    spec.parentIndex = getParentIndexForSource(compute, source);
                    break;
                }
            }
            return spec;
        }
    }

    if (!sparseSources.empty()) {
        const IteratorSource* selected = nullptr;
        for (const auto& source : sparseSources) {
            if (!getParentIndexForSource(compute, source).empty()) {
                selected = &source;
                break;
            }
        }
        if (!selected) {
            selected = &sparseSources.front();
        }
        spec.kind = scheduled::LoopKind::Sparse;
        spec.driverTensor = selected->tensorName;
        spec.parentIndex = getParentIndexForSource(compute, *selected);
        if (!node.isReduction && isSampledDenseContraction(compute)) {
            spec.accumStrategy = AccumulationStrategy::ScalarAccumulator;
        }
        return spec;
    }

    spec.kind = scheduled::LoopKind::Dense;
    spec.parentIndex = findPreferredDenseParentIndex(compute, node, sources);
    populateWorkspaceOwnership(compute, spec);
    return spec;
}

// Derive the full loop-lowering spec map for a compute statement.
// Each spec is derived directly from iterator semantics and output behavior.
std::unordered_map<std::string, LoopLoweringSpec> deriveAllLoopLoweringSpecs(
    const semantic::Compute& compute,
    const std::vector<std::string>& ordered) {
    std::unordered_map<std::string, LoopLoweringSpec> specs;
    for (const auto& indexName : ordered) {
        const IteratorNode* node = findIteratorNode(compute, indexName);
        if (!node) {
            return specs;
        }
        specs[indexName] = deriveLoopLoweringSpec(compute, *node);
    }
    return specs;
}

std::vector<std::string> orderedIteratorNames(const semantic::Compute& compute) {
    std::vector<std::string> ordered = compute.freeIndices;
    ordered.insert(ordered.end(), compute.reductionIndices.begin(), compute.reductionIndices.end());
    return ordered;
}

std::vector<std::string> loopChildrenFor(
    const std::vector<std::string>& ordered,
    const std::unordered_map<std::string, LoopLoweringSpec>& specs,
    const std::string& parentIndex) {
    std::vector<std::string> children;
    for (const auto& indexName : ordered) {
        auto it = specs.find(indexName);
        if (it == specs.end()) {
            continue;
        }
        if (it->second.parentIndex == parentIndex) {
            children.push_back(indexName);
        }
    }
    return children;
}

std::unique_ptr<ir::IRExpr> makeScalarExpr(const std::string& name) {
    return std::make_unique<ir::IRScalarVar>(name);
}

std::unique_ptr<ir::IRExpr> makeIndexedExpr(
    const std::string& baseName,
    std::unique_ptr<ir::IRExpr> index) {
    auto expr = std::make_unique<ir::IRIndexedAccess>(baseName);
    expr->indices.push_back(std::move(index));
    return expr;
}

std::unique_ptr<ir::IRExpr> makeOutputPatternIndexExpr(
    const semantic::Compute& compute,
    const std::string& pointerVar) {
    const std::string baseName = (compute.output.format == ir::Format::CSC)
        ? compute.output.name + "->row_idx"
        : compute.output.name + "->col_idx";
    return makeIndexedExpr(baseName, makeScalarExpr(pointerVar));
}

std::unique_ptr<ir::IRExpr> makeOutputPatternStartExpr(
    const semantic::Compute& compute,
    const std::string& outerIndex) {
    const std::string baseName = (compute.output.format == ir::Format::CSC)
        ? compute.output.name + "->col_ptr"
        : compute.output.name + "->row_ptr";
    return makeIndexedExpr(baseName, makeScalarExpr(outerIndex));
}

std::unique_ptr<ir::IRExpr> makeOutputPatternEndExpr(
    const semantic::Compute& compute,
    const std::string& outerIndex) {
    const std::string baseName = (compute.output.format == ir::Format::CSC)
        ? compute.output.name + "->col_ptr"
        : compute.output.name + "->row_ptr";
    return makeIndexedExpr(
        baseName,
        std::make_unique<ir::IRBinaryOp>(
            ir::IRBinaryOp::ADD,
            makeScalarExpr(outerIndex),
            std::make_unique<ir::IRConstant>(1.0)));
}

// Derive the symbolic runtime upper-bound string for a dense loop over indexName.
// Moves the codegen-side bound-inference logic into the scheduler so emitDenseLoop
// can read loop.runtimeBound directly instead of pattern-matching compute shape.
std::string deriveRuntimeBound(const semantic::Compute& compute, const std::string& indexName) {
    bool sparseOutput = (compute.outputStrategy == ir::OutputStrategy::SparseFixedPattern ||
                         compute.outputStrategy == ir::OutputStrategy::HashPerRow);

    if (sparseOutput) {
        // Output outer loop → rows, inner loop → cols
        if (!compute.output.indices.empty() && indexName == compute.output.indices[0]) {
            return compute.output.name + "->rows";
        }
        if (compute.output.indices.size() > 1 && indexName == compute.output.indices[1]) {
            return compute.output.name + "->cols";
        }
        // SDDMM sampled: reduction index is "K" (explicit param name in generated signature)
        if (isSampledDenseContraction(compute)) {
            bool inOutputIndices = std::find(compute.output.indices.begin(),
                                             compute.output.indices.end(),
                                             indexName) != compute.output.indices.end();
            if (!inOutputIndices) {
                return "K";
            }
        }
    }

    // Search sparse inputs first (preferred: bound from loaded matrix dimensions)
    for (const auto& tensor : compute.inputs) {
        if (tensor.format == ir::Format::Dense) continue;
        for (size_t p = 0; p < tensor.indices.size(); p++) {
            if (tensor.indices[p] == indexName) {
                if (p == 0) return tensor.name + "->rows";
                if (p == 1) return tensor.name + "->cols";
            }
        }
    }

    // Fallback: dimension parameter variable
    return "N_" + indexName;
}

std::unique_ptr<scheduled::Loop> buildGenericLoopNest(
    const semantic::Compute& compute,
    const std::unordered_map<std::string, ir::Tensor>& tensorMap,
    const std::vector<std::string>& ordered,
    const std::unordered_map<std::string, LoopLoweringSpec>& specs,
    const std::string& indexName,
    std::unordered_map<std::string, std::string> sparsePointerMap,
    bool forceRandomAccess,
    scheduled::Compute* computeCtx = nullptr) {
    auto specIt = specs.find(indexName);
    if (specIt == specs.end()) {
        return nullptr;
    }
    const LoopLoweringSpec& spec = specIt->second;

    std::unique_ptr<scheduled::Loop> loop;
    if (spec.kind == scheduled::LoopKind::Sparse) {
        loop = makeSparseLoop(indexName, spec.upper);
        if (spec.mergeStrategy == ir::MergeStrategy::Union ||
            spec.mergeStrategy == ir::MergeStrategy::Intersection) {
            const std::string formatTensor = !spec.mergedTensors.empty()
                ? spec.mergedTensors.front()
                : spec.driverTensor;
            configureSparseMergeEmission(
                *loop,
                spec.mergeStrategy,
                spec.mergedTensors,
                spec.parentIndex,
                findScheduledTensorFormat(compute, formatTensor));
        } else {
            configureSparseIteratorEmission(
                *loop,
                spec.driverTensor,
                spec.parentIndex,
                findScheduledTensorFormat(compute, spec.driverTensor));
        }
        if (spec.mergeStrategy == ir::MergeStrategy::Intersection) {
            for (const auto& tensorName : spec.mergedTensors) {
                sparsePointerMap[tensorName] = "p" + tensorName;
            }
        } else if (spec.driverTensor != compute.output.name && !spec.driverTensor.empty()) {
            sparsePointerMap[spec.driverTensor] = "p" + spec.driverTensor;
        }
    } else {
        loop = makeDenseLoop(indexName, spec.upper);
        loop->runtimeBound = deriveRuntimeBound(compute, indexName);
        configureDenseLoopEmission(*loop);
    }

    const bool useRandomAccess =
        forceRandomAccess || spec.mergeStrategy == ir::MergeStrategy::Union ||
        compute.outputStrategy == ir::OutputStrategy::SparseFixedPattern;

    // ScalarAccumulator: factor out sparse value, accumulate reduction into scalar.
    if (spec.accumStrategy == AccumulationStrategy::ScalarAccumulator) {
        const ir::Tensor* sparseInput = findSparseInput(compute);
        // Always strip the sparse INPUT from the expression (e.g. S in SDDMM).
        const std::string sparseInputName =
            sparseInput ? sparseInput->name : spec.driverTensor;

        auto denseExpr = stripSparseFactor(*compute.rhs, sparseInputName);
        if (!denseExpr) {
            return loop;
        }

        const auto& reductionIndex = compute.reductionIndices.front();
        auto [reductionExpr, invariantExpr] =
            splitExprByIndexDependency(*denseExpr, reductionIndex);
        if (!reductionExpr) {
            return loop;
        }

        // preStmt: double sum = 0.0;
        loop->preStmts.push_back(std::make_unique<ir::IRScalarDecl>("sum", 0.0));

        // Build children (the reduction loop subtree).
        auto children = loopChildrenFor(ordered, specs, indexName);
        for (const auto& childName : children) {
            auto child = buildGenericLoopNest(
                compute, tensorMap, ordered, specs, childName, sparsePointerMap, useRandomAccess);
            if (!child) {
                return nullptr;
            }
            loop->children.push_back(std::move(child));
        }

        // Find the leaf reduction loop and attach sum accumulation.
        std::function<scheduled::Loop*(scheduled::Loop*)> findLeaf =
            [&](scheduled::Loop* l) -> scheduled::Loop* {
                if (l->children.empty()) return l;
                return findLeaf(l->children.back().get());
            };
        if (!loop->children.empty()) {
            scheduled::Loop* leaf = findLeaf(loop->children.back().get());
            leaf->postStmts.clear();
            leaf->postStmts.push_back(std::make_unique<ir::IRAssign>(
                std::make_unique<ir::IRScalarVar>("sum"),
                lowerToOutputFillExpr(*reductionExpr, tensorMap),
                true));
        }

        // postStmt: output = sparse_vals * sum [* invariant]
        // For SparseFixedPattern, the output shares the sparse input's pattern,
        // so we read sparse_input->vals via the output pointer (pC).
        const std::string pointerVar = "p" +
            ((compute.outputStrategy == ir::OutputStrategy::SparseFixedPattern)
                ? compute.output.name
                : sparseInputName);
        auto sparseVals = std::make_unique<ir::IRTensorAccess>();
        sparseVals->tensorName = sparseInputName;
        sparseVals->isSparseVals = true;
        sparseVals->pointerVar = pointerVar;
        std::unique_ptr<ir::IRExpr> rhsExpr = std::make_unique<ir::IRBinaryOp>(
            ir::IRBinaryOp::MUL,
            std::move(sparseVals),
            std::make_unique<ir::IRScalarVar>("sum"));
        if (invariantExpr) {
            rhsExpr = std::make_unique<ir::IRBinaryOp>(
                ir::IRBinaryOp::MUL,
                std::move(rhsExpr),
                lowerToOutputFillExpr(*invariantExpr, tensorMap));
        }

        if (compute.outputStrategy == ir::OutputStrategy::SparseFixedPattern) {
            auto outVals = std::make_unique<ir::IRTensorAccess>();
            outVals->tensorName = compute.output.name;
            outVals->isSparseVals = true;
            outVals->pointerVar = "p" + compute.output.name;
            loop->postStmts.push_back(std::make_unique<ir::IRAssign>(
                std::move(outVals), std::move(rhsExpr), false));
        } else {
            auto outAccess = std::make_unique<ir::IRTensorAccess>(
                compute.lhs.tensorName, compute.lhs.indices);
            loop->postStmts.push_back(std::make_unique<ir::IRAssign>(
                std::move(outAccess), std::move(rhsExpr), false));
        }

        return loop;
    }

    // WorkspaceAccumulator: hash-based accumulation for two-sparse-input contractions.
    if (spec.ownsWorkspaceSetup && computeCtx) {
        const auto& sparseA = compute.inputs[0];
        const auto& sparseB = compute.inputs[1];
        const auto& outerIndex = indexName;
        const auto& innerIndex = (compute.output.format == ir::Format::CSC)
            ? compute.freeIndices[0]
            : compute.freeIndices[1];

        // Prologue: workspace allocation (sized from spec metadata).
        const std::string& sizeTensor = spec.workspaceSizeTensor;
        const std::string& sizeAxis = spec.workspaceSizeAxis;
        const std::string sizeVar = (sizeAxis == "rows") ? "M" : "N";
        computeCtx->prologueStmts.push_back(
            std::make_unique<ir::IRVarDecl>(sizeVar, "int", sizeTensor + "->" + sizeAxis));
        computeCtx->prologueStmts.push_back(
            std::make_unique<ir::IRVarDecl>("acc", "double*",
                "(double*)calloc((size_t)" + sizeVar + ", sizeof(double))"));
        computeCtx->prologueStmts.push_back(
            std::make_unique<ir::IRVarDecl>("marked", "unsigned char*",
                "(unsigned char*)calloc((size_t)" + sizeVar + ", sizeof(unsigned char))"));
        computeCtx->prologueStmts.push_back(
            std::make_unique<ir::IRVarDecl>("touched", "int*",
                "(int*)malloc((size_t)" + sizeVar + " * sizeof(int))"));

        // Epilogue: workspace deallocation.
        computeCtx->epilogueStmts.push_back(std::make_unique<ir::IRFreeStmt>("touched"));
        computeCtx->epilogueStmts.push_back(std::make_unique<ir::IRFreeStmt>("marked"));
        computeCtx->epilogueStmts.push_back(std::make_unique<ir::IRFreeStmt>("acc"));

        // preStmt: touched_count init.
        loop->preStmts.push_back(std::make_unique<ir::IRVarDecl>("touched_count", "int", "0"));

        // Build children (the sparse reduction loops).
        auto children = loopChildrenFor(ordered, specs, indexName);
        for (const auto& childName : children) {
            auto child = buildGenericLoopNest(
                compute, tensorMap, ordered, specs, childName, sparsePointerMap, useRandomAccess);
            if (!child) {
                return nullptr;
            }
            loop->children.push_back(std::move(child));
        }

        // Find the leaf and attach hash-accumulate body.
        std::function<scheduled::Loop*(scheduled::Loop*)> findLeaf =
            [&](scheduled::Loop* l) -> scheduled::Loop* {
                if (l->children.empty()) return l;
                return findLeaf(l->children.back().get());
            };
        if (!loop->children.empty()) {
            scheduled::Loop* leaf = findLeaf(loop->children.back().get());
            leaf->postStmts.clear();
            auto markCheck = std::make_unique<ir::IRIfStmt>(
                std::make_unique<ir::IRCompareExpr>(
                    ir::IRCompareExpr::EQ,
                    makeIndexedExpr("marked", makeScalarExpr(innerIndex)),
                    std::make_unique<ir::IRConstant>(0.0)));
            markCheck->thenBody.push_back(std::make_unique<ir::IRAssign>(
                makeIndexedExpr("marked", makeScalarExpr(innerIndex)),
                std::make_unique<ir::IRConstant>(1.0),
                false));
            markCheck->thenBody.push_back(std::make_unique<ir::IRAssign>(
                makeIndexedExpr("touched", makeScalarExpr("touched_count")),
                makeScalarExpr(innerIndex),
                false));
            markCheck->thenBody.push_back(std::make_unique<ir::IRAssign>(
                makeScalarExpr("touched_count"),
                std::make_unique<ir::IRConstant>(1.0),
                true));
            leaf->postStmts.push_back(std::move(markCheck));
            auto lhsAcc = makeIndexedExpr("acc", makeScalarExpr(innerIndex));
            auto sparseMul = std::make_unique<ir::IRBinaryOp>(
                ir::IRBinaryOp::MUL,
                std::make_unique<ir::IRTensorAccess>(sparseA.name, std::vector<std::string>{}),
                std::make_unique<ir::IRTensorAccess>(sparseB.name, std::vector<std::string>{}));
            auto* lhsTensor = dynamic_cast<ir::IRTensorAccess*>(sparseMul->lhs.get());
            lhsTensor->isSparseVals = true;
            lhsTensor->pointerVar = "p" + sparseA.name;
            auto* rhsTensor = dynamic_cast<ir::IRTensorAccess*>(sparseMul->rhs.get());
            rhsTensor->isSparseVals = true;
            rhsTensor->pointerVar = "p" + sparseB.name;
            leaf->postStmts.push_back(std::make_unique<ir::IRAssign>(
                std::move(lhsAcc),
                std::move(sparseMul),
                true));
        }

        // postStmt: gather-and-clear loop.
        const std::string pointerVar = "p" + compute.output.name;
        auto gatherLoop = std::make_unique<ir::IRForStmt>(
            pointerVar,
            makeOutputPatternStartExpr(compute, outerIndex),
            makeOutputPatternEndExpr(compute, outerIndex));
        auto outputVals = std::make_unique<ir::IRTensorAccess>();
        outputVals->tensorName = compute.output.name;
        outputVals->isSparseVals = true;
        outputVals->pointerVar = pointerVar;
        gatherLoop->body.push_back(std::make_unique<ir::IRAssign>(
            std::move(outputVals),
            makeIndexedExpr("acc", makeOutputPatternIndexExpr(compute, pointerVar)),
            false));
        loop->postStmts.push_back(std::move(gatherLoop));

        auto clearLoop = std::make_unique<ir::IRForStmt>(
            "t",
            std::make_unique<ir::IRConstant>(0.0),
            makeScalarExpr("touched_count"));
        clearLoop->body.push_back(std::make_unique<ir::IRAssign>(
            makeIndexedExpr(
                "acc",
                makeIndexedExpr("touched", makeScalarExpr("t"))),
            std::make_unique<ir::IRConstant>(0.0),
            false));
        clearLoop->body.push_back(std::make_unique<ir::IRAssign>(
            makeIndexedExpr(
                "marked",
                makeIndexedExpr("touched", makeScalarExpr("t"))),
            std::make_unique<ir::IRConstant>(0.0),
            false));
        loop->postStmts.push_back(std::move(clearLoop));

        return loop;
    }

    auto children = loopChildrenFor(ordered, specs, indexName);
    for (const auto& childName : children) {
        auto child = buildGenericLoopNest(
            compute, tensorMap, ordered, specs, childName, sparsePointerMap, useRandomAccess);
        if (!child) {
            return nullptr;
        }
        loop->children.push_back(std::move(child));
    }

    if (children.empty()) {
        if (compute.outputStrategy == ir::OutputStrategy::SparseFixedPattern) {
            auto outVals = std::make_unique<ir::IRTensorAccess>();
            outVals->tensorName = compute.output.name;
            outVals->isSparseVals = true;
            outVals->pointerVar = "p" + compute.output.name;
            loop->postStmts.push_back(std::make_unique<ir::IRAssign>(
                std::move(outVals),
                lowerToOutputFillExpr(*compute.rhs, tensorMap, sparsePointerMap),
                false));
        } else {
            auto lhsExpr = std::make_unique<ir::IRTensorAccess>(
                compute.lhs.tensorName, compute.lhs.indices);
            std::unique_ptr<ir::IRExpr> rhsExpr = useRandomAccess
                ? lowerToOutputFillExpr(*compute.rhs, tensorMap, sparsePointerMap)
                : lowerToLegacyExpr(*compute.rhs, sparsePointerMap);
            loop->postStmts.push_back(std::make_unique<ir::IRAssign>(
                std::move(lhsExpr),
                std::move(rhsExpr),
                !compute.reductionIndices.empty()));
        }
    }

    return loop;
}

void markExternallyBound(scheduled::Loop* loop,
                         const std::vector<std::string>& enclosingIndices) {
    if (!loop) return;

    scheduled::Loop* current = loop;
    for (const auto& index : enclosingIndices) {
        if (!current || current->indexName != index) {
            break;
        }
        current->isExternallyBound = true;
        if (current->children.empty()) {
            break;
        }
        current = current->children[0].get();
    }
}

std::unique_ptr<scheduled::Region> scheduleRegion(
    const semantic::Region& region,
    const std::vector<std::string>& enclosingIndices) {
    std::vector<std::string> childIndices = enclosingIndices;
    childIndices.insert(childIndices.end(), region.indices.begin(), region.indices.end());

    auto lowered = std::make_unique<scheduled::Region>(region.tensors, region.indices);
    lowered->runtimeBounds = region.runtimeBounds;
    for (const auto& stmt : region.body) {
        lowered->addStatement(scheduleStmt(*stmt, childIndices));
    }
    return lowered;
}

std::unique_ptr<scheduled::Stmt> scheduleStmt(
    const semantic::Stmt& stmt,
    const std::vector<std::string>& enclosingIndices) {
    if (auto* decl = dynamic_cast<const semantic::Declaration*>(&stmt)) {
        return std::make_unique<scheduled::Declaration>(decl->tensor);
    }
    if (auto* call = dynamic_cast<const semantic::Call*>(&stmt)) {
        auto lowered = std::make_unique<scheduled::Call>(call->functionName);
        for (const auto& arg : call->args) {
            lowered->args.push_back(arg->clone());
        }
        return lowered;
    }
    if (auto* compute = dynamic_cast<const semantic::Compute*>(&stmt)) {
        auto lowered = scheduleCompute(*compute);
        if (lowered && lowered->rootLoop) {
            markExternallyBound(lowered->rootLoop.get(), enclosingIndices);
        }
        return lowered;
    }
    if (auto* region = dynamic_cast<const semantic::Region*>(&stmt)) {
        return scheduleRegion(*region, enclosingIndices);
    }
    return std::make_unique<scheduled::Call>("noop");
}

class ExpressionAnalyzer : public SparseTensorCompiler::ASTVisitor {
public:
    ir::RootOpKind rootOp = ir::RootOpKind::MULT;
    bool isFused = false;
    std::string fusionFunction;
    std::vector<std::string> tensorNames;
    bool rootOpSet = false;

    void visit(SparseTensorCompiler::Program&) override {}
    void visit(SparseTensorCompiler::Declaration&) override {}
    void visit(SparseTensorCompiler::Computation&) override {}
    void visit(SparseTensorCompiler::CallStatement&) override {}
    void visit(SparseTensorCompiler::ForStatement&) override {}

    void visit(SparseTensorCompiler::TensorAccess& node) override {
        tensorNames.push_back(node.tensorName);
    }

    void visit(SparseTensorCompiler::FunctionCall& node) override {
        isFused = true;
        fusionFunction = node.functionName;
        for (auto& arg : node.arguments) {
            arg->accept(*this);
        }
    }

    void visit(SparseTensorCompiler::BinaryOp& node) override {
        if (!rootOpSet) {
            rootOp = (node.op == SparseTensorCompiler::BinaryOp::ADD)
                ? ir::RootOpKind::ADD
                : ir::RootOpKind::MULT;
            rootOpSet = true;
        }
        node.left->accept(*this);
        node.right->accept(*this);
    }

    void visit(SparseTensorCompiler::Number&) override {}
    void visit(SparseTensorCompiler::Identifier&) override {}
};

ir::ExpressionInfo analyzeExpression(
    const SparseTensorCompiler::Expression& rhs,
    const std::unordered_map<std::string, ir::Tensor>& tensors) {
    ir::ExpressionInfo info;

    ExpressionAnalyzer analyzer;
    const_cast<SparseTensorCompiler::Expression&>(rhs).accept(analyzer);

    info.rootOp = analyzer.rootOp;
    info.isFused = analyzer.isFused;
    info.fusionFunction = analyzer.fusionFunction;
    info.numTensorAccesses = static_cast<int>(analyzer.tensorNames.size());

    for (const auto& name : analyzer.tensorNames) {
        auto it = tensors.find(name);
        if (it != tensors.end() &&
            (it->second.format == ir::Format::CSR || it->second.format == ir::Format::CSC)) {
            info.numSparseInputs++;
            info.sparseInputNames.push_back(name);
        } else {
            info.numDenseInputs++;
            info.denseInputNames.push_back(name);
        }
    }

    return info;
}

} // namespace

std::unique_ptr<Expr> TensorRead::clone() const {
    return std::make_unique<TensorRead>(tensorName, indices);
}

std::unique_ptr<Expr> Constant::clone() const {
    return std::make_unique<Constant>(value);
}

std::unique_ptr<Expr> ScalarRef::clone() const {
    return std::make_unique<ScalarRef>(name);
}

std::unique_ptr<Expr> BinaryExpr::clone() const {
    return std::make_unique<BinaryExpr>(op, lhs->clone(), rhs->clone());
}

std::unique_ptr<Expr> CallExpr::clone() const {
    auto cloned = std::make_unique<CallExpr>(functionName);
    for (const auto& arg : args) {
        cloned->args.push_back(arg->clone());
    }
    return cloned;
}

namespace semantic {

std::unique_ptr<Stmt> Declaration::clone() const {
    return std::make_unique<Declaration>(tensor);
}

std::unique_ptr<Stmt> Call::clone() const {
    auto cloned = std::make_unique<Call>(functionName);
    for (const auto& arg : args) {
        cloned->args.push_back(arg->clone());
    }
    return cloned;
}

std::unique_ptr<Stmt> Compute::clone() const {
    auto cloned = std::make_unique<Compute>();
    cloned->lhs = lhs;
    cloned->output = output;
    cloned->inputs = inputs;
    cloned->rhs = rhs ? rhs->clone() : nullptr;
    cloned->iteratorGraph = iteratorGraph;
    cloned->freeIndices = freeIndices;
    cloned->reductionIndices = reductionIndices;
    cloned->exprInfo = exprInfo;
    cloned->outputStrategy = outputStrategy;
    return cloned;
}

std::unique_ptr<Stmt> Region::clone() const {
    auto cloned = std::make_unique<Region>(tensors, indices);
    cloned->runtimeBounds = runtimeBounds;
    for (const auto& stmt : body) {
        cloned->body.push_back(stmt->clone());
    }
    return cloned;
}

} // namespace semantic

namespace scheduled {

std::unique_ptr<Loop> Loop::clone() const {
    auto cloned = std::make_unique<Loop>();
    cloned->indexName = indexName;
    cloned->lower = lower;
    cloned->upper = upper;
    cloned->runtimeBound = runtimeBound;
    cloned->kind = kind;
    cloned->headerKind = headerKind;
    cloned->lowerExpr = lowerExpr;
    cloned->upperExpr = upperExpr;
    cloned->bindingVarName = bindingVarName;
    cloned->bindingExpr = bindingExpr;
    cloned->iterator = iterator;
    cloned->merge = merge;
    cloned->block = block;
    cloned->isExternallyBound = isExternallyBound;
    for (const auto& stmt : preStmts) {
        cloned->preStmts.push_back(stmt->clone());
    }
    for (const auto& stmt : postStmts) {
        cloned->postStmts.push_back(stmt->clone());
    }
    for (const auto& child : children) {
        cloned->children.push_back(child->clone());
    }
    return cloned;
}

std::unique_ptr<Stmt> Declaration::clone() const {
    return std::make_unique<Declaration>(tensor);
}

std::unique_ptr<Stmt> Call::clone() const {
    auto cloned = std::make_unique<Call>(functionName);
    for (const auto& arg : args) {
        cloned->args.push_back(arg->clone());
    }
    return cloned;
}

std::unique_ptr<Stmt> Compute::clone() const {
    auto cloned = std::make_unique<Compute>();
    cloned->lhs = lhs;
    cloned->output = output;
    cloned->inputs = inputs;
    cloned->rhs = rhs ? rhs->clone() : nullptr;
    cloned->exprInfo = exprInfo;
    cloned->outputStrategy = outputStrategy;
    cloned->outputPattern = outputPattern;
    cloned->patternSources = patternSources;
    for (const auto& stmt : prologueStmts) {
        cloned->prologueStmts.push_back(stmt->clone());
    }
    for (const auto& stmt : epilogueStmts) {
        cloned->epilogueStmts.push_back(stmt->clone());
    }
    cloned->optimizations = optimizations;
    cloned->fullyLowered = fullyLowered;
    cloned->rootLoop = rootLoop ? rootLoop->clone() : nullptr;
    return cloned;
}

std::unique_ptr<Stmt> Region::clone() const {
    auto cloned = std::make_unique<Region>(tensors, indices);
    cloned->runtimeBounds = runtimeBounds;
    for (const auto& stmt : body) {
        cloned->body.push_back(stmt->clone());
    }
    return cloned;
}

} // namespace scheduled

std::unique_ptr<semantic::Program> lowerToSemanticProgram(
    const SparseTensorCompiler::Program& ast) {
    auto declarations = collectDeclarations(ast.statements);
    auto program = std::make_unique<semantic::Program>();
    for (const auto& stmt : ast.statements) {
        program->addStatement(lowerSemanticStmt(*stmt, declarations));
    }
    return program;
}

std::unique_ptr<semantic::Compute> lowerComputationToSemantic(
    const SparseTensorCompiler::Computation& computation,
    const std::unordered_map<std::string, ir::Tensor>& declarations) {
    if (!computation.lhs || !computation.rhs) {
        return nullptr;
    }

    auto compute = std::make_unique<semantic::Compute>();
    compute->lhs.tensorName = computation.lhs->tensorName;
    compute->lhs.indices = computation.lhs->indices;
    compute->output = buildTensorUse(computation.lhs->tensorName, computation.lhs->indices, declarations);
    compute->rhs = lowerSemanticExpr(*computation.rhs);

    std::vector<AccessInfo> accesses;
    collectTensorAccesses(*computation.rhs, accesses);

    std::unordered_set<std::string> seenInputs;
    std::vector<AccessInfo> uniqueAccesses;
    for (const auto& access : accesses) {
        if (seenInputs.insert(access.tensorName).second) {
            uniqueAccesses.push_back(access);
            compute->inputs.push_back(buildTensorUse(access.tensorName, access.indices, declarations));
        }
    }

    compute->freeIndices = computation.lhs->indices;
    auto rhsIndices = collectRhsIndices(accesses);
    for (const auto& index : rhsIndices) {
        if (std::find(compute->freeIndices.begin(), compute->freeIndices.end(), index)
                == compute->freeIndices.end()) {
            compute->reductionIndices.push_back(index);
        }
    }

    compute->exprInfo = analyzeExpression(*computation.rhs, declarations);
    compute->outputStrategy = deriveOutputStrategy(*compute);

    std::vector<std::string> iteratorOrder = compute->freeIndices;
    iteratorOrder.insert(iteratorOrder.end(),
                         compute->reductionIndices.begin(),
                         compute->reductionIndices.end());

    for (const auto& index : iteratorOrder) {
        IteratorNode node;
        node.indexName = index;
        node.upper = inferDimension(index, declarations, compute->output, accesses);
        node.isReduction =
            std::find(compute->reductionIndices.begin(),
                      compute->reductionIndices.end(),
                      index) != compute->reductionIndices.end();

        int sparseSources = 0;
        for (const auto& access : accesses) {
            auto it = declarations.find(access.tensorName);
            ir::Format format = (it != declarations.end()) ? it->second.format : ir::Format::Dense;
            for (size_t pos = 0; pos < access.indices.size(); pos++) {
                if (access.indices[pos] != index) {
                    continue;
                }
                IteratorSource source;
                source.tensorName = access.tensorName;
                source.format = format;
                source.accessPosition = static_cast<int>(pos);
                source.sparse = isSparseAccessPosition(format, static_cast<int>(pos));
                if (source.sparse) {
                    sparseSources++;
                }
                node.sources.push_back(source);
            }
        }

        node.kind = (sparseSources > 0) ? IteratorKind::Sparse : IteratorKind::Dense;
        if (sparseSources >= 2) {
            node.merge = (compute->exprInfo.rootOp == ir::RootOpKind::ADD)
                ? MergeKind::Union
                : MergeKind::Intersection;
        }

        compute->iteratorGraph.iterators.push_back(std::move(node));
    }

    return compute;
}

std::unique_ptr<scheduled::Compute> scheduleCompute(
    const semantic::Compute& compute) {
    auto lowered = std::make_unique<scheduled::Compute>();
    lowered->lhs = compute.lhs;
    lowered->output = compute.output;
    lowered->inputs = compute.inputs;
    lowered->rhs = compute.rhs ? compute.rhs->clone() : nullptr;
    lowered->exprInfo = compute.exprInfo;
    lowered->outputStrategy = compute.outputStrategy;
    lowered->outputPattern = deriveOutputPattern(compute);
    lowered->patternSources = derivePatternSources(compute);

    const auto tensorMap = buildTensorMap(compute.output, compute.inputs);

    std::vector<std::string> ordered = orderedIteratorNames(compute);
    auto specs = deriveAllLoopLoweringSpecs(compute, ordered);
    if (specs.size() != ordered.size()) {
        return lowered;
    }

    auto rootIndices = loopChildrenFor(ordered, specs, "");
    if (rootIndices.empty()) {
        return lowered;
    }
    lowered->rootLoop = buildGenericLoopNest(
        compute, tensorMap, ordered, specs, rootIndices.front(), {},
        isPointwiseUnion(compute), lowered.get());
    for (size_t i = 1; i < rootIndices.size() && lowered->rootLoop; i++) {
        auto extraRoot = buildGenericLoopNest(
            compute, tensorMap, ordered, specs, rootIndices[i], {},
            isPointwiseUnion(compute), lowered.get());
        if (!extraRoot) {
            return lowered;
        }
        lowered->rootLoop->children.push_back(std::move(extraRoot));
    }

    if (!lowered->rootLoop) {
        return lowered;
    }

    lowered->fullyLowered = true;
    return lowered;
}

std::unique_ptr<scheduled::Compute> lowerFirstComputationToScheduled(
    const SparseTensorCompiler::Program& ast) {
    auto semanticProgram = lowerToSemanticProgram(ast);
    if (!semanticProgram) {
        return nullptr;
    }

    auto scheduledProgram = scheduleProgram(*semanticProgram);
    if (!scheduledProgram) {
        return nullptr;
    }

    auto* scheduledCompute = findFirstScheduledCompute(scheduledProgram->statements);
    if (!scheduledCompute) {
        return nullptr;
    }

    auto cloned = scheduledCompute->clone();
    return std::unique_ptr<scheduled::Compute>(
        dynamic_cast<scheduled::Compute*>(cloned.release()));
}

std::unique_ptr<scheduled::Compute> lowerFirstComputationToScheduledOptimized(
    const SparseTensorCompiler::Program& ast,
    const opt::OptConfig& cfg) {
    auto scheduled = lowerFirstComputationToScheduled(ast);
    if (!scheduled) {
        return nullptr;
    }

    opt::applyOptimizations(*scheduled, cfg);
    return scheduled;
}

// Program scheduling applies compute scheduling per-statement.
// No cross-statement optimization occurs. Region/external binding
// is layered after generic compute scheduling.
std::unique_ptr<scheduled::Program> scheduleProgram(
    const semantic::Program& prog) {
    auto scheduledProg = std::make_unique<scheduled::Program>();
    std::vector<std::string> noIndices;
    for (const auto& stmt : prog.statements) {
        scheduledProg->addStatement(scheduleStmt(*stmt, noIndices));
    }
    return scheduledProg;
}

// ============================================================================
// renderScheduledCompute — text dump of a scheduled::Compute for golden artifacts
// ============================================================================

namespace {

std::string loopKindLabel(scheduled::LoopKind kind) {
    switch (kind) {
        case scheduled::LoopKind::Dense:  return "dense";
        case scheduled::LoopKind::Sparse: return "sparse";
        case scheduled::LoopKind::Block:  return "block";
    }
    return "unknown";
}

void renderLoop(std::ostringstream& out, const scheduled::Loop& loop, int depth) {
    std::string indent(static_cast<size_t>(depth) * 2, ' ');
    out << indent << "for " << loop.indexName
        << " in [" << loop.lower << ", " << loop.upper << ") ("
        << loopKindLabel(loop.kind) << "):\n";

    if (loop.children.empty()) {
        // Leaf — emit body placeholder from preStmts/postStmts
        std::string bodyIndent(static_cast<size_t>(depth + 1) * 2, ' ');
        bool emitted = false;
        for (const auto& stmt : loop.preStmts) {
            out << bodyIndent << ir::renderStmt(*stmt) << "\n";
            emitted = true;
        }
        for (const auto& stmt : loop.postStmts) {
            out << bodyIndent << ir::renderStmt(*stmt) << "\n";
            emitted = true;
        }
        if (!emitted) {
            out << bodyIndent << "body: <empty>\n";
        }
    } else {
        for (const auto& child : loop.children) {
            renderLoop(out, *child, depth + 1);
        }
    }
}

} // anonymous namespace

std::string renderScheduledCompute(const scheduled::Compute& compute) {
    std::ostringstream out;

    out << "========================================\n";
    out << "Operation: compute\n";
    out << "========================================\n\n";

    // Output tensor
    out << "Output:\n";
    out << "  Tensor " << compute.output.name << " : "
        << ir::formatToString(compute.output.format) << "<";
    for (size_t i = 0; i < compute.output.dims.size(); ++i) {
        if (i > 0) out << ", ";
        out << compute.output.dims[i];
    }
    out << "> [";
    for (size_t i = 0; i < compute.output.indices.size(); ++i) {
        if (i > 0) out << ", ";
        out << compute.output.indices[i];
    }
    out << "]\n\n";

    // Input tensors
    out << "Inputs:\n";
    for (const auto& t : compute.inputs) {
        out << "  Tensor " << t.name << " : "
            << ir::formatToString(t.format) << "<";
        for (size_t i = 0; i < t.dims.size(); ++i) {
            if (i > 0) out << ", ";
            out << t.dims[i];
        }
        out << "> [";
        for (size_t i = 0; i < t.indices.size(); ++i) {
            if (i > 0) out << ", ";
            out << t.indices[i];
        }
        out << "]\n";
    }

    // Loop nest
    out << "\nLoop Nest:\n";
    if (compute.rootLoop) {
        renderLoop(out, *compute.rootLoop, 1);
    } else {
        out << "  <no loop nest>\n";
    }

    // Optimizations
    out << "\nOptimizations:\n";
    const auto& opts = compute.optimizations;
    out << "  Format-correctness reordering: "
        << (opts.reorderingApplied ? "applied" : "not needed") << "\n";
    out << "  Interchange: "
        << (opts.interchangeApplied ? "applied" : "not applied") << "\n";
    if (opts.blockingApplied) {
        out << "  Blocking: index=" << opts.tiledIndex
            << ", block_size=" << opts.blockSize << "\n";
    } else {
        out << "  Blocking: not applied\n";
    }

    return out.str();
}

} // namespace sparseir
