#include "scheduled_optimizations.h"

#include <algorithm>

namespace opt {
namespace {

using sparseir::scheduled::Compute;
using sparseir::scheduled::Loop;
using sparseir::scheduled::LoopKind;
using sparseir::scheduled::Program;
using sparseir::scheduled::Region;
using sparseir::scheduled::Stmt;

const ir::Tensor* findSparseTensor(const Compute& compute) {
    for (const auto& input : compute.inputs) {
        if (input.format == ir::Format::CSR || input.format == ir::Format::CSC) {
            return &input;
        }
    }
    return nullptr;
}

int countSparseInputs(const Compute& compute) {
    if (compute.exprInfo.numSparseInputs > 0 || compute.exprInfo.numDenseInputs > 0) {
        return compute.exprInfo.numSparseInputs;
    }

    int count = 0;
    for (const auto& input : compute.inputs) {
        if (input.format == ir::Format::CSR || input.format == ir::Format::CSC) {
            count++;
        }
    }
    return count;
}

int countDenseInputs(const Compute& compute) {
    if (compute.exprInfo.numSparseInputs > 0 || compute.exprInfo.numDenseInputs > 0) {
        return compute.exprInfo.numDenseInputs;
    }

    int count = 0;
    for (const auto& input : compute.inputs) {
        if (input.format == ir::Format::Dense) {
            count++;
        }
    }
    return count;
}

std::vector<std::string> collectLoopOrder(const Loop* loop) {
    std::vector<std::string> order;
    const Loop* current = loop;
    while (current) {
        order.push_back(current->indexName);
        if (!current->children.empty()) {
            current = current->children[0].get();
        } else {
            break;
        }
    }
    return order;
}

std::vector<std::string> computeNaturalOrder(const ir::Tensor& tensor) {
    if (tensor.indices.size() < 2) {
        return tensor.indices;
    }

    const std::string& idx0 = tensor.indices[0];
    const std::string& idx1 = tensor.indices[1];

    if (tensor.format == ir::Format::CSR) {
        return (idx0 < idx1) ? std::vector<std::string>{idx0, idx1}
                             : std::vector<std::string>{idx1, idx0};
    }
    if (tensor.format == ir::Format::CSC) {
        return (idx0 < idx1) ? std::vector<std::string>{idx1, idx0}
                             : std::vector<std::string>{idx0, idx1};
    }

    return {idx0, idx1};
}

bool canSwapLoops(Loop* root) {
    if (!root || root->children.empty()) {
        return false;
    }
    auto* child = root->children[0].get();
    if (!child) {
        return false;
    }
    if (child->kind == LoopKind::Sparse) {
        return false;
    }
    return root->kind != LoopKind::Sparse && child->kind != LoopKind::Sparse;
}

void swapLoopHeaders(Loop& lhs, Loop& rhs) {
    std::swap(lhs.indexName, rhs.indexName);
    std::swap(lhs.lower, rhs.lower);
    std::swap(lhs.upper, rhs.upper);
    std::swap(lhs.runtimeBound, rhs.runtimeBound);
    std::swap(lhs.kind, rhs.kind);
    std::swap(lhs.headerKind, rhs.headerKind);
    std::swap(lhs.lowerExpr, rhs.lowerExpr);
    std::swap(lhs.upperExpr, rhs.upperExpr);
    std::swap(lhs.bindingVarName, rhs.bindingVarName);
    std::swap(lhs.bindingExpr, rhs.bindingExpr);
    std::swap(lhs.iterator, rhs.iterator);
    std::swap(lhs.merge, rhs.merge);
    std::swap(lhs.block, rhs.block);
    std::swap(lhs.driverTensor, rhs.driverTensor);
    std::swap(lhs.parentIndexOverride, rhs.parentIndexOverride);
    std::swap(lhs.mergeStrategy, rhs.mergeStrategy);
    std::swap(lhs.mergedTensors, rhs.mergedTensors);
    std::swap(lhs.isExternallyBound, rhs.isExternallyBound);
    std::swap(lhs.tileBlockSize, rhs.tileBlockSize);
}

void copyLoopHeader(const Loop& src, Loop& dst) {
    dst.indexName = src.indexName;
    dst.lower = src.lower;
    dst.upper = src.upper;
    dst.runtimeBound = src.runtimeBound;
    dst.kind = src.kind;
    dst.headerKind = src.headerKind;
    dst.lowerExpr = src.lowerExpr;
    dst.upperExpr = src.upperExpr;
    dst.bindingVarName = src.bindingVarName;
    dst.bindingExpr = src.bindingExpr;
    dst.iterator = src.iterator;
    dst.merge = src.merge;
    dst.block = src.block;
    dst.driverTensor = src.driverTensor;
    dst.parentIndexOverride = src.parentIndexOverride;
    dst.mergeStrategy = src.mergeStrategy;
    dst.mergedTensors = src.mergedTensors;
    dst.isExternallyBound = src.isExternallyBound;
    dst.tileBlockSize = src.tileBlockSize;
}

void configureBlockLoopEmission(Loop& blockLoop, const Loop& denseLoop) {
    const std::string originalIdx = denseLoop.indexName;
    const std::string blockVar = blockLoop.indexName;
    const std::string startVar = originalIdx + "_start";
    const std::string endVar = originalIdx + "_end";
    const int blockSize = blockLoop.tileBlockSize;
    const std::string runtimeUpperBound = blockLoop.runtimeBound.empty()
        ? std::to_string(blockLoop.upper * blockSize)
        : blockLoop.runtimeBound;

    blockLoop.headerKind = sparseir::scheduled::LoopHeaderKind::Block;
    blockLoop.block.blockVar = blockVar;
    blockLoop.block.blockSize = blockSize;
    blockLoop.block.tripCountExpr =
        "(" + runtimeUpperBound + " + " + std::to_string(blockSize - 1) + ") / " +
        std::to_string(blockSize);
    blockLoop.block.startVar = startVar;
    blockLoop.block.endVar = endVar;
    blockLoop.block.innerIndexName = originalIdx;
    blockLoop.block.innerLowerExpr = startVar;
    blockLoop.block.innerUpperExpr = endVar;
}

bool swapRootChildLoops(Loop* root) {
    if (!canSwapLoops(root)) {
        return false;
    }
    swapLoopHeaders(*root, *root->children[0]);
    return true;
}

bool isBlockLoopIndex(const std::string& indexName) {
    return indexName.find("_block") != std::string::npos;
}

std::unique_ptr<Loop>* findLoopSlotByName(std::unique_ptr<Loop>* rootSlot,
                                          const std::string& indexName) {
    if (!rootSlot || !rootSlot->get()) {
        return nullptr;
    }
    if ((*rootSlot)->indexName == indexName) {
        return rootSlot;
    }
    for (auto& child : (*rootSlot)->children) {
        if (auto* found = findLoopSlotByName(&child, indexName)) {
            return found;
        }
    }
    return nullptr;
}

bool blockLoopByIndex(Compute& compute, int blockSize, const std::string& targetIndex) {
    auto* targetSlot = findLoopSlotByName(&compute.rootLoop, targetIndex);
    if (!targetSlot || !targetSlot->get()) {
        return false;
    }

    auto& target = *targetSlot;
    if (target->kind != LoopKind::Dense) {
        return false;
    }

    int upperBound = target->upper;
    int numBlocks = (upperBound + blockSize - 1) / blockSize;
    // Save runtimeBound before move so the block loop can inherit it.
    std::string savedRuntimeBound = target->runtimeBound;

    auto blockLoop = std::make_unique<Loop>();
    blockLoop->indexName = target->indexName + "_block";
    blockLoop->lower = 0;
    blockLoop->upper = numBlocks;
    blockLoop->runtimeBound = savedRuntimeBound;  // original dense-loop bound
    blockLoop->kind = LoopKind::Block;
    blockLoop->tileBlockSize = blockSize;
    configureBlockLoopEmission(*blockLoop, *target);
    blockLoop->children.push_back(std::move(target));

    target = std::move(blockLoop);
    return true;
}

bool isInterchangeLegal(Loop* middle, Loop* inner) {
    return !(middle->kind == LoopKind::Sparse && inner->kind == LoopKind::Sparse);
}

std::unique_ptr<ir::IRExpr> substituteScalarVar(const ir::IRExpr& expr,
                                                const std::string& varName,
                                                const ir::IRExpr& replacement) {
    if (auto* access = dynamic_cast<const ir::IRTensorAccess*>(&expr)) {
        return access->clone();
    }
    if (auto* constant = dynamic_cast<const ir::IRConstant*>(&expr)) {
        return constant->clone();
    }
    if (auto* scalar = dynamic_cast<const ir::IRScalarVar*>(&expr)) {
        if (scalar->name == varName) {
            return replacement.clone();
        }
        return scalar->clone();
    }
    if (auto* binary = dynamic_cast<const ir::IRBinaryOp*>(&expr)) {
        auto lowered = std::make_unique<ir::IRBinaryOp>();
        lowered->op = binary->op;
        lowered->lhs = substituteScalarVar(*binary->lhs, varName, replacement);
        lowered->rhs = substituteScalarVar(*binary->rhs, varName, replacement);
        return lowered;
    }
    if (auto* call = dynamic_cast<const ir::IRFuncCall*>(&expr)) {
        auto lowered = std::make_unique<ir::IRFuncCall>(call->name);
        for (const auto& arg : call->args) {
            lowered->args.push_back(substituteScalarVar(*arg, varName, replacement));
        }
        return lowered;
    }
    return expr.clone();
}

bool fuseStructuredAccumulatorPattern(const Loop& sparseLoop,
                                      const Loop& denseLoop,
                                      std::unique_ptr<ir::IRStmt>& fusedStmt) {
    if (sparseLoop.preStmts.size() != 1 || sparseLoop.postStmts.size() != 1 ||
        denseLoop.postStmts.size() != 1) {
        return false;
    }

    auto* decl = dynamic_cast<const ir::IRScalarDecl*>(sparseLoop.preStmts[0].get());
    auto* accum = dynamic_cast<const ir::IRAssign*>(denseLoop.postStmts[0].get());
    auto* finalize = dynamic_cast<const ir::IRAssign*>(sparseLoop.postStmts[0].get());
    if (!decl || !accum || !finalize || !accum->accumulate) {
        return false;
    }

    auto* accumLhs = dynamic_cast<const ir::IRScalarVar*>(accum->lhs.get());
    if (!accumLhs || accumLhs->name != decl->varName) {
        return false;
    }

    auto substituted = substituteScalarVar(*finalize->rhs, decl->varName, *accum->rhs);
    fusedStmt = std::make_unique<ir::IRAssign>(finalize->lhs->clone(),
                                               std::move(substituted),
                                               true);
    return true;
}

void renderLoopStmts(const Loop& loop, std::string& preBody, std::string& body) {
    ir::renderStmtsToStrings(loop.preStmts, loop.postStmts, preBody, body);
}

bool fuseAccumulatorPattern(
    const std::string& preBody,
    const std::string& innerBody,
    const std::string& postBody,
    std::string& fusedBody
) {
    if (preBody.find("double sum = 0.0;") == std::string::npos) return false;
    auto sumPos = innerBody.find("sum += ");
    if (sumPos == std::string::npos) return false;
    if (postBody.find("sum") == std::string::npos) return false;

    std::string innerExpr = innerBody.substr(sumPos + 7);
    auto semiPos = innerExpr.rfind(';');
    if (semiPos != std::string::npos) innerExpr = innerExpr.substr(0, semiPos);
    while (!innerExpr.empty() && innerExpr.back() == ' ') innerExpr.pop_back();
    while (!innerExpr.empty() && innerExpr.front() == ' ') innerExpr = innerExpr.substr(1);

    fusedBody = postBody;
    auto eqPos = fusedBody.find("= ");
    if (eqPos != std::string::npos && eqPos > 0 && fusedBody[eqPos - 1] != '+' &&
        fusedBody[eqPos - 1] != '!' && fusedBody[eqPos - 1] != '=' &&
        fusedBody[eqPos - 1] != '<' && fusedBody[eqPos - 1] != '>') {
        fusedBody.replace(eqPos, 2, "+= ");
    }
    auto fusedSumPos = fusedBody.find("sum");
    if (fusedSumPos != std::string::npos) {
        fusedBody.replace(fusedSumPos, 3, innerExpr);
    }

    return true;
}

bool tryInterchangeAtDenseNode(Loop* node, bool allowBlockWrappedDenseInner) {
    if (!node || node->kind == LoopKind::Sparse || node->children.empty()) {
        return false;
    }
    if (isBlockLoopIndex(node->indexName)) {
        return false;
    }

    auto* middle = node->children[0].get();
    if (!middle || middle->children.empty()) {
        return false;
    }
    auto* inner = middle->children[0].get();
    if (!inner) {
        return false;
    }

    if (!isInterchangeLegal(middle, inner)) {
        return false;
    }

    if (middle->kind == LoopKind::Sparse && inner->kind != LoopKind::Sparse) {
        if (!allowBlockWrappedDenseInner && isBlockLoopIndex(inner->indexName)) {
            return false;
        }

        std::string middlePre, middlePost, innerPre, innerPost;
        renderLoopStmts(*middle, middlePre, middlePost);
        renderLoopStmts(*inner, innerPre, innerPost);

        if (allowBlockWrappedDenseInner && isBlockLoopIndex(inner->indexName) &&
            innerPost.empty() && !inner->children.empty() &&
            !middlePre.empty() && !middlePost.empty()) {
            auto* innerPayload = inner->children[0].get();
            if (innerPayload && innerPayload->kind != LoopKind::Sparse) {
                std::string payloadPre, payloadPost, fusedBody;
                std::unique_ptr<ir::IRStmt> fusedStmt;
                renderLoopStmts(*innerPayload, payloadPre, payloadPost);
                if (fuseStructuredAccumulatorPattern(*middle, *innerPayload, fusedStmt) ||
                    fuseAccumulatorPattern(middlePre, payloadPost, middlePost, fusedBody)) {
                    auto sparseLoop = std::move(node->children[0]);
                    auto blockLoop = std::move(sparseLoop->children[0]);
                    auto denseLoop = std::move(blockLoop->children[0]);

                    auto newSparse = std::make_unique<Loop>();
                    copyLoopHeader(*sparseLoop, *newSparse);
                    if (fusedStmt) {
                        newSparse->postStmts.push_back(std::move(fusedStmt));
                    } else {
                        newSparse->postStmts.push_back(std::make_unique<ir::IRRawStmt>(fusedBody));
                    }

                    denseLoop->preStmts.clear();
                    denseLoop->postStmts.clear();
                    denseLoop->children.clear();
                    denseLoop->children.push_back(std::move(newSparse));
                    blockLoop->children.clear();
                    blockLoop->children.push_back(std::move(denseLoop));
                    node->children.clear();
                    node->children.push_back(std::move(blockLoop));
                    return true;
                }
            }
        }

        auto sparseLoop = std::move(node->children[0]);
        auto denseLoop = std::move(sparseLoop->children[0]);

        std::string densePre, densePost, fusedBody;
        std::unique_ptr<ir::IRStmt> fusedStmt;
        renderLoopStmts(*denseLoop, densePre, densePost);
        bool fusedStructured = fuseStructuredAccumulatorPattern(*sparseLoop, *denseLoop, fusedStmt);
        bool fusedString = !middlePre.empty() && !middlePost.empty() &&
            fuseAccumulatorPattern(middlePre, densePost, middlePost, fusedBody);
        bool fused = fusedStructured || fusedString;

        auto newSparse = std::make_unique<Loop>();
        copyLoopHeader(*sparseLoop, *newSparse);
        newSparse->children = std::move(denseLoop->children);
        if (fused) {
            if (fusedStmt) {
                newSparse->postStmts.push_back(std::move(fusedStmt));
            } else {
                newSparse->postStmts.push_back(std::make_unique<ir::IRRawStmt>(fusedBody));
            }
        } else {
            for (auto& stmt : sparseLoop->preStmts) {
                newSparse->preStmts.push_back(std::move(stmt));
            }
            for (auto& stmt : denseLoop->postStmts) {
                newSparse->postStmts.push_back(std::move(stmt));
            }
        }

        auto newDense = std::make_unique<Loop>();
        copyLoopHeader(*denseLoop, *newDense);
        newDense->children.push_back(std::move(newSparse));

        node->children.clear();
        node->children.push_back(std::move(newDense));
        return true;
    }

    if (middle->kind != LoopKind::Sparse && inner->kind == LoopKind::Sparse) {
        auto denseLoop = std::move(node->children[0]);
        auto sparseLoop = std::move(denseLoop->children[0]);

        auto newDense = std::make_unique<Loop>();
        copyLoopHeader(*denseLoop, *newDense);
        newDense->children = std::move(sparseLoop->children);
        for (auto& stmt : sparseLoop->postStmts) {
            newDense->postStmts.push_back(std::move(stmt));
        }

        auto newSparse = std::make_unique<Loop>();
        copyLoopHeader(*sparseLoop, *newSparse);
        newSparse->children.push_back(std::move(newDense));
        for (auto& stmt : sparseLoop->preStmts) {
            newSparse->preStmts.push_back(std::move(stmt));
        }

        node->children.clear();
        node->children.push_back(std::move(newSparse));
        return true;
    }

    return false;
}

bool applyOneInterchangeDFS(Loop* node, bool allowBlockWrappedDenseInner) {
    if (!node) {
        return false;
    }
    if (tryInterchangeAtDenseNode(node, allowBlockWrappedDenseInner)) {
        return true;
    }
    for (auto& child : node->children) {
        if (applyOneInterchangeDFS(child.get(), allowBlockWrappedDenseInner)) {
            return true;
        }
    }
    return false;
}

std::string chooseBlockingIndex(const Compute& compute, const OptConfig& config) {
    const int sparseInputs = countSparseInputs(compute);
    const int denseInputs = countDenseInputs(compute);

    if (compute.outputStrategy == ir::OutputStrategy::DenseArray &&
        sparseInputs == 1 &&
        denseInputs == 1 &&
        compute.output.indices.size() >= 2) {
        return compute.output.indices[1];
    }

    if (compute.outputStrategy == ir::OutputStrategy::DenseArray &&
        sparseInputs == 1 &&
        denseInputs == 2) {
        const Loop* current = compute.rootLoop.get();
        while (current) {
            if (current->kind == LoopKind::Dense &&
                std::find(compute.output.indices.begin(), compute.output.indices.end(),
                          current->indexName) == compute.output.indices.end()) {
                return current->indexName;
            }
            current = current->children.empty() ? nullptr : current->children[0].get();
        }
    }

    return compute.rootLoop ? compute.rootLoop->indexName : "";
}

void applyOptimizationsToStmt(Stmt& stmt, const OptConfig& config) {
    if (auto* compute = dynamic_cast<Compute*>(&stmt)) {
        applyOptimizations(*compute, config);
        return;
    }
    if (auto* region = dynamic_cast<Region*>(&stmt)) {
        for (auto& bodyStmt : region->body) {
            applyOptimizationsToStmt(*bodyStmt, config);
        }
    }
}

} // namespace

void applyReordering(Compute& compute) {
    if (compute.optimizations.reorderingApplied || !compute.rootLoop) {
        return;
    }

    const ir::Tensor* sparseTensor = findSparseTensor(compute);
    if (!sparseTensor) {
        return;
    }

    auto currentOrder = collectLoopOrder(compute.rootLoop.get());
    auto naturalOrder = computeNaturalOrder(*sparseTensor);
    if (currentOrder.size() != naturalOrder.size()) {
        return;
    }
    if (currentOrder == naturalOrder) {
        return;
    }
    if (currentOrder.size() == 2 &&
        currentOrder[0] == naturalOrder[1] &&
        currentOrder[1] == naturalOrder[0] &&
        swapRootChildLoops(compute.rootLoop.get())) {
        compute.optimizations.reorderingApplied = true;
        compute.optimizations.originalOrder = currentOrder;
        compute.optimizations.newOrder = collectLoopOrder(compute.rootLoop.get());
    }
}

void applyBlocking(Compute& compute, const OptConfig& config) {
    if (!config.enableBlocking || compute.optimizations.blockingApplied || !compute.rootLoop) {
        return;
    }

    int blockSize = config.blockSize;
    int bs2 = (config.blockSize2 > 0) ? config.blockSize2 : blockSize;
    const int sparseInputs = countSparseInputs(compute);
    const int denseInputs = countDenseInputs(compute);

    if (compute.outputStrategy == ir::OutputStrategy::DenseArray &&
        sparseInputs == 1 &&
        denseInputs == 1 &&
        compute.output.indices.size() >= 2) {
        if (config.enable2DBlocking) {
            if (!blockLoopByIndex(compute, blockSize, compute.output.indices[0])) return;
            if (!blockLoopByIndex(compute, bs2, compute.output.indices[1])) return;
            compute.optimizations.blockingApplied = true;
            compute.optimizations.blocking2DApplied = true;
            compute.optimizations.blockSize = blockSize;
            compute.optimizations.tiledIndex = compute.output.indices[0];
            compute.optimizations.tiledIndices = {compute.output.indices[0], compute.output.indices[1]};
            compute.optimizations.blockSizes = {blockSize, bs2};
            return;
        }
        if (!blockLoopByIndex(compute, blockSize, compute.output.indices[1])) return;
        compute.optimizations.blockingApplied = true;
        compute.optimizations.blockSize = blockSize;
        compute.optimizations.tiledIndex = compute.output.indices[1];
        return;
    }

    if (compute.outputStrategy == ir::OutputStrategy::DenseArray &&
        sparseInputs == 1 &&
        denseInputs == 2) {
        std::string reductionIdx = chooseBlockingIndex(compute, config);
        if (config.enable2DBlocking) {
            if (!blockLoopByIndex(compute, blockSize, compute.output.indices[0])) return;
            if (!blockLoopByIndex(compute, bs2, reductionIdx)) return;
            compute.optimizations.blockingApplied = true;
            compute.optimizations.blocking2DApplied = true;
            compute.optimizations.blockSize = blockSize;
            compute.optimizations.tiledIndex = compute.output.indices[0];
            compute.optimizations.tiledIndices = {compute.output.indices[0], reductionIdx};
            compute.optimizations.blockSizes = {blockSize, bs2};
            return;
        }
        if (!blockLoopByIndex(compute, blockSize, reductionIdx)) return;
        compute.optimizations.blockingApplied = true;
        compute.optimizations.blockSize = blockSize;
        compute.optimizations.tiledIndex = reductionIdx;
        return;
    }

    std::string target = chooseBlockingIndex(compute, config);
    if (target.empty() || !blockLoopByIndex(compute, blockSize, target)) {
        return;
    }
    compute.optimizations.blockingApplied = true;
    compute.optimizations.blockSize = blockSize;
    compute.optimizations.tiledIndex = target;
}

void applyLoopInterchange(Compute& compute, const OptConfig& config) {
    if (!config.enableInterchange || !compute.rootLoop || compute.rootLoop->children.empty()) {
        return;
    }
    const bool allowBlockWrappedDenseInner =
        (config.order == OptOrder::B_THEN_I && config.enableBlocking);
    if (applyOneInterchangeDFS(compute.rootLoop.get(), allowBlockWrappedDenseInner)) {
        compute.optimizations.interchangeApplied = true;
    }
}

void applyOptimizations(Compute& compute, const OptConfig& config) {
    applyReordering(compute);

    if (!config.enableInterchange && !config.enableBlocking) {
        return;
    }

    switch (config.order) {
        case OptOrder::I_THEN_B:
            if (config.enableInterchange) applyLoopInterchange(compute, config);
            if (config.enableBlocking) applyBlocking(compute, config);
            break;
        case OptOrder::B_THEN_I:
            if (config.enableBlocking) applyBlocking(compute, config);
            if (config.enableInterchange) applyLoopInterchange(compute, config);
            break;
        case OptOrder::I_B_I:
            if (config.enableInterchange) applyLoopInterchange(compute, config);
            if (config.enableBlocking) applyBlocking(compute, config);
            if (config.enableInterchange) applyLoopInterchange(compute, config);
            break;
        default:
            if (config.enableInterchange) applyLoopInterchange(compute, config);
            if (config.enableBlocking) applyBlocking(compute, config);
            break;
    }
}

void applyOptimizations(Program& program, const OptConfig& config) {
    for (auto& stmt : program.statements) {
        applyOptimizationsToStmt(*stmt, config);
    }
}

} // namespace opt
