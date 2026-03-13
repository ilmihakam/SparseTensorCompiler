#include "scheduled_optimizations.h"

#include <algorithm>
#include <unordered_map>

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
    dst.requiredOuterLoops = src.requiredOuterLoops;
    dst.isExternallyBound = src.isExternallyBound;
}

void appendUnique(std::vector<std::string>& values, const std::string& value) {
    if (value.empty()) {
        return;
    }
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

void configureBlockLoopEmission(Loop& blockLoop, const Loop& denseLoop) {
    const std::string originalIdx = denseLoop.indexName;
    const std::string blockVar = blockLoop.indexName;
    const std::string startVar = originalIdx + "_start";
    const std::string endVar = originalIdx + "_end";
    const int blockSize = blockLoop.block.blockSize;
    const std::string runtimeUpperBound = blockLoop.runtimeBound.empty()
        ? std::to_string(blockLoop.upper * blockSize)
        : blockLoop.runtimeBound;

    blockLoop.headerKind = sparseir::scheduled::LoopHeaderKind::Block;
    blockLoop.block.targetKind = sparseir::scheduled::BlockTargetKind::DenseIndex;
    blockLoop.block.blockVar = blockVar;
    blockLoop.block.blockSize = blockSize;
    blockLoop.block.baseIndexName = originalIdx;
    blockLoop.block.tileLevel = denseLoop.kind == LoopKind::Block ? denseLoop.block.tileLevel + 1 : 1;
    blockLoop.block.tripCountExpr =
        "(" + runtimeUpperBound + " + " + std::to_string(blockSize - 1) + ") / " +
        std::to_string(blockSize);
    blockLoop.block.startVar = startVar;
    blockLoop.block.endVar = endVar;
    blockLoop.block.innerIndexName = originalIdx;
    blockLoop.block.innerLowerExpr = startVar;
    blockLoop.block.innerUpperExpr = endVar;
}

void configureSparseIteratorBlockEmission(Loop& blockLoop, const Loop& sparseLoop) {
    const std::string pointerVar = sparseLoop.iterator.pointerVar;
    const int blockSize = blockLoop.block.blockSize;

    blockLoop.headerKind = sparseir::scheduled::LoopHeaderKind::Block;
    blockLoop.block.targetKind = sparseir::scheduled::BlockTargetKind::SparseIteratorPosition;
    blockLoop.block.blockVar = pointerVar + "_block";
    blockLoop.block.blockSize = blockSize;
    blockLoop.block.baseIndexName = sparseLoop.indexName;
    blockLoop.block.tileLevel = sparseLoop.kind == LoopKind::Block ? sparseLoop.block.tileLevel + 1 : 1;
    blockLoop.block.startVar = pointerVar + "_start";
    blockLoop.block.endVar = pointerVar + "_end";
    blockLoop.block.innerIndexName = sparseLoop.indexName;
    blockLoop.block.innerLowerExpr = blockLoop.block.startVar;
    blockLoop.block.innerUpperExpr = blockLoop.block.endVar;
    blockLoop.block.sparseBeginExpr = sparseLoop.iterator.beginExpr;
    blockLoop.block.sparseEndExpr = sparseLoop.iterator.endExpr;
    blockLoop.block.tripCountExpr =
        "(((" + sparseLoop.iterator.endExpr + ") - (" + sparseLoop.iterator.beginExpr + ")) + " +
        std::to_string(blockSize - 1) + ") / " + std::to_string(blockSize);
}

void configureSparseMergeBlockEmission(Loop& blockLoop, const Loop& sparseLoop) {
    blockLoop.headerKind = sparseir::scheduled::LoopHeaderKind::Block;
    blockLoop.block.targetKind = sparseir::scheduled::BlockTargetKind::SparseMergeSteps;
    blockLoop.block.blockVar = sparseLoop.indexName + "_merge_block";
    blockLoop.block.blockSize = blockLoop.block.blockSize;
    blockLoop.block.baseIndexName = sparseLoop.indexName;
    blockLoop.block.tileLevel = sparseLoop.kind == LoopKind::Block ? sparseLoop.block.tileLevel + 1 : 1;
    blockLoop.block.innerIndexName = sparseLoop.indexName;
}

bool isBlockLoopIndex(const std::string& indexName) {
    return indexName.find("_block") != std::string::npos;
}

const Loop* findLoopByName(const Loop* root, const std::string& indexName) {
    if (!root) {
        return nullptr;
    }
    if (root->indexName == indexName) {
        return root;
    }
    for (const auto& child : root->children) {
        if (const Loop* found = findLoopByName(child.get(), indexName)) {
            return found;
        }
    }
    return nullptr;
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

bool isBlockableTarget(const Loop* loop, bool allowSparseTargets) {
    if (!loop) {
        return false;
    }
    if (loop->kind == LoopKind::Dense) {
        return true;
    }
    if (!allowSparseTargets) {
        return false;
    }
    return loop->headerKind == sparseir::scheduled::LoopHeaderKind::SparseIterator ||
           loop->headerKind == sparseir::scheduled::LoopHeaderKind::SparseMerge;
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
    blockLoop->block.blockSize = blockSize;
    configureBlockLoopEmission(*blockLoop, *target);
    blockLoop->children.push_back(std::move(target));
    appendUnique(blockLoop->children[0]->requiredOuterLoops, blockLoop->indexName);

    target = std::move(blockLoop);
    return true;
}

bool blockSparseIteratorLoopByIndex(Compute& compute, int blockSize, const std::string& targetIndex) {
    auto* targetSlot = findLoopSlotByName(&compute.rootLoop, targetIndex);
    if (!targetSlot || !targetSlot->get()) {
        return false;
    }

    auto& target = *targetSlot;
    if (target->headerKind != sparseir::scheduled::LoopHeaderKind::SparseIterator) {
        return false;
    }

    auto blockLoop = std::make_unique<Loop>();
    blockLoop->indexName = target->indexName + "_pos_block";
    blockLoop->kind = LoopKind::Block;
    blockLoop->block.blockSize = blockSize;
    configureSparseIteratorBlockEmission(*blockLoop, *target);
    blockLoop->children.push_back(std::move(target));
    appendUnique(blockLoop->children[0]->requiredOuterLoops, blockLoop->indexName);
    target = std::move(blockLoop);
    return true;
}

bool blockSparseMergeLoopByIndex(Compute& compute, int blockSize, const std::string& targetIndex) {
    auto* targetSlot = findLoopSlotByName(&compute.rootLoop, targetIndex);
    if (!targetSlot || !targetSlot->get()) {
        return false;
    }

    auto& target = *targetSlot;
    if (target->headerKind != sparseir::scheduled::LoopHeaderKind::SparseMerge) {
        return false;
    }

    auto blockLoop = std::make_unique<Loop>();
    blockLoop->indexName = target->indexName + "_pos_block";
    blockLoop->kind = LoopKind::Block;
    blockLoop->block.blockSize = blockSize;
    configureSparseMergeBlockEmission(*blockLoop, *target);
    blockLoop->children.push_back(std::move(target));
    appendUnique(blockLoop->children[0]->requiredOuterLoops, blockLoop->indexName);
    target = std::move(blockLoop);
    return true;
}

bool blockTargetByIndex(Compute& compute,
                        int blockSize,
                        const std::string& targetIndex,
                        bool allowSparseTargets) {
    auto* targetSlot = findLoopSlotByName(&compute.rootLoop, targetIndex);
    if (!targetSlot || !targetSlot->get()) {
        return false;
    }

    const auto& target = *targetSlot->get();
    if (target.kind == LoopKind::Dense) {
        return blockLoopByIndex(compute, blockSize, targetIndex);
    }
    if (!allowSparseTargets) {
        return false;
    }
    if (target.headerKind == sparseir::scheduled::LoopHeaderKind::SparseIterator) {
        return blockSparseIteratorLoopByIndex(compute, blockSize, targetIndex);
    }
    if (target.headerKind == sparseir::scheduled::LoopHeaderKind::SparseMerge) {
        return blockSparseMergeLoopByIndex(compute, blockSize, targetIndex);
    }
    return false;
}

bool blockPositionTargetByIndex(Compute& compute, int blockSize, const std::string& targetIndex) {
    auto* targetSlot = findLoopSlotByName(&compute.rootLoop, targetIndex);
    if (!targetSlot || !targetSlot->get()) {
        return false;
    }

    const auto& target = *targetSlot->get();
    if (target.headerKind == sparseir::scheduled::LoopHeaderKind::SparseIterator) {
        return blockSparseIteratorLoopByIndex(compute, blockSize, targetIndex);
    }
    if (target.headerKind == sparseir::scheduled::LoopHeaderKind::SparseMerge) {
        return blockSparseMergeLoopByIndex(compute, blockSize, targetIndex);
    }
    return false;
}

void collectSparseLoopNames(const Loop* loop, std::vector<std::string>& names) {
    if (!loop) {
        return;
    }
    if (loop->headerKind == sparseir::scheduled::LoopHeaderKind::SparseIterator ||
        loop->headerKind == sparseir::scheduled::LoopHeaderKind::SparseMerge) {
        names.push_back(loop->indexName);
    }
    for (const auto& child : loop->children) {
        collectSparseLoopNames(child.get(), names);
    }
}

bool loopRequiresOuter(const Loop& loop, const std::string& outerIndex) {
    return std::find(loop.requiredOuterLoops.begin(), loop.requiredOuterLoops.end(), outerIndex) !=
           loop.requiredOuterLoops.end();
}

bool isAdjacentInterchangeLegal(const Loop& outer, const Loop& inner) {
    if (loopRequiresOuter(inner, outer.indexName)) {
        return false;
    }
    if (outer.kind == LoopKind::Sparse && inner.kind == LoopKind::Sparse) {
        return false;
    }
    return true;
}

std::unique_ptr<ir::IRExpr> substituteAccumulatorRef(const ir::IRExpr& expr,
                                                     const std::string& accumulatorName,
                                                     const ir::IRExpr& replacement) {
    if (auto* access = dynamic_cast<const ir::IRTensorAccess*>(&expr)) {
        return access->clone();
    }
    if (auto* constant = dynamic_cast<const ir::IRConstant*>(&expr)) {
        return constant->clone();
    }
    if (auto* scalar = dynamic_cast<const ir::IRScalarVar*>(&expr)) {
        if (scalar->name == accumulatorName) {
            return replacement.clone();
        }
        return scalar->clone();
    }
    if (auto* accumRef = dynamic_cast<const ir::IRAccumulatorRef*>(&expr)) {
        if (accumRef->name == accumulatorName) {
            return replacement.clone();
        }
        return accumRef->clone();
    }
    if (auto* binary = dynamic_cast<const ir::IRBinaryOp*>(&expr)) {
        auto lowered = std::make_unique<ir::IRBinaryOp>();
        lowered->op = binary->op;
        lowered->lhs = substituteAccumulatorRef(*binary->lhs, accumulatorName, replacement);
        lowered->rhs = substituteAccumulatorRef(*binary->rhs, accumulatorName, replacement);
        return lowered;
    }
    if (auto* call = dynamic_cast<const ir::IRFuncCall*>(&expr)) {
        auto lowered = std::make_unique<ir::IRFuncCall>(call->name);
        for (const auto& arg : call->args) {
            lowered->args.push_back(substituteAccumulatorRef(*arg, accumulatorName, replacement));
        }
        return lowered;
    }
    if (auto* indexed = dynamic_cast<const ir::IRIndexedAccess*>(&expr)) {
        auto lowered = std::make_unique<ir::IRIndexedAccess>(indexed->baseName);
        for (const auto& index : indexed->indices) {
            lowered->indices.push_back(substituteAccumulatorRef(*index, accumulatorName, replacement));
        }
        return lowered;
    }
    if (auto* compare = dynamic_cast<const ir::IRCompareExpr*>(&expr)) {
        return std::make_unique<ir::IRCompareExpr>(
            compare->op,
            substituteAccumulatorRef(*compare->lhs, accumulatorName, replacement),
            substituteAccumulatorRef(*compare->rhs, accumulatorName, replacement));
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

    auto* init = dynamic_cast<const ir::IRAccumulatorInit*>(sparseLoop.preStmts[0].get());
    auto* update = dynamic_cast<const ir::IRAccumulatorUpdate*>(denseLoop.postStmts[0].get());
    auto* finalize = dynamic_cast<const ir::IRAccumulatorFinalize*>(sparseLoop.postStmts[0].get());
    if (!init || !update || !finalize || !update->rhs) {
        return false;
    }

    if (update->accumulatorName != init->accumulatorName) {
        return false;
    }

    auto substituted = substituteAccumulatorRef(*finalize->rhs, init->accumulatorName, *update->rhs);
    fusedStmt = std::make_unique<ir::IRAssign>(finalize->lhs->clone(),
                                               std::move(substituted),
                                               true);
    return true;
}

void moveStmtList(std::vector<std::unique_ptr<ir::IRStmt>>& dst,
                  std::vector<std::unique_ptr<ir::IRStmt>>& src) {
    for (auto& stmt : src) {
        dst.push_back(std::move(stmt));
    }
    src.clear();
}

bool interchangeSparseWithDense(std::unique_ptr<Loop>* outerSlot) {
    auto outer = std::move(*outerSlot);
    auto inner = std::move(outer->children[0]);

    std::unique_ptr<ir::IRStmt> fusedStmt;
    const bool fused = fuseStructuredAccumulatorPattern(*outer, *inner, fusedStmt);

    outer->children = std::move(inner->children);
    if (fused) {
        outer->preStmts.clear();
        outer->postStmts.clear();
        outer->postStmts.push_back(std::move(fusedStmt));
    } else {
        moveStmtList(outer->postStmts, inner->postStmts);
    }

    inner->preStmts.clear();
    inner->postStmts.clear();
    inner->children.clear();
    inner->children.push_back(std::move(outer));
    *outerSlot = std::move(inner);
    return true;
}

bool interchangeSparseWithBlock(std::unique_ptr<Loop>* outerSlot) {
    auto outer = std::move(*outerSlot);
    auto blockLoop = std::move(outer->children[0]);
    if (blockLoop->children.empty()) {
        return false;
    }

    auto payload = std::move(blockLoop->children[0]);
    std::unique_ptr<ir::IRStmt> fusedStmt;
    const bool canFuse = payload->kind != LoopKind::Sparse &&
                         fuseStructuredAccumulatorPattern(*outer, *payload, fusedStmt);

    if (canFuse) {
        outer->children = std::move(payload->children);
        outer->preStmts.clear();
        outer->postStmts.clear();
        outer->postStmts.push_back(std::move(fusedStmt));
        payload->preStmts.clear();
        payload->postStmts.clear();
        payload->children.clear();
        payload->children.push_back(std::move(outer));
        blockLoop->children.clear();
        blockLoop->children.push_back(std::move(payload));
        *outerSlot = std::move(blockLoop);
        return true;
    }

    outer->children.clear();
    outer->children.push_back(std::move(payload));
    blockLoop->children.clear();
    blockLoop->children.push_back(std::move(outer));
    *outerSlot = std::move(blockLoop);
    return true;
}

bool interchangeDenseWithSparse(std::unique_ptr<Loop>* outerSlot) {
    auto outer = std::move(*outerSlot);
    auto inner = std::move(outer->children[0]);

    outer->children = std::move(inner->children);
    moveStmtList(outer->postStmts, inner->postStmts);

    inner->children.clear();
    inner->children.push_back(std::move(outer));
    *outerSlot = std::move(inner);
    return true;
}

bool interchangeGenericPair(std::unique_ptr<Loop>* outerSlot) {
    auto outer = std::move(*outerSlot);
    auto inner = std::move(outer->children[0]);
    outer->children = std::move(inner->children);
    inner->children.clear();
    inner->children.push_back(std::move(outer));
    *outerSlot = std::move(inner);
    return true;
}

bool interchangeAdjacentPair(std::unique_ptr<Loop>* outerSlot) {
    if (!outerSlot || !outerSlot->get() || (*outerSlot)->children.empty()) {
        return false;
    }

    Loop* outer = outerSlot->get();
    Loop* inner = outer->children[0].get();
    if (!inner || !isAdjacentInterchangeLegal(*outer, *inner)) {
        return false;
    }

    if (outer->kind == LoopKind::Sparse && inner->kind == LoopKind::Block) {
        return interchangeSparseWithBlock(outerSlot);
    }
    if (outer->kind == LoopKind::Sparse && inner->kind != LoopKind::Sparse) {
        return interchangeSparseWithDense(outerSlot);
    }
    if (outer->kind != LoopKind::Sparse && inner->kind == LoopKind::Sparse) {
        return interchangeDenseWithSparse(outerSlot);
    }
    return interchangeGenericPair(outerSlot);
}

bool tryInterchangeAtDenseNode(Loop* node, bool allowBlockWrappedDenseInner) {
    if (!node || node->kind != LoopKind::Dense || node->children.empty()) {
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
    if (!allowBlockWrappedDenseInner &&
        middle->kind == LoopKind::Sparse &&
        inner->kind == LoopKind::Block) {
        return false;
    }
    return interchangeAdjacentPair(&node->children[0]);
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

bool flattenPerfectNestSlots(std::unique_ptr<Loop>* rootSlot,
                             std::vector<std::unique_ptr<Loop>*>& slots) {
    if (!rootSlot || !rootSlot->get()) {
        return false;
    }
    slots.clear();
    std::unique_ptr<Loop>* current = rootSlot;
    while (current && current->get()) {
        slots.push_back(current);
        if ((*current)->children.size() > 1) {
            return false;
        }
        current = (*current)->children.empty() ? nullptr : &(*current)->children[0];
    }
    return !slots.empty();
}

bool isValidTargetOrder(const std::vector<std::string>& currentOrder,
                        const std::vector<std::string>& targetOrder) {
    if (currentOrder.size() != targetOrder.size()) {
        return false;
    }

    std::vector<std::string> lhs = currentOrder;
    std::vector<std::string> rhs = targetOrder;
    std::sort(lhs.begin(), lhs.end());
    std::sort(rhs.begin(), rhs.end());
    if (lhs != rhs) {
        return false;
    }
    return std::adjacent_find(rhs.begin(), rhs.end()) == rhs.end();
}

bool reorderTowardTarget(Compute& compute, const std::vector<std::string>& targetOrder) {
    if (!compute.rootLoop || targetOrder.empty()) {
        return false;
    }

    const std::vector<std::string> initialOrder = collectLoopOrder(compute.rootLoop.get());
    if (!isValidTargetOrder(initialOrder, targetOrder)) {
        return false;
    }
    if (initialOrder == targetOrder) {
        compute.optimizations.interchangeRequestedOrder = targetOrder;
        compute.optimizations.interchangeOriginalOrder = initialOrder;
        compute.optimizations.interchangeFinalOrder = initialOrder;
        return false;
    }

    std::unordered_map<std::string, size_t> targetPositions;
    for (size_t i = 0; i < targetOrder.size(); ++i) {
        targetPositions[targetOrder[i]] = i;
    }

    bool changed = false;
    bool progress = true;
    while (progress) {
        progress = false;
        std::vector<std::unique_ptr<Loop>*> slots;
        if (!flattenPerfectNestSlots(&compute.rootLoop, slots)) {
            break;
        }

        std::vector<std::string> currentOrder = collectLoopOrder(compute.rootLoop.get());
        for (size_t i = 0; i + 1 < currentOrder.size(); ++i) {
            if (targetPositions[currentOrder[i]] <= targetPositions[currentOrder[i + 1]]) {
                continue;
            }
            if (!interchangeAdjacentPair(slots[i])) {
                continue;
            }
            changed = true;
            progress = true;
            break;
        }
    }

    compute.optimizations.interchangeOriginalOrder = initialOrder;
    compute.optimizations.interchangeRequestedOrder = targetOrder;
    compute.optimizations.interchangeFinalOrder = collectLoopOrder(compute.rootLoop.get());
    return changed && compute.optimizations.interchangeFinalOrder == targetOrder;
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
        interchangeAdjacentPair(&compute.rootLoop)) {
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

    if (config.enable2DBlocking && config.block2DTargets.size() == 2) {
        if (config.block2DTargets[0] == config.block2DTargets[1]) {
            return;
        }
        const Loop* firstTarget = findLoopByName(compute.rootLoop.get(), config.block2DTargets[0]);
        const Loop* secondTarget = findLoopByName(compute.rootLoop.get(), config.block2DTargets[1]);
        if (!isBlockableTarget(firstTarget, true) || !isBlockableTarget(secondTarget, true)) {
            return;
        }
        if (!blockTargetByIndex(compute, blockSize, config.block2DTargets[0], true)) return;
        if (!blockTargetByIndex(compute, bs2, config.block2DTargets[1], true)) return;
        compute.optimizations.blockingApplied = true;
        compute.optimizations.blocking2DApplied = true;
        compute.optimizations.blockSize = blockSize;
        compute.optimizations.tiledIndex = config.block2DTargets[0];
        compute.optimizations.tiledIndices = config.block2DTargets;
        compute.optimizations.blockSizes = {blockSize, bs2};
        return;
    }

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

void applyPositionBlocking(Compute& compute, const OptConfig& config) {
    if (!config.enablePositionBlocking || compute.optimizations.positionBlockingApplied || !compute.rootLoop) {
        return;
    }

    std::vector<std::string> targets = config.positionBlockTargets;
    if (targets.empty()) {
        collectSparseLoopNames(compute.rootLoop.get(), targets);
    }

    const std::vector<std::string>& explicit2DTargets = config.block2DTargets;
    std::vector<std::string> appliedTargets;
    for (const auto& target : targets) {
        if (std::find(appliedTargets.begin(), appliedTargets.end(), target) != appliedTargets.end()) {
            continue;
        }
        if (std::find(explicit2DTargets.begin(), explicit2DTargets.end(), target) != explicit2DTargets.end()) {
            continue;
        }
        if (blockPositionTargetByIndex(compute, config.positionBlockSize, target)) {
            appliedTargets.push_back(target);
        }
    }

    if (!appliedTargets.empty()) {
        compute.optimizations.positionBlockingApplied = true;
        compute.optimizations.positionBlockSize = config.positionBlockSize;
        compute.optimizations.positionTiledIndices = std::move(appliedTargets);
    }
}

void applyLoopInterchange(Compute& compute, const OptConfig& config) {
    if (!config.enableInterchange || !compute.rootLoop || compute.rootLoop->children.empty()) {
        return;
    }
    if (!config.interchangeTargetOrder.empty()) {
        if (reorderTowardTarget(compute, config.interchangeTargetOrder)) {
            compute.optimizations.interchangeApplied = true;
        }
        return;
    }

    compute.optimizations.interchangeOriginalOrder = collectLoopOrder(compute.rootLoop.get());
    compute.optimizations.interchangeRequestedOrder.clear();
    const bool allowBlockWrappedDenseInner =
        (config.order == OptOrder::B_THEN_I && config.enableBlocking);
    if (applyOneInterchangeDFS(compute.rootLoop.get(), allowBlockWrappedDenseInner)) {
        compute.optimizations.interchangeApplied = true;
        compute.optimizations.interchangeFinalOrder = collectLoopOrder(compute.rootLoop.get());
    }
}

void applyOptimizations(Compute& compute, const OptConfig& config) {
    applyReordering(compute);

    if (!config.enableInterchange && !config.enableBlocking && !config.enablePositionBlocking) {
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

    if (config.enablePositionBlocking) {
        applyPositionBlocking(compute, config);
    }
}

void applyOptimizations(Program& program, const OptConfig& config) {
    for (auto& stmt : program.statements) {
        applyOptimizationsToStmt(*stmt, config);
    }
}

} // namespace opt
