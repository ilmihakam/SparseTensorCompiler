/**
 * Code Generator Implementation
 *
 * Generates complete, standalone C programs from optimized IR.
 * Supports optimization configurations:
 * 1. Baseline (no optimizations)
 * 2. Loop interchange only
 * 3. Loop blocking only
 * 4. Interchange + blocking (scheduled)
 */

#include "codegen.h"
#include "ir.h"
#include "optimizations.h"
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace codegen {

namespace {

bool hasNestedSparseLoops(const sparseir::scheduled::Loop* loop) {
    if (!loop) return false;
    if (loop->kind == sparseir::scheduled::LoopKind::Sparse) {
        for (const auto& child : loop->children) {
            if (child && child->kind == sparseir::scheduled::LoopKind::Sparse) {
                return true;
            }
            if (hasNestedSparseLoops(child.get())) {
                return true;
            }
        }
    }
    for (const auto& child : loop->children) {
        if (hasNestedSparseLoops(child.get())) {
            return true;
        }
    }
    return false;
}

bool isExplicitRuntimeParam(const std::string& bound) {
    if (bound.empty()) return false;
    if (bound.find("->") != std::string::npos) return false;
    return std::find_if(bound.begin(), bound.end(), [](unsigned char c) {
               return !std::isdigit(c);
           }) != bound.end();
}

void collectLoopBounds(const sparseir::scheduled::Loop* loop,
                       std::unordered_map<std::string, std::string>& bounds,
                       std::vector<std::string>& order) {
    if (!loop) return;
    if (!loop->runtimeBound.empty() && !bounds.count(loop->indexName)) {
        bounds.emplace(loop->indexName, loop->runtimeBound);
        order.push_back(loop->indexName);
    }
    for (const auto& child : loop->children) {
        collectLoopBounds(child.get(), bounds, order);
    }
}

int inferDefaultDimensionValue(const sparseir::scheduled::Compute& compute,
                               const std::string& indexName) {
    for (const auto& tensor : compute.inputs) {
        for (size_t pos = 0; pos < tensor.indices.size() && pos < tensor.dims.size(); ++pos) {
            if (tensor.indices[pos] == indexName && tensor.dims[pos] > 0) {
                return tensor.dims[pos];
            }
        }
    }
    for (size_t pos = 0; pos < compute.output.indices.size() && pos < compute.output.dims.size(); ++pos) {
        if (compute.output.indices[pos] == indexName && compute.output.dims[pos] > 0) {
            return compute.output.dims[pos];
        }
    }
    return 0;
}

EmissionContext buildEmissionContext(const sparseir::scheduled::Compute& compute) {
    EmissionContext context;
    for (const auto& t : compute.inputs) {
        if (t.format != ir::Format::Dense) context.sparseInputs.push_back(t);
        else                               context.denseInputs.push_back(t);
    }
    context.output = compute.output;
    context.outputStrategy = compute.outputStrategy;
    context.outputPattern = compute.outputPattern;
    context.hasNestedSparseTraversal = hasNestedSparseLoops(compute.rootLoop.get());

    for (const auto& tensor : context.sparseInputs) {
        if (tensor.format == ir::Format::CSR || tensor.format == ir::Format::CSC) {
            context.primarySparseFormat = tensor.format;
            break;
        }
    }
    if (context.sparseInputs.empty() &&
        (compute.output.format == ir::Format::CSR || compute.output.format == ir::Format::CSC)) {
        context.primarySparseFormat = compute.output.format;
    }

    std::vector<std::string> boundOrder;
    collectLoopBounds(compute.rootLoop.get(), context.indexBounds, boundOrder);

    std::vector<std::string> allIndices;
    auto rememberIndex = [&allIndices](const std::string& indexName) {
        if (std::find(allIndices.begin(), allIndices.end(), indexName) == allIndices.end()) {
            allIndices.push_back(indexName);
        }
    };
    for (const auto& tensor : compute.inputs) {
        for (const auto& indexName : tensor.indices) {
            rememberIndex(indexName);
        }
    }
    for (const auto& indexName : compute.output.indices) {
        rememberIndex(indexName);
    }

    for (const auto& indexName : allIndices) {
        if (context.indexBounds.count(indexName)) {
            continue;
        }

        bool assigned = false;
        for (const auto& tensor : compute.inputs) {
            if (tensor.format == ir::Format::Dense) {
                continue;
            }
            for (size_t pos = 0; pos < tensor.indices.size(); ++pos) {
                if (tensor.indices[pos] != indexName) {
                    continue;
                }
                context.indexBounds[indexName] =
                    tensor.name + (pos == 0 ? "->rows" : "->cols");
                assigned = true;
                break;
            }
            if (assigned) {
                break;
            }
        }

        if (!assigned) {
            for (size_t pos = 0; pos < compute.output.indices.size(); ++pos) {
                if (compute.output.indices[pos] != indexName) {
                    continue;
                }
                if (compute.output.format == ir::Format::CSR ||
                    compute.output.format == ir::Format::CSC) {
                    context.indexBounds[indexName] =
                        compute.output.name + (pos == 0 ? "->rows" : "->cols");
                } else {
                    context.indexBounds[indexName] = "N_" + indexName;
                }
                assigned = true;
                break;
            }
        }

        if (!assigned) {
            context.indexBounds[indexName] = "N_" + indexName;
        }
    }

    std::unordered_set<std::string> seenParams;
    for (const auto& indexName : allIndices) {
        auto it = context.indexBounds.find(indexName);
        if (it == context.indexBounds.end() || !isExplicitRuntimeParam(it->second)) {
            continue;
        }
        if (!seenParams.insert(it->second).second) {
            continue;
        }
        context.explicitDimensions.push_back(
            RuntimeDimension{indexName, it->second, inferDefaultDimensionValue(compute, indexName)});
    }

    return context;
}

struct LoopEmissionState {
    std::unordered_map<std::string, std::pair<std::string, std::string>> denseBounds;
    std::unordered_map<std::string, std::pair<std::string, std::string>> sparseIteratorBounds;
    std::unordered_map<std::string, int> mergeChunkSizes;
};

} // namespace

// ============================================================================
// CodeGenerator Implementation
// ============================================================================

CodeGenerator::CodeGenerator(std::ostream& out)
    : out_(out)
    , indent_(0)
    , currentScheduledCompute_(nullptr)
    , config_(nullptr)
{
}

void CodeGenerator::prepareComputeEmission(const sparseir::scheduled::Compute& compute,
                                           const opt::OptConfig& config) {
    currentScheduledCompute_ = &compute;
    config_ = &config;
    context_ = buildEmissionContext(compute);
    indent_ = 0;
}

void CodeGenerator::resetComputeEmission() {
    currentScheduledCompute_ = nullptr;
    config_ = nullptr;
    context_ = EmissionContext{};
    indent_ = 0;
}

// ----------------------------------------------------------------------------
// Indentation Management
// ----------------------------------------------------------------------------

void CodeGenerator::increaseIndent() {
    indent_++;
}

void CodeGenerator::decreaseIndent() {
    if (indent_ > 0) {
        indent_--;
    }
}

void CodeGenerator::emitIndent() {
    for (int i = 0; i < indent_; i++) {
        out_ << "    ";  // 4 spaces per indent level
    }
}

// ----------------------------------------------------------------------------
// Emission Helpers
// ----------------------------------------------------------------------------

void CodeGenerator::emit(const std::string& code) {
    out_ << code;
}

void CodeGenerator::emitLine(const std::string& code) {
    emitIndent();
    out_ << code << std::endl;
}

void CodeGenerator::emitLine() {
    out_ << std::endl;
}

// ----------------------------------------------------------------------------
// Main Entry Point
// ----------------------------------------------------------------------------

void CodeGenerator::generateKernel(const sparseir::scheduled::Compute& compute,
                                   const opt::OptConfig& config) {
    prepareComputeEmission(compute, config);

    emitLine("void " + getKernelSignature() + " {");
    increaseIndent();
    emitScheduledComputeBody(compute, true);
    decreaseIndent();
    emitLine("}");
    emitLine();

    resetComputeEmission();
}

void CodeGenerator::emitInlineScheduledCompute(
    const sparseir::scheduled::Compute& compute,
    int indentLevel) {
    const sparseir::scheduled::Compute* previousScheduled = currentScheduledCompute_;
    const opt::OptConfig* previousConfig = config_;
    EmissionContext previousContext = context_;
    int previousIndent = indent_;

    currentScheduledCompute_ = &compute;
    context_ = buildEmissionContext(compute);
    indent_ = indentLevel;

    emitScheduledComputeBody(compute, true);

    currentScheduledCompute_ = previousScheduled;
    config_ = previousConfig;
    context_ = std::move(previousContext);
    indent_ = previousIndent;
}

void CodeGenerator::emitKernel() {
    emitLine("// ============================================");
    emitLine("// Optimized Kernel");
    emitLine("// ============================================");
    emitLine();

    if (isSparseOutputMode()) {
        emitSparseAssembly();
    }

    std::string signature = getKernelSignature();
    emitLine("void " + signature + " {");
    increaseIndent();
    if (currentScheduledCompute_) {
        emitScheduledComputeBody(*currentScheduledCompute_, true);
    }
    decreaseIndent();
    emitLine("}");
    emitLine();
}

void CodeGenerator::emitVerification() {
    // Verification is part of main function, not a separate section
    // Keeping this as placeholder for now
}

// ----------------------------------------------------------------------------
// Loop Generation Methods (Stubs - will implement in Phase 3)
// ----------------------------------------------------------------------------

void CodeGenerator::emitScheduledLoop(const sparseir::scheduled::Loop& loop) {
    LoopEmissionState initialState;

    std::function<void(const sparseir::scheduled::Loop&, LoopEmissionState&)> emitLoop =
        [&](const sparseir::scheduled::Loop& current, LoopEmissionState& state) {
            auto emitLoopBody = [&](LoopEmissionState& bodyState) {
                for (const auto& stmt : current.preStmts) {
                    emitIRStmt(*stmt);
                }
                for (const auto& child : current.children) {
                    emitLoop(*child, bodyState);
                }
                for (const auto& stmt : current.postStmts) {
                    emitIRStmt(*stmt);
                }
            };

            if (current.headerKind == sparseir::scheduled::LoopHeaderKind::Block) {
                const auto& block = current.block;
                if (block.targetKind == sparseir::scheduled::BlockTargetKind::DenseIndex) {
                    const std::string& blockVar = block.blockVar;
                    const std::string& startVar = block.startVar;
                    const std::string& endVar = block.endVar;
                    const int blockSize = block.blockSize;

                    emitIndent();
                    out_ << "for (int " << blockVar << " = 0; " << blockVar
                         << " < " << block.tripCountExpr << "; " << blockVar << "++) {" << std::endl;
                    increaseIndent();

                    emitLine("int " + startVar + " = " + blockVar + " * " + std::to_string(blockSize) + ";");
                    emitLine("int " + endVar + " = (" + startVar + " + " + std::to_string(blockSize) +
                             " < " + current.runtimeBound + ") ? " + startVar + " + " +
                             std::to_string(blockSize) + " : " + current.runtimeBound + ";");

                    LoopEmissionState childState = state;
                    childState.denseBounds[block.innerIndexName] = {
                        block.innerLowerExpr,
                        block.innerUpperExpr,
                    };
                    emitLoopBody(childState);

                    decreaseIndent();
                    emitIndent();
                    out_ << "}" << std::endl;
                } else if (block.targetKind == sparseir::scheduled::BlockTargetKind::SparseIteratorPosition) {
                    const std::string& blockVar = block.blockVar;
                    const std::string& startVar = block.startVar;
                    const std::string& endVar = block.endVar;
                    const int blockSize = block.blockSize;

                    emitIndent();
                    out_ << "for (int " << blockVar << " = 0; " << blockVar
                         << " < " << block.tripCountExpr << "; " << blockVar << "++) {" << std::endl;
                    increaseIndent();

                    emitLine("int " + startVar + " = (" + block.sparseBeginExpr + ") + " + blockVar + " * " +
                             std::to_string(blockSize) + ";");
                    emitLine("int " + endVar + " = (" + startVar + " + " + std::to_string(blockSize) +
                             " < " + block.sparseEndExpr + ") ? " + startVar + " + " +
                             std::to_string(blockSize) + " : " + block.sparseEndExpr + ";");

                    LoopEmissionState childState = state;
                    childState.sparseIteratorBounds[block.innerIndexName] = {
                        block.innerLowerExpr,
                        block.innerUpperExpr,
                    };
                    emitLoopBody(childState);

                    decreaseIndent();
                    emitIndent();
                    out_ << "}" << std::endl;
                } else {
                    LoopEmissionState childState = state;
                    childState.mergeChunkSizes[block.innerIndexName] = block.blockSize;
                    emitLoopBody(childState);
                }
                return;
            }

            if (current.headerKind == sparseir::scheduled::LoopHeaderKind::DenseFor) {
                std::string lowerBound = current.lowerExpr.empty()
                    ? std::to_string(current.lower)
                    : current.lowerExpr;
                std::string upperBound = current.upperExpr.empty()
                    ? (current.runtimeBound.empty()
                        ? std::to_string(current.upper)
                        : current.runtimeBound)
                    : current.upperExpr;

                auto boundsIt = state.denseBounds.find(current.indexName);
                if (boundsIt != state.denseBounds.end()) {
                    lowerBound = boundsIt->second.first;
                    upperBound = boundsIt->second.second;
                }

                const bool emitsHeader = !current.isExternallyBound;
                if (emitsHeader) {
                    emitIndent();
                    out_ << "for (int " << current.indexName << " = " << lowerBound
                         << "; " << current.indexName << " < " << upperBound
                         << "; " << current.indexName << "++) {" << std::endl;
                    increaseIndent();
                }

                emitLoopBody(state);

                if (emitsHeader) {
                    decreaseIndent();
                    emitIndent();
                    out_ << "}" << std::endl;
                }
                return;
            }

            if (current.headerKind == sparseir::scheduled::LoopHeaderKind::SparseMerge) {
                for (const auto& term : current.merge.terms) {
                    emitLine("int " + term.pointerVar + " = " + term.beginExpr + ";");
                    emitLine("int " + term.endVar + " = " + term.endExpr + ";");
                }

                std::string cond;
                const char* join =
                    (current.merge.strategy == ir::MergeStrategy::Union) ? " || " : " && ";
                for (size_t i = 0; i < current.merge.terms.size(); ++i) {
                    if (i > 0) cond += join;
                    cond += current.merge.terms[i].pointerVar + " < " + current.merge.terms[i].endVar;
                }
                const auto chunkIt = state.mergeChunkSizes.find(current.indexName);
                const bool chunked = chunkIt != state.mergeChunkSizes.end();
                const std::string chunkCounter = current.indexName + "_chunk_steps";
                emitLine("while (" + cond + ") {");
                increaseIndent();
                if (chunked) {
                    emitLine("int " + chunkCounter + " = 0;");
                    emitLine("while ((" + cond + ") && " + chunkCounter + " < " +
                             std::to_string(chunkIt->second) + ") {");
                    increaseIndent();
                }

                if (current.merge.strategy == ir::MergeStrategy::Intersection) {
                    for (const auto& term : current.merge.terms) {
                        emitLine("int " + term.boundIndexVar + " = " + term.candidateExpr + ";");
                    }
                    const std::string firstIdx = current.merge.terms.front().boundIndexVar;
                    emitLine("int min_idx = " + firstIdx + ";");
                    emitLine("int max_idx = " + firstIdx + ";");
                    for (size_t i = 1; i < current.merge.terms.size(); ++i) {
                        const std::string& idxVar = current.merge.terms[i].boundIndexVar;
                        emitLine("if (" + idxVar + " < min_idx) min_idx = " + idxVar + ";");
                        emitLine("if (" + idxVar + " > max_idx) max_idx = " + idxVar + ";");
                    }
                    emitLine("if (min_idx == max_idx) {");
                    increaseIndent();
                    emitLine("int " + current.indexName + " = min_idx;");
                    emitLoopBody(state);
                    for (const auto& term : current.merge.terms) {
                        emitLine(term.advanceOnMatchStmt);
                    }
                    decreaseIndent();
                    emitLine("} else {");
                    increaseIndent();
                    for (const auto& term : current.merge.terms) {
                        emitLine(term.advanceIfLessThanMaxStmt);
                    }
                    decreaseIndent();
                    emitLine("}");
                } else {
                    emitLine("int has_idx = 0;");
                    emitLine("int " + current.indexName + " = 0;");
                    for (const auto& term : current.merge.terms) {
                        emitLine("if (" + term.pointerVar + " < " + term.endVar + ") {");
                        increaseIndent();
                        emitLine("int candidate = " + term.candidateExpr + ";");
                        emitLine("if (!has_idx || candidate < " + current.indexName + ") {");
                        increaseIndent();
                        emitLine(current.indexName + " = candidate;");
                        emitLine("has_idx = 1;");
                        decreaseIndent();
                        emitLine("}");
                        decreaseIndent();
                        emitLine("}");
                    }
                    emitLoopBody(state);
                    for (const auto& term : current.merge.terms) {
                        emitLine("if (" + term.pointerVar + " < " + term.endVar +
                                 " && " + term.matchExpr + ") " + term.advanceOnMatchStmt);
                    }
                }

                if (chunked) {
                    emitLine(chunkCounter + "++;");
                    decreaseIndent();
                    emitLine("}");
                }

                decreaseIndent();
                emitLine("}");
                return;
            }

            const std::string ptrVar = current.iterator.pointerVar;
            std::string beginExpr = current.iterator.beginExpr;
            std::string endExpr = current.iterator.endExpr;
            auto sparseBoundsIt = state.sparseIteratorBounds.find(current.indexName);
            if (sparseBoundsIt != state.sparseIteratorBounds.end()) {
                beginExpr = sparseBoundsIt->second.first;
                endExpr = sparseBoundsIt->second.second;
            }

            emitIndent();
            out_ << "for (int " << ptrVar << " = " << beginExpr << "; "
                 << ptrVar << " < " << endExpr << "; " << ptrVar
                 << "++) {" << std::endl;
            increaseIndent();

            emitLine("int " + current.bindingVarName + " = " + current.bindingExpr + ";");
            emitLoopBody(state);

            decreaseIndent();
            emitIndent();
            out_ << "}" << std::endl;
        };

    emitLoop(loop, initialState);
}

// ----------------------------------------------------------------------------
// Visitor Pattern Implementation (Stubs - will implement in Phase 6)
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Helper Methods (Stubs - will implement as needed)
// ----------------------------------------------------------------------------

std::string CodeGenerator::getComputeFunctionName() const {
    return "compute";
}

std::string CodeGenerator::getReferenceFunctionName() const {
    return "reference";
}

std::string CodeGenerator::getAssemblyFunctionName() const {
    return "assemble_output";
}

const std::vector<ir::Tensor>& CodeGenerator::getActiveInputs() const {
    static const std::vector<ir::Tensor> emptyInputs;
    return currentScheduledCompute_ ? currentScheduledCompute_->inputs : emptyInputs;
}

const ir::Tensor& CodeGenerator::getActiveOutput() const {
    static const ir::Tensor emptyOutput{};
    return currentScheduledCompute_ ? currentScheduledCompute_->output : emptyOutput;
}

const ir::LoopOptimizations& CodeGenerator::getActiveOptimizations() const {
    static const ir::LoopOptimizations emptyOptimizations{};
    return currentScheduledCompute_ ? currentScheduledCompute_->optimizations : emptyOptimizations;
}

const std::vector<std::unique_ptr<ir::IRStmt>>& CodeGenerator::getActivePrologueStmts() const {
    static const std::vector<std::unique_ptr<ir::IRStmt>> emptyStmts;
    return currentScheduledCompute_ ? currentScheduledCompute_->prologueStmts : emptyStmts;
}

const std::vector<std::unique_ptr<ir::IRStmt>>& CodeGenerator::getActiveEpilogueStmts() const {
    static const std::vector<std::unique_ptr<ir::IRStmt>> emptyStmts;
    return currentScheduledCompute_ ? currentScheduledCompute_->epilogueStmts : emptyStmts;
}

ir::Format CodeGenerator::getTensorFormat(const std::string& tensorName) const {
    if (tensorName.empty()) {
        return ir::Format::CSR;
    }

    const auto& output = getActiveOutput();
    if (!output.name.empty() && tensorName == output.name) {
        return output.format;
    }

    for (const auto& tensor : getActiveInputs()) {
        if (tensor.name == tensorName) {
            return tensor.format;
        }
    }

    return ir::Format::CSR;
}

std::string CodeGenerator::getKernelSignature() const {
    return getScheduledKernelSignature();
}

std::string CodeGenerator::getScheduledKernelSignature() const {
    return buildStructuralSignature(getComputeFunctionName(), false);
}

std::string CodeGenerator::getReferenceSignature() const {
    return getScheduledReferenceSignature();
}

std::string CodeGenerator::getScheduledReferenceSignature() const {
    return buildStructuralSignature(getReferenceFunctionName(), true);
}

std::string CodeGenerator::getIndexBoundExpr(const std::string& indexName) const {
    auto it = context_.indexBounds.find(indexName);
    if (it != context_.indexBounds.end()) {
        return it->second;
    }
    return "N_" + indexName;
}

int CodeGenerator::getDefaultDimensionValue(const std::string& indexName) const {
    for (const auto& dim : context_.explicitDimensions) {
        if (dim.indexName == indexName) {
            return dim.defaultValue;
        }
    }
    return 0;
}

bool CodeGenerator::usesSampledDenseTraversal() const {
    return context_.sparseInputs.size() == 1 &&
           context_.denseInputs.size() == 2 &&
           context_.output.indices.size() == 2;
}

bool CodeGenerator::usesPairwiseSparseInputFlow() const {
    return context_.sparseInputs.size() == 2;
}

bool CodeGenerator::usesNestedSparseProductFlow() const {
    return context_.hasNestedSparseTraversal ||
           context_.outputStrategy == ir::OutputStrategy::HashPerRow;
}

std::string CodeGenerator::buildStructuralSignature(const std::string& funcName,
                                                    bool forReference) const {
    const auto& inputs = getActiveInputs();
    const auto& output = getActiveOutput();
    if (inputs.empty() && output.name.empty()) return funcName + "()";

    std::vector<std::string> params;
    std::set<std::string> seenTensors;

    for (const auto& tensor : inputs) {
        if (seenTensors.count(tensor.name)) continue;
        seenTensors.insert(tensor.name);

        if (tensor.format != ir::Format::Dense) {
            // Sparse matrix
            params.push_back("const SparseMatrix* " + tensor.name);
        } else if (tensor.indices.size() >= 2) {
            // Dense 2D matrix
            params.push_back("const double** " + tensor.name);
        } else {
            // Dense 1D vector
            params.push_back("const double* " + tensor.name);
        }
    }

    if (!output.name.empty() && !seenTensors.count(output.name)) {
        if (context_.hasSparseOutput()) {
            params.push_back((forReference ? "const SparseMatrix* " : "SparseMatrix* ") + output.name);
            if (forReference) {
                params.push_back("double* " + output.name + "_ref_vals");
            }
        } else if (output.indices.size() >= 2) {
            params.push_back("double** " + output.name);
        } else {
            params.push_back("double* " + output.name);
        }
    }

    for (const auto& dim : context_.explicitDimensions) {
        params.push_back("int " + dim.paramName);
    }

    std::string result = funcName + "(";
    for (size_t i = 0; i < params.size(); i++) {
        if (i > 0) result += ", ";
        result += params[i];
    }
    result += ")";
    return result;
}

void CodeGenerator::emitScheduledComputeBody(const sparseir::scheduled::Compute& compute,
                                             bool emitOptimizationComments) {
    if (emitOptimizationComments) {
        const auto& optimizations = compute.optimizations;
        if (optimizations.reorderingApplied) {
            emitLine("// Format-correctness reordering applied");
        }
        if (optimizations.interchangeApplied) {
            emitLine("// Optimization: loop interchange applied");
        }
        if (optimizations.blockingApplied) {
            emitLine("// Optimization: loop blocking (block_size=" +
                     std::to_string(optimizations.blockSize) + ")");
        }
        if (optimizations.positionBlockingApplied) {
            emitLine("// Optimization: sparse position blocking (block_size=" +
                     std::to_string(optimizations.positionBlockSize) + ")");
        }
    }

    for (const auto& stmt : compute.prologueStmts) {
        emitIRStmt(*stmt);
    }

    if (compute.rootLoop) {
        emitScheduledLoop(*compute.rootLoop);
    }

    for (const auto& stmt : compute.epilogueStmts) {
        emitIRStmt(*stmt);
    }
}

// (Dedicated kernel emitters removed — visitor path handles all kernels)

// ============================================================================
// IRExprEmitter Implementation
// ============================================================================

void IRExprEmitter::visit(const ir::IRTensorAccess& n) {
    if (n.isSparseVals) {
        // Codegen-ready: use -> accessor for sparse matrix struct
        result = n.tensorName + "->vals[" + n.pointerVar + "]";
    } else if (n.useRandomAccess && n.indices.size() == 2) {
        std::string func = n.randomAccessFunc.empty() ? "sp_csr_get" : n.randomAccessFunc;
        result = func + "(" + n.tensorName + ", " + n.indices[0] + ", " + n.indices[1] + ")";
    } else if (n.indices.size() == 1) {
        result = n.tensorName + "[" + n.indices[0] + "]";
    } else if (n.indices.size() == 2) {
        result = n.tensorName + "[" + n.indices[0] + "][" + n.indices[1] + "]";
    } else if (n.indices.empty()) {
        result = n.tensorName;
    } else {
        // 3+ indices: chain brackets
        result = n.tensorName;
        for (const auto& idx : n.indices) {
            result += "[" + idx + "]";
        }
    }
}

void IRExprEmitter::visit(const ir::IRConstant& n) {
    if (n.value == std::floor(n.value) && std::abs(n.value) < 1e15) {
        std::ostringstream oss;
        oss << static_cast<long long>(n.value);
        result = oss.str();
    } else {
        std::ostringstream oss;
        oss << n.value;
        result = oss.str();
    }
}

void IRExprEmitter::visit(const ir::IRBinaryOp& n) {
    IRExprEmitter lhsE, rhsE;
    n.lhs->accept(lhsE);
    n.rhs->accept(rhsE);
    std::string opStr = (n.op == ir::IRBinaryOp::ADD) ? " + " : " * ";
    result = lhsE.result + opStr + rhsE.result;
}

void IRExprEmitter::visit(const ir::IRScalarVar& n) {
    result = n.name;
}

void IRExprEmitter::visit(const ir::IRFuncCall& n) {
    // Inline known activation functions
    if (n.name == "relu" && n.args.size() == 1) {
        IRExprEmitter argE;
        n.args[0]->accept(argE);
        result = "((" + argE.result + ") > 0 ? (" + argE.result + ") : 0)";
        return;
    }
    if (n.name == "sigmoid" && n.args.size() == 1) {
        IRExprEmitter argE;
        n.args[0]->accept(argE);
        result = "(1.0 / (1.0 + exp(-(" + argE.result + "))))";
        return;
    }

    // Unknown function: emit as direct call
    result = n.name + "(";
    for (size_t i = 0; i < n.args.size(); i++) {
        if (i > 0) result += ", ";
        IRExprEmitter argE;
        n.args[i]->accept(argE);
        result += argE.result;
    }
    result += ")";
}

void IRExprEmitter::visit(const ir::IRIndexedAccess& n) {
    result = n.baseName;
    for (const auto& index : n.indices) {
        IRExprEmitter indexEmitter;
        index->accept(indexEmitter);
        result += "[" + indexEmitter.result + "]";
    }
}

void IRExprEmitter::visit(const ir::IRCompareExpr& n) {
    IRExprEmitter lhsE, rhsE;
    n.lhs->accept(lhsE);
    n.rhs->accept(rhsE);
    const char* op = (n.op == ir::IRCompareExpr::EQ) ? " == " : " < ";
    result = lhsE.result + op + rhsE.result;
}

void IRExprEmitter::visit(const ir::IRAccumulatorRef& n) {
    result = n.name;
}

// ============================================================================
// Structured IR Statement Emission
// ============================================================================

std::string CodeGenerator::emitIRExpr(const ir::IRExpr& expr) {
    IRExprEmitter emitter;
    const_cast<ir::IRExpr&>(expr).accept(emitter);
    return emitter.result;
}

void CodeGenerator::emitIRStmt(const ir::IRStmt& stmt) {
    if (auto* decl = dynamic_cast<const ir::IRScalarDecl*>(&stmt)) {
        std::ostringstream oss;
        oss << "double " << decl->varName << " = ";
        if (decl->initValue == 0.0) {
            oss << "0.0";
        } else {
            oss << decl->initValue;
        }
        oss << ";";
        emitLine(oss.str());
        return;
    }
    if (auto* init = dynamic_cast<const ir::IRAccumulatorInit*>(&stmt)) {
        std::ostringstream oss;
        oss << "double " << init->accumulatorName << " = ";
        if (init->initValue == 0.0) {
            oss << "0.0";
        } else {
            oss << init->initValue;
        }
        oss << ";";
        emitLine(oss.str());
        return;
    }
    if (auto* assign = dynamic_cast<const ir::IRAssign*>(&stmt)) {
        std::string lhsStr = emitIRExpr(*assign->lhs);
        std::string rhsStr = emitIRExpr(*assign->rhs);
        std::string op = assign->accumulate ? " += " : " = ";
        emitLine(lhsStr + op + rhsStr + ";");
        return;
    }
    if (auto* update = dynamic_cast<const ir::IRAccumulatorUpdate*>(&stmt)) {
        emitLine(update->accumulatorName + " += " + emitIRExpr(*update->rhs) + ";");
        return;
    }
    if (auto* call = dynamic_cast<const ir::IRCallStmt*>(&stmt)) {
        std::string result = call->functionName + "(";
        for (size_t i = 0; i < call->args.size(); i++) {
            if (i > 0) result += ", ";
            result += emitIRExpr(*call->args[i]);
        }
        result += ");";
        emitLine(result);
        return;
    }
    if (auto* finalize = dynamic_cast<const ir::IRAccumulatorFinalize*>(&stmt)) {
        emitLine(emitIRExpr(*finalize->lhs) + " = " + emitIRExpr(*finalize->rhs) + ";");
        return;
    }
    if (auto* raw = dynamic_cast<const ir::IRRawStmt*>(&stmt)) {
        std::stringstream lines(raw->code);
        std::string line;
        while (std::getline(lines, line)) {
            emitLine(line);
        }
        return;
    }
    if (auto* varDecl = dynamic_cast<const ir::IRVarDecl*>(&stmt)) {
        emitLine(varDecl->type + " " + varDecl->varName + " = " + varDecl->initExpr + ";");
        return;
    }
    if (auto* freeStmt = dynamic_cast<const ir::IRFreeStmt*>(&stmt)) {
        emitLine("free(" + freeStmt->varName + ");");
        return;
    }
    if (auto* ifStmt = dynamic_cast<const ir::IRIfStmt*>(&stmt)) {
        emitLine("if (" + emitIRExpr(*ifStmt->condition) + ") {");
        increaseIndent();
        for (const auto& bodyStmt : ifStmt->thenBody) {
            emitIRStmt(*bodyStmt);
        }
        decreaseIndent();
        emitLine("}");
        return;
    }
    if (auto* forStmt = dynamic_cast<const ir::IRForStmt*>(&stmt)) {
        emitLine(
            "for (int " + forStmt->loopVar + " = " + emitIRExpr(*forStmt->lower) +
            "; " + forStmt->loopVar + " < " + emitIRExpr(*forStmt->upper) +
            "; " + forStmt->loopVar + "++) {");
        increaseIndent();
        for (const auto& bodyStmt : forStmt->body) {
            emitIRStmt(*bodyStmt);
        }
        decreaseIndent();
        emitLine("}");
        return;
    }
    emitLine("/* unknown stmt */");
}

// ============================================================================
// Public API Functions
// ============================================================================

std::string generateKernelCode(const sparseir::scheduled::Compute& compute,
                               const opt::OptConfig& config) {
    std::ostringstream oss;
    CodeGenerator gen(oss);
    gen.generateKernel(compute, config);
    return oss.str();
}

bool generateKernelToFile(const sparseir::scheduled::Compute& compute,
                          const opt::OptConfig& config,
                          const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    CodeGenerator gen(file);
    gen.generateKernel(compute, config);
    file.close();
    return true;
}

} // namespace codegen
