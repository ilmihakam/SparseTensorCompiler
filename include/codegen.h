#pragma once

#include <string>
#include <unordered_map>
#include <ostream>
#include <sstream>
#include "ir.h"
#include "optimizations.h"
#include "semantic_ir.h"

namespace codegen {

struct RuntimeDimension {
    std::string indexName;
    std::string paramName;
    int defaultValue = 0;
};

/**
 * EmissionContext: structural metadata extracted once from scheduled IR.
 *
 * This is descriptive only. It records tensor inventory, output behavior,
 * loop-owned runtime bounds, and explicit runtime parameters needed by
 * signatures/reference/main emission.
 */
struct EmissionContext {
    std::vector<ir::Tensor> sparseInputs;
    std::vector<ir::Tensor> denseInputs;
    ir::Tensor output;
    ir::OutputStrategy outputStrategy = ir::OutputStrategy::DenseArray;
    sparseir::OutputPatternKind outputPattern = sparseir::OutputPatternKind::None;
    ir::Format primarySparseFormat = ir::Format::CSR;
    std::unordered_map<std::string, std::string> indexBounds;
    std::vector<RuntimeDimension> explicitDimensions;
    bool hasNestedSparseTraversal = false;

    bool hasSparseOutput() const {
        return outputStrategy == ir::OutputStrategy::SparseFixedPattern ||
               outputStrategy == ir::OutputStrategy::HashPerRow;
    }

    bool needsSeparateAssembly() const {
        return hasSparseOutput();
    }
};

/**
 * IRExprEmitter: Visitor that emits C code directly from structured IR expressions.
 *
 * Unlike ExprRenderer in ir.cpp (which uses `_vals[` placeholder notation),
 * this emitter produces final codegen-ready C with `->vals[` accessor syntax.
 * It also inlines known functions (relu, sigmoid) as C ternary/expressions.
 */
class IRExprEmitter : public ir::IRExprVisitor {
public:
    std::string result;

    void visit(const ir::IRTensorAccess& n) override;
    void visit(const ir::IRConstant& n) override;
    void visit(const ir::IRBinaryOp& n) override;
    void visit(const ir::IRScalarVar& n) override;
    void visit(const ir::IRFuncCall& n) override;
    void visit(const ir::IRIndexedAccess& n) override;
    void visit(const ir::IRCompareExpr& n) override;
    void visit(const ir::IRAccumulatorRef& n) override;
};

/**
 * Code Generator for Sparse Tensor Compiler.
 *
 * Generates complete, standalone C programs from IR with optional
 * optimizations applied. Supports all 4 benchmark configurations:
 * - Baseline (no optimizations)
 * - Interchange only
 * - Blocking only
 * - Both optimizations
 */
class CodeGenerator {
public:
    explicit CodeGenerator(std::ostream& out);
    void generate(const sparseir::scheduled::Compute& compute, const opt::OptConfig& config);
    void generateKernel(const sparseir::scheduled::Compute& compute,
                        const opt::OptConfig& config);
    void emitInlineScheduledCompute(const sparseir::scheduled::Compute& compute,
                                    int indentLevel = 0);

    // Accessors for testing
    int getIndentLevel() const { return indent_; }

private:
    std::ostream& out_;
    int indent_ = 0;
    const sparseir::scheduled::Compute* currentScheduledCompute_ = nullptr;
    const opt::OptConfig* config_ = nullptr;
    EmissionContext context_;

    // Emission helpers
    void emit(const std::string& code);
    void emitLine(const std::string& code);
    void emitLine();  // Empty line
    void emitIndent();
    void increaseIndent();
    void decreaseIndent();
    void prepareComputeEmission(const sparseir::scheduled::Compute& compute,
                                const opt::OptConfig& config);
    void resetComputeEmission();

    // Section generators
    void emitHeader();
    void emitStructDefinitions();
    void emitMatrixMarketLoader();
    void emitCSRConversion();  // COO -> CSR converter
    void emitCSCConversion();  // COO -> CSC converter
    void emitTimingHarness();
    void emitTimingStatistics();  // Helper for timing statistics (used in main)
    void emitFeatureExtraction(); // Emits compute_features() C function for matrix characterization
    void emitKernel();
    void emitReferenceKernel();
    void emitScheduledReferenceKernel();
    void emitVerification();
    void emitMain();
    void emitScheduledMain();

    // Loop generators
    void emitScheduledLoop(const sparseir::scheduled::Loop& loop);
    void emitScheduledComputeBody(const sparseir::scheduled::Compute& compute,
                                  bool emitOptimizationComments);

    std::string getComputeFunctionName() const;
    std::string getReferenceFunctionName() const;
    std::string getAssemblyFunctionName() const;
    std::string getKernelSignature() const;
    std::string getScheduledKernelSignature() const;
    std::string getReferenceSignature() const;
    std::string getScheduledReferenceSignature() const;
    const std::vector<ir::Tensor>& getActiveInputs() const;
    const ir::Tensor& getActiveOutput() const;
    const ir::LoopOptimizations& getActiveOptimizations() const;
    const std::vector<std::unique_ptr<ir::IRStmt>>& getActivePrologueStmts() const;
    const std::vector<std::unique_ptr<ir::IRStmt>>& getActiveEpilogueStmts() const;
    ir::Format getTensorFormat(const std::string& tensorName) const;

    // (Dedicated kernel emitters removed — visitor path handles all kernels)

    // Dynamic sparse-output support
    void emitHashMapStructs();   // typedef HashEntry, RowHash
    void emitHashMapHelpers();   // init, accumulate, free

    // Sparse-output support — dispatched by outputPattern / outputStrategy
    bool isSparseOutputMode() const;
    ir::Format getPrimarySparseFormat() const;
    void emitSparseAccessHelpers();
    void emitSparseOutputHelpers();
    // Assembly helpers (behavior-oriented names)
    void emitUnionAssembly();
    void emitIntersectionAssembly();
    void emitDynamicRowAssembly();
    void emitSampledAssembly();
    void emitSparseAssembly();
    // Reference helpers (behavior-oriented names)
    void emitUnionSparseReference();
    void emitIntersectionSparseReference();
    void emitDynamicRowSparseReference();
    void emitSampledSparseReference();

    // Structural signature / harness helpers
    std::string buildStructuralSignature(const std::string& funcName, bool forReference) const;
    std::string getIndexBoundExpr(const std::string& indexName) const;
    int getDefaultDimensionValue(const std::string& indexName) const;
    bool usesSampledDenseTraversal() const;
    bool usesPairwiseSparseInputFlow() const;
    bool usesNestedSparseProductFlow() const;
    void emitDenseOutputMain();
    void emitPairwiseSparseOutputMain(const std::string& computeFn,
                                      const std::string& referenceFn,
                                      const std::string& assemblyFn,
                                      const std::string& formatLabel);
    void emitSampledOutputMain(const std::string& computeFn,
                               const std::string& referenceFn,
                               const std::string& assemblyFn,
                               const std::string& formatLabel);

    // Structured IR emission (Phase C expanded)
    void emitIRStmt(const ir::IRStmt& stmt);
    std::string emitIRExpr(const ir::IRExpr& expr);
};

/**
 * Convenience function to generate a full standalone C program to a string.
 *
 * This preserves the benchmark-facing output that includes runtime harness code.
 */
std::string generateCode(const sparseir::scheduled::Compute& compute,
                         const opt::OptConfig& config);

/**
 * Convenience function to generate only the compute kernel function to a string.
 *
 * The emitted text contains only the optimized compute function, with no loader,
 * reference kernel, timing harness, or main().
 */
std::string generateKernelCode(const sparseir::scheduled::Compute& compute,
                               const opt::OptConfig& config);

/**
 * Generate a full standalone C program to a file.
 */
bool generateToFile(const sparseir::scheduled::Compute& compute,
                    const opt::OptConfig& config,
                    const std::string& filename);

/**
 * Generate only the compute kernel function to a file.
 */
bool generateKernelToFile(const sparseir::scheduled::Compute& compute,
                          const opt::OptConfig& config,
                          const std::string& filename);

/**
 * Generate a complete standalone C program.
 *
 * The scheduled-program overload is canonical.
 */
bool generateProgramToFile(const sparseir::scheduled::Program& prog,
                           const opt::OptConfig& config,
                           const std::string& filename);

} // namespace codegen
