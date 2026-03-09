#include "codegen.h"
#include "output_assembly_shared.h"

#include <algorithm>

namespace codegen {

namespace {

detail::EmitIndentedLine makeCodegenLineEmitter(std::ostream& out, int indent) {
    return [&out, indent](int extra, const std::string& text) {
        for (int i = 0; i < indent + extra; ++i) {
            out << "    ";
        }
        out << text << '\n';
    };
}

template <typename BodyFn>
void emitAssemblyWrapper(std::ostream& out,
                         int& indent,
                         const std::string& signature,
                         BodyFn&& body) {
    auto emitLine = [&out, &indent](const std::string& text) {
        for (int i = 0; i < indent; ++i) {
            out << "    ";
        }
        out << text << '\n';
    };

    emitLine(signature);
    ++indent;
    body(makeCodegenLineEmitter(out, indent));
    --indent;
    emitLine("}");
    out << '\n';
}

}  // namespace

void CodeGenerator::emitHashMapStructs() {
    emitLine("// ============================================");
    emitLine("// Hash Map for SpGEMM Accumulation");
    emitLine("// ============================================");
    emitLine();
    emitLine("typedef struct {");
    increaseIndent();
    emitLine("int col;");
    emitLine("double val;");
    emitLine("char occupied;");
    decreaseIndent();
    emitLine("} HashEntry;");
    emitLine();
    emitLine("typedef struct {");
    increaseIndent();
    emitLine("HashEntry* entries;");
    emitLine("int capacity;");
    emitLine("int count;");
    decreaseIndent();
    emitLine("} RowHash;");
    emitLine();
}

void CodeGenerator::emitHashMapHelpers() {
    emitLine("static void init_row_hash(RowHash* rh, int capacity) {");
    increaseIndent();
    emitLine("rh->capacity = capacity;");
    emitLine("rh->count = 0;");
    emitLine("rh->entries = (HashEntry*)calloc(capacity, sizeof(HashEntry));");
    decreaseIndent();
    emitLine("}");
    emitLine();

    emitLine("static void hash_accumulate(RowHash* rh, int col, double val) {");
    increaseIndent();
    emitLine("int cap = rh->capacity;");
    emitLine("unsigned int h = (unsigned int)col % cap;");
    emitLine("for (int probe = 0; probe < cap; probe++) {");
    increaseIndent();
    emitLine("int idx = (h + probe) % cap;");
    emitLine("if (!rh->entries[idx].occupied) {");
    increaseIndent();
    emitLine("rh->entries[idx].col = col;");
    emitLine("rh->entries[idx].val = val;");
    emitLine("rh->entries[idx].occupied = 1;");
    emitLine("rh->count++;");
    emitLine("return;");
    decreaseIndent();
    emitLine("}");
    emitLine("if (rh->entries[idx].col == col) {");
    increaseIndent();
    emitLine("rh->entries[idx].val += val;");
    emitLine("return;");
    decreaseIndent();
    emitLine("}");
    decreaseIndent();
    emitLine("}");
    emitLine("// Table full — should not happen if capacity is sufficient");
    emitLine("fprintf(stderr, \"Error: hash table full\\n\");");
    decreaseIndent();
    emitLine("}");
    emitLine();

    emitLine("static int col_cmp(const void* a, const void* b) {");
    increaseIndent();
    emitLine("return ((const HashEntry*)a)->col - ((const HashEntry*)b)->col;");
    decreaseIndent();
    emitLine("}");
    emitLine();

    emitLine("static void free_row_hash(RowHash* rh) {");
    increaseIndent();
    emitLine("free(rh->entries);");
    decreaseIndent();
    emitLine("}");
    emitLine();
}

bool CodeGenerator::isSparseOutputMode() const {
    return context_.hasSparseOutput();
}

ir::Format CodeGenerator::getPrimarySparseFormat() const {
    return context_.primarySparseFormat;
}

void CodeGenerator::emitSparseOutputHelpers() {
    emitLine("static int cmp_int_asc(const void* a, const void* b) {");
    increaseIndent();
    emitLine("int ai = *(const int*)a;");
    emitLine("int bi = *(const int*)b;");
    emitLine("return (ai > bi) - (ai < bi);");
    decreaseIndent();
    emitLine("}");
    emitLine();
    emitLine("static void zero_sparse_values(SparseMatrix* C) {");
    increaseIndent();
    emitLine("if (C && C->vals && C->nnz > 0) memset(C->vals, 0, (size_t)C->nnz * sizeof(double));");
    decreaseIndent();
    emitLine("}");
    emitLine();

    const auto& inputs = getActiveInputs();
    const bool needsCSRGet = std::any_of(
        inputs.begin(), inputs.end(),
        [](const ir::Tensor& tensor) { return tensor.format == ir::Format::CSR; });
    const bool needsCSCGet = std::any_of(
        inputs.begin(), inputs.end(),
        [](const ir::Tensor& tensor) { return tensor.format == ir::Format::CSC; });

    if (needsCSRGet) {
        emitLine("static double sp_csr_get(const SparseMatrix* A, int row, int col) {");
        increaseIndent();
        emitLine("for (int p = A->row_ptr[row]; p < A->row_ptr[row + 1]; p++) {");
        increaseIndent();
        emitLine("if (A->col_idx[p] == col) return A->vals[p];");
        decreaseIndent();
        emitLine("}");
        emitLine("return 0.0;");
        decreaseIndent();
        emitLine("}");
        emitLine();
    }

    if (needsCSCGet) {
        emitLine("static double sp_csc_get(const SparseMatrix* A, int row, int col) {");
        increaseIndent();
        emitLine("for (int p = A->col_ptr[col]; p < A->col_ptr[col + 1]; p++) {");
        increaseIndent();
        emitLine("if (A->row_idx[p] == row) return A->vals[p];");
        decreaseIndent();
        emitLine("}");
        emitLine("return 0.0;");
        decreaseIndent();
        emitLine("}");
        emitLine();
    }

    emitLine("static double max_abs_error_sparse(const double* actual, const double* expected, int nnz) {");
    increaseIndent();
    emitLine("double max_error = 0.0;");
    emitLine("for (int p = 0; p < nnz; p++) {");
    increaseIndent();
    emitLine("double err = actual[p] - expected[p];");
    emitLine("if (err < 0.0) err = -err;");
    emitLine("if (err > max_error) max_error = err;");
    decreaseIndent();
    emitLine("}");
    emitLine("return max_error;");
    decreaseIndent();
    emitLine("}");
    emitLine();
}

void CodeGenerator::emitUnionAssembly() {
    const bool isCSC = (getPrimarySparseFormat() == ir::Format::CSC);

    emitAssemblyWrapper(
        out_,
        indent_,
        "void " + getAssemblyFunctionName() +
            "(const SparseMatrix* A, const SparseMatrix* B, SparseMatrix* C) {",
        [&](const detail::EmitIndentedLine& emitRelative) {
            detail::emitMergeAssemblyBody(emitRelative, "C", "A", "B", isCSC, true, "out");
        });
}

void CodeGenerator::emitIntersectionAssembly() {
    const bool isCSC = (getPrimarySparseFormat() == ir::Format::CSC);

    emitAssemblyWrapper(
        out_,
        indent_,
        "void " + getAssemblyFunctionName() +
            "(const SparseMatrix* A, const SparseMatrix* B, SparseMatrix* C) {",
        [&](const detail::EmitIndentedLine& emitRelative) {
            detail::emitMergeAssemblyBody(emitRelative, "C", "A", "B", isCSC, false, "out");
        });
}

void CodeGenerator::emitDynamicRowAssembly() {
    const bool isCSC = (getPrimarySparseFormat() == ir::Format::CSC);

    emitAssemblyWrapper(
        out_,
        indent_,
        "void " + getAssemblyFunctionName() +
            "(const SparseMatrix* A, const SparseMatrix* B, SparseMatrix* C) {",
        [&](const detail::EmitIndentedLine& emitRelative) {
            detail::emitDynamicRowAssemblyBody(emitRelative, "C", "A", "B", isCSC, "out");
        });
}

void CodeGenerator::emitSampledAssembly() {
    const bool isCSC = (getPrimarySparseFormat() == ir::Format::CSC);

    emitAssemblyWrapper(
        out_,
        indent_,
        "void " + getAssemblyFunctionName() + "(const SparseMatrix* S, SparseMatrix* C) {",
        [&](const detail::EmitIndentedLine& emitRelative) {
            detail::emitSampledAssemblyBody(emitRelative, "C", "S", isCSC);
        });
}

void CodeGenerator::emitSparseAssembly() {
    if (!currentScheduledCompute_) {
        return;
    }

    switch (currentScheduledCompute_->outputPattern) {
        case sparseir::OutputPatternKind::Union:
            emitUnionAssembly();
            return;
        case sparseir::OutputPatternKind::Intersection:
            emitIntersectionAssembly();
            return;
        case sparseir::OutputPatternKind::Sampled:
            emitSampledAssembly();
            return;
        case sparseir::OutputPatternKind::DynamicRowAccumulator:
            emitDynamicRowAssembly();
            return;
        case sparseir::OutputPatternKind::None:
            return;
    }
}

}  // namespace codegen
