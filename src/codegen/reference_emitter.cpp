#include "codegen.h"

namespace codegen {

void CodeGenerator::emitReferenceKernel() {
    emitLine("// ============================================");
    emitLine("// Reference Kernel (naive, for verification)");
    emitLine("// ============================================");
    emitLine();

    std::string refSignature = getReferenceSignature();
    emitLine("void " + refSignature + " {");
    increaseIndent();

    emitScheduledReferenceKernel();

    decreaseIndent();
    emitLine("}");
    emitLine();
}

void CodeGenerator::emitScheduledReferenceKernel() {
    const auto& compute = *currentScheduledCompute_;
    if (!isSparseOutputMode()) {
        if (currentScheduledCompute_->rootLoop) {
            emitScheduledLoop(*currentScheduledCompute_->rootLoop);
        }
        return;
    }

    switch (compute.outputPattern) {
        case sparseir::OutputPatternKind::Union:
            emitUnionSparseReference();
            break;
        case sparseir::OutputPatternKind::Intersection:
            emitIntersectionSparseReference();
            break;
        case sparseir::OutputPatternKind::Sampled:
            emitSampledSparseReference();
            break;
        case sparseir::OutputPatternKind::DynamicRowAccumulator:
            emitDynamicRowSparseReference();
            break;
        case sparseir::OutputPatternKind::None:
            if (currentScheduledCompute_->rootLoop) {
                emitScheduledLoop(*currentScheduledCompute_->rootLoop);
            }
            break;
    }
}

void CodeGenerator::emitUnionSparseReference() {
    const bool isCSC = (getPrimarySparseFormat() == ir::Format::CSC);
    if (isCSC) {
        emitLine("for (int j = 0; j < C->cols; j++) {");
        increaseIndent();
        emitLine("int pA = A->col_ptr[j], pB = B->col_ptr[j];");
        emitLine("int endA = A->col_ptr[j + 1], endB = B->col_ptr[j + 1];");
        emitLine("for (int pC = C->col_ptr[j]; pC < C->col_ptr[j + 1]; pC++) {");
        increaseIndent();
        emitLine("int i = C->row_idx[pC];");
        emitLine("double a_val = 0.0, b_val = 0.0;");
        emitLine("while (pA < endA && A->row_idx[pA] < i) pA++;");
        emitLine("while (pB < endB && B->row_idx[pB] < i) pB++;");
        emitLine("if (pA < endA && A->row_idx[pA] == i) { a_val = A->vals[pA]; pA++; }");
        emitLine("if (pB < endB && B->row_idx[pB] == i) { b_val = B->vals[pB]; pB++; }");
        emitLine("C_ref_vals[pC] = a_val + b_val;");
        decreaseIndent();
        emitLine("}");
        decreaseIndent();
        emitLine("}");
    } else {
        emitLine("for (int i = 0; i < C->rows; i++) {");
        increaseIndent();
        emitLine("int pA = A->row_ptr[i], pB = B->row_ptr[i];");
        emitLine("int endA = A->row_ptr[i + 1], endB = B->row_ptr[i + 1];");
        emitLine("for (int pC = C->row_ptr[i]; pC < C->row_ptr[i + 1]; pC++) {");
        increaseIndent();
        emitLine("int j = C->col_idx[pC];");
        emitLine("double a_val = 0.0, b_val = 0.0;");
        emitLine("while (pA < endA && A->col_idx[pA] < j) pA++;");
        emitLine("while (pB < endB && B->col_idx[pB] < j) pB++;");
        emitLine("if (pA < endA && A->col_idx[pA] == j) { a_val = A->vals[pA]; pA++; }");
        emitLine("if (pB < endB && B->col_idx[pB] == j) { b_val = B->vals[pB]; pB++; }");
        emitLine("C_ref_vals[pC] = a_val + b_val;");
        decreaseIndent();
        emitLine("}");
        decreaseIndent();
        emitLine("}");
    }
}

void CodeGenerator::emitIntersectionSparseReference() {
    const bool isCSC = (getPrimarySparseFormat() == ir::Format::CSC);
    if (isCSC) {
        emitLine("for (int j = 0; j < C->cols; j++) {");
        increaseIndent();
        emitLine("int pA = A->col_ptr[j], pB = B->col_ptr[j];");
        emitLine("int endA = A->col_ptr[j + 1], endB = B->col_ptr[j + 1];");
        emitLine("for (int pC = C->col_ptr[j]; pC < C->col_ptr[j + 1]; pC++) {");
        increaseIndent();
        emitLine("int i = C->row_idx[pC];");
        emitLine("double a_val = 0.0, b_val = 0.0;");
        emitLine("while (pA < endA && A->row_idx[pA] < i) pA++;");
        emitLine("while (pB < endB && B->row_idx[pB] < i) pB++;");
        emitLine("if (pA < endA && A->row_idx[pA] == i) { a_val = A->vals[pA]; pA++; }");
        emitLine("if (pB < endB && B->row_idx[pB] == i) { b_val = B->vals[pB]; pB++; }");
        emitLine("C_ref_vals[pC] = a_val * b_val;");
        decreaseIndent();
        emitLine("}");
        decreaseIndent();
        emitLine("}");
    } else {
        emitLine("for (int i = 0; i < C->rows; i++) {");
        increaseIndent();
        emitLine("int pA = A->row_ptr[i], pB = B->row_ptr[i];");
        emitLine("int endA = A->row_ptr[i + 1], endB = B->row_ptr[i + 1];");
        emitLine("for (int pC = C->row_ptr[i]; pC < C->row_ptr[i + 1]; pC++) {");
        increaseIndent();
        emitLine("int j = C->col_idx[pC];");
        emitLine("double a_val = 0.0, b_val = 0.0;");
        emitLine("while (pA < endA && A->col_idx[pA] < j) pA++;");
        emitLine("while (pB < endB && B->col_idx[pB] < j) pB++;");
        emitLine("if (pA < endA && A->col_idx[pA] == j) { a_val = A->vals[pA]; pA++; }");
        emitLine("if (pB < endB && B->col_idx[pB] == j) { b_val = B->vals[pB]; pB++; }");
        emitLine("C_ref_vals[pC] = a_val * b_val;");
        decreaseIndent();
        emitLine("}");
        decreaseIndent();
        emitLine("}");
    }
}

void CodeGenerator::emitDynamicRowSparseReference() {
    const bool isCSC = (getPrimarySparseFormat() == ir::Format::CSC);
    if (isCSC) {
        emitLine("int M = A->rows;");
        emitLine("double* acc = (double*)calloc((size_t)M, sizeof(double));");
        emitLine("unsigned char* marked = (unsigned char*)calloc((size_t)M, sizeof(unsigned char));");
        emitLine("int* touched = (int*)malloc((size_t)M * sizeof(int));");
        emitLine("for (int j = 0; j < C->cols; j++) {");
        increaseIndent();
        emitLine("int touched_count = 0;");
        emitLine("for (int pB = B->col_ptr[j]; pB < B->col_ptr[j + 1]; pB++) {");
        increaseIndent();
        emitLine("int k = B->row_idx[pB];");
        emitLine("double b_val = B->vals[pB];");
        emitLine("for (int pA = A->col_ptr[k]; pA < A->col_ptr[k + 1]; pA++) {");
        increaseIndent();
        emitLine("int i = A->row_idx[pA];");
        emitLine("if (!marked[i]) { marked[i] = 1; touched[touched_count++] = i; }");
        emitLine("acc[i] += A->vals[pA] * b_val;");
        decreaseIndent();
        emitLine("}");
        decreaseIndent();
        emitLine("}");
        emitLine("for (int pC = C->col_ptr[j]; pC < C->col_ptr[j + 1]; pC++) {");
        increaseIndent();
        emitLine("int i = C->row_idx[pC];");
        emitLine("C_ref_vals[pC] = acc[i];");
        decreaseIndent();
        emitLine("}");
        emitLine("for (int t = 0; t < touched_count; t++) {");
        increaseIndent();
        emitLine("int i = touched[t];");
        emitLine("acc[i] = 0.0;");
        emitLine("marked[i] = 0;");
        decreaseIndent();
        emitLine("}");
        decreaseIndent();
        emitLine("}");
        emitLine("free(touched);");
        emitLine("free(marked);");
        emitLine("free(acc);");
    } else {
        emitLine("int N = B->cols;");
        emitLine("double* acc = (double*)calloc((size_t)N, sizeof(double));");
        emitLine("unsigned char* marked = (unsigned char*)calloc((size_t)N, sizeof(unsigned char));");
        emitLine("int* touched = (int*)malloc((size_t)N * sizeof(int));");
        emitLine("for (int i = 0; i < C->rows; i++) {");
        increaseIndent();
        emitLine("int touched_count = 0;");
        emitLine("for (int pA = A->row_ptr[i]; pA < A->row_ptr[i + 1]; pA++) {");
        increaseIndent();
        emitLine("int k = A->col_idx[pA];");
        emitLine("double a_val = A->vals[pA];");
        emitLine("for (int pB = B->row_ptr[k]; pB < B->row_ptr[k + 1]; pB++) {");
        increaseIndent();
        emitLine("int j = B->col_idx[pB];");
        emitLine("if (!marked[j]) { marked[j] = 1; touched[touched_count++] = j; }");
        emitLine("acc[j] += a_val * B->vals[pB];");
        decreaseIndent();
        emitLine("}");
        decreaseIndent();
        emitLine("}");
        emitLine("for (int pC = C->row_ptr[i]; pC < C->row_ptr[i + 1]; pC++) {");
        increaseIndent();
        emitLine("int j = C->col_idx[pC];");
        emitLine("C_ref_vals[pC] = acc[j];");
        decreaseIndent();
        emitLine("}");
        emitLine("for (int t = 0; t < touched_count; t++) {");
        increaseIndent();
        emitLine("int j = touched[t];");
        emitLine("acc[j] = 0.0;");
        emitLine("marked[j] = 0;");
        decreaseIndent();
        emitLine("}");
        decreaseIndent();
        emitLine("}");
        emitLine("free(touched);");
        emitLine("free(marked);");
        emitLine("free(acc);");
    }
}

void CodeGenerator::emitSampledSparseReference() {
    const bool isCSC = (getPrimarySparseFormat() == ir::Format::CSC);
    if (isCSC) {
        emitLine("for (int j = 0; j < C->cols; j++) {");
        increaseIndent();
        emitLine("for (int p = C->col_ptr[j]; p < C->col_ptr[j + 1]; p++) {");
        increaseIndent();
        emitLine("int i = C->row_idx[p];");
        emitLine("double sum = 0.0;");
        emitLine("for (int k = 0; k < K; k++) sum += D[i][k] * E[k][j];");
        emitLine("C_ref_vals[p] = S->vals[p] * sum;");
        decreaseIndent();
        emitLine("}");
        decreaseIndent();
        emitLine("}");
    } else {
        emitLine("for (int i = 0; i < C->rows; i++) {");
        increaseIndent();
        emitLine("for (int p = C->row_ptr[i]; p < C->row_ptr[i + 1]; p++) {");
        increaseIndent();
        emitLine("int j = C->col_idx[p];");
        emitLine("double sum = 0.0;");
        emitLine("for (int k = 0; k < K; k++) sum += D[i][k] * E[k][j];");
        emitLine("C_ref_vals[p] = S->vals[p] * sum;");
        decreaseIndent();
        emitLine("}");
        decreaseIndent();
        emitLine("}");
    }
}

} // namespace codegen
