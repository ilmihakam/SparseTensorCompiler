#include "output_assembly_shared.h"

namespace codegen {
namespace {

detail::EmitIndentedLine makeStreamLineEmitter(std::ostream& out, int indent) {
    return [&out, indent](int extra, const std::string& text) {
        for (int i = 0; i < indent + extra; ++i) {
            out << "    ";
        }
        out << text << '\n';
    };
}

}  // namespace

namespace detail {

void emitMergeAssemblyBody(const EmitIndentedLine& emitLine,
                           const std::string& output,
                           const std::string& left,
                           const std::string& right,
                           bool isCSC,
                           bool isUnion,
                           const std::string& outputPositionVar) {
    emitLine(0, output + "->rows = " + left + "->rows;");
    emitLine(0, output + "->cols = " + left + "->cols;");
    if (isCSC) {
        emitLine(0, output + "->col_ptr = (int*)calloc((size_t)" + output +
                        "->cols + 1, sizeof(int));");
        emitLine(0, output + "->row_ptr = " + output + "->col_ptr;");
        emitLine(0, "for (int j = 0; j < " + output + "->cols; j++) {");
        emitLine(1, "int p" + left + " = " + left + "->col_ptr[j], p" + right +
                        " = " + right + "->col_ptr[j];");
        emitLine(1, "int end" + left + " = " + left + "->col_ptr[j + 1], end" +
                        right + " = " + right + "->col_ptr[j + 1];");
        emitLine(1, "int count = 0;");
        emitLine(1, "while (p" + left + " < end" + left + " && p" + right +
                        " < end" + right + ") {");
        emitLine(2, "int i" + left + " = " + left + "->row_idx[p" + left +
                        "], i" + right + " = " + right + "->row_idx[p" + right + "];");
        if (isUnion) {
            emitLine(2, "count++;");
            emitLine(2, "if (i" + left + " == i" + right + ") { p" + left +
                            "++; p" + right + "++; }");
            emitLine(2, "else if (i" + left + " < i" + right + ") { p" + left + "++; }");
            emitLine(2, "else { p" + right + "++; }");
        } else {
            emitLine(2, "if (i" + left + " == i" + right + ") { count++; p" + left +
                            "++; p" + right + "++; }");
            emitLine(2, "else if (i" + left + " < i" + right + ") { p" + left + "++; }");
            emitLine(2, "else { p" + right + "++; }");
        }
        emitLine(1, "}");
        if (isUnion) {
            emitLine(1, "count += (end" + left + " - p" + left + ") + (end" +
                            right + " - p" + right + ");");
        }
        emitLine(1, output + "->col_ptr[j + 1] = " + output + "->col_ptr[j] + count;");
        emitLine(0, "}");
        emitLine(0, output + "->nnz = " + output + "->col_ptr[" + output + "->cols];");
        emitLine(0, output + "->row_idx = (int*)malloc((size_t)" + output +
                        "->nnz * sizeof(int));");
        emitLine(0, output + "->col_idx = " + output + "->row_idx;");
        emitLine(0, output + "->vals = (double*)calloc((size_t)" + output +
                        "->nnz, sizeof(double));");
        emitLine(0, "for (int j = 0; j < " + output + "->cols; j++) {");
        emitLine(1, "int p" + left + " = " + left + "->col_ptr[j], p" + right +
                        " = " + right + "->col_ptr[j];");
        emitLine(1, "int end" + left + " = " + left + "->col_ptr[j + 1], end" +
                        right + " = " + right + "->col_ptr[j + 1];");
        emitLine(1, "int " + outputPositionVar + " = " + output + "->col_ptr[j];");
        emitLine(1, "while (p" + left + " < end" + left + " && p" + right +
                        " < end" + right + ") {");
        emitLine(2, "int i" + left + " = " + left + "->row_idx[p" + left +
                        "], i" + right + " = " + right + "->row_idx[p" + right + "];");
        if (isUnion) {
            emitLine(2, "if (i" + left + " == i" + right + ") { " + output +
                            "->row_idx[" + outputPositionVar + "++] = i" + left +
                            "; p" + left + "++; p" + right + "++; }");
            emitLine(2, "else if (i" + left + " < i" + right + ") { " + output +
                            "->row_idx[" + outputPositionVar + "++] = i" + left +
                            "; p" + left + "++; }");
            emitLine(2, "else { " + output + "->row_idx[" + outputPositionVar +
                            "++] = i" + right + "; p" + right + "++; }");
            emitLine(1, "}");
            emitLine(1, "while (p" + left + " < end" + left + ") " + output +
                            "->row_idx[" + outputPositionVar + "++] = " + left +
                            "->row_idx[p" + left + "++];");
            emitLine(1, "while (p" + right + " < end" + right + ") " + output +
                            "->row_idx[" + outputPositionVar + "++] = " + right +
                            "->row_idx[p" + right + "++];");
        } else {
            emitLine(2, "if (i" + left + " == i" + right + ") { " + output +
                            "->row_idx[" + outputPositionVar + "++] = i" + left +
                            "; p" + left + "++; p" + right + "++; }");
            emitLine(2, "else if (i" + left + " < i" + right + ") { p" + left + "++; }");
            emitLine(2, "else { p" + right + "++; }");
            emitLine(1, "}");
        }
        emitLine(0, "}");
        return;
    }

    emitLine(0, output + "->row_ptr = (int*)calloc((size_t)" + output +
                    "->rows + 1, sizeof(int));");
    emitLine(0, output + "->col_ptr = " + output + "->row_ptr;");
    emitLine(0, "for (int i = 0; i < " + output + "->rows; i++) {");
    emitLine(1, "int p" + left + " = " + left + "->row_ptr[i], p" + right +
                    " = " + right + "->row_ptr[i];");
    emitLine(1, "int end" + left + " = " + left + "->row_ptr[i + 1], end" +
                    right + " = " + right + "->row_ptr[i + 1];");
    emitLine(1, "int count = 0;");
    emitLine(1, "while (p" + left + " < end" + left + " && p" + right +
                    " < end" + right + ") {");
    emitLine(2, "int j" + left + " = " + left + "->col_idx[p" + left +
                    "], j" + right + " = " + right + "->col_idx[p" + right + "];");
    if (isUnion) {
        emitLine(2, "count++;");
        emitLine(2, "if (j" + left + " == j" + right + ") { p" + left +
                        "++; p" + right + "++; }");
        emitLine(2, "else if (j" + left + " < j" + right + ") { p" + left + "++; }");
        emitLine(2, "else { p" + right + "++; }");
    } else {
        emitLine(2, "if (j" + left + " == j" + right + ") { count++; p" + left +
                        "++; p" + right + "++; }");
        emitLine(2, "else if (j" + left + " < j" + right + ") { p" + left + "++; }");
        emitLine(2, "else { p" + right + "++; }");
    }
    emitLine(1, "}");
    if (isUnion) {
        emitLine(1, "count += (end" + left + " - p" + left + ") + (end" +
                        right + " - p" + right + ");");
    }
    emitLine(1, output + "->row_ptr[i + 1] = " + output + "->row_ptr[i] + count;");
    emitLine(0, "}");
    emitLine(0, output + "->nnz = " + output + "->row_ptr[" + output + "->rows];");
    emitLine(0, output + "->col_idx = (int*)malloc((size_t)" + output +
                    "->nnz * sizeof(int));");
    emitLine(0, output + "->row_idx = " + output + "->col_idx;");
    emitLine(0, output + "->vals = (double*)calloc((size_t)" + output +
                    "->nnz, sizeof(double));");
    emitLine(0, "for (int i = 0; i < " + output + "->rows; i++) {");
    emitLine(1, "int p" + left + " = " + left + "->row_ptr[i], p" + right +
                    " = " + right + "->row_ptr[i];");
    emitLine(1, "int end" + left + " = " + left + "->row_ptr[i + 1], end" +
                    right + " = " + right + "->row_ptr[i + 1];");
    emitLine(1, "int " + outputPositionVar + " = " + output + "->row_ptr[i];");
    emitLine(1, "while (p" + left + " < end" + left + " && p" + right +
                    " < end" + right + ") {");
    emitLine(2, "int j" + left + " = " + left + "->col_idx[p" + left +
                    "], j" + right + " = " + right + "->col_idx[p" + right + "];");
    if (isUnion) {
        emitLine(2, "if (j" + left + " == j" + right + ") { " + output +
                        "->col_idx[" + outputPositionVar + "++] = j" + left +
                        "; p" + left + "++; p" + right + "++; }");
        emitLine(2, "else if (j" + left + " < j" + right + ") { " + output +
                        "->col_idx[" + outputPositionVar + "++] = j" + left +
                        "; p" + left + "++; }");
        emitLine(2, "else { " + output + "->col_idx[" + outputPositionVar +
                        "++] = j" + right + "; p" + right + "++; }");
        emitLine(1, "}");
        emitLine(1, "while (p" + left + " < end" + left + ") " + output +
                        "->col_idx[" + outputPositionVar + "++] = " + left +
                        "->col_idx[p" + left + "++];");
        emitLine(1, "while (p" + right + " < end" + right + ") " + output +
                        "->col_idx[" + outputPositionVar + "++] = " + right +
                        "->col_idx[p" + right + "++];");
    } else {
        emitLine(2, "if (j" + left + " == j" + right + ") { " + output +
                        "->col_idx[" + outputPositionVar + "++] = j" + left +
                        "; p" + left + "++; p" + right + "++; }");
        emitLine(2, "else if (j" + left + " < j" + right + ") { p" + left + "++; }");
        emitLine(2, "else { p" + right + "++; }");
        emitLine(1, "}");
    }
    emitLine(0, "}");
}

void emitDynamicRowAssemblyBody(const EmitIndentedLine& emitLine,
                                const std::string& output,
                                const std::string& left,
                                const std::string& right,
                                bool isCSC,
                                const std::string& outputPositionVar) {
    emitLine(0, output + "->rows = " + left + "->rows;");
    emitLine(0, output + "->cols = " + right + "->cols;");
    if (isCSC) {
        emitLine(0, "int M = " + left + "->rows;");
        emitLine(0, "int N = " + right + "->cols;");
        emitLine(0, output + "->col_ptr = (int*)calloc((size_t)N + 1, sizeof(int));");
        emitLine(0, output + "->row_ptr = " + output + "->col_ptr;");
        emitLine(0, "int* marker = (int*)malloc((size_t)M * sizeof(int));");
        emitLine(0, "for (int i = 0; i < M; i++) marker[i] = -1;");
        emitLine(0, "for (int j = 0; j < N; j++) {");
        emitLine(1, "int count = 0;");
        emitLine(1, "for (int p" + right + " = " + right + "->col_ptr[j]; p" + right +
                        " < " + right + "->col_ptr[j + 1]; p" + right + "++) {");
        emitLine(2, "int k = " + right + "->row_idx[p" + right + "];");
        emitLine(2, "for (int p" + left + " = " + left + "->col_ptr[k]; p" + left +
                        " < " + left + "->col_ptr[k + 1]; p" + left + "++) {");
        emitLine(3, "int i = " + left + "->row_idx[p" + left + "];");
        emitLine(3, "if (marker[i] != j) { marker[i] = j; count++; }");
        emitLine(2, "}");
        emitLine(1, "}");
        emitLine(1, output + "->col_ptr[j + 1] = " + output + "->col_ptr[j] + count;");
        emitLine(0, "}");
        emitLine(0, output + "->nnz = " + output + "->col_ptr[N];");
        emitLine(0, output + "->row_idx = (int*)malloc((size_t)" + output +
                        "->nnz * sizeof(int));");
        emitLine(0, output + "->col_idx = " + output + "->row_idx;");
        emitLine(0, output + "->vals = (double*)calloc((size_t)" + output +
                        "->nnz, sizeof(double));");
        emitLine(0, "for (int i = 0; i < M; i++) marker[i] = -1;");
        emitLine(0, "int* touched = (int*)malloc((size_t)M * sizeof(int));");
        emitLine(0, "for (int j = 0; j < N; j++) {");
        emitLine(1, "int touched_count = 0;");
        emitLine(1, "for (int p" + right + " = " + right + "->col_ptr[j]; p" + right +
                        " < " + right + "->col_ptr[j + 1]; p" + right + "++) {");
        emitLine(2, "int k = " + right + "->row_idx[p" + right + "];");
        emitLine(2, "for (int p" + left + " = " + left + "->col_ptr[k]; p" + left +
                        " < " + left + "->col_ptr[k + 1]; p" + left + "++) {");
        emitLine(3, "int i = " + left + "->row_idx[p" + left + "];");
        emitLine(3, "if (marker[i] != j) { marker[i] = j; touched[touched_count++] = i; }");
        emitLine(2, "}");
        emitLine(1, "}");
        emitLine(1, "qsort(touched, (size_t)touched_count, sizeof(int), cmp_int_asc);");
        emitLine(1, "int " + outputPositionVar + " = " + output + "->col_ptr[j];");
        emitLine(1, "for (int t = 0; t < touched_count; t++) " + output +
                        "->row_idx[" + outputPositionVar + " + t] = touched[t];");
        emitLine(0, "}");
        emitLine(0, "free(touched);");
        emitLine(0, "free(marker);");
        return;
    }

    emitLine(0, "int M = " + left + "->rows;");
    emitLine(0, "int N = " + right + "->cols;");
    emitLine(0, output + "->row_ptr = (int*)calloc((size_t)M + 1, sizeof(int));");
    emitLine(0, output + "->col_ptr = " + output + "->row_ptr;");
    emitLine(0, "int* marker = (int*)malloc((size_t)N * sizeof(int));");
    emitLine(0, "for (int j = 0; j < N; j++) marker[j] = -1;");
    emitLine(0, "for (int i = 0; i < M; i++) {");
    emitLine(1, "int count = 0;");
    emitLine(1, "for (int p" + left + " = " + left + "->row_ptr[i]; p" + left +
                    " < " + left + "->row_ptr[i + 1]; p" + left + "++) {");
    emitLine(2, "int k = " + left + "->col_idx[p" + left + "];");
    emitLine(2, "for (int p" + right + " = " + right + "->row_ptr[k]; p" + right +
                    " < " + right + "->row_ptr[k + 1]; p" + right + "++) {");
    emitLine(3, "int j = " + right + "->col_idx[p" + right + "];");
    emitLine(3, "if (marker[j] != i) { marker[j] = i; count++; }");
    emitLine(2, "}");
    emitLine(1, "}");
    emitLine(1, output + "->row_ptr[i + 1] = " + output + "->row_ptr[i] + count;");
    emitLine(0, "}");
    emitLine(0, output + "->nnz = " + output + "->row_ptr[M];");
    emitLine(0, output + "->col_idx = (int*)malloc((size_t)" + output +
                    "->nnz * sizeof(int));");
    emitLine(0, output + "->row_idx = " + output + "->col_idx;");
    emitLine(0, output + "->vals = (double*)calloc((size_t)" + output +
                    "->nnz, sizeof(double));");
    emitLine(0, "for (int j = 0; j < N; j++) marker[j] = -1;");
    emitLine(0, "int* touched = (int*)malloc((size_t)N * sizeof(int));");
    emitLine(0, "for (int i = 0; i < M; i++) {");
    emitLine(1, "int touched_count = 0;");
    emitLine(1, "for (int p" + left + " = " + left + "->row_ptr[i]; p" + left +
                    " < " + left + "->row_ptr[i + 1]; p" + left + "++) {");
    emitLine(2, "int k = " + left + "->col_idx[p" + left + "];");
    emitLine(2, "for (int p" + right + " = " + right + "->row_ptr[k]; p" + right +
                    " < " + right + "->row_ptr[k + 1]; p" + right + "++) {");
    emitLine(3, "int j = " + right + "->col_idx[p" + right + "];");
    emitLine(3, "if (marker[j] != i) { marker[j] = i; touched[touched_count++] = j; }");
    emitLine(2, "}");
    emitLine(1, "}");
    emitLine(1, "qsort(touched, (size_t)touched_count, sizeof(int), cmp_int_asc);");
    emitLine(1, "int " + outputPositionVar + " = " + output + "->row_ptr[i];");
    emitLine(1, "for (int t = 0; t < touched_count; t++) " + output +
                    "->col_idx[" + outputPositionVar + " + t] = touched[t];");
    emitLine(0, "}");
    emitLine(0, "free(touched);");
    emitLine(0, "free(marker);");
}

void emitSampledAssemblyBody(const EmitIndentedLine& emitLine,
                             const std::string& output,
                             const std::string& sampled,
                             bool isCSC) {
    emitLine(0, output + "->rows = " + sampled + "->rows;");
    emitLine(0, output + "->cols = " + sampled + "->cols;");
    emitLine(0, output + "->nnz = " + sampled + "->nnz;");
    if (isCSC) {
        emitLine(0, output + "->col_ptr = (int*)malloc(((size_t)" + sampled +
                        "->cols + 1) * sizeof(int));");
        emitLine(0, "memcpy(" + output + "->col_ptr, " + sampled + "->col_ptr, ((size_t)" +
                        sampled + "->cols + 1) * sizeof(int));");
        emitLine(0, output + "->row_ptr = " + output + "->col_ptr;");
        emitLine(0, output + "->row_idx = (int*)malloc((size_t)" + sampled +
                        "->nnz * sizeof(int));");
        emitLine(0, "memcpy(" + output + "->row_idx, " + sampled + "->row_idx, (size_t)" +
                        sampled + "->nnz * sizeof(int));");
        emitLine(0, output + "->col_idx = " + output + "->row_idx;");
    } else {
        emitLine(0, output + "->row_ptr = (int*)malloc(((size_t)" + sampled +
                        "->rows + 1) * sizeof(int));");
        emitLine(0, "memcpy(" + output + "->row_ptr, " + sampled + "->row_ptr, ((size_t)" +
                        sampled + "->rows + 1) * sizeof(int));");
        emitLine(0, output + "->col_ptr = " + output + "->row_ptr;");
        emitLine(0, output + "->col_idx = (int*)malloc((size_t)" + sampled +
                        "->nnz * sizeof(int));");
        emitLine(0, "memcpy(" + output + "->col_idx, " + sampled + "->col_idx, (size_t)" +
                        sampled + "->nnz * sizeof(int));");
        emitLine(0, output + "->row_idx = " + output + "->col_idx;");
    }
    emitLine(0, output + "->vals = (double*)calloc((size_t)" + sampled +
                    "->nnz, sizeof(double));");
}

}  // namespace detail

void emitScheduledOutputAssembly(std::ostream& out,
                                 const sparseir::scheduled::Compute& compute,
                                 int indent) {
    if (compute.outputStrategy == ir::OutputStrategy::DenseArray) {
        return;
    }

    auto emitLine = makeStreamLineEmitter(out, indent);
    const bool isCSC = compute.output.format == ir::Format::CSC;
    const std::string& output = compute.output.name;

    if (compute.outputPattern == sparseir::OutputPatternKind::Sampled &&
        !compute.patternSources.empty()) {
        detail::emitSampledAssemblyBody(
            emitLine, output, compute.patternSources[0], isCSC);
        return;
    }

    if (compute.outputPattern == sparseir::OutputPatternKind::Union ||
        compute.outputPattern == sparseir::OutputPatternKind::Intersection) {
        if (compute.patternSources.size() < 2) {
            return;
        }
        detail::emitMergeAssemblyBody(emitLine,
                                      output,
                                      compute.patternSources[0],
                                      compute.patternSources[1],
                                      isCSC,
                                      compute.outputPattern == sparseir::OutputPatternKind::Union,
                                      "out_pos");
        return;
    }

    if (compute.outputPattern == sparseir::OutputPatternKind::DynamicRowAccumulator &&
        compute.patternSources.size() >= 2) {
        detail::emitDynamicRowAssemblyBody(emitLine,
                                           output,
                                           compute.patternSources[0],
                                           compute.patternSources[1],
                                           isCSC,
                                           "out_pos");
    }
}

}  // namespace codegen
