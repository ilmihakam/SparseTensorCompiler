#include "codegen.h"
#include "output_assembly_shared.h"

#include <algorithm>
#include <fstream>
#include <set>

namespace codegen {

namespace {

std::string emitProgExpr(
    const SparseTensorCompiler::Expression& expr,
    const std::unordered_map<std::string, std::string>& tensorTypes)
{
    using namespace SparseTensorCompiler;

    if (auto* ta = dynamic_cast<const TensorAccess*>(&expr)) {
        auto it = tensorTypes.find(ta->tensorName);
        std::string type = (it != tensorTypes.end()) ? it->second : "Dense";

        if ((type == "CSR" || type == "CSC") && ta->indices.size() == 2) {
            std::string func = (type == "CSR") ? "sp_csr_get" : "sp_csc_get";
            return func + "(" + ta->tensorName + ", " +
                   ta->indices[0] + ", " + ta->indices[1] + ")";
        }
        if (ta->indices.size() == 1) {
            return ta->tensorName + "[" + ta->indices[0] + "]";
        }
        if (ta->indices.size() == 2) {
            return ta->tensorName + "[" +
                   ta->indices[0] + " * " + ta->tensorName + "_ncols + " +
                   ta->indices[1] + "]";
        }
        return ta->tensorName;
    }

    if (auto* num = dynamic_cast<const Number*>(&expr)) {
        return num->value;
    }

    if (auto* id = dynamic_cast<const Identifier*>(&expr)) {
        return id->name;
    }

    if (auto* binop = dynamic_cast<const BinaryOp*>(&expr)) {
        std::string lhs = emitProgExpr(*binop->left, tensorTypes);
        std::string rhs = emitProgExpr(*binop->right, tensorTypes);
        std::string op = (binop->op == BinaryOp::ADD) ? " + " : " * ";
        return "(" + lhs + op + rhs + ")";
    }

    if (auto* fc = dynamic_cast<const FunctionCall*>(&expr)) {
        std::string res = fc->functionName + "(";
        for (size_t i = 0; i < fc->arguments.size(); i++) {
            if (i > 0) res += ", ";
            res += emitProgExpr(*fc->arguments[i], tensorTypes);
        }
        res += ")";
        return res;
    }

    return "0.0";
}

struct ProgramTensorInfo {
    std::string type;
    std::vector<int> dims;

    bool isSparse() const { return type == "CSR" || type == "CSC"; }
};

std::string emitScheduledProgExpr(
    const sparseir::Expr& expr,
    const std::unordered_map<std::string, std::string>& tensorTypes)
{
    if (auto* ta = dynamic_cast<const sparseir::TensorRead*>(&expr)) {
        auto it = tensorTypes.find(ta->tensorName);
        std::string type = (it != tensorTypes.end()) ? it->second : "Dense";

        if ((type == "CSR" || type == "CSC") && ta->indices.size() == 2) {
            std::string func = (type == "CSR") ? "sp_csr_get" : "sp_csc_get";
            return func + "(" + ta->tensorName + ", " +
                   ta->indices[0] + ", " + ta->indices[1] + ")";
        }
        if (ta->indices.size() == 1) {
            return ta->tensorName + "[" + ta->indices[0] + "]";
        }
        if (ta->indices.size() == 2) {
            return ta->tensorName + "[" +
                   ta->indices[0] + " * " + ta->tensorName + "_ncols + " +
                   ta->indices[1] + "]";
        }
        return ta->tensorName;
    }

    if (auto* num = dynamic_cast<const sparseir::Constant*>(&expr)) {
        std::ostringstream oss;
        oss << num->value;
        return oss.str();
    }

    if (auto* ref = dynamic_cast<const sparseir::ScalarRef*>(&expr)) {
        return ref->name;
    }

    if (auto* binop = dynamic_cast<const sparseir::BinaryExpr*>(&expr)) {
        std::string lhs = emitScheduledProgExpr(*binop->lhs, tensorTypes);
        std::string rhs = emitScheduledProgExpr(*binop->rhs, tensorTypes);
        std::string op = (binop->op == sparseir::ExprBinaryOp::Add) ? " + " : " * ";
        return "(" + lhs + op + rhs + ")";
    }

    if (auto* call = dynamic_cast<const sparseir::CallExpr*>(&expr)) {
        std::string result = call->functionName + "(";
        for (size_t i = 0; i < call->args.size(); i++) {
            if (i > 0) result += ", ";
            result += emitScheduledProgExpr(*call->args[i], tensorTypes);
        }
        result += ")";
        return result;
    }

    return "0.0";
}

std::string inferArgType(
    const SparseTensorCompiler::Expression& arg,
    const std::unordered_map<std::string, std::string>& tensorTypes)
{
    using namespace SparseTensorCompiler;
    if (auto* id = dynamic_cast<const Identifier*>(&arg)) {
        auto it = tensorTypes.find(id->name);
        if (it != tensorTypes.end()) {
            const std::string& t = it->second;
            if (t == "CSR" || t == "CSC") return "SparseMatrix*";
            return "double*";
        }
    }
    return "double";
}

std::string inferScheduledArgType(
    const sparseir::Expr& arg,
    const std::unordered_map<std::string, std::string>& tensorTypes)
{
    if (auto* ref = dynamic_cast<const sparseir::ScalarRef*>(&arg)) {
        auto it = tensorTypes.find(ref->name);
        if (it != tensorTypes.end()) {
            const std::string& t = it->second;
            if (t == "CSR" || t == "CSC") return "SparseMatrix*";
            return "double*";
        }
    }
    if (auto* tensor = dynamic_cast<const sparseir::TensorRead*>(&arg)) {
        auto it = tensorTypes.find(tensor->tensorName);
        if (it != tensorTypes.end()) {
            const std::string& t = it->second;
            if (tensor->indices.empty() && (t == "CSR" || t == "CSC")) return "SparseMatrix*";
            if (tensor->indices.empty()) return "double*";
        }
    }
    return "double";
}

std::string getProgramTensorDimExpr(const ir::Tensor& tensor,
                                    size_t position,
                                    const std::unordered_map<std::string, ProgramTensorInfo>& tmap,
                                    const std::string& refSparse)
{
    if (tensor.format == ir::Format::CSR || tensor.format == ir::Format::CSC) {
        if (position == 0) return tensor.name + "->rows";
        if (position == 1) return tensor.name + "->cols";
    }

    if (position < tensor.dims.size() && tensor.dims[position] > 0) {
        return std::to_string(tensor.dims[position]);
    }

    if (tensor.format == ir::Format::Dense && tensor.dims.size() >= 2 && position == 1) {
        return tensor.name + "_ncols";
    }

    if (!refSparse.empty()) {
        auto it = tmap.find(refSparse);
        if (it != tmap.end() && it->second.isSparse()) {
            if (position == 0) return refSparse + "->rows";
            if (position == 1) return refSparse + "->cols";
        }
    }

    return "0";
}

std::string pickReferenceSparseTensor(
    const sparseir::scheduled::Compute& compute,
    const std::unordered_map<std::string, ProgramTensorInfo>& tmap)
{
    for (const auto& input : compute.inputs) {
        auto it = tmap.find(input.name);
        if (it != tmap.end() && it->second.isSparse()) {
            return input.name;
        }
    }
    auto outIt = tmap.find(compute.output.name);
    if (outIt != tmap.end() && outIt->second.isSparse()) {
        return compute.output.name;
    }
    return "";
}

void emitScheduledRuntimeAliases(
    std::ostream& out,
    const sparseir::scheduled::Compute& compute,
    const std::unordered_map<std::string, ProgramTensorInfo>& tmap,
    int indent)
{
    auto ind = [&](int extra = 0) {
        for (int i = 0; i < indent + extra; i++) out << "    ";
    };

    const std::string refSparse = pickReferenceSparseTensor(compute, tmap);
    std::set<std::string> emittedBounds;
    for (const auto& index : compute.output.indices) {
        auto it = std::find(compute.output.indices.begin(), compute.output.indices.end(), index);
        size_t pos = static_cast<size_t>(std::distance(compute.output.indices.begin(), it));
        if (!emittedBounds.insert(index).second) continue;
        ind();
        out << "int N_" << index << " = "
            << getProgramTensorDimExpr(compute.output, pos, tmap, refSparse) << ";\n";
    }
    for (const auto& input : compute.inputs) {
        if (input.format != ir::Format::Dense) continue;
        for (size_t pos = 0; pos < input.indices.size(); pos++) {
            const auto& index = input.indices[pos];
            if (std::find(compute.output.indices.begin(), compute.output.indices.end(), index) !=
                compute.output.indices.end()) {
                continue;
            }
            if (!emittedBounds.insert(index).second) continue;
            ind();
            out << "int N_" << index << " = "
                << getProgramTensorDimExpr(input, pos, tmap, refSparse) << ";\n";
        }
    }

    if (compute.exprInfo.numSparseInputs == 1 && compute.output.indices.size() == 2) {
        ind();
        out << "int C_cols = "
            << getProgramTensorDimExpr(compute.output, 1, tmap, refSparse) << ";\n";
    }
    if (compute.exprInfo.numSparseInputs >= 2 && compute.output.indices.size() == 2) {
        ind();
        out << "int rows = "
            << getProgramTensorDimExpr(compute.output, 0, tmap, refSparse) << ";\n";
        ind();
        out << "int cols = "
            << getProgramTensorDimExpr(compute.output, 1, tmap, refSparse) << ";\n";
    }
    if (compute.exprInfo.numSparseInputs == 1 &&
        compute.output.indices.size() == 2 &&
        compute.outputStrategy != ir::OutputStrategy::DenseArray) {
        ind();
        out << "int rows = "
            << getProgramTensorDimExpr(compute.output, 0, tmap, refSparse) << ";\n";
        ind();
        out << "int cols = "
            << getProgramTensorDimExpr(compute.output, 1, tmap, refSparse) << ";\n";
    }
    if (compute.exprInfo.numSparseInputs == 1 &&
        compute.exprInfo.numDenseInputs >= 2 &&
        !compute.output.indices.empty() &&
        compute.output.indices.size() == 2) {
        for (const auto& input : compute.inputs) {
            if (input.format == ir::Format::Dense && input.indices.size() == 2 &&
                input.indices[0] != compute.output.indices[0] &&
                input.indices[1] != compute.output.indices[1]) {
                ind();
                out << "int K = "
                    << getProgramTensorDimExpr(input, 0, tmap, refSparse) << ";\n";
                break;
            }
        }
    }
}

void collectTensorNamesFromScheduledExpr(
    const sparseir::Expr& expr,
    std::set<std::string>& tensors)
{
    if (auto* tensor = dynamic_cast<const sparseir::TensorRead*>(&expr)) {
        tensors.insert(tensor->tensorName);
        return;
    }
    if (auto* scalar = dynamic_cast<const sparseir::ScalarRef*>(&expr)) {
        tensors.insert(scalar->name);
        return;
    }
    if (auto* binary = dynamic_cast<const sparseir::BinaryExpr*>(&expr)) {
        collectTensorNamesFromScheduledExpr(*binary->lhs, tensors);
        collectTensorNamesFromScheduledExpr(*binary->rhs, tensors);
        return;
    }
    if (auto* call = dynamic_cast<const sparseir::CallExpr*>(&expr)) {
        for (const auto& arg : call->args) {
            collectTensorNamesFromScheduledExpr(*arg, tensors);
        }
    }
}

void collectScheduledTensorUsage(
    const std::vector<std::unique_ptr<sparseir::scheduled::Stmt>>& stmts,
    std::set<std::string>& inputTensors,
    std::set<std::string>& ignoredOutputs)
{
    for (const auto& stmt : stmts) {
        if (auto* compute = dynamic_cast<const sparseir::scheduled::Compute*>(stmt.get())) {
            for (const auto& input : compute->inputs) {
                inputTensors.insert(input.name);
            }
            ignoredOutputs.insert(compute->output.name);
        } else if (auto* call = dynamic_cast<const sparseir::scheduled::Call*>(stmt.get())) {
            for (const auto& arg : call->args) {
                collectTensorNamesFromScheduledExpr(*arg, inputTensors);
            }
        } else if (auto* region = dynamic_cast<const sparseir::scheduled::Region*>(stmt.get())) {
            collectScheduledTensorUsage(region->body, inputTensors, ignoredOutputs);
        }
    }
}

void collectScheduledCallTargets(
    const std::vector<std::unique_ptr<sparseir::scheduled::Stmt>>& stmts,
    std::set<std::string>& callTargets)
{
    for (const auto& stmt : stmts) {
        if (auto* call = dynamic_cast<const sparseir::scheduled::Call*>(stmt.get())) {
            callTargets.insert(call->functionName);
        } else if (auto* region = dynamic_cast<const sparseir::scheduled::Region*>(stmt.get())) {
            collectScheduledCallTargets(region->body, callTargets);
        }
    }
}

bool findFirstScheduledCallForSignature(
    const std::vector<std::unique_ptr<sparseir::scheduled::Stmt>>& stmts,
    const std::string& functionName,
    const std::unordered_map<std::string, std::string>& tensorTypes,
    std::string& params)
{
    for (const auto& stmt : stmts) {
        if (auto* call = dynamic_cast<const sparseir::scheduled::Call*>(stmt.get())) {
            if (call->functionName != functionName) {
                continue;
            }
            for (size_t index = 0; index < call->args.size(); ++index) {
                if (index > 0) params += ", ";
                params += inferScheduledArgType(*call->args[index], tensorTypes);
            }
            return true;
        }
        if (auto* region = dynamic_cast<const sparseir::scheduled::Region*>(stmt.get())) {
            if (findFirstScheduledCallForSignature(region->body, functionName, tensorTypes, params)) {
                return true;
            }
        }
    }
    return false;
}

void emitScheduledStmtC(
    std::ostream& out,
    const sparseir::scheduled::Stmt& stmt,
    const std::unordered_map<std::string, ProgramTensorInfo>& tmap,
    const std::unordered_map<std::string, std::string>& tensorTypes,
    int indent)
{
    auto ind = [&](int extra = 0) {
        for (int i = 0; i < indent + extra; ++i) out << "    ";
    };

    if (dynamic_cast<const sparseir::scheduled::Declaration*>(&stmt)) {
        return;
    }

    if (auto* region = dynamic_cast<const sparseir::scheduled::Region*>(&stmt)) {
        for (size_t i = 0; i < region->indices.size(); ++i) {
            const std::string& bound = (i < region->runtimeBounds.size())
                ? region->runtimeBounds[i]
                : std::string("0");
            ind();
            out << "for (int " << region->indices[i] << " = 0; "
                << region->indices[i] << " < " << bound << "; "
                << region->indices[i] << "++) {\n";
        }
        for (const auto& bodyStmt : region->body) {
            emitScheduledStmtC(out, *bodyStmt, tmap, tensorTypes,
                               indent + static_cast<int>(region->indices.size()));
        }
        for (size_t i = 0; i < region->indices.size(); ++i) {
            ind(static_cast<int>(region->indices.size() - i - 1));
            out << "}\n";
        }
        return;
    }

    if (auto* call = dynamic_cast<const sparseir::scheduled::Call*>(&stmt)) {
        ind();
        out << call->functionName << "(";
        for (size_t i = 0; i < call->args.size(); ++i) {
            if (i > 0) out << ", ";
            out << emitScheduledProgExpr(*call->args[i], tensorTypes);
        }
        out << ");\n";
        return;
    }

    if (auto* compute = dynamic_cast<const sparseir::scheduled::Compute*>(&stmt)) {
        ind();
        out << "{\n";
        emitScheduledOutputAssembly(out, *compute, indent + 1);
        emitScheduledRuntimeAliases(out, *compute, tmap, indent + 1);
        CodeGenerator generator(out);
        generator.emitInlineScheduledCompute(*compute, indent + 1);
        ind();
        out << "}\n";
    }
}

}  // namespace

bool generateProgramToFile(
    const sparseir::scheduled::Program& prog,
    const opt::OptConfig& /*cfg*/,
    const std::string& filename)
{
    std::ofstream out(filename);
    if (!out.is_open()) return false;

    std::unordered_map<std::string, ProgramTensorInfo> tmap;
    std::unordered_map<std::string, std::string> tensorTypes;
    std::vector<std::string> declOrder;
    std::vector<std::string> sparseNames;
    std::vector<std::string> sparseInputNames;
    std::vector<std::string> denseNames;
    std::set<std::string> inputTensors;

    std::set<std::string> ignoredOutputs;
    collectScheduledTensorUsage(prog.statements, inputTensors, ignoredOutputs);

    for (const auto& stmt : prog.statements) {
        auto* decl = dynamic_cast<const sparseir::scheduled::Declaration*>(stmt.get());
        if (!decl) continue;

        ProgramTensorInfo info;
        info.type = ir::formatToString(decl->tensor.format);
        info.dims = decl->tensor.dims;
        tmap[decl->tensor.name] = info;
        tensorTypes[decl->tensor.name] = info.type;
        declOrder.push_back(decl->tensor.name);
        if (info.isSparse()) {
            sparseNames.push_back(decl->tensor.name);
            if (inputTensors.count(decl->tensor.name) > 0) {
                sparseInputNames.push_back(decl->tensor.name);
            }
        } else {
            denseNames.push_back(decl->tensor.name);
        }
    }

    std::set<std::string> callTargets;
    collectScheduledCallTargets(prog.statements, callTargets);

    bool hasCsr = false, hasCsc = false;
    for (const auto& name : sparseInputNames) {
        if (tmap[name].type == "CSR") hasCsr = true;
        if (tmap[name].type == "CSC") hasCsc = true;
    }

    out << "// ============================================\n"
        << "// Generated by SparseTensorCompiler (Scheduled Program Mode)\n"
        << "// ============================================\n\n"
        << "#include <stdio.h>\n"
        << "#include <stdlib.h>\n"
        << "#include <string.h>\n"
        << "#include <math.h>\n\n";

    out << "// ---- Sparse Matrix ----\n"
        << "typedef struct {\n"
        << "    int rows, cols, nnz;\n"
        << "    int *row_ptr, *col_idx;  /* CSR */\n"
        << "    int *col_ptr, *row_idx;  /* CSC */\n"
        << "    double *vals;\n"
        << "} SparseMatrix;\n\n";

    out << "// ---- Matrix Market Loader ----\n"
        << "static SparseMatrix* load_matrix_market(const char* filename) {\n"
        << "    FILE* f = fopen(filename, \"r\");\n"
        << "    if (!f) { fprintf(stderr, \"Cannot open %s\\n\", filename); return NULL; }\n"
        << "    char line[1024];\n"
        << "    while (fgets(line, sizeof(line), f) && line[0] == '%') {}\n"
        << "    int M, N, NNZ;\n"
        << "    if (sscanf(line, \"%d %d %d\", &M, &N, &NNZ) != 3) { fclose(f); return NULL; }\n"
        << "    int* ri = (int*)malloc(NNZ * sizeof(int));\n"
        << "    int* ci = (int*)malloc(NNZ * sizeof(int));\n"
        << "    double* vv = (double*)malloc(NNZ * sizeof(double));\n"
        << "    for (int p = 0; p < NNZ; p++) {\n"
        << "        double val = 1.0;\n"
        << "        int r, c;\n"
        << "        int rc = fscanf(f, \"%d %d %lf\", &r, &c, &val);\n"
        << "        if (rc < 2) { val = 1.0; }\n"
        << "        ri[p] = r - 1; ci[p] = c - 1; vv[p] = val;\n"
        << "    }\n"
        << "    fclose(f);\n"
        << "    SparseMatrix* A = (SparseMatrix*)calloc(1, sizeof(SparseMatrix));\n"
        << "    A->rows = M; A->cols = N; A->nnz = NNZ;\n"
        << "    A->row_ptr = (int*)calloc(M + 1, sizeof(int));\n"
        << "    A->col_idx = (int*)malloc(NNZ * sizeof(int));\n"
        << "    A->vals    = (double*)malloc(NNZ * sizeof(double));\n"
        << "    A->col_ptr = NULL; A->row_idx = NULL;\n"
        << "    for (int p = 0; p < NNZ; p++) A->row_ptr[ri[p] + 1]++;\n"
        << "    for (int i = 0; i < M; i++) A->row_ptr[i+1] += A->row_ptr[i];\n"
        << "    int* tmp = (int*)calloc(M, sizeof(int));\n"
        << "    for (int p = 0; p < NNZ; p++) {\n"
        << "        int row = ri[p];\n"
        << "        int pos = A->row_ptr[row] + tmp[row]++;\n"
        << "        A->col_idx[pos] = ci[p];\n"
        << "        A->vals[pos]    = vv[p];\n"
        << "    }\n"
        << "    free(tmp); free(ri); free(ci); free(vv);\n"
        << "    return A;\n"
        << "}\n\n"
        << "static void free_sparse(SparseMatrix* m) {\n"
        << "    if (!m) return;\n"
        << "    free(m->row_ptr); free(m->col_idx);\n"
        << "    free(m->col_ptr); free(m->row_idx);\n"
        << "    free(m->vals); free(m);\n"
        << "}\n\n"
        << "static int cmp_int_asc(const void* a, const void* b) {\n"
        << "    int ia = *(const int*)a;\n"
        << "    int ib = *(const int*)b;\n"
        << "    return (ia > ib) - (ia < ib);\n"
        << "}\n\n";

    if (hasCsr) {
        out << "static double sp_csr_get(const SparseMatrix* A, int row, int col) {\n"
            << "    for (int p = A->row_ptr[row]; p < A->row_ptr[row+1]; p++)\n"
            << "        if (A->col_idx[p] == col) return A->vals[p];\n"
            << "    return 0.0;\n"
            << "}\n\n";
    }
    if (hasCsc) {
        out << "static double sp_csc_get(const SparseMatrix* A, int row, int col) {\n"
            << "    for (int p = A->col_ptr[col]; p < A->col_ptr[col+1]; p++)\n"
            << "        if (A->row_idx[p] == row) return A->vals[p];\n"
            << "    return 0.0;\n"
            << "}\n\n";
    }

    for (const auto& functionName : callTargets) {
        std::string params;
        findFirstScheduledCallForSignature(prog.statements, functionName, tensorTypes, params);
        out << "extern void " << functionName << "(" << params << ");\n";
    }
    if (!callTargets.empty()) out << "\n";

    out << "int main(int argc, char** argv) {\n";

    int numSparse = static_cast<int>(sparseInputNames.size());
    out << "    if (argc < " << (numSparse + 1) << ") {\n"
        << "        fprintf(stderr, \"Usage: %s";
    for (int i = 0; i < numSparse; i++) out << " <matrix" << (i + 1) << ".mtx>";
    out << "\\n\", argv[0]);\n"
        << "        return 1;\n"
        << "    }\n\n";

    out << "    // Declare tensors\n";
    for (const auto& name : declOrder) {
        const auto& info = tmap.at(name);
        if (info.isSparse()) {
            out << "    SparseMatrix* " << name << " = NULL;\n";
        } else {
            out << "    double* " << name << " = NULL;\n";
            if (info.dims.size() >= 2) {
                int ncols = (info.dims[1] > 0) ? info.dims[1] : 0;
                out << "    int " << name << "_ncols = " << ncols << ";\n";
            }
        }
    }
    out << "\n";

    if (!sparseInputNames.empty()) {
        out << "    // Load sparse matrices\n";
        int argIdx = 1;
        for (const auto& name : sparseInputNames) {
            out << "    " << name << " = load_matrix_market(argv[" << argIdx++ << "]);\n";
            out << "    if (!" << name << ") { fprintf(stderr, \"Failed to load "
                << name << "\\n\"); return 1; }\n";
        }
        out << "\n";
    }

    if (!denseNames.empty()) {
        out << "    // Allocate dense tensors\n";
        std::string refSparse = sparseInputNames.empty() ? "" : sparseInputNames[0];
        for (const auto& name : denseNames) {
            const auto& info = tmap.at(name);
            if (info.dims.size() == 1) {
                std::string sz = (info.dims[0] > 0)
                    ? std::to_string(info.dims[0])
                    : (!refSparse.empty() ? refSparse + "->cols" : "0 /* unknown */");
                out << "    " << name << " = (double*)calloc(" << sz << ", sizeof(double));\n";
            } else if (info.dims.size() == 2) {
                std::string rows = (info.dims[0] > 0)
                    ? std::to_string(info.dims[0])
                    : (!refSparse.empty() ? refSparse + "->rows" : "0");
                std::string cols = (info.dims[1] > 0)
                    ? std::to_string(info.dims[1])
                    : (!refSparse.empty() ? refSparse + "->cols" : "0");
                if (info.dims[1] == 0 && !refSparse.empty()) {
                    out << "    " << name << "_ncols = " << cols << ";\n";
                }
                out << "    " << name << " = (double*)calloc(" << rows << " * " << cols
                    << ", sizeof(double));\n";
            } else {
                out << "    " << name << " = (double*)calloc(1, sizeof(double));\n";
            }
        }

        if (!sparseInputNames.empty()) {
            out << "\n    // Initialize dense inputs with 1.0\n";
            for (const auto& name : denseNames) {
                const auto& info = tmap.at(name);
                if (info.dims.size() == 1 && inputTensors.count(name) > 0) {
                    std::string sz = (info.dims[0] > 0)
                        ? std::to_string(info.dims[0])
                        : sparseInputNames[0] + "->cols";
                    out << "    for (int _i = 0; _i < " << sz << "; _i++) "
                        << name << "[_i] = 1.0;\n";
                }
            }
        }
        out << "\n";
    }

    out << "    // Execute program\n";
    for (const auto& stmt : prog.statements) {
        if (dynamic_cast<const sparseir::scheduled::Declaration*>(stmt.get())) continue;
        emitScheduledStmtC(out, *stmt, tmap, tensorTypes, 1);
    }
    out << "\n";

    out << "    // Cleanup\n";
    for (const auto& name : denseNames) {
        out << "    free(" << name << ");\n";
    }
    for (const auto& name : sparseNames) {
        out << "    free_sparse(" << name << ");\n";
    }
    out << "    return 0;\n"
        << "}\n";

    out.close();
    return true;
}

}  // namespace codegen
