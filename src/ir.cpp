#include "ir.h"

#include <cmath>
#include <sstream>
#include <unordered_set>

namespace ir {

void IRTensorAccess::accept(IRExprVisitor& v) const { v.visit(*this); }

std::unique_ptr<IRExpr> IRTensorAccess::clone() const {
    auto copy = std::make_unique<IRTensorAccess>(tensorName, indices);
    copy->isSparseVals = isSparseVals;
    copy->pointerVar = pointerVar;
    copy->useRandomAccess = useRandomAccess;
    copy->randomAccessFunc = randomAccessFunc;
    return copy;
}

void IRConstant::accept(IRExprVisitor& v) const { v.visit(*this); }

std::unique_ptr<IRExpr> IRConstant::clone() const {
    return std::make_unique<IRConstant>(value);
}

void IRBinaryOp::accept(IRExprVisitor& v) const { v.visit(*this); }

std::unique_ptr<IRExpr> IRBinaryOp::clone() const {
    return std::make_unique<IRBinaryOp>(op, lhs->clone(), rhs->clone());
}

void IRScalarVar::accept(IRExprVisitor& v) const { v.visit(*this); }

std::unique_ptr<IRExpr> IRScalarVar::clone() const {
    return std::make_unique<IRScalarVar>(name);
}

void IRFuncCall::accept(IRExprVisitor& v) const { v.visit(*this); }

std::unique_ptr<IRExpr> IRFuncCall::clone() const {
    auto copy = std::make_unique<IRFuncCall>(name);
    for (const auto& arg : args) {
        copy->args.push_back(arg->clone());
    }
    return copy;
}

void IRIndexedAccess::accept(IRExprVisitor& v) const { v.visit(*this); }

std::unique_ptr<IRExpr> IRIndexedAccess::clone() const {
    auto copy = std::make_unique<IRIndexedAccess>(baseName);
    for (const auto& index : indices) {
        copy->indices.push_back(index->clone());
    }
    return copy;
}

void IRCompareExpr::accept(IRExprVisitor& v) const { v.visit(*this); }

std::unique_ptr<IRExpr> IRCompareExpr::clone() const {
    return std::make_unique<IRCompareExpr>(op, lhs->clone(), rhs->clone());
}

std::unique_ptr<IRStmt> IRScalarDecl::clone() const {
    return std::make_unique<IRScalarDecl>(varName, initValue);
}

std::unique_ptr<IRStmt> IRAssign::clone() const {
    return std::make_unique<IRAssign>(lhs->clone(), rhs->clone(), accumulate);
}

std::unique_ptr<IRStmt> IRCallStmt::clone() const {
    auto copy = std::make_unique<IRCallStmt>(functionName);
    for (const auto& arg : args) {
        copy->args.push_back(arg->clone());
    }
    return copy;
}

std::unique_ptr<IRStmt> IRRawStmt::clone() const {
    return std::make_unique<IRRawStmt>(code);
}

std::unique_ptr<IRStmt> IRVarDecl::clone() const {
    return std::make_unique<IRVarDecl>(varName, type, initExpr);
}

std::unique_ptr<IRStmt> IRFreeStmt::clone() const {
    return std::make_unique<IRFreeStmt>(varName);
}

std::unique_ptr<IRStmt> IRIfStmt::clone() const {
    auto copy = std::make_unique<IRIfStmt>(condition ? condition->clone() : nullptr);
    for (const auto& stmt : thenBody) {
        copy->thenBody.push_back(stmt->clone());
    }
    return copy;
}

std::unique_ptr<IRStmt> IRForStmt::clone() const {
    auto copy = std::make_unique<IRForStmt>(
        loopVar,
        lower ? lower->clone() : nullptr,
        upper ? upper->clone() : nullptr);
    for (const auto& stmt : body) {
        copy->body.push_back(stmt->clone());
    }
    return copy;
}

namespace {

class ExprRenderer : public IRExprVisitor {
public:
    std::string result;

    void visit(const IRTensorAccess& n) override {
        if (n.isSparseVals) {
            result = n.tensorName + "_vals[" + n.pointerVar + "]";
            return;
        }
        if (n.useRandomAccess && n.indices.size() == 2) {
            const std::string& func = n.randomAccessFunc.empty() ? "sp_csr_get" : n.randomAccessFunc;
            result = func + "(" + n.tensorName + ", " + n.indices[0] + ", " + n.indices[1] + ")";
            return;
        }
        if (n.indices.empty()) {
            result = n.tensorName;
            return;
        }
        result = n.tensorName;
        for (const auto& index : n.indices) {
            result += "[" + index + "]";
        }
    }

    void visit(const IRConstant& n) override {
        if (n.value == std::floor(n.value) && std::abs(n.value) < 1e15) {
            std::ostringstream oss;
            oss << static_cast<long long>(n.value);
            result = oss.str();
            return;
        }
        std::ostringstream oss;
        oss << n.value;
        result = oss.str();
    }

    void visit(const IRBinaryOp& n) override {
        ExprRenderer lhsRenderer;
        ExprRenderer rhsRenderer;
        n.lhs->accept(lhsRenderer);
        n.rhs->accept(rhsRenderer);
        result = lhsRenderer.result + (n.op == IRBinaryOp::ADD ? " + " : " * ") + rhsRenderer.result;
    }

    void visit(const IRScalarVar& n) override {
        result = n.name;
    }

    void visit(const IRFuncCall& n) override {
        result = n.name + "(";
        for (size_t i = 0; i < n.args.size(); ++i) {
            if (i > 0) {
                result += ", ";
            }
            ExprRenderer argRenderer;
            n.args[i]->accept(argRenderer);
            result += argRenderer.result;
        }
        result += ")";
    }

    void visit(const IRIndexedAccess& n) override {
        result = n.baseName;
        for (const auto& index : n.indices) {
            ExprRenderer indexRenderer;
            index->accept(indexRenderer);
            result += "[" + indexRenderer.result + "]";
        }
    }

    void visit(const IRCompareExpr& n) override {
        ExprRenderer lhsRenderer;
        ExprRenderer rhsRenderer;
        n.lhs->accept(lhsRenderer);
        n.rhs->accept(rhsRenderer);
        const char* op = (n.op == IRCompareExpr::EQ) ? " == " : " < ";
        result = lhsRenderer.result + op + rhsRenderer.result;
    }
};

} // namespace

std::string renderExpr(const IRExpr& expr) {
    ExprRenderer renderer;
    const_cast<IRExpr&>(expr).accept(renderer);
    return renderer.result;
}

std::string renderStmt(const IRStmt& stmt) {
    if (auto* decl = dynamic_cast<const IRScalarDecl*>(&stmt)) {
        std::ostringstream oss;
        oss << "double " << decl->varName << " = ";
        if (decl->initValue == 0.0) {
            oss << "0.0";
        } else {
            oss << decl->initValue;
        }
        oss << ";";
        return oss.str();
    }
    if (auto* assign = dynamic_cast<const IRAssign*>(&stmt)) {
        return renderExpr(*assign->lhs) +
               (assign->accumulate ? " += " : " = ") +
               renderExpr(*assign->rhs) + ";";
    }
    if (auto* call = dynamic_cast<const IRCallStmt*>(&stmt)) {
        std::string rendered = call->functionName + "(";
        for (size_t i = 0; i < call->args.size(); ++i) {
            if (i > 0) {
                rendered += ", ";
            }
            rendered += renderExpr(*call->args[i]);
        }
        rendered += ");";
        return rendered;
    }
    if (auto* raw = dynamic_cast<const IRRawStmt*>(&stmt)) {
        return raw->code;
    }
    if (auto* varDecl = dynamic_cast<const IRVarDecl*>(&stmt)) {
        return varDecl->type + " " + varDecl->varName + " = " + varDecl->initExpr + ";";
    }
    if (auto* freeStmt = dynamic_cast<const IRFreeStmt*>(&stmt)) {
        return "free(" + freeStmt->varName + ");";
    }
    if (auto* ifStmt = dynamic_cast<const IRIfStmt*>(&stmt)) {
        std::string rendered = "if (" + renderExpr(*ifStmt->condition) + ") {\n";
        for (const auto& bodyStmt : ifStmt->thenBody) {
            rendered += "  " + renderStmt(*bodyStmt) + "\n";
        }
        rendered += "}";
        return rendered;
    }
    if (auto* forStmt = dynamic_cast<const IRForStmt*>(&stmt)) {
        std::string rendered = "for (int " + forStmt->loopVar + " = " +
            renderExpr(*forStmt->lower) + "; " + forStmt->loopVar + " < " +
            renderExpr(*forStmt->upper) + "; " + forStmt->loopVar + "++) {\n";
        for (const auto& bodyStmt : forStmt->body) {
            rendered += "  " + renderStmt(*bodyStmt) + "\n";
        }
        rendered += "}";
        return rendered;
    }
    return "/* unknown stmt */";
}

void renderStmtsToStrings(const std::vector<std::unique_ptr<IRStmt>>& preStmts,
                          const std::vector<std::unique_ptr<IRStmt>>& postStmts,
                          std::string& preBody,
                          std::string& body) {
    preBody.clear();
    body.clear();

    for (const auto& stmt : preStmts) {
        if (!preBody.empty()) {
            preBody += " ";
        }
        preBody += renderStmt(*stmt);
    }
    for (const auto& stmt : postStmts) {
        if (!body.empty()) {
            body += " ";
        }
        body += renderStmt(*stmt);
    }
}

std::string formatToString(Format f) {
    switch (f) {
        case Format::Dense: return "Dense";
        case Format::CSR: return "CSR";
        case Format::CSC: return "CSC";
        default: return "Unknown";
    }
}

std::string rootOpKindToString(RootOpKind k) {
    switch (k) {
        case RootOpKind::ADD: return "ADD";
        case RootOpKind::MULT: return "MULT";
        default: return "unknown";
    }
}

std::string mergeStrategyToString(MergeStrategy m) {
    switch (m) {
        case MergeStrategy::None: return "none";
        case MergeStrategy::Union: return "union";
        case MergeStrategy::Intersection: return "intersection";
        default: return "unknown";
    }
}

void printExpressionInfo(std::ostream& out, const ExpressionInfo& info) {
    out << "Expression Info:\n";
    out << "  Root operator: " << rootOpKindToString(info.rootOp) << "\n";
    out << "  Tensor accesses: " << info.numTensorAccesses << "\n";
    out << "  Sparse inputs: " << info.numSparseInputs;
    if (!info.sparseInputNames.empty()) {
        out << " (";
        for (size_t i = 0; i < info.sparseInputNames.size(); ++i) {
            if (i > 0) {
                out << ", ";
            }
            out << info.sparseInputNames[i];
        }
        out << ")";
    }
    out << "\n";
    out << "  Dense inputs: " << info.numDenseInputs;
    if (!info.denseInputNames.empty()) {
        out << " (";
        for (size_t i = 0; i < info.denseInputNames.size(); ++i) {
            if (i > 0) {
                out << ", ";
            }
            out << info.denseInputNames[i];
        }
        out << ")";
    }
    out << "\n";
    if (info.isFused) {
        out << "  Fused: " << info.fusionFunction << "\n";
    }
}

} // namespace ir
