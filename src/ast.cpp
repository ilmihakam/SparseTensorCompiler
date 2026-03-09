#include "ast.h"

namespace SparseTensorCompiler {

// ============================================
// Program Node
// ============================================
void Program::accept(ASTVisitor& visitor) {
    visitor.visit(*this);
}

// ============================================
// Declaration Statement
// ============================================
void Declaration::accept(ASTVisitor& visitor) {
    visitor.visit(*this);
}

// ============================================
// Computation Statement
// ============================================
void Computation::accept(ASTVisitor& visitor) {
    visitor.visit(*this);
}

// ============================================
// Call Statement
// ============================================
void CallStatement::accept(ASTVisitor& visitor) {
    visitor.visit(*this);
}

// ============================================
// For Statement
// ============================================
void ForStatement::accept(ASTVisitor& visitor) {
    visitor.visit(*this);
}

// ============================================
// Tensor Access Expression
// ============================================
void TensorAccess::accept(ASTVisitor& visitor) {
    visitor.visit(*this);
}

// ============================================
// Function Call Expression
// ============================================
void FunctionCall::accept(ASTVisitor& visitor) {
    visitor.visit(*this);
}

// ============================================
// Binary Operation Expression
// ============================================
void BinaryOp::accept(ASTVisitor& visitor) {
    visitor.visit(*this);
}

// ============================================
// Number Literal Expression
// ============================================
void Number::accept(ASTVisitor& visitor) {
    visitor.visit(*this);
}

// ============================================
// Identifier Expression
// ============================================
void Identifier::accept(ASTVisitor& visitor) {
    visitor.visit(*this);
}

} // namespace SparseTensorCompiler
