#ifndef AST_H
#define AST_H

#include <string>
#include <vector>
#include <memory>

namespace SparseTensorCompiler {

// ============================================
// Forward Declarations
// ============================================
class ASTNode;
class Statement;
class Expression;
class Declaration;
class Computation;
class CallStatement;
class ForStatement;
class TensorAccess;
class FunctionCall;
class BinaryOp;
class Number;
class Identifier;
class Program;

// ============================================
// Base AST Node Class
// ============================================
class ASTNode {
public:
    virtual ~ASTNode() = default;

    // Virtual method for visitor pattern (useful for IR lowering later)
    virtual void accept(class ASTVisitor& visitor) = 0;
};

// ============================================
// Statement Base Class
// ============================================
class Statement : public ASTNode {
public:
    virtual ~Statement() = default;
};

// ============================================
// Expression Base Class
// ============================================
class Expression : public ASTNode {
public:
    virtual ~Expression() = default;
};

// ============================================
// Program Node - Root of AST
// ============================================
class Program : public ASTNode {
public:
    std::vector<std::unique_ptr<Statement>> statements;

    Program() = default;

    void addStatement(std::unique_ptr<Statement> stmt) {
        statements.push_back(std::move(stmt));
    }

    void accept(ASTVisitor& visitor) override;
};

// ============================================
// Declaration Statement
// FORMAT: tensor A : Dense<10, 20>;
// ============================================
class Declaration : public Statement {
public:
    std::string tensorName;              // "A"
    std::string tensorType;              // Dense, CSR, COO, CSC, ELLPACK, DIA (required)
    std::vector<std::string> shape;      // Shape dimensions: [10, 20, 30] (optional)

    // Constructor: tensor A : Dense<10, 20>;
    Declaration(const std::string& name, const std::string& type,
                const std::vector<std::string>& shapeList = std::vector<std::string>())
        : tensorName(name), tensorType(type), shape(shapeList) {}

    void accept(ASTVisitor& visitor) override;
};

// ============================================
// Computation Statement
// Format: compute C[i] = A[i] + B[i];
// ============================================
class Computation : public Statement {
public:
    std::unique_ptr<TensorAccess> lhs;
    std::unique_ptr<Expression> rhs;

    Computation(std::unique_ptr<TensorAccess> left, std::unique_ptr<Expression> right)
        : lhs(std::move(left)), rhs(std::move(right)) {}

    void accept(ASTVisitor& visitor) override;
};

// ============================================
// Call Statement
// Format: call optimize(A);
// ============================================
class CallStatement : public Statement {
public:
    std::string functionName;
    std::vector<std::unique_ptr<Expression>> arguments;

    CallStatement(const std::string& name) : functionName(name) {}

    void addArgument(std::unique_ptr<Expression> arg) {
        arguments.push_back(std::move(arg));
    }

    void accept(ASTVisitor& visitor) override;
};

// ============================================
// For Statement
// Format: for [A, B] [i, j] { statements }
// ============================================
class ForStatement : public Statement {
public:
    std::vector<std::string> tensors;
    std::vector<std::string> indices;
    std::vector<std::unique_ptr<Statement>> body;

    ForStatement(const std::vector<std::string>& tensorList,
                 const std::vector<std::string>& indexList)
        : tensors(tensorList), indices(indexList) {}

    void addStatement(std::unique_ptr<Statement> stmt) {
        body.push_back(std::move(stmt));
    }

    void accept(ASTVisitor& visitor) override;
};

// ============================================
// Tensor Access Expression
// Format: A[i, j]
// ============================================
class TensorAccess : public Expression {
public:
    std::string tensorName;
    std::vector<std::string> indices;

    TensorAccess(const std::string& name, const std::vector<std::string>& idx)
        : tensorName(name), indices(idx) {}

    void accept(ASTVisitor& visitor) override;
};

// ============================================
// Function Call Expression
// Format: relu(A[i])
// ============================================
class FunctionCall : public Expression {
public:
    std::string functionName;
    std::vector<std::unique_ptr<Expression>> arguments;

    FunctionCall(const std::string& name) : functionName(name) {}

    void addArgument(std::unique_ptr<Expression> arg) {
        arguments.push_back(std::move(arg));
    }

    void accept(ASTVisitor& visitor) override;
};

// ============================================
// Binary Operation Expression
// Format: A[i] + B[i] or A[i] * B[i]
// ============================================
class BinaryOp : public Expression {
public:
    enum Operator { ADD, MULT };

    Operator op;
    std::unique_ptr<Expression> left;
    std::unique_ptr<Expression> right;

    BinaryOp(Operator operation,
             std::unique_ptr<Expression> lhs,
             std::unique_ptr<Expression> rhs)
        : op(operation), left(std::move(lhs)), right(std::move(rhs)) {}

    void accept(ASTVisitor& visitor) override;
};

// ============================================
// Number Literal Expression
// Format: 42 or 3.14
// ============================================
class Number : public Expression {
public:
    std::string value;

    explicit Number(const std::string& val) : value(val) {}

    void accept(ASTVisitor& visitor) override;
};

// ============================================
// Identifier Expression
// Format: A (bare identifier without indices)
// ============================================
class Identifier : public Expression {
public:
    std::string name;

    explicit Identifier(const std::string& n) : name(n) {}

    void accept(ASTVisitor& visitor) override;
};

// ============================================
// Visitor Interface (for future IR lowering)
// ============================================
class ASTVisitor {
public:
    virtual ~ASTVisitor() = default;

    virtual void visit(Program& node) = 0;
    virtual void visit(Declaration& node) = 0;
    virtual void visit(Computation& node) = 0;
    virtual void visit(CallStatement& node) = 0;
    virtual void visit(ForStatement& node) = 0;
    virtual void visit(TensorAccess& node) = 0;
    virtual void visit(FunctionCall& node) = 0;
    virtual void visit(BinaryOp& node) = 0;
    virtual void visit(Number& node) = 0;
    virtual void visit(Identifier& node) = 0;
};

} // namespace SparseTensorCompiler

#endif // AST_H
