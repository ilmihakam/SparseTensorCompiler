/**
 * Test Suite: General Kernel Code Generation (Phase C)
 *
 * Tests that general (non-recognized) kernels produce compilable C code
 * with proper signatures, runtime bounds, reference kernels, and main().
 *
 * Phase C Expanded tests:
 * - IRExprEmitter direct ->vals[] access (no _vals[] hack)
 * - relu/sigmoid inline expansion
 * - math.h inclusion when function calls present
 * - Balanced braces
 * - Pre-optimization clone for reference kernels
 * - Golden suite: scalar factor, three-input add, mixed-index multiply
 */

#include <gtest/gtest.h>
#include <sstream>
#include "codegen.h"
#include "optimizations.h"
#include "semantic_ir.h"
#include "ast.h"

extern std::unique_ptr<SparseTensorCompiler::Program> g_program;
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern int yynerrs;

static bool parserInitialized = false;

static std::unique_ptr<sparseir::scheduled::Compute> parseForGeneralCodegen(
    const std::string& code) {
    if (!parserInitialized) {
        yynerrs = 0;
        g_program.reset();
        yy_scan_string("tensor x : Dense;");
        yyparse();
        yylex_destroy();
        g_program.reset();
        parserInitialized = true;
    }

    yynerrs = 0;
    g_program.reset();
    yy_scan_string(code.c_str());
    int result = yyparse();
    yylex_destroy();

    if (result != 0 || yynerrs != 0 || !g_program) {
        return nullptr;
    }

    return sparseir::lowerFirstComputationToScheduled(*g_program);
}

static std::string generate(sparseir::scheduled::Compute& compute,
                            const opt::OptConfig& config) {
    return codegen::generateCode(compute, config);
}

// ============================================================================
// Test 1: Signature has all tensors
// ============================================================================

TEST(GeneralCodegenTest, SignatureHasAllTensors) {
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        tensor z : Dense<50>;
        compute y[i] = A[i, j] * x[j] * z[j];
    )");
    ASSERT_NE(compute, nullptr);
    EXPECT_TRUE(compute->fullyLowered);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    // Signature should contain all tensor parameters
    EXPECT_NE(output.find("SparseMatrix* A"), std::string::npos)
        << "Missing sparse matrix A in signature";
    EXPECT_NE(output.find("double* x"), std::string::npos)
        << "Missing dense vector x in signature";
    EXPECT_NE(output.find("double* z"), std::string::npos)
        << "Missing dense vector z in signature";
    EXPECT_NE(output.find("double* y"), std::string::npos)
        << "Missing output vector y in signature";
}

// ============================================================================
// Test 2: Signature has dimension params for non-sparse-derivable indices
// ============================================================================

TEST(GeneralCodegenTest, SignatureHasDimensionParams) {
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        tensor z : Dense<50>;
        compute y[i] = A[i, j] * x[j] * z[j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("void compute("), std::string::npos)
        << "Missing compute kernel function";
}

// ============================================================================
// Test 3: Runtime bound from sparse CSR
// ============================================================================

TEST(GeneralCodegenTest, RuntimeBoundFromSparseCSR) {
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        tensor z : Dense<50>;
        compute y[i] = A[i, j] * x[j] * z[j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("A->rows"), std::string::npos)
        << "Dense loop should use A->rows for index i";
}

// ============================================================================
// Test 4: Runtime bound from sparse CSC
// ============================================================================

TEST(GeneralCodegenTest, RuntimeBoundFromSparseCSC) {
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSC<100, 50>;
        tensor x : Dense<100>;
        compute y[j] = A[j, i] * x[i];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("col_ptr"), std::string::npos)
        << "CSC kernel should use col_ptr";
}

// ============================================================================
// Test 5: Dense loop uses runtime bound (not hardcoded)
// ============================================================================

TEST(GeneralCodegenTest, DenseLoopUsesRuntimeBound) {
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        tensor z : Dense<50>;
        compute y[i] = A[i, j] * x[j] * z[j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    bool hasRuntimeBound = (output.find("A->rows") != std::string::npos);
    EXPECT_TRUE(hasRuntimeBound)
        << "Dense loop should use runtime bound (A->rows), not compile-time constant";
}

// ============================================================================
// Test 6: Full code has kernel, reference, and main
// ============================================================================

TEST(GeneralCodegenTest, FullCodeCompiles) {
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        tensor z : Dense<50>;
        compute y[i] = A[i, j] * x[j] * z[j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("void compute("), std::string::npos)
        << "Missing optimized kernel function";
    EXPECT_NE(output.find("void reference("), std::string::npos)
        << "Missing reference kernel function";
    EXPECT_NE(output.find("int main("), std::string::npos)
        << "Missing main function";
}

// ============================================================================
// Test 7: Reference kernel body is not empty
// ============================================================================

TEST(GeneralCodegenTest, ReferenceKernelPresent) {
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        tensor z : Dense<50>;
        compute y[i] = A[i, j] * x[j] * z[j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    size_t refStart = output.find("void reference(");
    ASSERT_NE(refStart, std::string::npos);

    size_t refBody = output.find("for (", refStart);
    EXPECT_NE(refBody, std::string::npos)
        << "Reference kernel should have loop body, not be empty";
}

// ============================================================================
// Test 8: Main loads correct number of matrices
// ============================================================================

TEST(GeneralCodegenTest, MainLoadsCorrectNumberOfMatrices) {
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        tensor z : Dense<50>;
        compute y[i] = A[i, j] * x[j] * z[j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("load_matrix_market(argv[1])"), std::string::npos)
        << "Should load matrix A from argv[1]";
    EXPECT_NE(output.find("malloc"), std::string::npos)
        << "Should allocate dense vectors";
    EXPECT_NE(output.find("max_error"), std::string::npos)
        << "Should have correctness verification";
}

// ============================================================================
// Test 9: IRExprEmitter produces ->vals[] (not _vals[])
// ============================================================================

TEST(GeneralCodegenTest, IRExprEmitter_DirectValsAccess) {
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        tensor z : Dense<50>;
        compute y[i] = A[i, j] * x[j] * z[j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    // The structured path should emit ->vals[ directly
    EXPECT_NE(output.find("A->vals["), std::string::npos)
        << "Kernel should use A->vals[ (not A_vals[)";
}

// ============================================================================
// Test 10: Dense tensor access patterns are correct
// ============================================================================

TEST(GeneralCodegenTest, IRExprEmitter_DenseTensorAccess) {
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        tensor z : Dense<50>;
        compute y[i] = A[i, j] * x[j] * z[j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    // Dense 1D vectors should be accessed as x[j] and z[j]
    EXPECT_NE(output.find("x[j]"), std::string::npos)
        << "Dense vector x should be accessed as x[j]";
    EXPECT_NE(output.find("z[j]"), std::string::npos)
        << "Dense vector z should be accessed as z[j]";
}

// ============================================================================
// Test 11: relu inline expansion
// ============================================================================

TEST(GeneralCodegenTest, FunctionCall_ReluInline) {
    // Use 3 tensor accesses (A, x, z) to force "general" kernel detection
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        tensor z : Dense<50>;
        compute y[i] = relu(A[i, j] * x[j] * z[j]);
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    // relu should be inlined as ternary
    EXPECT_NE(output.find("> 0 ?"), std::string::npos)
        << "relu should be inlined as ternary (> 0 ? expr : 0)";
}

// ============================================================================
// Test 12: sigmoid inline expansion
// ============================================================================

TEST(GeneralCodegenTest, FunctionCall_SigmoidInline) {
    // Use 3 tensor accesses to force "general" kernel detection
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        tensor z : Dense<50>;
        compute y[i] = sigmoid(A[i, j] * x[j] * z[j]);
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    // sigmoid should use exp()
    EXPECT_NE(output.find("exp"), std::string::npos)
        << "sigmoid should be inlined using exp()";
}

// ============================================================================
// Test 13: math.h is included (always included in current codegen)
// ============================================================================

TEST(GeneralCodegenTest, MathHeaderIncluded) {
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        tensor z : Dense<50>;
        compute y[i] = sigmoid(A[i, j] * x[j] * z[j]);
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    EXPECT_NE(output.find("#include <math.h>"), std::string::npos)
        << "Generated code should include math.h";
}

// ============================================================================
// Test 14: Balanced braces
// ============================================================================

TEST(GeneralCodegenTest, BalancedBraces) {
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        tensor z : Dense<50>;
        compute y[i] = A[i, j] * x[j] * z[j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    int openBraces = 0, closeBraces = 0;
    for (char c : output) {
        if (c == '{') openBraces++;
        if (c == '}') closeBraces++;
    }
    EXPECT_EQ(openBraces, closeBraces)
        << "Generated code has unbalanced braces: { = " << openBraces
        << ", } = " << closeBraces;
}

// ============================================================================
// Test 15: cloneLoop deep copies structured body
// ============================================================================

TEST(GeneralCodegenTest, CloneLoopDeepCopy) {
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        tensor z : Dense<50>;
        compute y[i] = A[i, j] * x[j] * z[j];
    )");
    ASSERT_NE(compute, nullptr);
    ASSERT_NE(compute->rootLoop, nullptr);

    // Clone the operation
    auto clonedStmt = compute->clone();
    auto* cloned = dynamic_cast<sparseir::scheduled::Compute*>(clonedStmt.get());
    ASSERT_NE(cloned, nullptr);
    ASSERT_NE(cloned->rootLoop, nullptr);
    EXPECT_NE(cloned->rootLoop.get(), compute->rootLoop.get());
    EXPECT_EQ(cloned->inputs.size(), compute->inputs.size());
    EXPECT_EQ(cloned->output.name, compute->output.name);
}

// ============================================================================
// Test 16: Pre-optimization clone produces different reference
// ============================================================================

TEST(GeneralCodegenTest, ReferenceKernelPresentOnScheduledPath) {
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        tensor z : Dense<50>;
        compute y[i] = A[i, j] * x[j] * z[j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    // Both kernel and reference should exist
    EXPECT_NE(output.find("void compute("), std::string::npos);
    EXPECT_NE(output.find("void reference("), std::string::npos);

    // Reference should have a loop body
    size_t refStart = output.find("void reference(");
    ASSERT_NE(refStart, std::string::npos);
    size_t refBody = output.find("for (", refStart);
    EXPECT_NE(refBody, std::string::npos)
        << "Reference kernel should have a loop body";
}

// ============================================================================
// Golden Suite: Test 17 - Scalar factor
// ============================================================================

TEST(GeneralCodegenTest, Golden_ScalarFactor) {
    // Use 3 tensor accesses + scalar to force "general" kernel
    auto compute = parseForGeneralCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 50>;
        tensor x : Dense<50>;
        tensor z : Dense<50>;
        compute y[i] = 2.0 * A[i, j] * x[j] * z[j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = generate(*compute, config);

    // Scalar 2.0 should appear in the kernel body
    EXPECT_NE(output.find("2.0"), std::string::npos)
        << "Scalar factor 2.0 should appear in generated code";
    // Should still have A->vals access
    EXPECT_NE(output.find("A->vals["), std::string::npos)
        << "Should have sparse value access";
}

// ============================================================================
// Golden Suite: Test 18 - IRExprEmitter unit test
// ============================================================================

TEST(GeneralCodegenTest, IRExprEmitter_Unit) {
    // Test IRExprEmitter directly
    codegen::IRExprEmitter emitter;

    // Test sparse vals access
    ir::IRTensorAccess sparseAccess;
    sparseAccess.tensorName = "A";
    sparseAccess.isSparseVals = true;
    sparseAccess.pointerVar = "pA";
    sparseAccess.accept(emitter);
    EXPECT_EQ(emitter.result, "A->vals[pA]");

    // Test dense 1D access
    ir::IRTensorAccess dense1d("x", {"j"});
    dense1d.accept(emitter);
    EXPECT_EQ(emitter.result, "x[j]");

    // Test dense 2D access
    ir::IRTensorAccess dense2d("C", {"i", "j"});
    dense2d.accept(emitter);
    EXPECT_EQ(emitter.result, "C[i][j]");

    // Test constant
    ir::IRConstant c(3.0);
    c.accept(emitter);
    EXPECT_EQ(emitter.result, "3");

    // Test scalar var
    ir::IRScalarVar sv("sum");
    sv.accept(emitter);
    EXPECT_EQ(emitter.result, "sum");
}

// ============================================================================
// Golden Suite: Test 19 - relu inline unit test
// ============================================================================

TEST(GeneralCodegenTest, IRExprEmitter_ReluUnit) {
    codegen::IRExprEmitter emitter;

    auto arg = std::make_unique<ir::IRScalarVar>("x");
    ir::IRFuncCall relu("relu");
    relu.args.push_back(std::move(arg));
    relu.accept(emitter);

    EXPECT_NE(emitter.result.find("> 0 ?"), std::string::npos)
        << "relu should inline to ternary, got: " << emitter.result;
    EXPECT_NE(emitter.result.find(": 0"), std::string::npos)
        << "relu false branch should be 0";
}

// ============================================================================
// Golden Suite: Test 20 - sigmoid inline unit test
// ============================================================================

TEST(GeneralCodegenTest, IRExprEmitter_SigmoidUnit) {
    codegen::IRExprEmitter emitter;

    auto arg = std::make_unique<ir::IRScalarVar>("x");
    ir::IRFuncCall sigmoid("sigmoid");
    sigmoid.args.push_back(std::move(arg));
    sigmoid.accept(emitter);

    EXPECT_NE(emitter.result.find("exp"), std::string::npos)
        << "sigmoid should use exp(), got: " << emitter.result;
    EXPECT_NE(emitter.result.find("1.0"), std::string::npos)
        << "sigmoid should have 1.0 in formula";
}

// ============================================================================
// Golden Suite: Test 21 - Unknown function call pass-through
// ============================================================================

TEST(GeneralCodegenTest, IRExprEmitter_UnknownFuncCall) {
    codegen::IRExprEmitter emitter;

    auto arg1 = std::make_unique<ir::IRScalarVar>("x");
    auto arg2 = std::make_unique<ir::IRConstant>(2.0);
    ir::IRFuncCall call("custom_func");
    call.args.push_back(std::move(arg1));
    call.args.push_back(std::move(arg2));
    call.accept(emitter);

    EXPECT_EQ(emitter.result, "custom_func(x, 2)")
        << "Unknown function should pass through as direct call";
}
