/**
 * Test Suite: Code Generator Framework
 *
 * Tests the CodeGenerator class infrastructure including
 * indentation, emission helpers, and section generation.
 */

#include <gtest/gtest.h>
#include <sstream>
#include <vector>
#include "codegen.h"
#include "optimizations.h"
#include "scheduled_optimizations.h"
#include "semantic_ir.h"
#include "ast.h"

extern std::unique_ptr<SparseTensorCompiler::Program> g_program;
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern int yynerrs;

static bool parserInitialized = false;

std::unique_ptr<sparseir::scheduled::Compute> parseAndLowerForCodegen(const std::string& code) {
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

static std::string extractComputeBody(const std::string& code) {
    size_t sig = code.find("void compute(");
    if (sig == std::string::npos) return "";
    size_t bodyStart = code.find('{', sig);
    if (bodyStart == std::string::npos) return "";

    int depth = 0;
    size_t bodyEnd = std::string::npos;
    for (size_t i = bodyStart; i < code.size(); ++i) {
        if (code[i] == '{') depth++;
        else if (code[i] == '}') {
            depth--;
            if (depth == 0) {
                bodyEnd = i;
                break;
            }
        }
    }
    if (bodyEnd == std::string::npos || bodyEnd <= bodyStart + 1) return "";
    return code.substr(bodyStart + 1, bodyEnd - bodyStart - 1);
}

static std::string dedentCommonIndentation(const std::string& text) {
    std::istringstream in(text);
    std::string line;
    std::vector<std::string> lines;
    size_t commonIndent = std::string::npos;

    while (std::getline(in, line)) {
        lines.push_back(line);
        const size_t firstNonSpace = line.find_first_not_of(' ');
        if (firstNonSpace == std::string::npos) {
            continue;
        }
        if (commonIndent == std::string::npos || firstNonSpace < commonIndent) {
            commonIndent = firstNonSpace;
        }
    }

    if (commonIndent == std::string::npos) {
        return "";
    }

    std::ostringstream out;
    for (size_t i = 0; i < lines.size(); ++i) {
        std::string trimmed = lines[i];
        if (!trimmed.empty() && trimmed.find_first_not_of(' ') != std::string::npos &&
            trimmed.size() >= commonIndent) {
            trimmed = trimmed.substr(commonIndent);
        }
        while (!trimmed.empty() && (trimmed.back() == '\r' || trimmed.back() == ' ')) {
            trimmed.pop_back();
        }
        out << trimmed;
        if (i + 1 < lines.size()) out << '\n';
    }

    std::string result = out.str();
    while (!result.empty() && (result.front() == '\n' || result.front() == '\r')) {
        result.erase(result.begin());
    }
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
    }
    return result;
}

// ============================================================================
// CodeGenerator Construction Tests
// ============================================================================

/**
 * Test: Can instantiate CodeGenerator with output stream.
 */
TEST(CodegenFrameworkTest, CreateCodeGenerator) {
    std::ostringstream out;
    codegen::CodeGenerator gen(out);

    // Should be constructible and have initial state
    EXPECT_EQ(gen.getIndentLevel(), 0);
}

/**
 * Test: Indent increases/decreases correctly.
 */
TEST(CodegenFrameworkTest, IndentationTracking) {
    std::ostringstream out;
    codegen::CodeGenerator gen(out);

    EXPECT_EQ(gen.getIndentLevel(), 0);

    // After generation with nested structures, indent should return to 0
    auto op = parseAndLowerForCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    gen.generate(*op, config);

    // After full generation, indent should be back to 0
    EXPECT_EQ(gen.getIndentLevel(), 0);
}

// ============================================================================
// Output Generation Tests
// ============================================================================

/**
 * Test: Generated output contains standard includes.
 */
TEST(CodegenFrameworkTest, HeaderGeneration) {
    std::ostringstream out;
    codegen::CodeGenerator gen(out);

    auto op = parseAndLowerForCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    gen.generate(*op, config);

    std::string output = out.str();

    // Should contain standard includes
    EXPECT_NE(output.find("#include <stdio.h>"), std::string::npos);
    EXPECT_NE(output.find("#include <stdlib.h>"), std::string::npos);
    EXPECT_NE(output.find("#include <time.h>"), std::string::npos);
}

/**
 * Test: Injects comments showing optimization status.
 */
TEST(CodegenFrameworkTest, OptimizationComments_Baseline) {
    std::ostringstream out;
    codegen::CodeGenerator gen(out);

    auto op = parseAndLowerForCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    gen.generate(*op, config);

    std::string output = out.str();

    // Should indicate format-correctness status and optimization toggles
    EXPECT_NE(output.find("Format-correctness reordering"), std::string::npos);
    EXPECT_NE(output.find("interchange"), std::string::npos);
    EXPECT_NE(output.find("blocking"), std::string::npos);
}

/**
 * Test: Comments reflect format-correctness reordering status.
 */
TEST(CodegenFrameworkTest, OptimizationComments_FormatCorrectness) {
    std::ostringstream out;
    codegen::CodeGenerator gen(out);

    auto op = parseAndLowerForCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    opt::applyOptimizations(*op, config);
    gen.generate(*op, config);

    std::string output = out.str();

    // Should mention format-correctness reordering
    EXPECT_NE(output.find("Format-correctness reordering"), std::string::npos);
}

/**
 * Test: Comments reflect blocking configuration with block size.
 */
TEST(CodegenFrameworkTest, OptimizationComments_BlockingOnly) {
    std::ostringstream out;
    codegen::CodeGenerator gen(out);

    auto op = parseAndLowerForCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::blockingOnly(32);
    opt::applyOptimizations(*op, config);
    gen.generate(*op, config);

    std::string output = out.str();

    // Should mention blocking and size
    EXPECT_NE(output.find("blocking=ON"), std::string::npos);
    EXPECT_NE(output.find("32"), std::string::npos);
}

/**
 * Test: Generates timing harness function.
 */
TEST(CodegenFrameworkTest, TimingHarnessGeneration) {
    std::ostringstream out;
    codegen::CodeGenerator gen(out);

    auto op = parseAndLowerForCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    gen.generate(*op, config);

    std::string output = out.str();

    // Should contain timing function
    EXPECT_NE(output.find("get_time_ms"), std::string::npos);
    EXPECT_NE(output.find("clock"), std::string::npos);
}

TEST(CodegenFrameworkTest, InlineComputeMatchesGeneratedKernelBody_SpMM) {
    auto compute = parseAndLowerForCodegen(R"(
        tensor A : CSC<100, 100>;
        tensor B : Dense<100, 32>;
        tensor C : Dense<100, 32>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();

    std::ostringstream fullOut;
    codegen::CodeGenerator fullGen(fullOut);
    fullGen.generate(*compute, config);

    std::ostringstream inlineOut;
    codegen::CodeGenerator inlineGen(inlineOut);
    inlineGen.emitInlineScheduledCompute(*compute, 0);

    EXPECT_EQ(
        dedentCommonIndentation(extractComputeBody(fullOut.str())),
        dedentCommonIndentation(inlineOut.str()));
}

TEST(CodegenFrameworkTest, InlineComputeMatchesGeneratedKernelBody_SpGEMM) {
    auto compute = parseAndLowerForCodegen(R"(
        tensor A : CSR<100, 50>;
        tensor B : CSR<50, 100>;
        tensor C : CSR<100, 100>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();

    std::ostringstream fullOut;
    codegen::CodeGenerator fullGen(fullOut);
    fullGen.generate(*compute, config);

    std::ostringstream inlineOut;
    codegen::CodeGenerator inlineGen(inlineOut);
    inlineGen.emitInlineScheduledCompute(*compute, 0);

    EXPECT_EQ(
        dedentCommonIndentation(extractComputeBody(fullOut.str())),
        dedentCommonIndentation(inlineOut.str()));
}

/**
 * Test: Generates main() function skeleton.
 */
TEST(CodegenFrameworkTest, MainFunctionStructure) {
    std::ostringstream out;
    codegen::CodeGenerator gen(out);

    auto op = parseAndLowerForCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    gen.generate(*op, config);

    std::string output = out.str();

    // Should contain main function
    EXPECT_NE(output.find("int main("), std::string::npos);
    EXPECT_NE(output.find("argc"), std::string::npos);
    EXPECT_NE(output.find("argv"), std::string::npos);
}

/**
 * Test: Generates Matrix Market parsing code.
 */
TEST(CodegenFrameworkTest, MatrixMarketLoader) {
    std::ostringstream out;
    codegen::CodeGenerator gen(out);

    auto op = parseAndLowerForCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    gen.generate(*op, config);

    std::string output = out.str();

    // Should contain MM loader structures and functions
    EXPECT_NE(output.find("SparseMatrix"), std::string::npos);
    EXPECT_NE(output.find("load_matrix_market"), std::string::npos);
    EXPECT_NE(output.find("row_ptr"), std::string::npos);
    EXPECT_NE(output.find("col_idx"), std::string::npos);
}

/**
 * Test: Generates reference implementation for verification.
 */
TEST(CodegenFrameworkTest, ReferenceKernelGeneration) {
    std::ostringstream out;
    codegen::CodeGenerator gen(out);

    auto op = parseAndLowerForCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    gen.generate(*op, config);

    std::string output = out.str();

    EXPECT_NE(output.find("void reference("), std::string::npos);
}

// ============================================================================
// Convenience Function Tests
// ============================================================================

/**
 * Test: generateCode() convenience function returns string.
 */
TEST(CodegenFrameworkTest, GenerateCodeConvenience) {
    auto op = parseAndLowerForCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateCode(*op, config);

    // Should produce non-empty output
    EXPECT_FALSE(output.empty());
    EXPECT_NE(output.find("int main"), std::string::npos);
}

/**
 * Test: generateKernelCode() emits only the compute kernel.
 */
TEST(CodegenFrameworkTest, GenerateKernelCodeConvenience) {
    auto op = parseAndLowerForCodegen(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");
    ASSERT_NE(op, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string output = codegen::generateKernelCode(*op, config);

    EXPECT_FALSE(output.empty());
    EXPECT_NE(output.find("void compute("), std::string::npos);
    EXPECT_EQ(output.find("#include <stdio.h>"), std::string::npos);
    EXPECT_EQ(output.find("void reference("), std::string::npos);
    EXPECT_EQ(output.find("int main("), std::string::npos);
}

/**
 * Test: kernel-only emission matches the optimized compute function from the
 * full-program emitter.
 */
TEST(CodegenFrameworkTest, GenerateKernelCodeMatchesFullProgramKernel) {
    auto compute = parseAndLowerForCodegen(R"(
        tensor C : Dense<128, 64>;
        tensor A : CSC<128, 32>;
        tensor B : Dense<32, 64>;
        compute C[i, j] = A[i, k] * B[k, j];
    )");
    ASSERT_NE(compute, nullptr);

    opt::OptConfig config = opt::OptConfig::baseline();
    std::string fullProgram = codegen::generateCode(*compute, config);
    std::string kernelOnly = codegen::generateKernelCode(*compute, config);

    EXPECT_EQ(
        dedentCommonIndentation(extractComputeBody(fullProgram)),
        dedentCommonIndentation(extractComputeBody(kernelOnly)));
}
