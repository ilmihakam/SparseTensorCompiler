/**
 * Demo: Show what the code generator outputs
 */

#include <iostream>
#include <fstream>
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

std::unique_ptr<sparseir::scheduled::Compute> parseAndLower(const std::string& code) {
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

void generateExample(const std::string& name, opt::OptConfig config) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Configuration: " << name << "\n";
    std::cout << std::string(80, '=') << "\n\n";

    // Parse SpMV DSL
    auto op = parseAndLower(R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )");

    if (!op) {
        std::cout << "Failed to parse!\n";
        return;
    }

    // Apply optimizations
    opt::applyOptimizations(*op, config);

    // Generate code
    std::string code = codegen::generateCode(*op, config);

    // Show first ~100 lines or 3000 chars
    size_t showLength = std::min(3000UL, code.length());
    std::cout << code.substr(0, showLength);

    if (code.length() > showLength) {
        std::cout << "\n... (" << (code.length() - showLength) << " more bytes)\n";
    }

    // Save to file
    std::string filename = "generated_" + name + ".c";
    std::ofstream file(filename);
    file << code;
    file.close();

    std::cout << "\n✓ Saved to: " << filename << "\n";
}

int main() {
    std::cout << "SparseTensorCompiler - Code Generation Demo\n";
    std::cout << "============================================\n";

    // Example 1: Baseline (no optimizations)
    generateExample("baseline", opt::OptConfig::baseline());

    // Example 2: Blocking only
    generateExample("blocking", opt::OptConfig::blockingOnly(32));

    // Example 3: Both optimizations
    generateExample("both", opt::OptConfig::allOptimizations(32));

    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "To compile and test the generated code:\n";
    std::cout << "  gcc -O2 generated_baseline.c -o spmv_baseline\n";
    std::cout << "  ./spmv_baseline <matrix.mtx>\n";
    std::cout << std::string(80, '=') << "\n\n";

    return 0;
}
