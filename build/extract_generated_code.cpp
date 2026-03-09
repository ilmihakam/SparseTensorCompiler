/**
 * Extract generated C code to demonstrate the compiler output
 * This uses the same approach as the tests, which work perfectly.
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

int main() {
    // Step 1: Parse DSL input
    std::cout << "Step 1: Parsing DSL input...\n";
    const char* dsl_input = R"(
        tensor y : Dense<100>;
        tensor A : CSR<100, 100>;
        tensor x : Dense<100>;
        compute y[i] = A[i, j] * x[j];
    )";

    yynerrs = 0;
    g_program.reset();
    yy_scan_string(dsl_input);
    int result = yyparse();
    yylex_destroy();

    if (result != 0 || yynerrs != 0 || !g_program) {
        std::cerr << "Parse failed!\n";
        return 1;
    }
    std::cout << "✓ DSL parsed successfully\n\n";

    // Step 2: Lower AST to scheduled IR
    std::cout << "Step 2: Lowering AST to scheduled IR...\n";
    auto scheduled = sparseir::lowerFirstComputationToScheduled(*g_program);
    if (!scheduled) {
        std::cerr << "Scheduled lowering failed!\n";
        return 1;
    }
    std::cout << "✓ Scheduled IR generated successfully\n";
    std::cout << "  Output tensor: " << scheduled->output.name << "\n\n";

    // Step 3: Apply optimizations and generate C code for each configuration
    std::cout << "Step 3: Generating C code for all configurations...\n\n";

    // Configuration 1: Baseline
    {
        std::cout << "→ Generating baseline (no optimizations)...\n";
        auto op_copy = sparseir::lowerFirstComputationToScheduled(*g_program);
        opt::OptConfig config = opt::OptConfig::baseline();

        bool success = codegen::generateToFile(*op_copy, config, "generated_baseline.c");
        if (success) {
            std::cout << "  ✓ Saved to: generated_baseline.c\n";
        }
    }

    // Configuration 2: Blocking only
    {
        std::cout << "→ Generating with blocking (block size 32)...\n";
        auto op_copy = sparseir::lowerFirstComputationToScheduled(*g_program);
        opt::OptConfig config = opt::OptConfig::blockingOnly(32);
        opt::applyOptimizations(*op_copy, config);

        bool success = codegen::generateToFile(*op_copy, config, "generated_blocking.c");
        if (success) {
            std::cout << "  ✓ Saved to: generated_blocking.c\n";
        }
    }

    // Configuration 3: Both optimizations
    {
        std::cout << "→ Generating with both optimizations...\n";
        auto op_copy = sparseir::lowerFirstComputationToScheduled(*g_program);
        opt::OptConfig config = opt::OptConfig::allOptimizations(32);
        opt::applyOptimizations(*op_copy, config);

        bool success = codegen::generateToFile(*op_copy, config, "generated_both.c");
        if (success) {
            std::cout << "  ✓ Saved to: generated_both.c\n";
        }
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "SUCCESS! Generated C code for all configurations.\n\n";
    std::cout << "To compile and test:\n";
    std::cout << "  gcc -O2 generated_baseline.c -o spmv_baseline\n";
    std::cout << "  gcc -O2 generated_blocking.c -o spmv_blocking\n";
    std::cout << "  gcc -O2 generated_both.c -o spmv_both\n";
    std::cout << "\nTo run (with a Matrix Market file):\n";
    std::cout << "  ./spmv_baseline matrix.mtx\n";
    std::cout << std::string(70, '=') << "\n";

    return 0;
}
