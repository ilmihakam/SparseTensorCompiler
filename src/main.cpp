/**
 * SparseTensorCompiler - Main CLI Driver
 * Milestone 4: End-to-end compiler with command-line interface
 *
 * Usage: sparse_compiler <input.tc> [options]
 */

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>
#include <unistd.h>  // for isatty()
#include "codegen.h"
#include "ir.h"
#include "optimizations.h"
#include "scheduled_optimizations.h"
#include "ast.h"

// ============================================================================
// Version Information
// ============================================================================

#define VERSION_MAJOR 0
#define VERSION_MINOR 1
#define VERSION_PATCH 0

// ============================================================================
// External Parser Interface
// ============================================================================

extern std::unique_ptr<SparseTensorCompiler::Program> g_program;
extern int yyparse();
extern void yy_scan_string(const char* str);
extern void yylex_destroy();
extern int yynerrs;

// ============================================================================
// Configuration
// ============================================================================

struct CompilerConfig {
    std::string inputFile;
    std::string outputFile = "output.c";
    bool enableInterchange = false;  // Loop interchange for cache locality
    bool enableBlocking = false;     // Loop blocking/tiling
    int blockSize = 32;
    bool enable2DBlocking = false;   // 2D tiling (SpMM/SDDMM)
    int blockSize2 = 0;             // Second dimension block size (0 = same as blockSize)
    opt::OptOrder order = opt::OptOrder::I_THEN_B;  // Default: interchange then block
    bool verbose = false;
    bool showVersion = false;
    bool showHelp = false;
};

// ============================================================================
// Terminal Colors (optional, for better UX)
// ============================================================================

namespace Color {
    const char* RESET   = "\033[0m";
    const char* RED     = "\033[31m";
    const char* GREEN   = "\033[32m";
    const char* YELLOW  = "\033[33m";
    const char* BLUE    = "\033[34m";
    const char* MAGENTA = "\033[35m";
    const char* CYAN    = "\033[36m";
    const char* BOLD    = "\033[1m";

    // Disable colors if not a terminal
    void disableIfNotTTY() {
        if (!isatty(fileno(stdout))) {
            RESET = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = BOLD = "";
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

void printVersion() {
    std::cout << Color::BOLD << "SparseTensorCompiler" << Color::RESET
              << " v" << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH << "\n";
    std::cout << "Sparse tensor algebra DSL compiler with optimization scheduling\n";
}

void printUsage(const char* prog) {
    printVersion();
    std::cout << "\n" << Color::BOLD << "Usage:" << Color::RESET << "\n";
    std::cout << "  " << prog << " <input.tc> [options]\n\n";

    std::cout << Color::BOLD << "Options:" << Color::RESET << "\n";
    std::cout << "  -o <file>                Output file (default: output.c)\n";
    std::cout << "  --opt-interchange        Enable loop interchange optimization\n";
    std::cout << "  --opt-block=SIZE         Enable loop blocking with block size\n";
    std::cout << "  --opt-block-2d=SIZE      Enable 2D blocking (SpMM/SDDMM)\n";
    std::cout << "                           SIZE or SIZExSIZE (e.g., 32 or 32x64)\n";
    std::cout << "  --opt-all=SIZE           Enable all optimizations (default order)\n";
    std::cout << "  --opt-order=ORDER        Optimization scheduling order\n";
    std::cout << "                           (I_THEN_B, B_THEN_I, I_B_I)\n";
    std::cout << "  -v, --verbose            Show detailed compilation steps\n";
    std::cout << "  --version                Show version information\n";
    std::cout << "  -h, --help               Show this help message\n\n";

    std::cout << Color::BOLD << "Optimization Orders:" << Color::RESET << "\n";
    std::cout << "  " << Color::CYAN << "I_THEN_B" << Color::RESET
              << "  Interchange → Block (default, recommended)\n";
    std::cout << "  " << Color::CYAN << "B_THEN_I" << Color::RESET
              << "  Block → Interchange\n";
    std::cout << "  " << Color::CYAN << "I_B_I" << Color::RESET
              << "     Interchange → Block → Interchange\n";
    std::cout << "\n";

    std::cout << Color::BOLD << "Examples:" << Color::RESET << "\n";
    std::cout << "  # Baseline (no optimizations)\n";
    std::cout << "  " << prog << " spmv.tc -o spmv.c\n\n";
    std::cout << "  # Interchange only\n";
    std::cout << "  " << prog << " spmv.tc --opt-interchange -o spmv_interchange.c\n\n";
    std::cout << "  # Blocking only\n";
    std::cout << "  " << prog << " spmv.tc --opt-block=32 -o spmv_blocked.c\n\n";
    std::cout << "  # All optimizations with default order (I_THEN_B)\n";
    std::cout << "  " << prog << " spmv.tc --opt-all=32 -o spmv_optimized.c\n\n";
    std::cout << "  # Specify optimization order\n";
    std::cout << "  " << prog << " spmv.tc --opt-all=32 --opt-order=B_THEN_I -o spmv_b_then_i.c\n\n";
    std::cout << "  # Verbose output\n";
    std::cout << "  " << prog << " spmv.tc --opt-all=32 -v -o spmv.c\n\n";

    std::cout << Color::BOLD << "Benchmarking Workflow:" << Color::RESET << "\n";
    std::cout << "  # Generate all optimization configurations\n";
    std::cout << "  " << prog << " spmv.tc -o baseline.c\n";
    std::cout << "  " << prog << " spmv.tc --opt-interchange -o interchange.c\n";
    std::cout << "  " << prog << " spmv.tc --opt-block=32 -o blocking.c\n";
    std::cout << "  " << prog << " spmv.tc --opt-all=32 --opt-order=I_THEN_B -o i_then_b.c\n";
    std::cout << "  " << prog << " spmv.tc --opt-all=32 --opt-order=B_THEN_I -o b_then_i.c\n\n";
    std::cout << "  # Compile all versions\n";
    std::cout << "  gcc -O2 baseline.c -o baseline\n";
    std::cout << "  gcc -O2 interchange.c -o interchange\n";
    std::cout << "  gcc -O2 blocking.c -o blocking\n";
    std::cout << "  gcc -O2 i_then_b.c -o i_then_b\n";
    std::cout << "  gcc -O2 b_then_i.c -o b_then_i\n\n";
    std::cout << "  # Benchmark on the same matrix\n";
    std::cout << "  ./baseline matrix.mtx\n";
    std::cout << "  ./interchange matrix.mtx\n";
    std::cout << "  ./blocking matrix.mtx\n";
    std::cout << "  ./i_then_b matrix.mtx\n";
    std::cout << "  ./b_then_i matrix.mtx\n";
}

void printError(const std::string& msg) {
    std::cerr << Color::RED << Color::BOLD << "Error: " << Color::RESET
              << Color::RED << msg << Color::RESET << "\n";
}

void printWarning(const std::string& msg) {
    std::cerr << Color::YELLOW << "Warning: " << Color::RESET << msg << "\n";
}

void printSuccess(const std::string& msg) {
    std::cout << Color::GREEN << "✓ " << Color::RESET << msg << "\n";
}

void printStep(const std::string& msg) {
    std::cout << Color::CYAN << "→ " << Color::RESET << msg << "\n";
}

// ============================================================================
// Argument Parser
// ============================================================================

bool parseArguments(int argc, char** argv, CompilerConfig& config) {
    if (argc < 2) {
        return false;
    }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            config.showHelp = true;
            return true;
        } else if (arg == "--version") {
            config.showVersion = true;
            return true;
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--opt-interchange") {
            config.enableInterchange = true;
        } else if (arg.rfind("--opt-block=", 0) == 0) {
            config.enableBlocking = true;
            try {
                config.blockSize = std::stoi(arg.substr(12));
                if (config.blockSize <= 0) {
                    printError("Block size must be positive");
                    return false;
                }
            } catch (...) {
                printError("Invalid block size: " + arg.substr(12));
                return false;
            }
        } else if (arg.rfind("--opt-block-2d=", 0) == 0) {
            config.enableBlocking = true;
            config.enable2DBlocking = true;
            std::string sizeStr = arg.substr(15);
            auto xPos = sizeStr.find('x');
            try {
                if (xPos != std::string::npos) {
                    config.blockSize = std::stoi(sizeStr.substr(0, xPos));
                    config.blockSize2 = std::stoi(sizeStr.substr(xPos + 1));
                } else {
                    config.blockSize = std::stoi(sizeStr);
                    config.blockSize2 = 0;  // same as blockSize
                }
                if (config.blockSize <= 0 || (config.blockSize2 < 0)) {
                    printError("Block sizes must be positive");
                    return false;
                }
            } catch (...) {
                printError("Invalid 2D block size: " + sizeStr);
                return false;
            }
        } else if (arg.rfind("--opt-all=", 0) == 0) {
            config.enableInterchange = true;
            config.enableBlocking = true;
            try {
                config.blockSize = std::stoi(arg.substr(10));
                if (config.blockSize <= 0) {
                    printError("Block size must be positive");
                    return false;
                }
            } catch (...) {
                printError("Invalid block size: " + arg.substr(10));
                return false;
            }
        } else if (arg.rfind("--opt-order=", 0) == 0) {
            std::string orderStr = arg.substr(12);
            if (orderStr == "I_THEN_B") {
                config.order = opt::OptOrder::I_THEN_B;
            } else if (orderStr == "B_THEN_I") {
                config.order = opt::OptOrder::B_THEN_I;
            } else if (orderStr == "I_B_I") {
                config.order = opt::OptOrder::I_B_I;
            } else {
                printError("Invalid optimization order: " + orderStr);
                std::cerr << "Valid orders: I_THEN_B, B_THEN_I, I_B_I\n";
                return false;
            }
        } else if (arg == "-o" && i + 1 < argc) {
            config.outputFile = argv[++i];
        } else if (arg[0] != '-') {
            if (config.inputFile.empty()) {
                config.inputFile = arg;
            } else {
                printError("Multiple input files specified");
                return false;
            }
        } else {
            printError("Unknown option: " + arg);
            return false;
        }
    }

    if (!config.showHelp && !config.showVersion && config.inputFile.empty()) {
        printError("No input file specified");
        return false;
    }

    return true;
}

// ============================================================================
// Compiler Pipeline
// ============================================================================

bool compileFile(const CompilerConfig& config) {
    // Step 1: Read input file
    if (config.verbose) printStep("Reading input file: " + config.inputFile);

    std::ifstream file(config.inputFile);
    if (!file) {
        printError("Cannot open file: " + config.inputFile);
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();

    if (config.verbose) {
        std::cout << "  File size: " << content.size() << " bytes\n";
    }

    // Step 2: Parse DSL
    if (config.verbose) printStep("Parsing DSL");

    yynerrs = 0;
    g_program.reset();
    yy_scan_string(content.c_str());
    int result = yyparse();
    yylex_destroy();

    if (result != 0 || yynerrs != 0 || !g_program) {
        printError("Parse failed");
        if (yynerrs > 0) {
            std::cerr << "  Found " << yynerrs << " syntax error(s)\n";
        }
        return false;
    }

    printSuccess("Parse successful");

    // Step 3: Detect whether this program needs the multi-statement program path
    bool needsProgramPath = false;
    int computeCount = 0;
    for (const auto& stmt : g_program->statements) {
        if (dynamic_cast<SparseTensorCompiler::ForStatement*>(stmt.get()) ||
            dynamic_cast<SparseTensorCompiler::CallStatement*>(stmt.get())) {
            needsProgramPath = true;
        }
        if (dynamic_cast<SparseTensorCompiler::Computation*>(stmt.get())) {
            computeCount++;
        }
    }
    if (computeCount > 1) needsProgramPath = true;

    if (needsProgramPath) {
        // ---- Program path: multi-statement / for-loop / call ----
        if (config.verbose) printStep("Lowering to semantic/scheduled program IR");

        auto semanticProgram = sparseir::lowerToSemanticProgram(*g_program);
        if (!semanticProgram) {
            printError("Semantic program lowering failed");
            return false;
        }

        auto scheduledProgram = sparseir::scheduleProgram(*semanticProgram);
        if (!scheduledProgram) {
            printError("Scheduled program lowering failed");
            return false;
        }

        printSuccess("Scheduled program IR generated");

        opt::OptConfig optConfig;
        optConfig.enableInterchange = config.enableInterchange;
        optConfig.enableBlocking = config.enableBlocking;
        optConfig.blockSize = config.blockSize;
        optConfig.enable2DBlocking = config.enable2DBlocking;
        optConfig.blockSize2 = config.blockSize2;
        optConfig.order = config.order;
        optConfig.outputFile = config.outputFile;

        if (config.verbose) printStep("Applying scheduled optimizations");
        opt::applyOptimizations(*scheduledProgram, optConfig);

        if (config.verbose) printSuccess("Scheduled optimizations complete");

        if (config.verbose) printStep("Generating C code (scheduled program mode)");

        bool success = codegen::generateProgramToFile(*scheduledProgram, optConfig, config.outputFile);
        if (!success) {
            printError("Code generation failed");
            return false;
        }

        // Get file size for summary
        std::ifstream out2(config.outputFile, std::ifstream::ate | std::ifstream::binary);
        size_t fileSize2 = static_cast<size_t>(out2.tellg());
        out2.close();

        if (config.verbose) std::cout << "  Output size: " << fileSize2 << " bytes\n";

        printSuccess("C code generated (program mode)");

        std::cout << "\n" << Color::BOLD << std::string(60, '=') << Color::RESET << "\n";
        std::cout << Color::GREEN << Color::BOLD << "✓ COMPILATION SUCCESSFUL" << Color::RESET << "\n";
        std::cout << Color::BOLD << std::string(60, '=') << Color::RESET << "\n\n";
        std::cout << Color::BOLD << "Output:" << Color::RESET << " " << config.outputFile
                  << " (" << fileSize2 << " bytes)\n\n";
        std::cout << Color::BOLD << "Next steps:" << Color::RESET << "\n";
        std::cout << "  gcc -O2 " << config.outputFile << " -o program\n";
        std::cout << "  ./program <matrix.mtx>\n";

        return true;
    }

    // ---- Single-compute scheduled path ----
    if (config.verbose) printStep("Lowering AST to scheduled IR");

    opt::OptConfig optConfig;
    optConfig.enableInterchange = config.enableInterchange;
    optConfig.enableBlocking = config.enableBlocking;
    optConfig.blockSize = config.blockSize;
    optConfig.enable2DBlocking = config.enable2DBlocking;
    optConfig.blockSize2 = config.blockSize2;
    optConfig.order = config.order;
    optConfig.outputFile = config.outputFile;

    auto scheduled = sparseir::lowerFirstComputationToScheduledOptimized(*g_program, optConfig);
    if (!scheduled) {
        printError("Scheduled IR lowering failed");
        return false;
    }

    if (config.verbose) {
        std::cout << "  Output strategy: "
                  << (scheduled->outputStrategy == ir::OutputStrategy::DenseArray
                          ? "DenseArray"
                          : (scheduled->outputStrategy == ir::OutputStrategy::SparseFixedPattern
                                 ? "SparseFixedPattern"
                                 : "HashPerRow"))
                  << "\n";
        std::cout << "  Input tensors: " << scheduled->inputs.size() << "\n";
    }

    printSuccess("Scheduled IR generated");

    if (config.verbose || (!config.enableInterchange && !config.enableBlocking)) {
        std::cout << "  Format-correctness reordering: AUTO (when needed)\n";
        std::cout << "  Interchange: " << (config.enableInterchange ? "ON" : "OFF") << "\n";
        if (config.enable2DBlocking) {
            int bs2 = config.blockSize2 > 0 ? config.blockSize2 : config.blockSize;
            std::cout << "  Blocking: ON 2D (" << config.blockSize << "x" << bs2 << ")\n";
        } else {
            std::cout << "  Blocking: " << (config.enableBlocking ? "ON (size=" + std::to_string(config.blockSize) + ")" : "OFF") << "\n";
        }

        if (config.enableInterchange && config.enableBlocking) {
            std::string orderStr;
            switch (config.order) {
                case opt::OptOrder::I_THEN_B:
                    orderStr = "I_THEN_B (Interchange → Block)";
                    break;
                case opt::OptOrder::B_THEN_I:
                    orderStr = "B_THEN_I (Block → Interchange)";
                    break;
                case opt::OptOrder::I_B_I:
                    orderStr = "I_B_I (Interchange → Block → Interchange)";
                    break;
            }
            std::cout << "  Order: " << orderStr << "\n";
        }
    }

    if (config.verbose) {
        if (scheduled->optimizations.reorderingApplied) {
            std::cout << "  ✓ Format-correctness reordering applied\n";
        }
        if (scheduled->optimizations.interchangeApplied) {
            std::cout << "  ✓ Loop interchange applied\n";
        }
        if (scheduled->optimizations.blockingApplied) {
            std::cout << "  ✓ Blocking applied (size=" << scheduled->optimizations.blockSize << ")\n";
        }
    }

    printSuccess("Optimizations applied");

    // Step 5: Generate C code
    if (config.verbose) printStep("Generating C code");

    bool success = codegen::generateToFile(*scheduled, optConfig, config.outputFile);

    if (!success) {
        printError("Code generation failed");
        return false;
    }

    // Get file size
    std::ifstream out(config.outputFile, std::ifstream::ate | std::ifstream::binary);
    size_t fileSize = out.tellg();
    out.close();

    if (config.verbose) {
        std::cout << "  Output size: " << fileSize << " bytes\n";
    }

    printSuccess("C code generated");

    // Summary
    std::cout << "\n" << Color::BOLD << std::string(60, '=') << Color::RESET << "\n";
    std::cout << Color::GREEN << Color::BOLD << "✓ COMPILATION SUCCESSFUL" << Color::RESET << "\n";
    std::cout << Color::BOLD << std::string(60, '=') << Color::RESET << "\n\n";

    std::cout << Color::BOLD << "Output:" << Color::RESET << " " << config.outputFile
              << " (" << fileSize << " bytes)\n\n";

    std::cout << Color::BOLD << "Next steps:" << Color::RESET << "\n";
    std::cout << "  1. Compile to executable:\n";
    std::cout << "     " << Color::CYAN << "gcc -O2 " << config.outputFile << " -o program" << Color::RESET << "\n\n";
    std::cout << "  2. Run with a matrix:\n";
    std::cout << "     " << Color::CYAN << "./program <matrix.mtx>" << Color::RESET << "\n\n";

    if (!config.enableInterchange && !config.enableBlocking) {
        std::cout << Color::YELLOW << "Tip:" << Color::RESET
                  << " Try " << Color::CYAN << "--opt-all=32" << Color::RESET
                  << " for optimized performance\n";
    }

    return true;
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char** argv) {
    // Disable colors if not a terminal
    Color::disableIfNotTTY();

    // Parse command-line arguments
    CompilerConfig config;
    if (!parseArguments(argc, argv, config)) {
        printUsage(argv[0]);
        return 1;
    }

    // Handle special flags
    if (config.showHelp) {
        printUsage(argv[0]);
        return 0;
    }

    if (config.showVersion) {
        printVersion();
        return 0;
    }

    // Run compiler pipeline
    try {
        if (!compileFile(config)) {
            std::cerr << "\n" << Color::RED << "Compilation failed" << Color::RESET << "\n";
            return 1;
        }
    } catch (const std::exception& e) {
        printError(std::string("Unexpected error: ") + e.what());
        return 1;
    } catch (...) {
        printError("Unexpected error occurred");
        return 1;
    }

    return 0;
}
