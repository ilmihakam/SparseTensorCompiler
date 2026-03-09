cmake_minimum_required(VERSION 3.12)
project(SparseTensorCompiler
    VERSION 0.1.0
    LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall")

# Enable export of compile commands for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages for parser generation
find_package(FLEX REQUIRED)
find_package(BISON REQUIRED)

# Include directories
include_directories(include)
include_directories(${CMAKE_BINARY_DIR})  # For generated parser headers

# Generate lexer and parser
FLEX_TARGET(SparseCompilerLexer src/lexer.l ${CMAKE_BINARY_DIR}/lexer.cpp)
BISON_TARGET(SparseCompilerParser src/parser.y ${CMAKE_BINARY_DIR}/parser.cpp
    DEFINES_FILE ${CMAKE_BINARY_DIR}/parser_generated.h)

# Ensure lexer depends on parser for header generation
ADD_FLEX_BISON_DEPENDENCY(SparseCompilerLexer SparseCompilerParser)

# Source directories (exclude test files and main.cpp)
file(GLOB_RECURSE ALL_CPP_FILES "src/*.cpp" "src/*.c")
file(GLOB_RECURSE TEST_FILES "src/tests/*.cpp")
list(REMOVE_ITEM ALL_CPP_FILES ${TEST_FILES})
list(REMOVE_ITEM ALL_CPP_FILES "${CMAKE_SOURCE_DIR}/src/main.cpp")
set(SOURCES ${ALL_CPP_FILES})
file(GLOB_RECURSE HEADERS "include/*.hpp" "include/*.h")

# Add generated parser files to sources
set(GENERATED_SOURCES
    ${FLEX_SparseCompilerLexer_OUTPUTS}
    ${BISON_SparseCompilerParser_OUTPUTS}
)

# Create main library
set(ALL_SOURCES ${SOURCES} ${GENERATED_SOURCES})
if(ALL_SOURCES)
    add_library(sparse_tensor_core ${ALL_SOURCES} ${HEADERS})
    target_include_directories(sparse_tensor_core PUBLIC
        include
        ${CMAKE_BINARY_DIR}
    )
else()
    # Create library with just generated sources if no other sources exist
    add_library(sparse_tensor_core ${GENERATED_SOURCES} ${HEADERS})
    target_include_directories(sparse_tensor_core PUBLIC
        include
        ${CMAKE_BINARY_DIR}
    )
endif()

# ============================================================================
# Main Compiler Executable (Milestone 4)
# ============================================================================

add_executable(sparse_compiler src/main.cpp)
target_link_libraries(sparse_compiler sparse_tensor_core)
target_include_directories(sparse_compiler PRIVATE
    include
    ${CMAKE_BINARY_DIR}
)

# Install target
install(TARGETS sparse_compiler DESTINATION bin)

# Example executables (will be created as examples are added)
# add_subdirectory(examples)

# Testing
enable_testing()

# Find or add a testing framework
find_package(GTest QUIET)
if(GTest_FOUND)
    # Add lexer tests (disabled until test_lexer.cpp exists)
    # add_executable(test_lexer src/tests/lexer/test_lexer.cpp)
    # target_link_libraries(test_lexer sparse_tensor_core GTest::gtest GTest::gtest_main)
    # target_include_directories(test_lexer PRIVATE
    #     include
    #     ${CMAKE_BINARY_DIR}
    # )
    # add_test(NAME LexerTests COMMAND test_lexer)

    # Add basic token tests
    add_executable(test_basic_tokens src/tests/lexer/test_basic_tokens.cpp)
    target_link_libraries(test_basic_tokens sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_basic_tokens PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME BasicTokenTests COMMAND test_basic_tokens)

    # Add literals tests
    add_executable(test_literals src/tests/lexer/test_literals.cpp)
    target_link_libraries(test_literals sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_literals PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME LiteralsTests COMMAND test_literals)

    # Add sequences tests
    add_executable(test_sequences src/tests/lexer/test_sequences.cpp)
    target_link_libraries(test_sequences sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_sequences PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME SequencesTests COMMAND test_sequences)

    # Add tensor type tests
    add_executable(test_tensor_types src/tests/lexer/test_tensor_types.cpp)
    target_link_libraries(test_tensor_types sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_tensor_types PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME TensorTypeTests COMMAND test_tensor_types)

    # Add parser tests
    add_executable(test_parser_basic src/tests/parser/test_parser_basic.cpp)
    target_link_libraries(test_parser_basic sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_parser_basic PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME ParserBasicTests COMMAND test_parser_basic)

    # Add AST generation tests
    add_executable(test_ast_generation src/tests/ast/test_ast_generation.cpp)
    target_link_libraries(test_ast_generation sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_ast_generation PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME ASTGenerationTests COMMAND test_ast_generation)

    # Add typed declaration tests
    add_executable(test_typed_declarations src/tests/ast/test_typed_declarations.cpp)
    target_link_libraries(test_typed_declarations sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_typed_declarations PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME TypedDeclarationTests COMMAND test_typed_declarations)

    # Add parser AST integration tests
    add_executable(test_parser_ast_integration src/tests/parser/test_parser_ast_integration.cpp)
    target_link_libraries(test_parser_ast_integration sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_parser_ast_integration PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME ParserASTIntegrationTests COMMAND test_parser_ast_integration)

    # Add tensor declaration tests
    add_executable(test_tensor_declarations src/tests/parser/test_tensor_declarations.cpp)
    target_link_libraries(test_tensor_declarations sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_tensor_declarations PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME TensorDeclarationTests COMMAND test_tensor_declarations)

    # Add typed integration tests
    add_executable(test_typed_integration src/tests/parser/test_typed_integration.cpp)
    target_link_libraries(test_typed_integration sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_typed_integration PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME TypedIntegrationTests COMMAND test_typed_integration)

    # SpMM parsing tests
    add_executable(test_spmm_parsing src/tests/parser/test_spmm_parsing.cpp)
    target_link_libraries(test_spmm_parsing sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_spmm_parsing PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME SpMMParsingTests COMMAND test_spmm_parsing)

    # =========================================================================
    # IR Tests (Milestone 1)
    # =========================================================================

    # IR lowering tests
    add_executable(test_ir_lowering src/tests/ir/test_ir_lowering.cpp)
    target_link_libraries(test_ir_lowering sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_ir_lowering PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME IRLoweringTests COMMAND test_ir_lowering)

    # IR expression tree tests (Phase A)
    add_executable(test_ir_expr src/tests/ir/test_ir_expr.cpp)
    target_link_libraries(test_ir_expr sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_ir_expr PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME IRExprTests COMMAND test_ir_expr)

    # SpMM IR lowering tests
    add_executable(test_spmm_ir_lowering src/tests/ir/test_spmm_ir_lowering.cpp)
    target_link_libraries(test_spmm_ir_lowering sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_spmm_ir_lowering PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME SpMMIRLoweringTests COMMAND test_spmm_ir_lowering)

    # New kernel lowering tests
    add_executable(test_new_kernel_lowering src/tests/ir/test_new_kernel_lowering.cpp)
    target_link_libraries(test_new_kernel_lowering sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_new_kernel_lowering PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME NewKernelLoweringTests COMMAND test_new_kernel_lowering)

    # General loop nest lowering tests (Phase B)
    add_executable(test_general_lowering src/tests/ir/test_general_lowering.cpp)
    target_link_libraries(test_general_lowering sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_general_lowering PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME GeneralLoweringTests COMMAND test_general_lowering)

    # Semantic/scheduled IR refactor tests
    add_executable(test_semantic_ir src/tests/ir/test_semantic_ir.cpp)
    target_link_libraries(test_semantic_ir sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_semantic_ir PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME SemanticIRTests COMMAND test_semantic_ir)

    # =========================================================================
    # Optimization Tests (Milestone 2)
    # =========================================================================

    # OptConfig tests
    add_executable(test_opt_config src/tests/optimizations/test_opt_config.cpp)
    target_link_libraries(test_opt_config sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_opt_config PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME OptConfigTests COMMAND test_opt_config)

    # Format analyzer tests
    add_executable(test_format_analyzer src/tests/optimizations/test_format_analyzer.cpp)
    target_link_libraries(test_format_analyzer sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_format_analyzer PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME FormatAnalyzerTests COMMAND test_format_analyzer)

    # Reordering pass tests
    add_executable(test_reordering_pass src/tests/optimizations/test_reordering_pass.cpp)
    target_link_libraries(test_reordering_pass sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_reordering_pass PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME ReorderingPassTests COMMAND test_reordering_pass)

    # Blocking pass tests
    add_executable(test_blocking_pass src/tests/optimizations/test_blocking_pass.cpp)
    target_link_libraries(test_blocking_pass sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_blocking_pass PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME BlockingPassTests COMMAND test_blocking_pass)

    # Pass integration tests
    add_executable(test_pass_integration src/tests/optimizations/test_pass_integration.cpp)
    target_link_libraries(test_pass_integration sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_pass_integration PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME PassIntegrationTests COMMAND test_pass_integration)

    # Optimization scheduling tests
    add_executable(test_opt_scheduling src/tests/optimizations/test_opt_scheduling.cpp)
    target_link_libraries(test_opt_scheduling sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_opt_scheduling PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME OptSchedulingTests COMMAND test_opt_scheduling)

    # Loop interchange tests
    add_executable(test_loop_interchange src/tests/optimizations/test_loop_interchange.cpp)
    target_link_libraries(test_loop_interchange sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_loop_interchange PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME LoopInterchangeTests COMMAND test_loop_interchange)

    # =========================================================================
    # Code Generation Tests
    # =========================================================================

    # Framework tests
    add_executable(test_codegen_framework src/tests/codegen/test_codegen_framework.cpp)
    target_link_libraries(test_codegen_framework sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_codegen_framework PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME CodegenFrameworkTests COMMAND test_codegen_framework)

    # Dense loop tests
    add_executable(test_dense_loops src/tests/codegen/test_dense_loops.cpp)
    target_link_libraries(test_dense_loops sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_dense_loops PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME DenseLoopsTests COMMAND test_dense_loops)

    # Sparse loop tests
    add_executable(test_sparse_loops src/tests/codegen/test_sparse_loops.cpp)
    target_link_libraries(test_sparse_loops sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_sparse_loops PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME SparseLoopsTests COMMAND test_sparse_loops)

    # Blocking codegen tests
    add_executable(test_blocking_codegen src/tests/codegen/test_blocking_codegen.cpp)
    target_link_libraries(test_blocking_codegen sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_blocking_codegen PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME BlockingCodegenTests COMMAND test_blocking_codegen)

    # Blocking correctness tests (regression tests for blocking bugs)
    add_executable(test_blocking_correctness src/tests/codegen/test_blocking_correctness.cpp)
    target_link_libraries(test_blocking_correctness sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_blocking_correctness PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME BlockingCorrectnessTests COMMAND test_blocking_correctness)

    # Reordering correctness tests (TDD approach - define intended behavior)
    add_executable(test_reordering_correctness src/tests/codegen/test_reordering_correctness.cpp)
    target_link_libraries(test_reordering_correctness sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_reordering_correctness PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME ReorderingCorrectnessTests COMMAND test_reordering_correctness)

    # Comprehensive sanity check tests (from sanity_check.md)
    add_executable(test_sanity_check src/tests/codegen/test_sanity_check.cpp)
    target_link_libraries(test_sanity_check sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_sanity_check PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME SanityCheckTests COMMAND test_sanity_check)

    # Scheduled optimizations runtime correctness tests
    add_executable(test_scheduled_optimizations_correctness src/tests/codegen/test_scheduled_optimizations_correctness.cpp)
    target_link_libraries(test_scheduled_optimizations_correctness sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_scheduled_optimizations_correctness PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME ScheduledOptimizationsCorrectnessTests COMMAND test_scheduled_optimizations_correctness)

    # SpMV codegen tests
    add_executable(test_spmv_codegen src/tests/codegen/test_spmv_codegen.cpp)
    target_link_libraries(test_spmv_codegen sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_spmv_codegen PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME SpMVCodegenTests COMMAND test_spmv_codegen)

    # SpMM codegen tests
    add_executable(test_spmm_codegen src/tests/codegen/test_spmm_codegen.cpp)
    target_link_libraries(test_spmm_codegen sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_spmm_codegen PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME SpMMCodegenTests COMMAND test_spmm_codegen)

    # SpAdd codegen tests
    add_executable(test_spadd_codegen src/tests/codegen/test_spadd_codegen.cpp)
    target_link_libraries(test_spadd_codegen sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_spadd_codegen PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME SpAddCodegenTests COMMAND test_spadd_codegen)

    # SpElMul codegen tests
    add_executable(test_spelmul_codegen src/tests/codegen/test_spelmul_codegen.cpp)
    target_link_libraries(test_spelmul_codegen sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_spelmul_codegen PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME SpElMulCodegenTests COMMAND test_spelmul_codegen)

    # SpGEMM codegen tests
    add_executable(test_spgemm_codegen src/tests/codegen/test_spgemm_codegen.cpp)
    target_link_libraries(test_spgemm_codegen sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_spgemm_codegen PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME SpGEMMCodegenTests COMMAND test_spgemm_codegen)

    # SDDMM codegen tests
    add_executable(test_sddmm_codegen src/tests/codegen/test_sddmm_codegen.cpp)
    target_link_libraries(test_sddmm_codegen sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_sddmm_codegen PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME SDDMMCodegenTests COMMAND test_sddmm_codegen)

    # Visitor unification tests (Phase 1: all kernels through visitor)
    add_executable(test_visitor_unification src/tests/codegen/test_visitor_unification.cpp)
    target_link_libraries(test_visitor_unification sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_visitor_unification PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME VisitorUnificationTests COMMAND test_visitor_unification)

    # New kernel optimization tests (Phase 2: blocking/interchange for all kernels)
    add_executable(test_new_kernel_optimizations src/tests/optimizations/test_new_kernel_optimizations.cpp)
    target_link_libraries(test_new_kernel_optimizations sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_new_kernel_optimizations PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME NewKernelOptimizationsTests COMMAND test_new_kernel_optimizations)

    # Output format IR tests (Phase 3)
    add_executable(test_output_format src/tests/ir/test_output_format.cpp)
    target_link_libraries(test_output_format sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_output_format PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME OutputFormatTests COMMAND test_output_format)

    # SpGEMM hash output codegen tests (Phase 3)
    add_executable(test_spgemm_hash_output src/tests/codegen/test_spgemm_hash_output.cpp)
    target_link_libraries(test_spgemm_hash_output sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_spgemm_hash_output PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME SpGEMMHashOutputTests COMMAND test_spgemm_hash_output)

    # Realistic input configuration tests (Phase 4)
    add_executable(test_realistic_inputs src/tests/codegen/test_realistic_inputs.cpp)
    target_link_libraries(test_realistic_inputs sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_realistic_inputs PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME RealisticInputsTests COMMAND test_realistic_inputs)

    # General kernel codegen tests (Phase C)
    add_executable(test_general_codegen src/tests/codegen/test_general_codegen.cpp)
    target_link_libraries(test_general_codegen sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_general_codegen PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME GeneralCodegenTests COMMAND test_general_codegen)

    # Sparse-output config effects tests (visitor refactor)
    add_executable(test_sparse_output_config_effects src/tests/optimizations/test_sparse_output_config_effects.cpp)
    target_link_libraries(test_sparse_output_config_effects sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_sparse_output_config_effects PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME SparseOutputConfigEffectsTests COMMAND test_sparse_output_config_effects)

    # Sparse-output codegen shape tests (visitor refactor)
    add_executable(test_sparse_output_codegen src/tests/codegen/test_sparse_output_codegen.cpp)
    target_link_libraries(test_sparse_output_codegen sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_sparse_output_codegen PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    add_test(NAME SparseOutputCodegenTests COMMAND test_sparse_output_codegen)

    # Golden artifact tests (kernel codegen vs .kernel.c.txt files)
    add_executable(test_kernel_golden src/tests/codegen/test_kernel_golden.cpp)
    target_link_libraries(test_kernel_golden sparse_tensor_core GTest::gtest GTest::gtest_main)
    target_include_directories(test_kernel_golden PRIVATE
        include
        ${CMAKE_BINARY_DIR}
    )
    target_compile_definitions(test_kernel_golden PRIVATE SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    add_test(NAME KernelGoldenTests COMMAND test_kernel_golden)

    message(STATUS "Tests enabled with GTest")
else()
    message(STATUS "GTest not found. Tests will be skipped.")
    message(STATUS "To enable tests, install GTest or run: apt-get install libgtest-dev")
endif()

# Benchmark pipeline smoke tests (Python)
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME BenchmarkUnifiedOurs
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/benchmarks/scripts/test_unified_benchmark_pipeline.py --mode ours --kernel all
    )
    add_test(
        NAME BenchmarkUnifiedTACO
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/benchmarks/scripts/test_unified_benchmark_pipeline.py --mode taco --kernel all
    )
    add_test(
        NAME BenchmarkAllOrchestrator
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/benchmarks/scripts/benchmark_all.py
            --kernels spmv,spadd
            --formats csr
            --configs baseline
            --iterations 1
            --max-matrices 1
            --max-pairs 1
            --no-taco
            --output-dir ${CMAKE_BINARY_DIR}/benchmarks_csv
    )
    add_test(
        NAME Analyze2DBlockingUnit
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/benchmarks/scripts/test_analyze_2d_blocking_unit.py
    )
else()
    message(STATUS "Python3 not found. Benchmark pipeline tests will be skipped.")
endif()

# Custom targets for development
add_custom_target(format
    COMMAND find ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests -name "*.cpp" -o -name "*.hpp" -o -name "*.c" -o -name "*.h" | xargs clang-format -i
    COMMENT "Running clang-format on source files"
)

add_custom_target(check-format
    COMMAND find ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests -name "*.cpp" -o -name "*.hpp" -o -name "*.c" -o -name "*.h" | xargs clang-format --dry-run --Werror
    COMMENT "Checking code formatting"
)

# Documentation (if Doxygen is available)
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
        COMMENT "Generating documentation with Doxygen"
    )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project: ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Sources found: ${SOURCES}")
message(STATUS "  GTest available: ${GTest_FOUND}")
message(STATUS "  Doxygen available: ${DOXYGEN_FOUND}")
message(STATUS "")# Demo executable to extract generated code
add_executable(extract_generated_code extract_generated_code.cpp)
target_link_libraries(extract_generated_code sparse_tensor_core)
target_include_directories(extract_generated_code PRIVATE
    ../include
    ${CMAKE_BINARY_DIR}
)

# Simple DSL compiler executable
add_executable(compile_dsl compile_dsl.cpp)
target_link_libraries(compile_dsl sparse_tensor_core)
target_include_directories(compile_dsl PRIVATE
    ../include
    ${CMAKE_BINARY_DIR}
)
