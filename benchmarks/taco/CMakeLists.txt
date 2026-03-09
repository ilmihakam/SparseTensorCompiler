cmake_minimum_required(VERSION 3.12)
project(TACOBenchmarks)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TACO paths (user can override with -DTACO_DIR)
if(NOT DEFINED TACO_DIR)
    # Use environment variable HOME
    file(TO_CMAKE_PATH "$ENV{HOME}" HOME_DIR)
    set(TACO_DIR "${HOME_DIR}/taco")
endif()

# Set TACO include and library paths
set(TACO_INCLUDE_DIR "${TACO_DIR}/include")
set(TACO_LIB_DIR "${TACO_DIR}/build/lib")

# Verify TACO exists
if(NOT EXISTS "${TACO_INCLUDE_DIR}/taco.h")
    message(FATAL_ERROR "TACO not found at ${TACO_DIR}. Please set TACO_DIR to your TACO installation.")
endif()

# SpMV benchmark
add_executable(spmv_taco spmv_taco.cpp)
target_include_directories(spmv_taco PRIVATE ${TACO_INCLUDE_DIR})
target_link_directories(spmv_taco PRIVATE ${TACO_LIB_DIR})
target_link_libraries(spmv_taco taco)
target_compile_options(spmv_taco PRIVATE -O2 -march=native)

# SpMM benchmark
add_executable(spmm_taco spmm_taco.cpp)
target_include_directories(spmm_taco PRIVATE ${TACO_INCLUDE_DIR})
target_link_directories(spmm_taco PRIVATE ${TACO_LIB_DIR})
target_link_libraries(spmm_taco taco)
target_compile_options(spmm_taco PRIVATE -O2 -march=native)

# SpAdd benchmark
add_executable(spadd_taco spadd_taco.cpp)
target_include_directories(spadd_taco PRIVATE ${TACO_INCLUDE_DIR})
target_link_directories(spadd_taco PRIVATE ${TACO_LIB_DIR})
target_link_libraries(spadd_taco taco)
target_compile_options(spadd_taco PRIVATE -O2 -march=native)

# SpElMul benchmark
add_executable(spelmul_taco spelmul_taco.cpp)
target_include_directories(spelmul_taco PRIVATE ${TACO_INCLUDE_DIR})
target_link_directories(spelmul_taco PRIVATE ${TACO_LIB_DIR})
target_link_libraries(spelmul_taco taco)
target_compile_options(spelmul_taco PRIVATE -O2 -march=native)

# SpGEMM benchmark
add_executable(spgemm_taco spgemm_taco.cpp)
target_include_directories(spgemm_taco PRIVATE ${TACO_INCLUDE_DIR})
target_link_directories(spgemm_taco PRIVATE ${TACO_LIB_DIR})
target_link_libraries(spgemm_taco taco)
target_compile_options(spgemm_taco PRIVATE -O2 -march=native)

# SDDMM benchmark
add_executable(sddmm_taco sddmm_taco.cpp)
target_include_directories(sddmm_taco PRIVATE ${TACO_INCLUDE_DIR})
target_link_directories(sddmm_taco PRIVATE ${TACO_LIB_DIR})
target_link_libraries(sddmm_taco taco)
target_compile_options(sddmm_taco PRIVATE -O2 -march=native)
