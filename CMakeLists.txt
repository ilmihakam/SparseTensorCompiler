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

# Include directories
include_directories(include)

# Source directories
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.c")
file(GLOB_RECURSE HEADERS "include/*.hpp" "include/*.h")

# Create main library
if(SOURCES)
    add_library(sparse_tensor_core ${SOURCES} ${HEADERS})
    target_include_directories(sparse_tensor_core PUBLIC include)
endif()

# Example executables (will be created as examples are added)
# add_subdirectory(examples)

# Testing
enable_testing()

# Find or add a testing framework
find_package(GTest QUIET)
if(GTest_FOUND)
    add_subdirectory(tests)
else()
    message(STATUS "GTest not found. Tests will be skipped.")
    message(STATUS "To enable tests, install GTest or run: apt-get install libgtest-dev")
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
message(STATUS "")