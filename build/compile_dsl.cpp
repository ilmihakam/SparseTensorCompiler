/**
 * Legacy CLI wrapper for SparseTensorCompiler.
 *
 * This binary is retained for backward compatibility only.
 * It forwards all arguments to the main CLI: sparse_compiler.
 */

#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

static std::string resolveSiblingCompiler(const char* argv0) {
    if (!argv0) {
        return "sparse_compiler";
    }
    std::string path(argv0);
    auto pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return "sparse_compiler";
    }
    return path.substr(0, pos + 1) + "sparse_compiler";
}

int main(int argc, char** argv) {
    std::cerr << "Note: compile_dsl is deprecated. Forwarding to sparse_compiler.\n";

    std::string compilerPath = resolveSiblingCompiler(argc > 0 ? argv[0] : nullptr);

    std::vector<char*> args;
    args.reserve(static_cast<size_t>(argc) + 1);
    args.push_back(const_cast<char*>(compilerPath.c_str()));
    for (int i = 1; i < argc; i++) {
        args.push_back(argv[i]);
    }
    args.push_back(nullptr);

    // Try sibling path first
    execv(compilerPath.c_str(), args.data());

    // Fallback to PATH
    execvp("sparse_compiler", args.data());

    // If we got here, exec failed
    std::cerr << "Error: failed to launch sparse_compiler: " << std::strerror(errno) << "\n";
    return 1;
}
