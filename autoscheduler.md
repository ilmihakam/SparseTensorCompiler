Phase 3 target architecture — compiler-integrated auto-scheduler:

 ./sparse_compiler input.tc --auto-schedule matrix.mtx -o output.c

 The compiler would:
 1. Read the .mtx file and extract features internally (C++ in src/optimizations.cpp)
 2. Run heuristic function: OptConfig autoSchedule(const MatrixFeatures& features, const std::string& kernelType)
 3. Apply the chosen optimizations to the IR
 4. Generate code with the best schedule

 This means feature extraction logic will eventually exist in two places:
 - Generated C code (this work) — for benchmarking and data collection
 - C++ in the compiler (src/optimizations.cpp) — for the auto-scheduler (future Phase 3)

 The feature computation algorithms are identical; only the language differs. The generated C version serves as the prototype/reference for the
 eventual C++ compiler version.

 Files affected in Phase 3 (future, not this PR):
 - include/optimizations.h — MatrixFeatures struct, autoSchedule() declaration
 - src/optimizations.cpp — feature extraction from .mtx, heuristic rules
 - src/main.cpp — --auto-schedule <matrix.mtx> CLI flag, .mtx loading

 ---
 Future: Tier 2 Features (deferred)

 - LRU cache simulation: temporal reuse distance at L1/L2 capacities (needs C LRU implementation)
 - Sampled Jaccard: inter-row similarity (needs random sampling + sorted merge)
 - 2D blockiness: BCSR tendency metric (needs hash map in generated C)
 - RCM bandwidth reduction: orderability metric (needs graph BFS — likely standalone tool)