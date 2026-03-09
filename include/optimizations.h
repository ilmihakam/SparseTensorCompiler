#ifndef OPTIMIZATIONS_H
#define OPTIMIZATIONS_H

#include <string>
#include <vector>
#include "ir.h"

namespace opt {

// ============================================================================
// Optimization Scheduling Order
// ============================================================================

/**
 * Optimization scheduling order.
 *
 * IMPORTANT TERMINOLOGY:
 * - "Reordering": Format-correct IR traversal (CSR/CSC)
 *   This is NOT an optimization - it's required for CORRECTNESS.
 *   CSC MUST iterate by columns (col_ptr[j]), CSR by rows (row_ptr[i]).
 *   This is applied automatically when needed.
 *
 * - "Interchange" (enableInterchange): Loop interchange for CACHE LOCALITY
 *   This is the REAL OPTIMIZATION (like i,j,k → i,k,j in dense matmul).
 *   Changes loop order to improve memory access patterns.
 *
 * - "Blocking" (enableBlocking): Loop tiling for cache reuse
 *   Splits loops into blocks to fit data in cache.
 *
 * This enum controls the ORDER in which optimizations are applied.
 */
enum class OptOrder {
    I_THEN_B = 0,      // Interchange → Block (recommended)
    B_THEN_I = 1,      // Block → Interchange
    I_B_I = 2          // Interchange → Block → Interchange
};

// ============================================================================
// Configuration Structure
// ============================================================================

/**
 * Configuration for optimization passes.
 * Controls which optimizations are applied and their parameters.
 */
struct OptConfig {
    // Optimization toggles (format-correctness reordering is automatic)
    bool enableBlocking = false;     // [OPTIMIZATION] Loop blocking/tiling for cache locality
    bool enableInterchange = false;  // [OPTIMIZATION] Loop interchange for cache locality (i,k,j ↔ i,j,k)

    // Optimization parameters
    int blockSize = 32;              // Block size for tiling (fits in L1 cache)
    bool enable2DBlocking = false;   // 2D tiling for SpMM/SDDMM (tiles both dense loops)
    int blockSize2 = 0;              // Second dimension block size (0 = same as blockSize)

    // Scheduling order (controls order of optimization passes)
    OptOrder order = OptOrder::I_THEN_B;  // Default: interchange first, then block

    // Output settings
    std::string outputFile = "output.c";

    // Factory methods for common configurations
    static OptConfig baseline() {
        OptConfig config;
        config.enableBlocking = false;
        return config;
    }

    static OptConfig blockingOnly(int blockSize = 32) {
        OptConfig config;
        config.enableBlocking = true;
        config.blockSize = blockSize;
        return config;
    }

    static OptConfig interchangeOnly() {
        OptConfig config;
        config.enableBlocking = false;
        config.enableInterchange = true;
        return config;
    }

    static OptConfig allOptimizations(int blockSize = 32, OptOrder order = OptOrder::I_THEN_B) {
        OptConfig config;
        config.enableBlocking = true;        // Tiling optimization
        config.enableInterchange = true;     // Cache locality optimization
        config.blockSize = blockSize;
        config.order = order;
        return config;
    }

    // Convenience alias
    static OptConfig withBothOpts(int blockSize = 32, OptOrder order = OptOrder::I_THEN_B) {
        return allOptimizations(blockSize, order);
    }

    static OptConfig blocking2D(int bs1 = 32, int bs2 = 32) {
        OptConfig config;
        config.enableBlocking = true;
        config.enable2DBlocking = true;
        config.blockSize = bs1;
        config.blockSize2 = bs2;
        return config;
    }
};

// ============================================================================
// Format Analysis Functions
// ============================================================================

/**
 * Get the natural iteration order for a tensor format.
 *
 * @param format The tensor storage format
 * @return Vector of index names in natural order (outer to inner)
 *         CSR: ["row", "col"] - row outer, col inner
 *         CSC: ["col", "row"] - col outer, row inner
 *         Dense: ["row", "col"] - row-major default
 */
std::vector<std::string> getNaturalOrder(ir::Format format);

/**
 * Check if the outer index is iterated densely for a format.
 *
 * @param format The tensor storage format
 * @return true if outer index is dense (e.g., rows in CSR)
 */
bool isOuterIndexDense(ir::Format format);

/**
 * Check if the inner index is iterated sparsely for a format.
 *
 * @param format The tensor storage format
 * @return true if inner index is sparse (e.g., cols in CSR)
 */
bool isInnerIndexSparse(ir::Format format);

/**
 * Check if a tensor's access pattern needs reordering.
 *
 * Compares the tensor's index order against the natural order
 * for its storage format.
 *
 * @param tensor The tensor to analyze
 * @return true if reordering would improve access patterns
 */
bool needsReordering(const ir::Tensor& tensor);

} // namespace opt

#endif // OPTIMIZATIONS_H
