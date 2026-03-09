/**
 * Test Suite: Optimization Configuration
 *
 * Tests the OptConfig structure that controls which optimizations
 * are applied and their parameters.
 */

#include <gtest/gtest.h>
#include "optimizations.h"

// ============================================================================
// OptConfig Default Values Tests
// ============================================================================

/**
 * Test: Default configuration has all optimizations disabled.
 *
 * By default, we want baseline (unoptimized) code generation.
 */
TEST(OptConfigTest, DefaultsToNoOptimizations) {
    opt::OptConfig config;

    EXPECT_FALSE(config.enableBlocking);
    EXPECT_FALSE(config.enableInterchange);
}

/**
 * Test: Default block size is 32.
 *
 * 32 is a reasonable default that fits in L1 cache for most architectures.
 */
TEST(OptConfigTest, DefaultBlockSize) {
    opt::OptConfig config;

    EXPECT_EQ(config.blockSize, 32);
}

/**
 * Test: Default output file name.
 */
TEST(OptConfigTest, DefaultOutputFile) {
    opt::OptConfig config;

    EXPECT_EQ(config.outputFile, "output.c");
}

// ============================================================================
// OptConfig Modification Tests
// ============================================================================

/**
 * Test: Enable interchange only.
 */
TEST(OptConfigTest, EnableInterchangeOnly) {
    opt::OptConfig config;
    config.enableInterchange = true;

    EXPECT_FALSE(config.enableBlocking);
    EXPECT_TRUE(config.enableInterchange);
}

/**
 * Test: Enable blocking only.
 */
TEST(OptConfigTest, EnableBlockingOnly) {
    opt::OptConfig config;
    config.enableBlocking = true;

    EXPECT_TRUE(config.enableBlocking);
    EXPECT_FALSE(config.enableInterchange);
}

/**
 * Test: Enable both optimizations.
 */
TEST(OptConfigTest, EnableBothOptimizations) {
    opt::OptConfig config;
    config.enableBlocking = true;
    config.enableInterchange = true;

    EXPECT_TRUE(config.enableBlocking);
    EXPECT_TRUE(config.enableInterchange);
}

/**
 * Test: Custom block size.
 */
TEST(OptConfigTest, CustomBlockSize) {
    opt::OptConfig config;
    config.blockSize = 64;

    EXPECT_EQ(config.blockSize, 64);
}

/**
 * Test: Custom output file.
 */
TEST(OptConfigTest, CustomOutputFile) {
    opt::OptConfig config;
    config.outputFile = "spmv_kernel.c";

    EXPECT_EQ(config.outputFile, "spmv_kernel.c");
}

// ============================================================================
// Configuration Combinations for Benchmarking
// ============================================================================

/**
 * Test: Create baseline configuration (no opts).
 */
TEST(OptConfigTest, BaselineConfiguration) {
    opt::OptConfig config = opt::OptConfig::baseline();

    EXPECT_FALSE(config.enableBlocking);
    EXPECT_FALSE(config.enableInterchange);
}

/**
 * Test: Create blocking-only configuration.
 */
TEST(OptConfigTest, BlockingOnlyConfiguration) {
    opt::OptConfig config = opt::OptConfig::blockingOnly(64);

    EXPECT_TRUE(config.enableBlocking);
    EXPECT_FALSE(config.enableInterchange);
    EXPECT_EQ(config.blockSize, 64);
}

/**
 * Test: Create interchange-only configuration.
 */
TEST(OptConfigTest, InterchangeOnlyConfiguration) {
    opt::OptConfig config = opt::OptConfig::interchangeOnly();

    EXPECT_FALSE(config.enableBlocking);
    EXPECT_TRUE(config.enableInterchange);
}

/**
 * Test: Create full optimization configuration.
 */
TEST(OptConfigTest, FullOptimizationConfiguration) {
    opt::OptConfig config = opt::OptConfig::allOptimizations(32);

    EXPECT_TRUE(config.enableBlocking);
    EXPECT_TRUE(config.enableInterchange);
    EXPECT_EQ(config.blockSize, 32);
}

/**
 * Test: Configuration can be copied correctly.
 */
TEST(OptConfigTest, ConfigurationCopy) {
    opt::OptConfig original;
    original.enableBlocking = true;
    original.enableInterchange = true;
    original.blockSize = 128;
    original.outputFile = "custom.c";

    opt::OptConfig copy = original;

    EXPECT_EQ(copy.enableBlocking, original.enableBlocking);
    EXPECT_EQ(copy.enableInterchange, original.enableInterchange);
    EXPECT_EQ(copy.blockSize, original.blockSize);
    EXPECT_EQ(copy.outputFile, original.outputFile);
}

/**
 * Test: All defaults together in one verification.
 */
TEST(OptConfigTest, AllDefaultsTogether) {
    opt::OptConfig config;

    EXPECT_FALSE(config.enableBlocking);
    EXPECT_FALSE(config.enableInterchange);
    EXPECT_EQ(config.blockSize, 32);
    EXPECT_EQ(config.outputFile, "output.c");
}
