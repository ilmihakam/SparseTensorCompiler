# Optimization Scheduling Order - Implementation Summary

## Overview

This document summarizes scheduling order for **loop interchange** and **loop blocking**. The format-correctness reordering pass (CSR/CSC traversal) is always applied when needed and is not user-controlled.

## What Was Implemented

### 1. Core Infrastructure (`include/optimizations.h`)

**OptOrder enum (interchange + blocking):**
```cpp
enum class OptOrder {
    I_THEN_B = 3,  // Interchange → Block (default)
    B_THEN_I = 4,  // Block → Interchange
    I_B_I = 5      // Interchange → Block → Interchange
};
```

**OptConfig fields:**
```cpp
struct OptConfig {
    bool enableInterchange = false;
    bool enableBlocking = false;
    int blockSize = 32;
    OptOrder order = OptOrder::I_THEN_B;
    std::string outputFile = "output.c";
};
```

### 2. Scheduling Logic (`src/optimizations.cpp`)

```cpp
void applyOptimizations(ir::Operation& op, const OptConfig& config) {
    switch (config.order) {
        case OptOrder::I_THEN_B:
            if (config.enableInterchange) applyLoopInterchange(op, config);
            if (config.enableBlocking) applyBlocking(op, config);
            break;

        case OptOrder::B_THEN_I:
            if (config.enableBlocking) applyBlocking(op, config);
            if (config.enableInterchange) applyLoopInterchange(op, config);
            break;

        case OptOrder::I_B_I:
            if (config.enableInterchange) applyLoopInterchange(op, config);
            if (config.enableBlocking) applyBlocking(op, config);
            if (config.enableInterchange) applyLoopInterchange(op, config);
            break;
    }
}
```

### 3. CLI (`src/main.cpp`)

```bash
./sparse_compiler input.tc --opt-interchange
./sparse_compiler input.tc --opt-block=32
./sparse_compiler input.tc --opt-all=32 --opt-order=I_THEN_B
./sparse_compiler input.tc --opt-all=32 --opt-order=B_THEN_I
./sparse_compiler input.tc --opt-all=32 --opt-order=I_B_I
```

### 4. Codegen Annotation (`src/codegen.cpp`)

Generated C files include the chosen scheduling order in header comments when both interchange and blocking are enabled.

### 5. Test Coverage

`src/tests/optimizations/test_opt_scheduling.cpp` verifies the new orders and their effects on loop structure.

## Usage Examples

```bash
# Default order (Interchange → Block)
./sparse_compiler spmv.tc --opt-all=32 -o i_then_b.c

# Block → Interchange
./sparse_compiler spmv.tc --opt-all=32 --opt-order=B_THEN_I -o b_then_i.c

# Interchange → Block → Interchange
./sparse_compiler spmv.tc --opt-all=32 --opt-order=I_B_I -o i_b_i.c
```

## Key Design Decisions

- Format-correctness reordering is always applied when needed and is not a CLI option.
- Scheduling order only controls interchange + blocking.
- Default order is `I_THEN_B`.

## Files Modified

1. `include/optimizations.h`
2. `src/optimizations.cpp`
3. `src/codegen.cpp`
4. `src/main.cpp`
5. `src/tests/optimizations/test_opt_scheduling.cpp`

## Status

✅ **Implementation Complete**
