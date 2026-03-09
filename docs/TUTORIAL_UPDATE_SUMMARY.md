# TUTORIAL.md Update Summary

## Changes Made

### ✅ 1. Updated Compiler Name Throughout
- `./compile_dsl` → `./sparse_compiler`

### ✅ 2. Updated Output Format
- Matches current CLI output (Interchange/Blocking flags)

### ✅ 3. Added Verbose Mode Documentation
```bash
./sparse_compiler my_spmv.tc --opt-all=32 -v -o my_kernel.c
```

### ✅ 4. Updated Quick Reference
- `--opt-interchange` - Enable loop interchange (SpMM)
- `--opt-block=SIZE` - Enable blocking
- `--opt-all=SIZE` - Enable interchange + blocking
- `--opt-order=ORDER` - Set optimization order (`I_THEN_B`, `B_THEN_I`, `I_B_I`)

### ✅ 5. Updated Benchmarking Workflow
- Orders: `I_THEN_B`, `B_THEN_I`, `I_B_I`
- Configs: `baseline`, `interchange_only`, `block_only`, `i_then_b`, `b_then_i`, `i_b_i`

