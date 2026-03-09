# Optimization Scheduling - Status Report

## ✅ Current Status

Scheduling order for **loop interchange** and **blocking** is implemented and validated. Format-correctness reordering is always applied when needed and is not user-controlled.

## Test Results

- `ctest --output-on-failure` passes all tests in the repo (33/33).
- `OptSchedulingTests` cover `I_THEN_B`, `B_THEN_I`, and `I_B_I` orders.
- Runtime correctness tests pass for CSR and CSC with all scheduling orders.

## End-to-End CLI (Current)

```bash
./sparse_compiler test_order.tc --opt-all=32 --opt-order=I_THEN_B -o test_i_then_b.c
./sparse_compiler test_order.tc --opt-all=32 --opt-order=B_THEN_I -o test_b_then_i.c
./sparse_compiler test_order.tc --opt-all=32 --opt-order=I_B_I -o test_i_b_i.c
```

## Notes

- `compile_dsl` is now a deprecated wrapper that forwards to `sparse_compiler`.
- The default order is `I_THEN_B`.
