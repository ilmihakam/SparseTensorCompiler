---
name: Performance issue
about: Report performance problems or optimization opportunities
title: '[PERF] '
labels: performance
assignees: ''

---

**Performance Issue Description**
A clear description of the performance problem you've observed.

**Current Performance**
- Operation: [e.g. CSR matrix multiplication]
- Matrix dimensions: [e.g. 10000x10000]
- Sparsity: [e.g. 95% sparse, 0.05 density]
- Current timing: [e.g. 2.5 seconds]
- Memory usage: [e.g. 500MB peak]

**Expected Performance**
- Expected timing: [e.g. under 1 second]
- Comparison benchmark: [e.g. "SciPy CSR multiply takes 0.8s for same input"]
- Memory expectation: [e.g. under 200MB]

**Profiling Information**
If you have profiling data, please include:
- Hot spots identified
- Cache misses
- Memory allocation patterns
- CPU utilization

**Environment**
- OS: [e.g. Ubuntu 20.04]
- CPU: [e.g. Intel i7-9700K, ARM M1]
- RAM: [e.g. 16GB]
- Compiler: [e.g. GCC 11.2 with -O3]
- Sparse format: [e.g. CSR, COO]

**Potential Optimizations**
If you have ideas for improvements:
- [ ] Algorithm optimization
- [ ] Memory layout improvements
- [ ] Vectorization opportunities
- [ ] Parallelization opportunities
- [ ] Cache optimization

**Additional Context**
Any other relevant information about the performance issue.