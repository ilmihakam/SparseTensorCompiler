Matrix Selection Criteria for Phase 1
Goal: Maximize diversity in properties driving your optimizations:

Property	Why Matters	Target Range
Sparsity Density (nnz/M)	Blocking benefit	Low: <2, Med: 2‑10, High: >10
Column Span	Reordering sensitivity	Narrow: nnz cols < M/10, Wide: >M/2
Format	Your key distinction	5 CSR, 5 CSC
Natural/Inverted	Reordering trigger	Mix to test both
Size	Cache effects	Small: M<1k, Med: 1k‑10k

Recommended 10 Matrices (Phase 1)
text
1. **csr01.mtx** (circuit)     CSR  nnz/M=1.2  narrow cols  ← Low density baseline
2. **bcsstk01.mtx** (struct)   CSR  nnz/M=3.8  banded      ← Good for blocking  
3. **dwt_247.mtx** (web)       CSR  nnz/M=18.2 wide cols   ← High density reorder test
4. **kkt_power.mtx** (power)   CSR  nnz/M=6.1  irregular   ← Real app, med density
5. **thermal2.mtx** (thermal)  CSR  nnz/M=4.2  blocky      ← Block structure

6. **c-33.mtx** (web)          CSC  nnz/M=2.1  narrow      ← CSC natural
7. **dwt_992.mtx** (web)       CSC  nnz/M=12.4 wide        ← CSC high density
8. **G3_circuit.mtx** (circ)   CSC  nnz/M=5.6  banded      ← CSC blocking test
9. **majorbasis.mtx** (basis)  CSC  nnz/M=8.3  irregular   ← CSC reorder candidate
10. **nos4_2882.mtx** (chem)   CSC  nnz/M=3.9  structured  ← CSC app matrix
Download: http://sparse.tamu.edu/ → search by name → Matrix Market (.mtx)

Expected Properties Coverage
text
Matrix Mix:
- CSR: 5 (2 low, 2 med, 1 high nnz/M)
- CSC: 5 (same)
- Column span: 3 narrow (<10% cols), 4 med, 3 wide
- Size: 4 small (M<1k), 4 med (1k‑10k), 2 larger

This stresses:
- Reordering: inverted DSL cases
- Blocking: varying nnz/M  
- Format‑awareness: CSR vs CSC differences
- Correctness: diverse patterns

Phase 1 Success Criteria
text
✅ All config runs pass correctness
✅ Timing stable (stddev <5%)
✅ ≥2 matrices show reordering benefit (inverted cases)
✅ ≥3 matrices show blocking benefit (nnz/M >4)
✅ CSR vs CSC show different natural orders