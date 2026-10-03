# SYRK and Cholesky

## Summary

A SIMD kernel for transposed SYRK gives small Cholesky gains and larger panel
and leaf gains. It also repairs transposed-Lower beta=0 behavior.

## Setup

- Baseline: `ba4d7ed2d3bd8725817e02cebc8d1534e0a7d761`, after the TT GEMM
  milestone.
- Apple M1 Pro, darwin/arm64, Go 1.27.1, GOEXPERIMENT=simd, -pgo=off.
- Production routing stays under the existing ARM64 experiment gate. No AMD64
  promotion and no native AMD64 performance claim.

Public Cholesky calls Dpotrf. Its upper path uses transposed SYRK on diagonal
blocks. Its lower path uses NoTrans, and this change does not redirect it.

A three-second profile of mat Cholesky n=256 gave Dsyrk 16.72% cumulative
sampled CPU. Triangular solves in condition estimation used more CPU. Thus a
large SYRK leaf gain cannot give a large factorization gain.

## Implementation

For disjoint Trans/ConjTrans calls with n,k>=16, the SIMD kernel:

- keeps four portable SIMD vectors of each active triangular row in registers,
  then one-vector and scalar tails
- traverses k in increasing 64-term blocks across output rows
- applies beta exactly once, in the first block
- keeps, for each output, increasing-k accumulation, alpha-before-term scaling,
  and the exact-zero skip

### Aliasing

- The existing row-interval disjoint check runs before mutation. Real C/A
  overlap falls back to the original path.
- In LAPACK panels, A shares storage with C. Disjoint active row ranges stay
  eligible.
- A conservative check against the full C rectangle can reject overlap that is
  only in its inactive triangle. This limits performance, not results.

### Code generation

- The double-precision source generates single precision.
- Native simd128 disassembly shows four accumulators in V3-V6, four VFMLA
  operations per nonzero coefficient, no inner-loop calls, and no accumulator
  spills.
- Bounds checks remain. Stores/reloads between k blocks are intentional.
- The native frame is 208 bytes. The generic emulated variant is not native
  codegen evidence.

### Beta=0 repair

- In both precisions, the old transposed-Lower path multiplied C by zero. Thus
  unused NaN/Inf C values contaminated the output. It now clears the active
  triangle.
- Persistent tests fail on the unchanged baseline. They pass against Reference
  BLAS and the repair.
- Scalar argument and slice-validation order, alpha-zero handling, and NoTrans
  arithmetic do not change.

## Rejected experiments

1. Public GEMM for rectangular off-diagonal blocks: smaller D cases regressed
   11-33%; S cases regressed 16-67%. Removed, despite a modest Cholesky gain.
2. Direct SIMD without k blocking: compact leaves improved 8-45%. But actual
   lda=ldc=256 panels with k=128/192 regressed 5-7%, and Cholesky256 was
   inconclusive. The final 64-term blocking removes these regressions.

Better A-panel reuse was the reason for k blocking. Hardware counters did not
measure cache misses. Timing and loop structure support the reason; there is
no measured cache-hit rate.

## Results

- Independent detached baseline, identical persistent harnesses, prebuilt
  binaries, ten alternating samples per case.
- No concurrent builds, tests or profiles during timing; ordinary desktop
  activity continued.
- Sample time: SYRK leaves 100ms; actual-panel controls 200ms; final
  single-worker caller 500ms; four-worker callers 200ms.
- All SYRK cases allocate zero bytes. Cholesky stays at 72 B/op and three
  allocations on both revisions.

| Public Cholesky | Workers | Baseline | Candidate | Time change |
| --- | ---: | ---: | ---: | ---: |
| n=32 | 1 | 16.58 us | 16.56 us | inconclusive, p=.075 |
| n=128 | 1 | 252.4 us | 250.1 us | -0.92%, p<.001 |
| n=256 | 1 | 1.129 ms | 1.112 ms | -1.50%, p<.001 |
| n=32 | 4 | 16.56 us | 16.55 us | inconclusive, p=.271 |
| n=128 | 4 | 252.4 us | 250.1 us | -0.94%, p<.001 |
| n=256 | 4 | 1.129 ms | 1.112 ms | -1.44%, p<.001 |

These longer final results replace the noisier initial caller gains of 1.4-2.5%.

Panels with n=64, lda=ldc=256, alpha=-1, beta=1:

- Actual shared storage improves 8.10%, 7.31%, and 7.49% at k=64/128/192.
- Matching independent-storage controls improve 6.86-7.71%.

Compact/padded Trans leaves, both triangles and precisions:

- n=16/17,k=16: about 41-44% less time.
- n=64,k=16: D improves 5.5-10%; S improves 16.8-22.2%.
- n=64,k=256: D improves 7.5-10.8%; S improves 12.9-16.8%.
- n=4/15 fallback controls: mostly inconclusive. D Lower n=4 costs 0.84-1.44%
  more (about 3-5 ns). This cost stays with the beta-zero correctness fix.
- NoTrans n=64,k=256 controls: inconclusive in both precisions and triangles.

A leaf geomean is not application throughput, and the improvement is not
universal. The evidence covers only the tested dimensions and worker counts.

## Validation

Persistent tests cover triangle/transpose combinations, offset guards, padding,
real overlap rejection, shared backing, vector tails, and cache boundaries
63/64/65/127/128/129. They also cover signed zero, skipped NaN/Inf operands,
exceptional beta, alpha Inf with zero operands, and one-time beta scaling.

- Sequential-reference comparisons are exact, except NaN classification.
- Dyadic native differential tests require exact finite numerical equality.
  Dedicated tests also cover signed-zero semantics.
- Zero-beta tests include exceptional C across vector/cache blocks.

Reference: Reference-LAPACK v3.12.1 commit
`6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca`, BLAS/SRC/dsyrk.f. The opt-in
Darwin bridge calls Homebrew LP64 Reference BLAS, not Accelerate. Row-major
adaptation flips the triangle and transpose. Native SYRK plus Dpotrf/Dpotrs
differential tests passed three runs.

Also passed:

- full Go 1.27.1 SIMD suite
- affected packages: default Go 1.26.4, minimum Go 1.24.0, combined
  safe/noasm/bounds
- focused SYRK race tests; GODEBUG=simd=0 correctness
- BLAS/LAPACK vet; formatting and diff checks

No cross-compilation was requested.

## Reproduce

- Benchmarks: BenchmarkD/SsyrkSIMDShapes, BenchmarkDsyrkCholeskyPanel, and
  BenchmarkFactorization/Cholesky.
- Runner: go-optimisation/scripts/compare_benchmarks.py. Analysis: benchstat.
- Local evidence in `/tmp/gonum-syrk.7Cnqnn/`: `final-boundaries`,
  `final-wide`, `panel-blocked`, `final-consumer`, `four-worker-consumer`,
  profiles, final-codegen.txt, binary hashes, metadata, and rejected cohorts.
- This temporary evidence is not a repository fixture. The benchmarks and tests
  above are persistent and runnable.

## Remaining work

- This closes the measured TT-GEMM/SYRK goal, not all BLAS optimization.
- Measure native AMD64 before you change its production dispatch.
- Future targets: bounds-check cost, conservative inactive-triangle alias
  rejection, and Cholesky condition-estimation triangular solves.
- On future Go/SIMD releases, recheck vector widths, APIs, numerical gates and
  native crossovers before you change the experiment guards or cache cutoff.
