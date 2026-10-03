# Native SVD rotation follow-up

## Summary

Scalar cache blocking in Right/Variable Dlasr cuts 256-by-256 thin-vector
Dgesvd time by about half. The work also fixes a Left/Top/Backward
correctness bug and the shared Dlasr test.

## Setup

- Base: `150606198bafc4b32c207753acdfa6b43b2c288d`.
- Host: Apple M1 Pro, darwin/arm64, macOS 26.6.2. Not CPU-isolated.
- Primary toolchain: Go 1.27.1, `GOEXPERIMENT=simd`, `GOMAXPROCS=1`.
- Default comparison: installed Go 1.26.4 without the experiment.
- No AMD64 measurements or cross-compilation.

Method:

- The two source revisions used the same persistent benchmark harness.
- Native runs were serial, with six samples per case and alternating
  base/change order.
- Final rotation samples used 50ms. SVD samples used 100ms. Earlier screens
  used 100-200ms.
- benchstat interprets the results, not a single fastest sample.

The new base profile of 256-by-256 thin-vector Dgesvd put 70.96% of flat
samples in Dlasr and 75.72% cumulative in Dbdsqr. The main branch was
Right/Variable/Forward, which accesses columns of row-major matrices.

## Results

Median Dgesvd time with thin singular vectors, before and after the change:

| Toolchain | Matrix | Base | Changed | Time change | p-value |
| --- | --- | ---: | ---: | ---: | ---: |
| Go 1.27.1 SIMD | 64 × 64 | 932.7 µs | 927.5 µs | no significant change | 0.485 |
| Go 1.27.1 SIMD | 128 × 128 | 6.479 ms | 6.120 ms | -5.53% | 0.002 |
| Go 1.27.1 SIMD | 256 × 256 | 114.60 ms | 55.13 ms | -51.90% | 0.002 |
| Go 1.27.1 SIMD | 256 × 128 | 12.03 ms | 11.67 ms | -3.01% | 0.041 |
| Go 1.27.1 SIMD | 128 × 256 | 10.027 ms | 9.653 ms | -3.74% | 0.026 |
| Go 1.26.4 default | 64 × 64 | 1.042 ms | 1.042 ms | no significant change | 0.699 |
| Go 1.26.4 default | 256 × 256 | 120.48 ms | 60.90 ms | -49.46% | 0.002 |

- Each comparison has six samples per revision.
- 95% benchstat intervals for 256-by-256: SIMD ±4% base and ±6% changed;
  default ±1% and ±8%.
- Smaller default-toolchain gains are not significant.
- Values-only SVD cases did not change beyond noise.
- No measured SVD case had a significant regression.
- Thin-vector cases had zero allocations per operation.

Dlasr microbenchmarks:

- Dense Right/Variable/Forward, compact 256-by-256: 303.66 µs to 47.50 µs
  (-84.36%, p=0.002).
- Backward: 307.84 µs to 47.13 µs (-84.69%, p=0.002).
- Padded stride 259: no significant change in either direction, about 46 µs.
- The 63-by-65 fallback cases: no significant change.

Microbenchmark costs:

- Some blocked boundary cases regressed 2-8%.
- Sparse Backward 64-by-256 at stride 259 regressed from 1.567 µs to
  1.762 µs (+12.41%, p=0.002).
- Identity-only calls usually added about 2-3 ns.

These cases stay in the benchmarks. Use them to limit future dispatch tuning.

### Native comparison

Separate comparison with the same inputs:

| Matrix | Changed Gonum | Reference-LAPACK | Lower elapsed time |
| --- | ---: | ---: | --- |
| 32 × 24 | 105.80 µs | 73.04 µs | Reference-LAPACK |
| 24 × 32 | 101.00 µs | 80.17 µs | Reference-LAPACK |
| 96 × 64 | 1.327 ms | 1.088 ms | Reference-LAPACK |
| 64 × 96 | 1.210 ms | 1.059 ms | Reference-LAPACK |
| 256 × 256 | 54.64 ms | 89.61 ms | Gonum |

All five paired differences have p=0.002, n=6 per implementation. Gonum
takes about 39% less time in the large case. Reference-LAPACK takes 12-31%
less time in the smaller cases. This harness uses different inputs from the
Dgesvd benchmark above, so compare implementations within each table only.
The reference backend and timing boundaries are in Reproduce.

## Changes

- **Correctness fix.** The old Left/Top/Backward loop was duplicated and
  applied each rotation once per matrix column. This is not a valid
  before/after speed comparison. It is not the profiled SVD branch.
- **Test repair.** The shared Dlasr test had reversed copies, wrong
  bottom-pivot reference indices and a missing right-side transpose.
  Nonzero structured and seeded random fixtures now exercise all 12
  side/pivot/direction combinations.
- **Row blocks.** Right/Variable rotations use row blocks only when the two
  dimensions are at least 64 and all rotations are active.
  - Each row keeps its original rotation order.
  - Nominal blocks have 32 rows. A final remainder shorter than 16 rows
    merges into the block before it.
  - Small and identity-containing cases keep the original traversal.
  - An out-of-line helper keeps the fallback code small.
  - The matrix index increments by its stride, so there is no per-row
    address multiplication.

This is architecture-neutral scalar cache blocking, not a new SIMD kernel.
Generated ARM64 code uses scalar fused arithmetic and keeps bounds checks.
The SIMD BLAS configuration does not otherwise change.

Rejected options:

- A row-at-a-time candidate was approximately five times slower on large
  padded rotation cases. This is consistent with serial dependent arithmetic.
- Cutoff and tail sweeps rejected a 33-row crossover and a one-row final
  block.
- Measure compact and padded strides. The original compact 256-by-256
  rotation was much slower than the padded one.

## Reference and correctness

Source review pin: [Reference-LAPACK v3.12.1,
6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca](https://github.com/Reference-LAPACK/lapack/tree/6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca),
`SRC/dlasr.f`. The SVD bridge also calls Dbdsqr and Dgesvd.

- The installed Homebrew formula is `lapack 3.12.1_1`. Its runtime `ILAVER`
  reports **3.12.0**. We record the two identifiers separately.
- Its linked BLAS is the `libblas` of the Reference-LAPACK keg, not OpenBLAS
  or Accelerate.

Dlasr checks:

- All 12 Dlasr computational branches were source-reviewed and
  independently tested. Tests include mixed identities and shapes around the
  dispatch and tail-merge boundaries.
- Row-major leading-dimension validation is a deliberate adaptation of the
  column-major contract of Netlib.
- The exact duplicated-rotation regression failed before the repair.
- Padding, non-finite identity skips and signed zero keep exact checks.

The larger oracle cases caused cancellation-sensitive failures in a new
fixed output-relative comparison. One failure was in an unchanged branch.
The new test instead bounds absolute error by the input vector norm and
rotation count. It uses `2*gamma_(3*r)` for the two evaluations and rejects
non-finite results. Existing numerical tolerances are not relaxed.

Persistent Dbdsqr and Dgesvd tests:

- Dbdsqr: upper/lower matrices, optional transformed matrices and
  values-only cases.
- Dgesvd: square/tall/wide shapes, five vector-job combinations,
  tiny/ordinary/huge scales and a rank-deficient case.
- They check singular values, reconstruction and orthogonality. They do not
  require identical singular vectors.

This is not a full transitive SVD parity audit. The new differential suite
does not cover:

- Overwrite and mixed All/Store jobs.
- Minimum-workspace paths.
- Difficult non-convergence.
- Extreme or clustered bidiagonal cases.

## Reproduce

The optional bridge uses the existing `netlib && darwin && cgo` configuration
and Homebrew LAPACK paths. It adds no production CGo dependency.

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -tags netlib ./lapack/gonum \
  -run 'Test(Dlasr|DlasrNetlibDifferential|DbdsqrNetlib.*|DgesvdNetlib.*|NetlibRuntimeVersion)$' -count=3

GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd GOMAXPROCS=1 \
  go test -tags netlib ./lapack/gonum -run '^$' \
  -bench '^BenchmarkDgesvdNetlibKernels$' -benchmem -count=6 > native-svd.txt
benchstat -col /implementation native-svd.txt
```

The paired native benchmark:

- Computes Dgesvd with thin vectors on the same logical random matrix on the
  two sides.
- Each implementation queries and reuses its own preferred workspace.
- Includes input restoration on the two sides and the native CGo call.
- Excludes layout conversion and workspace allocation.
- Go allocation counters do not count native allocations.
- Compares Reference-LAPACK plus reference BLAS. It does not compare an
  optimized vendor library or the different Dgesdd algorithm.

Validation passed:

- Full default and SIMD suites.
- Affected LAPACK/mat `safe` and `noasm` suites, and LAPACK `bounds`.
- Focused Dlasr race tests.
- The persistent Netlib suite, three times.
- Formatting, imports, copyright and diff checks.
- A broader Dlasr/Dbdsqr/Dgesvd race run during integration.

## Limits

These results apply to this host and benchmark matrix, not to every SVD
workload. Not every rotation layout is faster.

## Remaining work

- Measure the shared traversal change on AMD64 before you claim a speedup
  there.
- Compare optimized OpenBLAS or Accelerate separately.
- Profile the remaining SVD cost before another kernel pass.
- Evaluate Dgesdd as a separate algorithmic project. Do not attribute a
  Dgesvd-versus-Dgesdd difference to SIMD.
