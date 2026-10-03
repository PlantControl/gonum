# Production BLAS SIMD performance pass

Measured 2026-09-05 on Apple M1 Pro, macOS 26.6.2 (25G83), darwin/arm64,
Go 1.27.1, `GOEXPERIMENT=simd`, `GODEBUG=simd=128`.

- Independent baseline: `4cbe23600778fdb3b89f8ddfccf5b50e10c494f8`, with the
  previous AMD64 patch import.
- Measured implementation: `d09a2b7bb77af78f0ad695d875e7fa4ebcd5495f` on
  `codex/arm64-simd-blas` (GEMV/norm milestone `27ce2051`, GEMM milestone
  `d09a2b7b`).

These are native ARM64 measurements, not AMD64 predictions.

## Public BLAS results

Times are medians. Negative percentages mean less time. Every changed timing
here has `p=0.002` and six samples per revision. B-transposed Dgemm
controls had no statistical change (`p=0.818`).

| Routine and shape | Baseline | Changed | Time change |
|---|---:|---:|---:|
| Dgemm NN, 10³ | 572.8 ns | 297.7 ns | -48.0% |
| Dgemm NN, 100³ | 217.3 µs | 145.6 µs | -33.0% |
| Dgemm NN, 1000³ | 190.1 ms | 147.5 ms | -22.4% |
| Dgemm TN, 100³ | 217.4 µs | 145.6 µs | -33.0% |
| Dgemv N, 10×10, unit increments | 72.98 ns | 57.47 ns | -21.3% |
| Dgemv T, 10×10, unit increments | 63.98 ns | 48.48 ns | -24.2% |
| Dgemv N, 100×100, unit increments | 5.577 µs | 1.934 µs | -65.3% |
| Dgemv N, 1000×1000, unit increments | 551.2 µs | 181.9 µs | -67.0% |
| Dgemv T, 1000×1000, unit increments | 211.7 µs | 194.3 µs | -8.2% |
| Dgemv T, 1000×10, unit increments | 5.316 µs | 2.758 µs | -48.1% |
| Dgemv T, 100×100, increments 2/3 | 5.639 µs | 4.479 µs | -20.6% |
| Dgemv T, 1000×1000, increments 2/3 | 514.2 µs | 400.3 µs | -22.2% |
| Dgemv T, 1000×10, increments 2/3 | 6.238 µs | 6.902 µs | +10.6% |
| Dnrm2, n=1000, unit increment | 2.456 µs | 781.0 ns | -68.2% |
| Dnrm2, n=100000, unit increment | 250.49 µs | 76.74 µs | -69.4% |
| Dnrm2, n=100000, increment 5 | 250.0 µs | 124.4 µs | -50.3% |

The short-row strided regression is not resolved. Its inner instruction count,
loads, checks and call/spill behavior match the baseline. Code placement and
register assignment differ. A frontend or operand
scheduling cause is a hypothesis. A separate f64 fixture measured only +3%
for that shape, so call context and binary layout are important.

Tiny strided public GEMV and norms add approximately 0.7–2 ns. Internal
length-three strided norms add approximately 1.5 ns (27%). The production gate
avoids the larger cost of the compensated vector kernel at these lengths.

### GEMM dispatch

| GEMM dispatch, GOMAXPROCS=4 | Baseline | Changed | Time change |
|---|---:|---:|---:|
| Dgemm, 60³ | 52.39 µs | 33.58 µs | -35.9% |
| Dgemm, 100³ | 163.7 µs | 106.6 µs | -34.9% |
| Dgemm, 160³ | 332.5 µs | 219.3 µs | -34.0% |
| Sgemm, 60³ | 32.18 µs | 19.64 µs | -39.0% |
| Sgemm, 100³ | 114.21 µs | 61.78 µs | -45.9% |
| Sgemm, 160³ | 220.7 µs | 127.9 µs | -42.1% |

- Both precisions also improved in every measured 1000×100×10, 100×1000×10
  and 128×128×8 dispatch case at `GOMAXPROCS=1` and `4` (7.7–36.0% less
  time).
- Direct neighboring-column tests at n=56,59,60,61,64 rejected an early tail
  implementation and confirmed the final full-column approach.
- With `GOMAXPROCS=4`, the forced blocked helper is approximately 8% slower
  than serial at 60³ (a single output tile). It wins at 100³ and 160³.
- Skinny shapes need separate policy calibration. Forced parallel Sgemm at
  128×128×8 is approximately 10% slower than its serial helper.
- Forced helpers omit public validation/beta setup, so these are not identical
  end-to-end policy comparisons. No parallel threshold changed.

## Consumer results

| Shape | Dgebrd time change | SVD values-only, baseline → changed | SVD thin vectors, baseline → changed |
|---|---:|---:|---:|
| 16×16 | -7.1% | 17.17 → 16.54 µs (-3.7%) | 39.03 → 37.06 µs (-5.1%) |
| 64×64 | -28.6% | 336.8 → 285.0 µs (-15.4%) | 1.0231 → 0.9398 ms (-8.1%) |
| 128×128 | -28.5% | 1.991 → 1.602 ms (-19.5%) | 7.183 → 6.410 ms (-10.8%) |
| 256×256 | -3.0% | 19.18 → 18.67 ms (-2.7%) | 116.4 → 114.7 ms (-1.5%) |
| 512×64 | -23.9% | 1.850 → 1.672 ms (-9.6%) | 4.321 → 3.867 ms (-10.5%) |
| 64×512 | -28.2% | 1.4511 → 0.9917 ms (-31.7%) | 3.445 → 2.576 ms (-25.2%) |
| 256×128 | -25.5% | 4.229 → 3.607 ms (-14.7%) | 13.24 → 12.04 ms (-9.1%) |
| 128×256 | -27.4% | 3.584 → 2.549 ms (-28.9%) | 11.696 → 9.993 ms (-14.6%) |

Allocation counts did not increase. Single-worker public BLAS, Dgebrd and
thin-vector SVD measured zero allocations. Values-only SVD keeps two small
allocations (40 bytes). Parallel GEMM with `GOMAXPROCS=4` keeps its existing
task allocations (10–66 per call for these shapes). The SIMD tiles do not
allocate.

## Candidate and cutoff checks

### Prefix scans

**Correction, 2026-09-05:** The historical CumProd timings in this section
used inputs that overflowed. They are not valid ordinary-data performance
evidence. The [tail/stride follow-up](RESULTS_TAIL_STRIDE.md) replaces the
benchmark inputs with finite reciprocal factors and reruns both revisions with
the same corrected harness. Its AMD64 results do not replace native ARM64
measurements. This input correction does not affect the CumSum results.

The final prefix comparison uses a fresh baseline build at the same base
commit, 75 ms samples and the unchanged candidate harness:

- CumSum takes 40.3% less time at n=31 and 42.8% less at n=4096
  (4.463 → 2.554 µs).
- CumProd takes 16.2% and 12.0% less (3.115 → 2.742 µs at n=4096).
- All four have `p=0.002`, n=6 and zero allocations.

This resolves the prefix timing regression in the
[previous import report](RESULTS_IMPORT.md), with no new scan order.

### Norm cutoff

These cutoff samples use 50 ms and six interleaved rounds.

- n=32 improves 32.7% contiguous and approximately 5.8% with increments 2/17.
  n=33 also improves.
- At n=4096 the gains are 67.2% contiguous and approximately 47% strided.
- Zero-leading length-4096 inputs have effectively no change. Length 32 adds
  approximately 1–1.5 ns.

### Strided GEMV boundary

- n=31 stays on the old loop. n=32/33 improves approximately 19–22% at both 10
  and 1000 rows.
- Same-binary transposed SIMD dispatch at 15×16, 31×16, 32×16, 32×17 and
  32×255 improves 5.7–36.3%.
- Excluded 15×17 and 31×17 have no statistical change. Excluded 31×255 still
  adds 3.4% in that fixture.

These are measured tradeoffs for this host, not universal wins.

## Implementation and dispatch

### GEMM

Portable Dgemm/Sgemm kernels keep a four-row, four-vector output tile in
registers across the inner dimension. Single-vector and scalar column tails use
the same accumulation strategy. Incomplete rows use existing helpers. They
do not repeatedly load/store C, and reuse B loads and broadcasts.
The existing Gonum generator makes the float32 source from float64.

Production GEMM selection is ARM64-only, with the SIMD experiment on. B must
be untransposed. A can be transposed. The call needs at least four rows, four
inner elements and four vectors of columns (8 doubles or 16 singles on this
host). C must not overlap A or B. Existing parallel cutoffs do not change.

### GEMV

Float64 GEMV uses shared portable candidates for disjoint unit-stride inputs
with both dimensions at least eight. Transposed GEMV also needs at least 32
rows or at most 16 columns. Short-wide shapes keep the fallback. Four-vector
unrolling makes the residual-row path faster, with the same multiply/add
order. Ger was not promoted, because its measured crossover did not justify
production selection.

The strided transposed GEMV fallback uses indexed, incrementally advanced
matrix rows from 32 columns. Short rows keep the original AXPY loop.
Value-range and unconditional indexed rewrites were rejected. On this CPU,
fewer bounds checks did not offset severe short-row regressions.

### Norms and prefixes

Float64 norms and distances use compensated portable sums of squares from
length 32. Zero-leading or extreme first values select the scaled recurrence
immediately. Unsafe sums found later retry that recurrence. Tests cover tiny
calls, NaNs, infinities, overflow/underflow and zero-increment behavior.

Prefix candidates peel their first lane step. This removes a redundant loop
condition and keeps the existing grouped scan order. Invariant AXPY broadcasts
are outside the loop. Neither change promotes AMD64 production dispatch.

Older Go releases, unsupported targets, `safe`, `noasm` and GCCGo keep their
non-SIMD production paths. AMD64 assembly stays the production baseline,
with the candidate comparison harness available. These changes
do not make an experimental Go API GA or set a portable performance cutoff.

## Method

Prebuilt test binaries ran sequentially. Alternate rounds reversed the
baseline/change order. Each before/after comparison uses six samples and
`benchstat`. Public BLAS uses 75 ms per sample. GEMM shape/worker and LAPACK
consumer comparisons use 50 ms. `GOMAXPROCS=1` is the default. Explicit GEMM
worker sub-benchmarks also set `GOMAXPROCS=4`. Compilation, tests and other
agents' timing jobs stopped before measurement.

Both checkouts have identical benchmark fixtures. The Dgemv repeated-call
benchmark uses beta=1, not beta=3, so its output stays finite. SVD and
bidiagonal benchmarks restore their input on each iteration, reuse queried
workspace and include the input copy in timing. The SVD vector case computes
thin U and VT. Fixtures use deterministic ordinary dense inputs, not every
possible matrix distribution or conditioning.

## Numerical and performance guardrails

An uncompensated norm promotion first failed the existing Dgesvd test for a
300-by-150, very-large-magnitude matrix, together with GEMV promotion. The
independent baseline passed. Each promotion alone passed. No SVD tolerance was relaxed. Compensation of summation error, and
of product error where SIMD MulAdd is fused, repaired the regression. The
method follows
[TwoProductFMA and Dot2](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf).
Portable MulAdd is not necessarily fused on every backend. Thus fixture-level
1-ULP checks do not claim universally correctly rounded norms.

Persistent tests cover a 256-bit norm oracle, cancellation and exponent
extremes, vector/tile/dispatch boundaries, all GEMM column remainders,
transpose and beta combinations, padding, overlapping slices,
positive/negative/zero increments, zero coefficients with NaN/Inf data, and
prefix carry/alias order.

An existing zero-beta transposed GEMV bug also surfaced. The scalar path
cleared past the logical output length. It now clears only the n outputs, with
internal and public BLAS regression tests.

Generated M1 code uses vector FMA for GEMM and norm product-error recovery.
The full GEMM tile and norm arithmetic helpers have no hot-loop vector spills.
Norm helpers inline. Bounds/address bookkeeping still exists, so this is not
a claim of an ideal hand-written microkernel. Scalar GEMM tails stay in
their declared precision, including float32 FMA.

Validation passed:

- full Go 1.27.1 SIMD suite
- full supported Go 1.26.4 suite
- affected internal-assembly, BLAS, LAPACK and mat tests under SIMD emulation,
  `safe` and `noasm`
- internal-assembly and BLAS race/bounds tests
- exact float32 regeneration
- formatting, import policy, copyright and diff checks

This pass did no new cross-compilation or native AMD64 timing.

## Go development branches

The [upstream investigation](UPSTREAM.md) records exact revisions and open
compiler changes, including master, Gerrit and dev.simd. The
[experimental SVE SGEMM](https://go-review.googlesource.com/c/go/+/827812)
gave the useful register-tiling and broadcast-reuse pattern. This work assumes
no SVE API or capability on the M1. Current loop-invariant-code-motion work
also supports explicit hoisting of invariant broadcasts today.

With future Go releases, recheck VEX spill/move encoding, FMA accumulation
forms, NEON broadcasts, float reductions and missing portable
widening/permutation operations. Rebenchmark native AMD64 assembly against
candidates before you change its dispatch. Open changes and accepted
experiments are not a GA schedule or proof that the compiler will select every
profitable algorithm.

## Remaining SVD bottleneck

The [subsequent SVD follow-up](RESULTS_SVD.md) repairs the Dlasr defects below
and measures a shared row-blocked rotation implementation.

A separate baseline CPU profile covered 256-by-256 SVD with thin vectors.
Dlasr had approximately 71% of flat samples and Dbdsqr 76% of cumulative
samples. Dgemv and Dgemm had approximately 12% and 9% cumulative. These are
sampled profile proportions, not predicted speedups. BLAS improvements help,
but rotation application is the next large SVD target, especially strided
column rotations and reuse across successive rotations.

Read-only inspection also found a pre-existing duplicated column loop in the
left/top/backward case of Dlasr, and a reversed fixture-copy direction in its
shared test. That case is not the variable-pivot path in the profile. Both
defects need a separate LAPACK correctness repair and independent reference
tests. This BLAS pass did not change them.

## Reproduce

Build the baseline in a separate worktree. Apply only these identical
benchmark changes from this pass:

- `blas/testblas/level2bench.go`
- the two LAPACK benchmark helpers and their wrappers
- the f64 norm/strided-GEMV benchmark files

Do not change the baseline implementation. Build each one time:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c ./blas/gonum -o blas.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c ./lapack/gonum -o lapack.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c ./internal/asm/f64 -o f64.test
```

Alternate the two binaries for at least six rounds with identical flags:

```sh
GOMAXPROCS=1 GODEBUG=simd=128 ./blas.test -test.run '^$' \
  -test.bench '^Benchmark(Dgemm|Dgemv|Dnrm2)' -test.benchtime=75ms -test.benchmem
GOMAXPROCS=1 GODEBUG=simd=128 ./lapack.test -test.run '^$' \
  -test.bench '^Benchmark(Dgesvd|Dgebrd)$' -test.benchtime=50ms -test.benchmem
benchstat baseline.txt changed.txt
```

For AMD64, use the native comparison procedure in [README.md](README.md).
Record CPU/vector width and affinity. Keep the default assembly run. Do not
read ARM64 `current`-versus-candidate results as assembly comparisons. On
ARM64, `current` now includes the selected production SIMD kernels.
