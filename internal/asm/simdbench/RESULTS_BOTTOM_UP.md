# Bottom-up BLAS optimization

## Summary

- A one-RHS path in DTRSM/STRSM cuts one-RHS solve time by 70.68–86.16%
  (p<0.001).
- Public one-RHS LU/Cholesky solves gain most. QR/LQ solves gain less.
- Controls show no significant regression, except one small DPOTRS case that a
  recheck did not reproduce.
- We rejected and reverted an AXPY leaf change.
- Allocation counts do not change.

## Scope and setup

The pass starts at the production BLAS Level 1 AXPY leaf. It then checks BLAS
triangular solves, LAPACK solves and public `mat` consumers. It does not change
LAPACK algorithms. AMD64 production assembly and opt-out implementations are
outside the change. Required workflow: `go-optimisation` and `gonum-simd`, with
dedicated Sol numerical/harness and code-generation reviews.

- Starting implementation: `3eb28d78306f776f4d2496e81db9b6f53026c938`.
- Clean benchmark checkpoint: `dd91d903edd614c0506fad8174e47dfcbba49b44`.
  It adds only persistent tests and short-RHS benchmark cases. Baseline
  binaries come from its detached clean worktree.
- Host: Apple M1 Pro, darwin/arm64.
- Benchmarks: Go 1.27.1 with `GOEXPERIMENT=simd`, `-pgo=off`, `GOMAXPROCS=1`.
  No repository PGO profile or GOFLAGS override was present.
- Default tests: installed Go 1.26.4.

Before the edits, we read Go's [release history](https://go.dev/doc/devel/release)
and the installed Go 1.27.1 sources again. Portable SIMD is still an
[experiment](https://github.com/golang/go/issues/78902). This pass uses the
existing native ARM64 leaf and adds no new experimental API dependency.

## Baseline profile

Public LU SolveTo, n=256, one RHS:

- 50.16% of sampled CPU time is cumulatively in f64.AxpyUnitary, with 12.13%
  in its overlap check.
- DTRSM calls this leaf once per coefficient, also for a one-value slice.
- The old leaf cannot inline (compiler cost 417, budget 80). It runs alias
  checks and vector setup before its one-element scalar tail.

## Rejected leaf candidates

The first candidate moved the original vector body into an unexported helper.
The exported wrapper handled only `len(x)==1`, in both precisions. Other
lengths kept the original arithmetic, bounds and overlap behavior.

- A below-vector-width loop wrapper had cost 83 and did not inline.
- The one-element wrapper had cost 77 and inlined into BLAS callers.
- DTRSM and leaf benchmark machine code showed direct scalar fused arithmetic
  and stores for that branch, and the native helper for other lengths.

Ten interleaved rounds at 150 ms rejected that wrapper (all p<0.001):

- float64 n=3/4/8/16 regressed 8.96/18.93/8.34/5.85%.
- float32 n=7/8/16 regressed 5.17/15.99/6.04%.
- The one-element leaf benchmark stayed near 3.42 ns in both precisions
  (inconclusive).

The fixture has a loop-carried data dependency, so it measures latency, not
independent-call throughput. Inlining was confirmed, so a second nested call
does not explain the regressions.

A screen (three rounds at 100 ms) also tried a separate in-function singleton
return and a sub-vector threshold in the existing fallback. Neither avoided
short-vector costs in all cases. Screens are diagnostic only. We reverted all
AXPY production changes. The leaf data stays as rejection evidence.

## Single-RHS BLAS specialization

The accepted candidate handles one RHS once, at the DTRSM/STRSM entry point.
The scalar work stays inside the solve, with no AXPY dispatch per coefficient.
AXPY and all multi-RHS arithmetic do not change.

It keeps the transpose-specific update order, the zero-coefficient skip, the
alpha timing and the reciprocal multiplication. A direct TRSV call cannot keep
these: its NoTrans path sums products before the subtraction and divides by the
diagonal.

The running value stays in a local variable:

- NoTrans rows: apply alpha, do the original ordered updates, multiply by the
  reciprocal, store once.
- Transposed rows: apply the reciprocal, update the remaining rows with the
  pre-alpha value, then apply alpha and store.

This removes the repeated AXPY calls and the conservative reload/store traffic.
It does not regroup sums. Active A/B entries must be disjoint, as in standard
TRSM use. Tests cover disjoint regions that share a backing allocation.

Dispatch occurs only in the existing experimental ARM64 BLAS configuration, for
left-sided solves with exactly one RHS. It occurs after validation and alpha-zero
handling. Multi-RHS and AMD64/default/opt-out arithmetic do not change. The
helper uses ordinary Go scalar operations, not new SIMD instructions. It removes
composition overhead that the SIMD backend exposed.

### Tests and checks

Persistent D/S solve tests compare to the old two-RHS path. They cover:

- all real transpose/triangle combinations, Unit/NonUnit;
- alpha=0/1/-0.75, stride/padding, extreme scales;
- zero coefficients with NaN/Inf sources, backward-order overflow;
- shared backing with B in A row padding.

Residual checks use the existing precision tolerances. Native DGETRS/DPOTRS
gates give an independent reference.

- Candidate implementation: `c284b8ac7f5e12543f4764a4345e6aede1f65d7a`. The
  final BLAS, LAPACK and mat binaries come from it, with PGO off.
- Later one-RHS native extreme-scale test cases do not change production or
  benchmark source.
- Pass: full standard/SIMD suites; safe/noasm/bounds BLAS/LAPACK/mat suites;
  focused race and emulation checks; strengthened native solve tests.
- Float32 generation is synchronized.
- Machine code shows scalar fused updates, and no AXPY, mallocgc or makeslice
  calls, in the specialized helpers. Existing bounds checks stay.

## Benchmark and numerical contracts

`BenchmarkAxpyUnitaryShort` calls the production operation directly at lengths
0/1/2/3/4/7/8/15/16/17/64/256/4096.

- Exact-binary nonzero fixtures and alternating alpha=+/-0.25 keep the input
  lifecycle bounded.
- Each iteration times one call and the alpha sign update.
- A b.N loop permits normal inlining. A result sink after the timer keeps the
  result. On this toolchain B.Loop uses KeepAlive transformations, so the older
  blanket inlining-ban explanation does not apply.
- Timing is in ns/op. SetBytes is the logical x-vector size, not total memory
  traffic.

Persistent tests cover ordinary and guarded short lengths, exact alias,
directional overlap where supported, immediate and partial-update panics, and
NaN/Inf/signed zero. They do not apply scalar overlap or bounds-check contracts
to AMD64 assembly. Leaf tests pass before and after the rejected wrapper, and with the original
implementation restored.

Acceptance uses the skill runner: prebuilt binaries, ten interleaved rounds,
matching harnesses, recorded binary hashes and runtime controls. No builds,
tests or profiles run during timing. A three-round public solve screen is
diagnostic only. It showed one-RHS gains and a small multi-RHS cost that needed
confirmation.

Local evidence is in `/tmp/gonum-bottom-up.In6f6t`. It is temporary. Use the
persistent benchmark entry points to reproduce.

## Results

### BLAS

Ten interleaved rounds, 100 ms per benchmark, GOMAXPROCS=1. All 24 single-RHS
cases improve with p<0.001. Each row is the range across U/N, U/T, L/N and L/T,
not an aggregate score.

| Routine | m | Time reduction, one RHS |
| --- | ---: | ---: |
| DTRSM | 32 | 72.57–84.52% |
| DTRSM | 128 | 71.40–84.77% |
| DTRSM | 256 | 70.68–85.17% |
| STRSM | 32 | 73.90–85.88% |
| STRSM | 128 | 72.48–86.13% |
| STRSM | 256 | 72.47–86.16% |

- DTRSM m=256 Upper/NoTrans: 132.60 to 38.88 us. Lower/Trans: 127.05 to
  18.84 us.
- No two-/sixteen-RHS control (24 total) has a significant regression.
- STRSM m=128, two-RHS Upper/Trans improves 1.69% (p=0.023). The other controls
  are inconclusive.
- All BLAS cases stay at zero B/op and zero allocs/op.

BLAS SHA-256 identities from the runner:

```text
baseline  32bcaebf0f1c11640c85e8b23ac0ef9cc85071e88d2d7de965b94d4e1665a388
candidate 754ffc59b871cdcca17631eacb5860bebf9a23157d716d328c0782316470197a
```

### Public solve consumers

Ten interleaved rounds at 150 ms, GOMAXPROCS=1. These are steady-state
`mat.SolveTo` calls on existing factorizations, without factorization time.
QR/SVD fixtures are tall, LQ is wide, LU/Cholesky are square. All gains have
p<0.001, n=10 per version.

| Public solve, one RHS | Size | Baseline us/op | Candidate us/op | Change |
| --- | ---: | ---: | ---: | ---: |
| QR | 128 | 439.2 | 412.4 | -6.09% |
| QR | 256 | 1734 | 1640 | -5.41% |
| LQ | 128 | 505.2 | 478.9 | -5.22% |
| LQ | 256 | 2140 | 2041 | -4.61% |
| LU | 128 | 69.68 | 18.30 | -73.73% |
| LU | 256 | 274.20 | 76.80 | -71.99% |
| Cholesky | 128 | 69.82 | 16.32 | -76.62% |
| Cholesky | 256 | 274.35 | 62.64 | -77.17% |

- Both one-RHS SVD controls and all ten sixteen-RHS controls are inconclusive.
- No public solve has a significant regression.
- Allocations do not change: LU/Cholesky zero, QR/LQ two, SVD seven.
- The SVD path does not use this triangular specialization.

### Reference-LAPACK solve comparison

Ten interleaved rounds at 100 ms, GOMAXPROCS=1. The reference is Homebrew
Reference-LAPACK 3.12.1_1 (ILAVER reports 3.12.0), not Accelerate or OpenBLAS.

- The native harness times matching RHS copies and solves. Factors, layout
  conversion and pivot conversion are outside timing.
- LP64 native pivots are prepared once.
- Native correctness tests compare solutions and scaled residuals, including
  one-RHS n=192 cases at RHS scales 1e-200 and 1e200.

All eight Gonum changes have p<0.001, n=10 per version. Reference times are
medians from the candidate binary. The comparison applies to these fixtures
only, not to backends in general.

| Solve, one RHS | n | Gonum before us | Gonum after us | Reference us | Gonum change |
| --- | ---: | ---: | ---: | ---: | ---: |
| DGETRS N | 128 | 69.85 | 18.38 | 7.691 | -73.69% |
| DGETRS T | 128 | 66.32 | 11.19 | 16.02 | -83.13% |
| DGETRS N | 256 | 275.22 | 76.37 | 30.35 | -72.25% |
| DGETRS T | 256 | 262.05 | 43.17 | 70.67 | -83.52% |
| DPOTRS U | 128 | 69.65 | 16.51 | 9.436 | -76.29% |
| DPOTRS L | 128 | 65.63 | 12.35 | 14.51 | -81.18% |
| DPOTRS U | 256 | 275.00 | 64.22 | 41.12 | -76.65% |
| DPOTRS L | 256 | 260.41 | 52.93 | 59.28 | -79.67% |

- Normal LU is still 2.39–2.52 times the Reference-LAPACK median. Upper
  Cholesky is 1.56–1.75 times.
- Transposed LU and lower Cholesky are below the reference medians.
- All native Reference-LAPACK before/after controls are inconclusive.
- Gonum sixteen-/sixty-four-RHS controls are inconclusive, except DPOTRS
  n=128, nrhs=16, Upper: 94.71 to 94.96 us, +0.27% (p=0.023). We keep this
  small regression in the record.
- All native cases stay allocation-free.

A ten-round, 300 ms recheck of that Upper/n=128/nrhs=16 case does not
reproduce the slowdown. Gonum: 93.66 to 93.51 us (p=0.739). Reference-LAPACK:
153.5 to 153.4 us (p=0.353). We do not discard the original result or pool it
with the recheck.

### Factorization controls

LU and Cholesky factorization-only controls at n=128/256 (ten rounds at
200 ms) show no significant change and no allocation change. This pass claims
faster solves, not faster factorization.

### Runtime controls

GOMAXPROCS=4, ten 100 ms rounds, m=256:

- One-RHS reductions: DTRSM 70.69–85.41%, STRSM 71.94–86.41% (p<0.001).
- All eight sixteen-RHS controls are inconclusive. All cases are
  allocation-free.
- This checks a runtime setting. The helper does not use four workers.

GOMAXPROCS=1 and GODEBUG=simd=0, ten 150 ms rounds, n=128/256:

- One-RHS public reductions: LU 72.41–73.78%, Cholesky 76.97–77.52%
  (p<0.001), with no allocations.
- The helper uses scalar Go and does not enter a software-emulated vector loop.
- We do not claim that GODEBUG disables all native architecture-specific leaves.

Runner SHA-256 identities for the other prebuilt packages:

```text
mat baseline     5f68dd7346290b90f620cc3bfffd63dde5cb90e255a02606e6efa4ddfd2b814d
mat candidate    c017bb65394f281143d8f514ed4ea39902a0936b481b8d26c75c64b082508354
lapack baseline  33e72dd38460ae55148a70a05e09dd41526d0f82f895463d9be5777440abc1bc
lapack candidate d394037d65f6f056ca2d1667c28bd4480f7fa9ce1912c6bc2163a77436de8891
```

## Limits

- GOGC and GOMEMLIMIT are unset. GODEBUG is unset except in the emulation run.
- These are local M1 Pro measurements, not cross-platform predictions.
- No AMD64 binary was cross-compiled or benchmarked.

## Next targets

We captured three-second profiles only after all timed acceptance runs.

- LU n=256/one-RHS: 97.82% of sampled CPU is flat in `dtrsmLeftVector`. No AXPY
  entry is in the hot frames. To close the normal-LU reference gap, improve the
  scalar triangular arithmetic and keep the numerical edge behavior.
- QR n=256/one-RHS: `AxpyInc` has 83.70% flat, below `gemvT` (90.22%
  cumulative), `Dlarft` and `Dormqr`. Setup factorization is only 0.82% of
  samples.

Next priority: a bounded strided-AXPY and transposed-GEMV fixture with those
real strides. Then run the Dlarft/Dormqr and public QR/LQ consumer gates.
Profiles identify candidates; they do not prove that a change wins.

- Keep two-/three-/four-RHS TRSM as a separate measured workload.
- Keep the SVD and factorization-only controls.
- Do not claim from this result that all Level 1, GEMM, LAPACK or AMD64 paths
  are optimized.

## Reproduce

Build the benchmark packages from clean baseline and candidate worktrees. Use
the same toolchain, tags and explicit PGO setting:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./blas/gonum -o blas.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./mat -o mat.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib -c ./lapack/gonum -o lapack.test
```

The native build needs the existing Darwin/cgo Homebrew Reference-LAPACK bridge.

1. Run the prebuilt binaries serially with the `go-optimisation` skill's
   `scripts/compare_benchmarks.py`.
2. Use `GOMAXPROCS=1`, `--rounds 10`, matching `--baseline` and `--candidate`
   paths, and a new `--output` directory.
3. Compare its `baseline.txt` and `candidate.txt` with benchstat.
4. Keep the metadata and raw output.
5. Do not run builds, tests or profiles during timing.

Selectors:

- BLAS: `^Benchmark[DS]trsmBlockedSizes$/m(32|128|256)$/n(1|2|16)$`,
  `--benchtime 100ms`. This covers one-, two- and sixteen-RHS controls, all four
  triangle/transpose orientations and both precisions. It runs only the size
  pairs that the benchmark defines, not the full Cartesian product.
- Public solves:
  `^BenchmarkFactorizationSolve$/(QR|LQ|LU|Cholesky|SVD)$/n=(128|256)$/nrhs=(1|16)$`,
  `--benchtime 150ms`.
- Native solves: `^BenchmarkD(getrs|potrs)Netlib$`, `--benchtime 100ms`.
- Factorization controls: `^BenchmarkFactorization$/(LU|Cholesky)$/n=(128|256)$`,
  `--benchtime 200ms`.
- GOMAXPROCS=4 BLAS: `^Benchmark[DS]trsmBlockedSizes$/m256$/n(1|16)$`, 100 ms.
- GODEBUG=simd=0 public:
  `^BenchmarkFactorizationSolve$/(LU|Cholesky)$/n=(128|256)$/nrhs=1$`, 150 ms.
- Native regression recheck: `^BenchmarkDpotrsNetlib$/n=128$/nrhs=16$/uplo=U$`,
  300 ms.

For diagnostic profiles, run the candidate mat binary. Replace QR with LU for
the LU profile:

```sh
GOMAXPROCS=1 ./mat.test -test.run='^$' -test.bench='^BenchmarkFactorizationSolve$/QR$/n=256$/nrhs=1$' -test.benchtime=3s -test.cpuprofile=qr.cpu
GOTOOLCHAIN=go1.27.1 go tool pprof -top ./mat.test qr.cpu
```

Do not profile during acceptance timing.
