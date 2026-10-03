# Bottom-up strided GEMV and QR

## Summary

A new ARM64-experimental GEMV helper speeds up strided-output reflector
products, QR solves and Dlarft. Allocations do not change. This work changes
BLAS composition, not a LAPACK algorithm or public API.

## Scope

Start point: `63e967af0426bdb8acd58a2f7bc2c7a6f871b6f6`. The work uses
`go-optimisation`, `gonum-simd` and dedicated Sol code/numerical reviewers.
The previous QR SolveTo n=256/one-RHS profile put 83.70% flat CPU in
`AxpyInc`, below transposed GEMV, Dlarft and Dormqr.

- Dlarft Forward/ColumnWise calls Dgemv(Trans) with input stride ldv and
  output stride ldt.
- Its matrix-row AXPY has unit input stride and strided output.
- Dormqr uses ldt=64 even when its reflector block width is 32. Benchmark
  that real workspace and packed ldt=32. The two fixtures are not
  interchangeable.

## Setup

- Host: Apple M1 Pro, darwin/arm64, macOS 26.6.2 (25G83), AC power. No CPU
  affinity or frequency pinning.
- Experimental builds: Go 1.27.1, GOEXPERIMENT=simd, explicit -pgo=off.
- Default tests: installed Go 1.26.4.
- No GOFLAGS override. The repository has no PGO profile.
- Native comparison: Homebrew Reference-LAPACK 3.12.1, not Accelerate or
  OpenBLAS.
- On 2026-09-06 we read again the official
  [release history](https://go.dev/doc/devel/release), the installed package
  documentation and the [portable SIMD proposal](https://github.com/golang/go/issues/78902).
  This work needs no new SIMD API.

Checkpoints:

- Clean baseline: `3e328a6074755a6e49798a94f2087bcf79b93aab`. It adds only
  the common benchmark/test harness to the start implementation.
- Candidate: implementation commit `c6416f1e`. Benchmark source is identical.
  Candidate-only tests exercise the new span validator.

Fixtures:

- AxpyInc benchmarks use exact-binary bounded inputs and alternating alpha.
- New GEMV benchmarks use beta=0 and give the same finite result each time.
- The old strided GEMV control used unbounded beta=1. We corrected it to
  beta=-1, which keeps repeated updates bounded.
- Dlarft benchmarks make valid Householder vectors before timing and
  overwrite T each iteration.
- Public solves reuse factorizations and include their normal API costs, not
  factorization.

Method:

- Compare identical prebuilt harnesses serially with the go-optimisation skill
  runner.
- Acceptance: ten interleaved rounds, GOMAXPROCS=1 unless stated,
  GOGC/GOMEMLIMIT unset. No concurrent builds, tests or profiles.
- Raw output and metadata are in the temporary directory
  `/tmp/gonum-strided-gemv.CXnTAM`. The persistent benchmarks permit reruns.

## Rejected leaf change

A unit-input-stride AxpyInc range loop removed repeated x index bounds
checks behind an overflow-safe valid-span gate, with the old invalid-input
fallback. The float64 and float32 bodies stayed
inlineable at cost 68/80, also in transposed GEMV. They used scalar fused
arithmetic and no per-row call.

Three-round, 100 ms diagnostic screens rejected it:

- Float64 length 17, output stride 32: 13.65 to 22.08 ns.
- GEMV 128x16 with reflector strides: 2.061 to 2.829 us.
- Public QR one-RHS n=128: 415.0 to 511.5 us. n=256: 1.632 to 2.018 ms.

Three samples are not enough for the acceptance significance test. These
medians support the rejection only. We fully reverted the two production
AXPY edits. Fewer bounds checks did not give faster generated code.

## GEMV helper

The gate is ARM64-experimental. It selects m>=8, 1<=n<=32, positive incX and
incY>1.

- The old contiguous dispatch and all fallbacks do not change.
- The gate validates complete active spans, without integer overflow, before
  it reads or writes data.
- Invalid, overlapping and unsupported-stride calls use the old helper, with
  its partial-write panic behavior.
- Read-only A and x can share storage, as in Dlarft.

Design:

- Four independent output values stay in scalar registers across matrix rows.
  Two-output and one-output tails handle the rest.
- Each output keeps its original update order, with a separate alpha*x scale.
- Beta zero starts with positive zero and does not read old y.
- The helper reduces strided output load/store traffic. It adds no wider SIMD
  instructions and does not regroup a dot-product reduction.

Generated code:

- Four FP accumulators, one scale multiplication and four scalar fused
  updates per row.
- No FP stack spills or AXPY calls in the row loop. Bounds checks remain.
- The non-inlined validator does three overflow-check divisions per
  qualifying call. Cheap shape/stride tests are outside it, so other calls do
  not pay for it.
- Not all bounds checks or setup overhead are removed.

## Tests

- Persistent leaf/GEMV tests cover real strides and boundaries.
- A new Dlarft conformance test reconstructs the reflector at block width 32
  with ldt=64.
- The reusable Dlarft benchmark is in lapack/testlapack/dlarft_bench.go. Its
  wrapper is in lapack/gonum/bench_test.go.
- Full default/SIMD tests pass. Affected safe/noasm/bounds suites pass.
- Focused race and GODEBUG=simd=0 checks pass.
- Independent reflector reconstruction and the native QR
  reconstruction/orthogonality tests pass.
- No tolerance is relaxed.
- The short-A panic fixture caps cloned slices. Without the cap, spare
  capacity permits the original two-index row reslice. The baseline and the
  candidate run the same corrected fixture.

## GEMV results

Ten interleaved 150 ms rounds, GOMAXPROCS=1, n=10 for each version. All
reflector-shaped cases improve with p<0.001. All cases stay at zero B/op and
allocs/op. Representative exact medians:

| m x n | incX / incY | Baseline ns/op | Candidate ns/op | Change |
| --- | --- | ---: | ---: | ---: |
| 128 x 1 | 128 / 32 | 423.8 | 160.7 | -62.09% |
| 128 x 7 | 128 / 32 | 1620.0 | 516.7 | -68.10% |
| 128 x 16 | 128 / 32 | 2139.0 | 715.1 | -66.57% |
| 128 x 31 | 128 / 32 | 3240 | 1485 | -54.18% |
| 128 x 16 | 128 / 64 | 2180.5 | 712.0 | -67.35% |
| 128 x 31 | 128 / 64 | 3305 | 1490 | -54.92% |
| 256 x 1 | 256 / 32 | 860.2 | 321.3 | -62.64% |
| 256 x 16 | 256 / 32 | 4402 | 1532 | -65.21% |
| 256 x 31 | 256 / 32 | 6650 | 3288 | -50.56% |
| 256 x 16 | 256 / 64 | 4594 | 1534 | -66.60% |
| 256 x 31 | 256 / 64 | 6884 | 3292 | -52.18% |

Controls:

- The two contiguous reflector controls are inconclusive.
- The old bounded incX=2/incY=3 sweep improves 23.61–68.96% at m=10,
  n=8..32, and 25.30–81.93% at m=1000, n=8..32 (all p<0.001).
- The n=1000 controls and m=1000/n=33 are inconclusive.
- The m=10/n=33 fallback goes from 159.1 to 159.8 ns, +0.44% (p<0.001).
  This small dispatch-boundary regression remains.
- The six admitted boundary cases at m=8/9 improve 43.82–72.39% (p<0.001).
- The m=7 control is inconclusive. The m=8/n=33 control improves 0.19%
  (p=0.001). This is too small to support an algorithmic claim.

## Public solve results

Ten interleaved 150 ms rounds, GOMAXPROCS=1, n=10 for each version. SolveTo
timings on existing factorizations. The four QR gains have p<0.001:

| QR size | RHS | Baseline us/op | Candidate us/op | Change |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 414.9 | 205.7 | -50.44% |
| 256 | 1 | 1631.3 | 824.3 | -49.47% |
| 128 | 16 | 757.7 | 548.1 | -27.66% |
| 256 | 16 | 2948 | 2133 | -27.65% |

All six LQ controls are inconclusive. Small significant increases in the
first control matrix are below. All other LU/Cholesky/SVD controls are
inconclusive.

| Control | Baseline us/op | Candidate us/op | Change | p |
| --- | ---: | ---: | ---: | ---: |
| QR 32, one RHS | 9.138 | 9.149 | +0.12% | 0.024 |
| QR 32, sixteen RHS | 33.53 | 33.59 | +0.21% | 0.010 |
| Cholesky 128, one RHS | 16.22 | 16.27 | +0.33% | 0.001 |
| SVD 128, one RHS | 203.1 | 203.8 | +0.34% | 0.029 |
| LU 256, one RHS | 76.33 | 76.42 | +0.12% | 0.011 |
| Cholesky 256, one RHS | 62.54 | 62.75 | +0.33% | 0.002 |

Allocations do not change in any public solve: LU/Cholesky zero, QR/LQ two,
SVD seven.

## Dlarft results

Reflector block width 32. Ten 150 ms rounds. Each improvement has p<0.001
and zero allocations:

| n | ldv | ldt | Baseline us/op | Candidate us/op | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 32 | 32 | 21.18 | 11.64 | -45.05% |
| 64 | 32 | 64 | 21.08 | 11.63 | -44.85% |
| 128 | 64 | 32 | 45.96 | 22.90 | -50.18% |
| 128 | 64 | 64 | 45.99 | 22.91 | -50.18% |
| 256 | 128 | 32 | 116.98 | 51.73 | -55.77% |
| 256 | 128 | 64 | 117.85 | 52.54 | -55.42% |
| 512 | 256 | 32 | 239.5 | 107.3 | -55.21% |
| 512 | 256 | 64 | 240.5 | 107.4 | -55.35% |

## Native DGEQRF results

Seven matrix shapes, ten 150 ms rounds:

- Gonum 256x256: 5.339 to 5.137 ms (-3.78%, p<0.001). Reference-LAPACK is
  6.891 ms in the candidate run.
- Gonum 128x128: 744.7 us versus reference 769.3 us. Its before/after change
  is inconclusive.
- Small cases trail the reference: 32x32 is 15.84 versus 10.32 us, 64x32 is
  36.34 versus 29.91 us.
- Gonum 128x256 first goes up 0.25% (1.723 to 1.727 ms, p=0.015).
- The untouched native 32x64 control also goes up 0.26% (23.09 to 23.15 us,
  p=0.009).
- Other native before/after cases are inconclusive. All native cases stay
  allocation-free.

## Factorization and runtime results

First public LU/Cholesky factorization controls at n=128/256 improve
1.22–3.39% (p<0.001 except Cholesky 256, p=0.002), with 72 B/op. These are
observed public-call effects. We did not change those factorization
algorithms.

GOMAXPROCS=4 (n=10, all p<0.001):

- QR one-RHS improves 51.75% at n=128 and 49.57% at n=256.
- QR sixteen-RHS improves 27.84% and 30.06%.
- LQ controls are inconclusive. Allocations do not change.
- The helper is serial. This checks the caller runtime setting only.

GODEBUG=simd=0 and GOMAXPROCS=1 (both p<0.001):

- QR n=256 one RHS improves 49.66% (1628.2 to 819.6 us).
- QR n=256 sixteen RHS improves 16.23% (4.973 to 4.166 ms).
- LQ controls are inconclusive. Allocations do not change.
- The sixteen-RHS gain is smaller: other work still uses the emulated
  portable backend.

## Rechecks and factorization controls

Separate ten-round 300 ms rechecks do not confirm significant slowdowns in
the first GEMV or public solve controls:

- GEMV m=10/n=33: 159.2 versus 159.9 ns (p=0.126). The small median
  difference remains.
- LU/Cholesky/SVD one-RHS cases at n=128/256 are inconclusive.
- QR n=32 one RHS is inconclusive. Sixteen RHS improves 0.61% (p=0.029).

Ten-round 200 ms native recheck:

- Gonum 128x256 is inconclusive (1.724 versus 1.727 ms, p=0.853).
- The unchanged Reference-LAPACK 32x64 control still goes up 0.27% (23.08 to
  23.14 us, p=0.007). We do not attribute it to a changed native algorithm.

Shape-qualified public factorization benchmarks, ten 150 ms rounds:

- QR square n=256 improves 3.87% (5.587 to 5.371 ms, p<0.001).
- QR tall n=256 improves 3.74% (13.39 to 12.89 ms, p=0.001).
- Both QR n=128 cases and LQ n=128 are inconclusive.
- Wide LQ n=256 goes up 0.36% (35.70 to 35.83 ms, p=0.029). A separate
  ten-round 300 ms recheck is inconclusive (35.77 to 35.72 ms, p=0.218).
- Thin SVD at n=128 is inconclusive for square and tall.
- Thin SVD at n=256: square improves 1.45% (57.16 to 56.33 ms), tall 2.51%
  (89.51 to 87.26 ms), both p=0.002, n=10.
- Allocation counts do not change in the factorization cohorts.

## Limits

- Ranges are per-case observations, not a workload-throughput aggregate.
- Many simultaneous per-case tests can give small false positives.
- We report rechecks separately. We do not pool them with the first samples.
- Native comparisons are case-specific. They make no claim about Accelerate
  or OpenBLAS.
- Small SVD factorization changes do not mean that SVD solves or all SVD
  modes improved.
- AMD64 assembly, float32 AXPY and default/safe/noasm implementations do not
  change. There is no native AMD64 performance claim and no cross-compilation.
- The new scalar helper stays behind the ARM64 experiment boundary. Other
  backends need native baseline comparisons before promotion.

## Remaining work

Separate three-second profiles ran after the main acceptance runs.

- QR one-RHS n=256: the new scalar GEMV helper has 68.88% flat CPU. Its
  bounds/address work and wider output tiles are possible future changes, not
  accepted changes. We stop before another ISA-specific implementation without
  comparison data.
- LQ one-RHS n=256: DotInc is 80.23% flat, below non-transposed gemvN, Dlarft
  and Dormlq. This is the next bottom-up target.
- In RowWise reflector products, A/X are contiguous and the output column is
  strided. Verify the exact strides before you make the next fixture.
- Profiles include setup (0.91% QR factorization, 2.62% LQ factorization).
  Steady-state benchmark timing does not include it.

## Binary identities

SHA-256 values recorded by the runner:

```text
f64 baseline     9346bb4c1909afbd9932ebf1437363010ba80e7020a39c324e9e4e20bfce46d9
f64 candidate    c274900c370f6e1ad7cce4fb8427dd9267f461ba967898f31260198e43d3a638
mat baseline     a5848112db5fd4dd796468fa31b3448e626885c82b9f4fc9061b8f9576ccb70f
mat candidate    965ce82b72b4655eba2c259ab3801c478b0c9eaf9cbdd9a8c81d578a35347570
lapack baseline  e98e8244ec331289d868a4cad1c0da3cb558bd92758aa0352ea09e6d54f30aaf
lapack candidate 7fab494d04308103e794026a04c726c4b5e702173f37cfca16d582f012ecb56b
native baseline  975f7211aef148857d6b675b1101ece7b97ebf1ef870536194cd0dbc779935e2
native candidate c40c18f0df274350b7fa45c5a5a87e80f92313e3aa3527e7f797ca01cdeeb132
```

## Reproduce

Build each package in the clean baseline and candidate worktrees. Use the
same installed toolchain and explicit PGO setting:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./internal/asm/f64 -o f64.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./mat -o mat.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./lapack/gonum -o lapack.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib -c ./lapack/gonum -o native.test
```

Native build:

- It needs the existing Darwin/cgo Homebrew Reference-LAPACK bridge.
- Dynamic links resolve to /opt/homebrew/opt/lapack/lib, not Accelerate or
  OpenBLAS.
- Native QR timing includes an equal matrix copy and factorization on the two
  sides. Layout conversion, workspace queries and allocation are outside the
  timer.
- Independent reconstruction/orthogonality tests check the two backends.
  They do not require identical Householder factors.

Run the scripts/compare_benchmarks.py file of the go-optimisation skill:

- Pass matching prebuilt --baseline/--candidate paths and a new --output
  directory.
- Pass --rounds 10 and --benchtime 150ms. Set GOMAXPROCS=1.
- Keep its raw output and metadata.
- Benchstat version: golang.org/x/perf v0.0.0-20260312031701-16a31bc5fbd0.

| Package | Exact --bench selector |
| --- | --- |
| f64 kernel/control | `^BenchmarkGemvT(Dlarft\|Arm64StridedBoundary\|Strided)$` |
| public solves | `^BenchmarkFactorizationSolve$/(QR\|LQ\|LU\|Cholesky\|SVD)$/n=(32\|128\|256)$/nrhs=(1\|16)$` |
| reflector construction | `^BenchmarkDlarft$` |
| native QR | `^BenchmarkDgeqrfNetlib$` |
| LU/Cholesky factors | `^BenchmarkFactorization$/(LU\|Cholesky)$/n=(128\|256)$` |
| QR/LQ factors | `^BenchmarkFactorization$/(QR\|LQ)$/shape=(square\|tall\|wide)$/n=(128\|256)$` |
| thin SVD factors | `^BenchmarkFactorization$/SVD$/kind=thin$/shape=(square\|tall)$/n=(128\|256)$` |

Runtime cohorts (ten 150 ms rounds each):

- GOMAXPROCS=4 solves:
  `^BenchmarkFactorizationSolve$/(QR|LQ)$/n=(128|256)$/nrhs=(1|16)$`.
- GODEBUG=simd=0, GOMAXPROCS=1:
  `^BenchmarkFactorizationSolve$/(QR|LQ)$/n=256$/nrhs=(1|16)$`.
  This selects portable emulation. It does not disable every
  architecture-specific native leaf. This scalar helper calls no emulated
  vector operations.

Focused rechecks use ten rounds with the same binaries and environment:

- 300 ms: `^BenchmarkGemvTStrided$/m=10$/n=33$`.
- 300 ms: `^BenchmarkFactorizationSolve$/(Cholesky|SVD|LU)$/n=(128|256)$/nrhs=1$`.
- 300 ms: `^BenchmarkFactorizationSolve$/QR$/n=32$/nrhs=(1|16)$`.
- 200 ms: `^BenchmarkDgeqrfNetlib$/m=(32|128)$/n=(64|256)$`.
- 300 ms: `^BenchmarkFactorization$/LQ$/shape=wide$/n=256$`.

For diagnostic profiles, run the candidate mat binary with GOMAXPROCS=1,
`-test.run='^$' -test.bench='^BenchmarkFactorizationSolve$/QR$/n=256$/nrhs=1$'`
and `-test.benchtime=3s -test.cpuprofile=qr.cpu`; replace QR with LQ for its
profile. Use Go 1.27.1 `go tool pprof -top` to read them. Do not overlap
profiles or validation builds with benchmark timing.
