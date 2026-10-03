# Large-row strided-output GEMV beneath SVD

## Summary

- Status: accepted after native performance and numerical validation.
- Implementation: `e1427ec0a2a1f5f9288ad2d4d1158c0c4d13581a`.
- Public GEMV for newly eligible shapes improves 50.27–73.06%.
- All public n=256 SVD workloads improve at P1 and P4.

## Setup

- Baseline: `7df53091aa34a94ed97a4210673b0f1c9965642d`. It was verified
  against the fetched `origin/codex/arm64-simd-blas` on 2026-09-06.
- The independent baseline worktree gets only the identical new public
  benchmark harness.

## Consumer

- The previous thin-wide n=256 SVD profile put 10.18% flat CPU in DotInc. All
  of it was sampled through gemvN.
- In the upper Dlabrd panel, the large call at `dlabrd.go:136` multiplies
  contiguous matrix rows by contiguous x.
- It writes an X-workspace column with stride 32. Its first panel shape is 255
  by 255.
- The existing four-row helper declined this call only because m exceeded 32.

## Change

- The ARM64 experimental dispatch and validator no longer have the upper row
  limit.
- The helper has the new name `gemvNStridedOutput`.
- The arithmetic body does not change. Four independent sequential row
  accumulators share each x load. Two-row and one-row tails follow.
- This is scalar load reuse, not a new vector reduction or SIMD API.
- The private validator now specializes for the caller-proven incX=1. This
  removes a redundant parameter/check. The general strided X-span calculation
  becomes n<=len(x).

The remaining gate requires:

- m>=4, n>=8, incX=1, positive incY>1 and valid lda;
- overflow-safe complete input/output spans;
- no Y overlap with bounded A/X spans.

Behavior that does not change:

- Conservative alias rejection keeps the original fallback, including
  short-slice panic/partial-write behavior.
- Read-only A/X sharing is permitted. Beta zero does not read old Y.
- DotInc itself, contiguous SIMD, AMD64 assembly, float32, default/safe/noasm
  and negative/zero stride behavior do not change.
- No public API or numerical tolerance changes.

## Method

- Host: Apple M1 Pro, darwin/arm64 v8.0, macOS 26.6.2 (25G83), AC power, 80%
  battery.
- No affinity or frequency pinning. Normal desktop background activity
  continues.
- Build: Go 1.27.1, GOEXPERIMENT=simd, empty GOFLAGS and -pgo=off.
- GOGC, GOMEMLIMIT and GODEBUG are unset during timing.
- Prebuilt binaries run baseline/candidate, then candidate/baseline, serially.
  No builds, tests or profiles run at the same time.
- Final cohorts: ten rounds, 100 ms per case, matching GOMAXPROCS and
  benchmark CPU settings.
- Benchstat is x/perf `v0.0.0-20260312031701-16a31bc5fbd0`.

- Percentages are time changes, not throughput aggregates.
- Many comparisons increase the false-positive risk.
- Inconclusive does not prove equivalence.

## Results

### Public GEMV, one worker

- All thirteen newly eligible shapes improve 50.27–73.06% (p<0.001).
- All twelve already-eligible/fallback controls are inconclusive.
- All 25 cases stay at zero B/op and allocs/op.

| m by n; incY | Baseline | Candidate | Time change |
| --- | ---: | ---: | ---: |
| 33 by 8; 2 | 168.60 ns | 83.84 ns | -50.27% |
| 33 by 32; 32 | 750.1 ns | 273.6 ns | -63.53% |
| 224 by 224; 32 | 55.17 us | 14.87 us | -73.04% |
| 255 by 255; 32 | 72.47 us | 20.01 us | -72.38% |
| 255 by 255; 32, padded | 72.52 us | 20.04 us | -72.37% |
| 512 by 255; 64 | 145.49 us | 39.45 us | -72.88% |

The harness includes:

- normal public validation/dispatch;
- finite exact-binary inputs and beta=0 stable overwrites;
- padding, m=31/32/33 boundaries and odd tails;
- output strides 2/32/64;
- strided-input and unit-stride controls.

### Public SVD, n=256

| Workload | P1 baseline → candidate | P1 change | P4 change |
| --- | ---: | ---: | ---: |
| Values, square | 15.76 → 11.83 ms | -24.94% | -25.23% |
| Thin vectors, square | 43.46 → 39.58 ms | -8.92% | -9.16% |
| Values, tall | 24.75 → 20.86 ms | -15.70% | -16.61% |
| Thin vectors, tall | 65.35 → 61.48 ms | -5.93% | -6.38% |
| Values, wide | 21.84 → 17.94 ms | -17.83% | -18.93% |
| Thin vectors, wide | 59.69 → 55.72 ms | -6.64% | -7.19% |

- All P1 differences have p<=0.001. All P4 differences have p<0.001.
- P1 covers 45 factorization cases. P4 covers 33 SVD/QR/LQ/LU/Cholesky cases.
- Smaller SVD cases are mostly inconclusive. At P4, all are inconclusive.
- P1 EigenSym at n=128 improves 3.50% without vectors and 1.23% with vectors.
  At n=256 the changes are -1.74%/-0.56%.
- Other controls are mostly inconclusive. Several favorable sub-percent changes
  are not attributed to this kernel.

### Adverse signals and rechecks

- P1 Eigen values-only n=32: +0.49%, p=0.035. Ten separate 300 ms rounds give
  130.2 → 130.7 us, inconclusive (p=0.075).
- Right/variable/backward Dlasr, compact 256 square: +0.20%, p=0.050. Ten
  separate 300 ms rounds give 43.44 → 43.58 us, inconclusive (p=0.165).
- P1 SVD values-only wide n=256: 5 → 6 allocs/op in the time-calibrated
  cohort. Ten rounds with exactly 20 iterations per binary give identical 5
  allocs/op and B/op, with time still -17.91%. This isolated calibrated-count
  shift shows no memory improvement or regression.

- No P4 timing or allocation-count regression is significant.
- Public B/op is effectively unchanged. A -0.05% QR n=128 byte change is not a
  memory claim.
- The 28-case rotation cohort removes the material regression of the first
  candidate. Its only adverse signal is the borderline Dlasr case above.
- The rechecks are not pooled with or substituted for the original samples.

### Native reference LAPACK/BLAS

- 256-square thin SVD, Gonum: 38.97 → 35.08 ms (-9.99%, p<0.001).
- Reference Netlib is unchanged at 89.41 → 89.42 ms.
- The four smaller shapes stay slower than reference Netlib. Their
  before/after differences are inconclusive.
- All Netlib before/after differences are inconclusive.

- This is Homebrew reference LAPACK/BLAS, not Accelerate or OpenBLAS.
- Timing includes input restoration and cgo calls.
- Each backend uses its native layout. Conversion/workspace setup is excluded.
- Source review is pinned to Reference-LAPACK v3.12.1. The linked runtime
  reports ILAVER=3.12.0.

## Rejected first candidate

The first candidate removed only the row cap:

- GEMV improved 50–73% and n=256 SVD improved 6–25%.
- Right/variable/forward Dlasr at m=64, n=32/33 became 8.35–9.38% slower.
- A separate ten-round 300 ms recheck confirmed 8.44–9.18%.
- Small Cholesky n=32 controls also kept increases of 0.34–0.45%.

- The shorter dispatch moved unchanged later machine code by -16 bytes.
- The sequential rotation entry moved from cache-line offset 0x20 to 0x10.
- This is an alignment correlation, not proof of causality.

The consumer/control gates of the Go optimization skill stopped the acceptance
of the attractive SVD-only win.

## Accepted follow-up

- It removes eight instructions/32 bytes of genuine validator work.
- Public call setup also changes. Later LAPACK symbols end up -32 bytes from
  baseline.
- The compute helper stays instruction-identical. It has four register-resident
  scalar FMA accumulators and no hot-loop calls or accumulator spills. Per-row
  bounds branches stay.
- DotUnitary, Ger and scalar GemvN keep their original placement.
- It uses no padding, alignment directive, artificial cap or LAPACK arithmetic
  change.

Placement changes are not a promised repair on future toolchains. Run the
native controls again after toolchain changes.

## Validation

Passed:

- full Go 1.27.1 SIMD and Go 1.26.4 default suites;
- native Go 1.24.0 affected packages;
- safe/noasm affected packages;
- portable-emulation GODEBUG=simd=0 affected packages;
- native Netlib Dgesvd/Dbdsqr differential tests, three times.

GODEBUG=simd=0 does not disable this scalar helper or all other archsimd
leaves.

Race tests:

- Full f64/BLAS race suites pass.
- Focused LAPACK/mat race tests cover Dgesvd, Dgebrd, Dlabrd, SVD and SVD
  solves.
- These use GOMAXPROCS=4 and `-p=2`. They are not a full LAPACK/mat race run.

Persistent tests prove large-row eligibility through 257 by 255. They compare
exact existing scalar arithmetic (NaNs by classification) and cover:

- tails, offsets, padding and guards;
- beta-zero NaN destinations;
- overflow/cancellation, subnormals, signed zero and Inf/NaN;
- read-only inputs;
- active X/later-A-row alias fallback;
- zero/negative/nonunit increments;
- short slices with spare capacity, including panic/partial-write parity.

- Independent review confirms that span arithmetic cannot overflow for
  admitted actual backing slices.
- Formatting, import policy, copyright and diff checks pass.
- A/X/Y do not escape.

## Post-change profile

A separate post-timing thin-wide n=256/P1 profile has 3.28 seconds of CPU
samples.

- DotInc falls from 10.18% flat in the prior profile to 3.35%.
- The expanded helper is 3.66% flat/3.96% cumulative.
- Dlasr stays at 34.15% cumulative, inside the 41.46% of Dbdsqr. Do not add
  these nested percentages.

This shows the remaining work. It does not show additional gains from this
change.

## Reproduce

- Raw evidence is local temporary data in `/tmp/gonum-gemvn-followup.GS0YGD`.
- The persistent harnesses permit new runs after it expires.
- The six-round 200 ms `svd-screen` and non-v2 cohorts are superseded, not
  acceptance data.

Build separate binaries from baseline and candidate, with the same harness:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./blas/gonum -o /tmp/blas.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./mat -o /tmp/mat.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib -c ./lapack/gonum -o /tmp/lapack.test
```

Run the go-optimisation skill's `compare_benchmarks.py`:

- Set separate `--baseline`/`--candidate` and a fresh `--output`.
- Set `--rounds 10` and `--benchtime 100ms`.
- Match GOMAXPROCS/`--cpu`.

```text
v2-gemv P1:              ^BenchmarkDgemvStridedOutput$
v2-mat-p1 P1:            ^BenchmarkFactorization$
v2-mat-p4 P4:            ^BenchmarkFactorization$/(SVD|QR|LQ|LU|Cholesky)$/.*
v2-netlib P1:            ^BenchmarkDgesvdNetlibKernels$
v2-rotation-controls P1: ^BenchmarkDlasr$/side=R/pivot=V/direct=(F|B)/pattern=dense/m=(32|33|64|256)$/n=(32|33|64|256)$
```

- The three rechecks use `v2-eigen-confirm`, `v2-allocation-confirm` and
  `v2-rotation-confirm`.
- Their metadata.json files record the exact selectors/iteration settings. The
  Adverse signals section describes them.
- The separate `v2-svd.cpu` profile uses
  `^BenchmarkFactorization$/SVD/kind=thin/shape=wide/n=256$`, P1, 3 s.

SHA256 of baseline/candidate binaries:

```text
BLAS base b792b4b37b7e7ba604803f714236376946d3c4723ce73304db9ddd19ac8bb52e
BLAS v2   003d2db790f82833b873ec9f656c01d2a7b9d9c3afa49627aa54680f7486df83
mat base  472376e8130061fe80bf732b9af988290e92d656ca69f77613e5245a76abc97b
mat v2    7ea79bcd4e7bf1c02d442c34e77f567cb81d9901d1c418b862119753f7b040a5
LAPACK base d75e67654b18ea158910a571f2af545f4d0dad01c90356b238358d6c114d2448
LAPACK v2   bffe57e48c85b44cd2bba033e4edd54a799bc8373038a318786e02bb396c4689
```

Identical new benchmark source SHA256:
`e07a6db139dba8be17ce3297869b50fa9ec0ba3ad27cf1ee5036a0086f7d21c9`.

## Limits

- No cross-compilation or native AMD64 timing was done.
- Go SIMD is still experimental.
- Before the edits, we refreshed the
  [Go 1.27 notes](https://go.dev/doc/go1.27#simd), the
  [portable SIMD proposal](https://github.com/golang/go/issues/78902) and the
  selected toolchain source.
- This patch adds no intrinsic.
