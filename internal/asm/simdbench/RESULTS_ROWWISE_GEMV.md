# Bottom-up RowWise GEMV and LQ

## Summary

A new ARM64-experimental GEMV helper speeds up strided-output RowWise
products, LQ solves and RowWise Dlarft (p<0.001). Allocations do not change.
Two small fallback regressions remain.

## Scope

Start point: `bc9a80b7e5d9133d2be149e8e38c0a72c05791ad`. The work follows
`go-optimisation` and `gonum-simd`. Dedicated Sol reviews covered the
implementation, the code generation and the numerics.

A new three-second LQ SolveTo profile at n=256/one RHS puts 80.63% flat CPU
in DotInc and 83.19% cumulative in gemvN. Profiles are separate from
acceptance timing.

Call chain: LQ.SolveTo -> Dormlq -> Dlarft(Forward, RowWise)
-> Dgemv(NoTrans) -> GemvN -> gemvN -> DotInc.

- Dlarft forms one column of T from m=i preceding reflectors (normally 1..31
  for block width 32).
- Each product has a long contiguous matrix-row tail and contiguous x, but
  strided y. The T workspace of Dormlq has ldt=64.
- The old contiguous SIMD gate requires incY=1. Thus these products used one
  sequential DotInc for each output row.

## Gate

A new ARM64-experimental gate selects calls with:

- 4<=m<=32, n>=8, incX=1 and positive incY>1.
- Valid lda and complete slice spans with no overflow.
- An output bounding span disjoint from the active A and x spans.

The gate is conservative: overlap only in output stride gaps still goes to
the fallback. A and x can share read-only storage. Invalid slices,
unsupported strides and overlapping outputs use the old fallback, with its
partial-write panic behavior.

## Helper design

- Processes four rows together, with two-row and one-row tails.
- Shares each x load. Each output keeps a separate sum in sequential order.
- Uses scalar arithmetic, not a horizontal SIMD reduction. The portable
  GemvNSIMD reduction reassociates terms, so it has a different numerical
  contract.
- Beta zero does not read old y. Nonzero beta keeps y*beta + alpha*sum.
- DotInc, AMD64, float32 and default/safe/noasm implementations do not change.

Generated code:

- Four scalar FMA accumulators stay in registers.
- The hot loop has no helper calls, divisions, allocations or FP stack spills.
- Bounds branches remain for the four matrix loads in each column.
- The non-inlined validator does two overflow-check divisions once for each
  eligible call. Cheap shape/stride tests skip it for other shapes.
- Not all bounds checks or dispatch overhead are removed.

## Setup

- Host: Apple M1 Pro, darwin/arm64, macOS 26.6.2 (25G83), AC power. No CPU
  affinity or frequency pinning.
- Native builds: Go 1.27.1, GOEXPERIMENT=simd, explicit -pgo=off. GOFLAGS is
  empty. The repository has no PGO profile.
- Default compatibility: Go 1.26.4.
- On 2026-09-06 we read again the
  [Go 1.27 release notes](https://go.dev/doc/go1.27#simd), the installed
  simd.Emulated documentation and the
  [portable SIMD proposal](https://github.com/golang/go/issues/78902).
  This work adds no new experimental API.

Checkpoints:

- Clean baseline: `684ac2fa7f76389fafc682e5da1c66c14b931b3a`. It adds only
  the common test/benchmark harness to the start revision.
- Candidate: `f8d215fb0f8a369373e114b64e9cd35cec9d339a`. Benchmark source is
  identical. The candidate adds one validator-only test.

Fixtures:

- Direct GEMV: finite exact-binary inputs, beta=0, the same bounded y each
  iteration. Controls: boundary, long-tail, contiguous and
  unsupported-shape/stride.
- RowWise Dlarft: Dgelq2 makes valid LQ reflectors before timing. Each
  iteration overwrites T. ColumnWise benchmark names do not change.
- Public solves reuse factorizations and include normal SolveTo costs only.
  Separate benchmarks measure factorization.

Method:

- The compare_benchmarks.py runner of the skill alternates baseline and
  candidate order with prebuilt binaries. It rejects failed or mismatched runs
  and records hashes and runtime settings.
- Acceptance: ten rounds on a quiet host, GOMAXPROCS=1 unless specified,
  GOGC/GOMEMLIMIT unset. No builds, tests or profiles overlap timed cohorts.
- Three-round 100 ms screens are diagnostic only.
- benchstat is golang.org/x/perf v0.0.0-20260312031701-16a31bc5fbd0.

Raw logs, profiles, disassembly and runner metadata are in the temporary
directory `/tmp/gonum-rowwise-gemv.JW8uih`. The persistent benchmarks and
source checkpoints permit reruns.

## Kernel results

Ten interleaved 150 ms rounds, GOMAXPROCS=1, n=10 for each version. All ten
eligible cases improve 30.93–73.68% with p<0.001. All cases stay at zero
B/op and allocs/op. Representative medians:

| m x n | incY | Baseline ns/op | Candidate ns/op | Change |
| --- | ---: | ---: | ---: | ---: |
| 4 x 8 | 32 | 23.01 | 15.53 | -32.49% |
| 5 x 9 | 64 | 29.25 | 20.20 | -30.93% |
| 31 x 16 | 64 | 297.5 | 133.2 | -55.23% |
| 4 x 128 | 64 | 509.9 | 149.0 | -70.79% |
| 31 x 128 | 32 | 3895 | 1245 | -68.04% |
| 31 x 256 | 64 | 8858 | 2678 | -69.77% |
| 32 x 256 | 64 | 9151 | 2408 | -73.68% |
| 31 x 512 | 64 | 18736 | 5565 | -70.30% |

Small fallback regressions, both p<0.001, reported separately from the gains
(extra 0.28–0.30 ns):

- m=3/n=8: 18.48 to 18.78 ns (+1.62%).
- m=4/n=7: 21.79 to 22.07 ns (+1.31%).

The other five contiguous or unsupported-shape/stride controls are
inconclusive.

## Public solve results

Ten interleaved 150 ms rounds, GOMAXPROCS=1, n=10 for each version. All four
useful-size LQ improvements have p<0.001:

| LQ size | RHS | Baseline us/op | Candidate us/op | Change |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 478.5 | 207.3 | -56.67% |
| 256 | 1 | 2036.8 | 825.8 | -59.45% |
| 128 | 16 | 893.3 | 622.1 | -30.35% |
| 256 | 16 | 3784 | 2576 | -31.93% |

- Both LQ n=32 cases and all QR/LU/SVD solve controls are inconclusive.
- Cholesky n=128/one RHS changes -0.18% (16.27 to 16.24 us, p=0.022), not
  attributed to the new helper. Other Cholesky controls are inconclusive.
- No solve control has a statistically significant slowdown.
- Allocations do not change: QR/LQ two, LU/Cholesky zero, SVD seven for each
  solve. These are steady-state SolveTo results, not factorization timings.

## Reflector construction results

Ten interleaved 150 ms rounds, GOMAXPROCS=1, k=32, n=10. All eight RowWise
Dlarft cases improve 50.14–68.69%, p<0.001, with zero allocations:

| n / ldv | ldt | Baseline us/op | Candidate us/op | Change |
| --- | ---: | ---: | ---: | ---: |
| 64 / 64 | 32 | 21.26 | 10.60 | -50.14% |
| 64 / 64 | 64 | 21.27 | 10.59 | -50.22% |
| 128 / 128 | 32 | 54.00 | 21.54 | -60.11% |
| 128 / 128 | 64 | 53.99 | 21.57 | -60.06% |
| 256 / 256 | 32 | 133.46 | 44.80 | -66.43% |
| 256 / 256 | 64 | 133.54 | 44.78 | -66.47% |
| 512 / 512 | 32 | 291.50 | 91.35 | -68.66% |
| 512 / 512 | 64 | 291.50 | 91.27 | -68.69% |

Unchanged ColumnWise n=64 controls show small increases, not attributed to
the new RowWise algorithm:

- ldt=32: 11.63 to 11.64 us (+0.07%, p=0.004).
- ldt=64: 11.61 to 11.64 us (+0.21%, p=0.001).

The other six ColumnWise controls are inconclusive. All ColumnWise cases stay
allocation-free. Separate ten-round 300 ms rechecks do not confirm the two
slowdowns: ldt=32 is 11.62 to 11.63 us (p=0.839), ldt=64 is 11.63 to 11.63 us
(p=0.616).

## Factorization results

Ten 150 ms rounds of separate factorization measurements:

- LQ wide n=256 improves 3.82% (35.71 to 34.35 ms, p=0.019), with a wider 6%
  candidate interval. A separate ten-round 300 ms recheck confirms it: 35.72
  to 34.35 ms (-3.84%, p<0.001), allocations unchanged.
- LQ n=128 and all four QR square/tall n=128/256 cases are inconclusive.
- Thin SVD square n=256 improves 1.02% (56.18 to 55.61 ms, p=0.011).
- Thin SVD wide n=256 improves 2.39% (82.29 to 80.32 ms, p=0.001).
- Thin SVD tall n=256 and all three n=128 shapes are inconclusive.
- Allocation counts and byte medians do not change in the two cohorts.

## Runtime controls

Ten 150 ms rounds for each setting. All gains have p<0.001. All QR controls
are inconclusive. Allocations do not change.

- GOMAXPROCS=4: LQ one RHS n=128/256 improves 56.69%/59.62%. Sixteen RHS
  improves 30.42%/33.84%.
- GOMAXPROCS=1 and GODEBUG=simd=0: LQ n=256 one RHS goes from 2039.3 to
  825.1 us (-59.54%). Sixteen RHS goes from 3.893 to 2.680 ms (-31.16%).

The new helper is scalar. Portable emulation does not disable all
architecture-specific leaves in the rest of Gonum.

## Numerical validation

Persistent tests cover:

- The m=3/4/5 and 31/32/33 boundaries, n=7/8/9 and longer tails.
- Offsets, ldt=32/64, beta-zero NaN outputs and exact signed zero.
- Ordered overflow/cancellation and subnormals.
- Bitwise comparisons with the old scalar path. NaNs are compared by class,
  not payload.
- Output gaps and guards, read-only A/x storage, true active A/x overlap and
  output overlap.
- Negative, zero and nonunit increments, short slices with capped capacities,
  and rejection of overflowing spans.

Existing consumer checks: independent Dlarft reflector reconstruction,
Dgelqf versus unblocked Dgelq2, Dormlq versus Dorml2, and public LQ
reconstruction/solve. No numerical tolerance is relaxed.

These checks pass:

- Full `go test -pgo=off ./...` on default Go 1.26.4 and experimental
  Go 1.27.1.
- Affected f64/BLAS/LAPACK/mat suites with the safe, noasm and bounds tags.
- Focused f64 race tests, and GODEBUG=simd=0 GEMV/LQ/QR/SVD and reflector
  tests.
- Common RowWise tests on the clean baseline.
- Formatting, import-policy, copyright and diff checks.

## Limits

- Inconclusive results do not prove equivalence.
- Many per-case tests increase the risk of false positives. Per-case changes
  are not an application throughput aggregate.
- Rechecks are separate measurements. We do not pool them with the first
  cohorts or use them in their place.
- Small SVD factorization effects do not show gains for all SVD modes or
  sizes.
- This work adds no Netlib LQ bridge. It makes no claim against optimized
  vendor BLAS.

## Remaining work

Three-second profiles ran only after timing finished.

LQ n=256/one RHS:

- DotInc is now 6.10% flat.
- The new GemvN helper is 59.59% flat / 61.05% cumulative. Its remaining
  bounds/address work is a possible future target.
- Transposed/transposed DGEMM is 22.97% cumulative, with strided AXPY work.
- LQ setup (Factorize 2.03% cumulative) is in the profile, not in SolveTo
  timing.

Thin wide SVD n=256:

- Dbdsqr is 38.05% cumulative and Dlasr 34.22% cumulative. Do not add these
  nested costs.
- Dlasr itself is 19.47% flat. Its blocked right-variable helper is 13.86%
  flat.
- DotUnitary is 21.83% cumulative, and SIMD DGEMM 12.39%.

Next measured LAPACK target: rotation application below bidiagonal SVD. Keep
GEMV/dot consumers as controls. These profiles give hypotheses only, not
accepted optimizations.

## Reproduce

Build the same packages at each checkpoint:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./internal/asm/f64 -o /tmp/f64.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./lapack/gonum -o /tmp/lapack.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./mat -o /tmp/mat.test
```

Use separate output paths for the two builds. There is no cross-compilation
and no native AMD64 timing. Other backends need native data before promotion.

Acceptance selectors (ten 150 ms rounds, default GOMAXPROCS=1):

```text
f64:    ^BenchmarkGemvNRowWiseDlarft$
lapack: ^BenchmarkDlarft$
mat:    ^BenchmarkFactorizationSolve$/(LQ|QR|LU|Cholesky|SVD)$/n=(32|128|256)$/nrhs=(1|16)$
mat:    ^BenchmarkFactorization$/(QR|LQ)$/shape=(square|tall|wide)$/n=(128|256)$
mat:    ^BenchmarkFactorization$/SVD$/kind=thin$/shape=(square|tall|wide)$/n=(128|256)$
```

Other cohorts:

- P4: `^BenchmarkFactorizationSolve$/(QR|LQ)$/n=(128|256)$/nrhs=(1|16)$`.
- Emulation: `^BenchmarkFactorizationSolve$/(QR|LQ)$/n=256$/nrhs=(1|16)$`.
- Focused rechecks (ten 300 ms rounds, same binaries):
  `^BenchmarkDlarft$/n=64$/ldv=32$/ldt=(32|64)$` and
  `^BenchmarkFactorization$/LQ$/shape=wide$/n=256$`.

Skill runner:

- Pass `--baseline`, `--candidate`, `--bench`, `--rounds 10`,
  `--benchtime 150ms` and a new `--output` directory.
- Run benchstat on its baseline.txt and candidate.txt.
- Record the same runtime environment on the two sides.
- Profile separately with the selected benchmark and
  `-test.run '^$' -test.benchtime=3s -test.cpuprofile=/tmp/profile.cpu`.

Final binary SHA-256 values:

```text
f64 baseline     32eea6cfd4c64a05f1a63d58d96d9cb7eb457aac9580db676640769fb2a3a58e
f64 candidate    94191a6eab57c1280e3632ccc18301f8de6de7cacb290c9fceb7bb768adf2fd3
lapack baseline  41892c8a6df42a69a13cdbaa4b1c5d61b0595461ac33d6ca9ddd5982c32d9d46
lapack candidate 3f9360653a429ccc6ebfedf0ee45a43292d82045ddfcf506bb85f195438373ad
mat baseline     8852efa8a6b8c0ba8ee7621d1db38bbb135d1c835f86dd2cd01649140acba319
mat candidate    d9b0b89e01fa6f859f523ac3def7c5f54508c50933a9e06b32958c0e0fd06374
```
