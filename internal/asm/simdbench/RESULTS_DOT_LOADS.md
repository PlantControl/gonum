# ARM64 transposed-B DGEMM and SVD consumers

## Summary

- Accepted: a paired-output ARM64 DGEMM SIMD kernel, committed as
  `937183b7fe1b8db17c0cb0458f9483530dbc1d56`.
- Rejected and removed: the broad DotUnitary replacement. It caused repeatable
  consumer regressions.
- The shared f64 production code does not change.

## Setup

- Date: 2026-09-06.
- Base commit: `8b8e472d4399bc7ec94526c4e071dfacdfbadf65`. This was also the
  fetched `origin/codex/arm64-simd-blas` tip before this work.
- The baseline implementation stays unchanged in an independent worktree.
- Both builds get the same persistent benchmark and dot regression harness,
  published in `486b6a88`. The implementation baseline is older than that
  harness-only commit.
- Host: Apple M1 Pro, darwin/arm64, macOS 26.6.2 (25G83), AC power.
- Build: Go 1.27.1, `GOEXPERIMENT=simd`, `-pgo=off`, empty GOFLAGS. GOGC,
  GOMEMLIMIT and GODEBUG are unset.
- Compatibility build: default Go 1.26.4.

Timing rules:

- Timing uses prebuilt binaries.
- The order alternates: baseline/candidate, then candidate/baseline.
- No compilation, tests or profiles run at the same time.
- No CPU affinity or frequency pinning.
- Normal desktop background activity continues. Timing excludes the indexing
  bursts that occurred during preparation.

Scope:

- The original target was `internal/asm/f64/dot_simd_arm64.go`.
- The accepted change reuses its arithmetic in an isolated transposed-B DGEMM
  SIMD kernel. The kernel uses ARM64 NEON `simd/archsimd`.
- Default, AMD64 assembly, safe, noasm and GCCGo selection do not change.
- There are no cross-compiled or native AMD64 measurements.
- Go SIMD is still experimental. Before the edits, we checked the
  [Go 1.27 notes](https://go.dev/doc/go1.27#simd) and the installed Go 1.27.1
  `src/simd/archsimd/slice_gen_arm64.go`.

## Why this leaf

The previous thin-wide n=256 public SVD profile had 3.30 seconds of CPU
samples. DotUnitary had 21.21% flat and 26.67% cumulative. Of its 0.88
cumulative seconds, 0.79 came through `dgemmSerialNotTrans`.

`Dlarfb` is the main DGEMM source (0.99 of 1.32 sampled DGEMM seconds):

- `Dorglq` and `Dgelqf` call it most.
- Wide SVD first does an LQ reduction of a 256-by-512 matrix.
- A right-side rowwise Householder block calls `Dgemm(NoTrans, Trans, ...)`.
- The first outputs are 224-by-32, with dot reduction length 480. Later lengths
  are 448 and below.
- The portable DGEMM tile does not accept transposed B, so this path calls the
  unitary dot leaf.

`Dgebrd` gives a smaller DGEMM part (0.17 sampled seconds). Here it operates on
the reduced 256-by-256 matrix. Its first trailing update has 224-by-224 outputs
and reduction length 32. An initial interpretation said length-32 dots
dominated this workload. We corrected this before timing.

## Rejected leaf prototype

### Change and contract

Only the load windows change:

- Each eight-product iteration makes fixed, capacity-bounded eight-element
  operand slices. Then it loads four fixed pairs.
- The second operand is first capped at its logical length, so a short slice
  cannot silently read through spare capacity.
- The four accumulators, fused multiply-add operations, reduction tree, paired
  remainder and scalar tail keep their order.
- There are no unsafe pointers, new dispatch or tuning threshold.

Go 1.27.1 disassembly of the main eight-product loop:

- The instruction count falls from roughly 90 to roughly 24.
- The 24 are eight vector loads, four vector FMAs and the block
  bounds/address/loop bookkeeping.
- Per-vector bounds branches and capacity masking go away.
- Block bounds checks stay. The loop is not bounds-check-free.
- There are no hot-loop calls or spills. Escape diagnostics show that both
  operands do not escape.

Persistent tests cover:

- all lengths 0 through 80, and larger tails through 1025;
- offsets and longer second operands;
- overlapping read-only slices and unchanged input storage;
- short operands with spare capacity;
- cancellation, overflow, subnormals, signed zero, infinity and NaN.

The reference models the accumulation order of this ARM64 kernel. It does not
claim bitwise agreement between architectures. These tests pass on the
unchanged baseline.

### Measurements

- All results are native interleaved comparisons, ten samples per binary per
  case.
- Percentages are time reductions, not application throughput.
- benchstat gives per-case significance. Many comparisons increase the chance
  of small false positives.
- All DGEMM cases allocate 0 B/op and 0 allocs/op.

#### Public transposed-B DGEMM, P1

Each benchmark runs 100 ms per round. This includes public validation, dispatch
and beta-zero output handling. Inputs stay finite and unchanged between calls.

| Output m by n; reduction k | Baseline | Candidate | Time change |
| --- | ---: | ---: | ---: |
| 96 by 32; 224 | 378.08 us | 95.94 us | -74.62% |
| 224 by 32; 480 | 1.8698 ms | 491.8 us | -73.70% |
| 224 by 32; 480, padded | 1.8731 ms | 504.8 us | -73.05% |
| 96 by 224; 32 | 411.1 us | 163.4 us | -60.24% |
| 224 by 224; 32 | 958.0 us | 381.4 us | -60.18% |
| 224 by 480; 32 | 2.0484 ms | 822.5 us | -59.85% |
| 224 by 480; 32, padded | 2.0593 ms | 893.8 us | -56.60% |

- All these differences have p<0.001.
- The traced consumer shapes are the first two Householder-block shapes and the
  square bidiagonal update. The other shapes and padding are coverage only. They
  make no claim about sampled call frequencies.
- Larger square/padded controls improve 69.66-71.53%.
- k=16 through 33 boundary cases improve 40.78-57.42%.
- No SIMD crossover changes.

Tiny unchanged-scalar cases show two significant regressions:

- 1-by-1/k=7: 24.63 to 24.92 ns (+0.29 ns, +1.18%, p=0.001).
- 4-by-4/k=7: 128.0 to 128.2 ns (+0.2 ns, +0.20%, p=0.029).
- Padded k=8 is inconclusive. k=15 is -0.52%.

We report these subnanosecond changes; no aggregate hides them. They do not
justify a new short-vector dispatch or arithmetic change.

#### Dot and public factorization

Clean `dot-accept` timings (100 ms):

- length 16: 9.705 to 4.973 ns (-48.76%);
- length 32: 18.290 to 6.213 ns (-66.03%);
- length 480: 259.80 to 63.98 ns (-75.37%);
- length 4096: 2219.5 to 621.1 ns (-72.02%).

All have p<0.001 and zero allocations. Offset-one results agree. All lengths
below 16 are inconclusive. These leaf improvements did not pass the consumer
gate.

Public n=256 SVD improves 8.37-19.80%, but smaller cases regress. A separate
ten-round, 300 ms confirmation shows the square/tall n=32/128 regressions again:

- singular values only: 2.67-9.25%;
- thin vectors: 2.81-4.72%;
- QR controls at n=32/128: 7.50-13.04%;
- LU controls at every measured size: 11.49-16.89%.

We do not use a geometric mean to accept this change.

Independent disassembly of Ger, Axpy, GEMV, rotations and inspected LAPACK
consumers shows:

- The arithmetic instructions do not change.
- The code placement changes after the smaller DotUnitary symbol.
- The Ger cache-line entry offset changes from zero to 32.

This is strong evidence of a text-layout effect, but not proof that entry
alignment explains every cycle of regression. We removed the replacement instead
of adding padding or alignment hacks.

## Accepted isolated DGEMM SIMD kernel

### Design

- The original shared f64 leaf does not change.
- The kernel pairs two output columns in
  `blas/gonum/dgemm_trans_simd_arm64.go`.
- Each eight-element step reuses four A vectors across two B rows. This gives
  twelve loads instead of sixteen.
- Eight vector accumulators keep the original dot order of each output.
- There are no accumulator spills, hot-loop calls or heap allocations.
- Three fixed-window bounds groups stay.
- The active-row disjointness check allows safe shared LAPACK panels. Before any
  output write, it rejects actual C/A or C/B overlap. This includes overlap only
  with the odd final B row.

### Dispatch

- Only NoTrans/Trans or NoTrans/ConjTrans enters the tile.
- The guard checks n>=2 and k>=16 before the call.
- At k=16, the original dot change to four-accumulator arithmetic stays.
- Odd output columns use the original DotUnitary.
- Alpha-zero behavior does not change.
- Non-ARM64 and float32 helpers return false. The single-precision generator
  explicitly maps the new helper name.
- No experimental types escape the implementation package or public API.

### Code layout

- The first paired-output prototype still called the helper on ineligible tiny
  matrices. This cost about 4-7 ns.
- We moved the cheap eligibility checks before the call. This removed most of
  that cost.
- The final code keeps the original f64 Dot/Ger/GEMV addresses.
- Inspected LAPACK rotation bodies keep their original instruction sequences
  and cache-line alignment, with a uniform +0x6c0 shift.
- We use no padding, alignment directives or architecture-independent leaf
  replacement to force those measurements.

### Final public DGEMM, P1

Ten rounds, 100 ms per case, public-call timing. All cases have 0 B/op and 0
allocs/op.

| Output m by n; reduction k | Baseline | Candidate | Time change |
| --- | ---: | ---: | ---: |
| 96 by 32; 224 | 377.33 us | 72.06 us | -80.90% |
| 224 by 32; 480 | 1.8703 ms | 362.6 us | -80.61% |
| 224 by 32; 480, padded | 1.8703 ms | 371.6 us | -80.13% |
| 224 by 224; 32 | 957.6 us | 224.5 us | -76.56% |
| 1 by 2; 16 | 41.44 ns | 30.12 ns | -27.33% |
| 1 by 3; 480 | 809.6 ns | 377.8 ns | -53.33% |

- All tabulated differences have p<0.001.
- Other length-32 panel shapes improve 74.99-76.63%.
- Long-reduction n=31/32/33 controls improve 77.55-80.67%, including odd output
  tails.
- Larger 128/129/256 reduction controls improve 76.56-79.80%.

Five ineligible tiny cases show significant overhead:

- 1-by-1/k=7: +0.28 ns;
- 1-by-1/k=16: +0.62 ns;
- 1-by-1/k=17: +0.57 ns;
- 4-by-4/k=7: about +0.3 ns;
- padded 4-by-4/k=8: about +0.3 ns.

The largest is +2.05%. The k=15 cases improve slightly. This is a bounded
subnanosecond dispatch tradeoff, not a universal improvement.

### Final public factorizations and Netlib

- All final cohorts use ten interleaved rounds at 100 ms per case.
- There are no significant timing regressions in the 45 P1 public
  factorization cases or the 33 P4 SVD/QR/LQ/LU/Cholesky cases.
- Smaller n=32/128 SVD/QR/LQ results are inconclusive. They do not prove exact
  equivalence.
- The table shows time reductions at n=256. All listed improvements have
  p<0.001.

| Public workload | P1 baseline → candidate | P1 change | P4 change |
| --- | ---: | ---: | ---: |
| SVD values, square | 17.61 → 15.75 ms | -10.56% | -3.33% |
| SVD thin, square | 49.06 → 43.47 ms | -11.40% | -5.33% |
| SVD values, tall | 31.09 → 24.72 ms | -20.49% | -7.16% |
| SVD thin, tall | 80.08 → 65.37 ms | -18.37% | -7.73% |
| SVD values, wide | 28.03 → 21.83 ms | -22.12% | -15.38% |
| SVD thin, wide | 73.66 → 59.58 ms | -19.12% | -14.16% |
| QR, square | 5.366 → 3.492 ms | -34.92% | -15.13% |
| QR, tall | 12.904 → 8.372 ms | -35.12% | -14.03% |
| LQ, wide | 34.29 → 19.67 ms | -42.63% | -28.08% |

Other P1 n=256 results:

- EigenSym with vectors improves 6.26% (30.11 → 28.22 ms).
- Eigen with right vectors improves 10.00% (62.56 → 56.31 ms).
- LU, Cholesky and eigenvalues-only controls are inconclusive at this size.
- We do not attribute sub-percent favorable changes in small P4 controls to
  this kernel.

- Public B/op is effectively unchanged.
- Isolated allocation-count shifts (including P1 LQ 9 → 10 and P4 LQ
  1218 → 1215) do not show a memory improvement or regression.
- The kernel itself allocates zero.

Native reference-BLAS/LAPACK comparison:

- n=256 square thin SVD: Gonum 44.46 → 38.89 ms (-12.53%, p<0.001).
- Netlib is unchanged at 89.61 → 89.52 ms.
- The other four smaller Gonum cases are inconclusive before/after. They stay
  about 1–41% slower than reference Netlib.
- No Netlib before/after case changes significantly.
- These benchmarks include timed input restoration and cgo calls. Each backend
  uses its native layout. Conversion and workspace setup are excluded.

### Post-change profile

A separate P1 thin-wide n=256 SVD profile ran after the timing. It has 3.34
seconds of CPU samples.

| Function | Flat | Cumulative |
| --- | ---: | ---: |
| Original DotUnitary, prior profile | 21.21% | 26.67% |
| Original DotUnitary, this profile | 2.10% | 2.69% |
| New paired kernel | 3.29% | 4.19% |
| Enclosing SIMD DGEMM dispatch | 12.28% | 21.26% |

Cumulative percentages overlap. Do not add them.

Remaining costs:

- Dlasr: 36.53% cumulative.
- DotInc: 10.18% flat. All sampled DotInc calls come through gemvN.
- Strided dot under non-transposed GEMV is a measured follow-up. This patch
  does not improve it.
- Ergo A7PB45 tracks a separate alpha-zero GEMM semantics concern. This patch
  does not change that behavior.

## Validation

- Full suites pass with Go 1.27.1 SIMD enabled and with default Go 1.26.4.
- Native minimum Go 1.24.0 tests pass for the affected f64, BLAS, LAPACK and
  mat packages.
- Affected BLAS/LAPACK/mat tests also pass under safe, noasm, race
  instrumentation and `GODEBUG=simd=0`. The last one changes portable SIMD
  emulation, not archsimd.
- The final affected race run uses GOMAXPROCS=4 and `-p=2`. It completed with
  no failures.
- Native Netlib Dgesvd/Dbdsqr differential tests pass three consecutive runs.
- New dot tests also pass against the unchanged baseline.

Persistent DGEMM regression tests cover:

- block boundaries through k=481 and odd output tails;
- both real transpose enums and alpha/beta combinations;
- offset and padded storage;
- exact existing per-output dot arithmetic;
- cancellation, overflow, subnormals, signed zero, infinities and NaNs;
- a scalar-tail FMA witness.

Alias tests reject active C/A or C/B overlap before mutation, including overlap
only with the odd final B row. They accept safe disjoint panels in shared
backing storage. Focused new tests pass normally and with race instrumentation.

- Scoped generation changes only the expected single-precision dispatch file.
- Formatting, diff whitespace, import policy and copyright checks pass.
- Final escape diagnostics show that A/B/C do not escape.
- No tolerances were relaxed.

Diagnostic-only runs:

- We stopped the first dot cohort after a background CPU burst. A replacement
  possibly overlapped the old child process briefly during shutdown.
- Both `dot/` and `dot-final/` are diagnostic only.
- The clean `dot-accept/` run confirmed the leaf improvement of the rejected
  prototype. It did not give production acceptance.
- The first paired-tile three-round screens and a stopped superseded race run
  are diagnostic only. Final acceptance uses the release cohorts below.

## Reproduce

Local raw evidence is in `/tmp/gonum-dot-followup.BVSDAE`. It is temporary and
not checked in.

Build separate binaries from the baseline and the candidate, each with the
common harness:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./internal/asm/f64 -o /tmp/f64.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./blas/gonum -o /tmp/blas.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./mat -o /tmp/mat.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib -c ./lapack/gonum -o /tmp/lapack.test
```

Run the go-optimisation skill's `compare_benchmarks.py`:

- Set separate `--baseline` and `--candidate`.
- Set a fresh `--output` and `--rounds 10`.
- Match `GOMAXPROCS` and `--cpu`.

Analyze raw samples with benchstat
`golang.org/x/perf@v0.0.0-20260312031701-16a31bc5fbd0`.

Final evidence directories (each has metadata, raw samples and summaries):

| Directory | Selector | P |
| --- | --- | ---: |
| release-dgemm | `^BenchmarkDgemmTransposedB$` | 1 |
| release-mat-p1 | `^BenchmarkFactorization$` | 1 |
| release-mat-p4 | `^BenchmarkFactorization$/(SVD\|QR\|LQ\|LU\|Cholesky)$/.*` | 4 |
| release-netlib | `^BenchmarkDgesvdNetlibKernels$` | 1 |

- The profile is `release-svd.cpu`, with selector
  `^BenchmarkFactorization$/SVD/kind=thin/shape=wide/n=256$`, P1 and 3 s
  benchtime. It is not mixed into the timing samples.

SHA256 of the exact measured binaries. They were built from the frozen source
before the commit metadata was added:

```text
tile-base-blas.test d97eda38da3e3c40d43918a07660b8852f9d1bdbd06cc6ea3677a4aee5bbe458
release-blas.test   ee6cd27e61f6a2169d5c7560ef1d2fc422dcca053d44c3909eb9545575fe6683
base-mat.test      010b168791999b566b569b148789f8de0d87f6422d2015d5a18570e1f58d638d
tile2-mat.test     01ce97bb9dcf605ceae0060546001eedd18ce28a0c4bc2d706e7cb84209fabd0
base-lapack.test   57d5b60c468c1329277327ccf22a8668ca5f6b4ed5298e41999bfdf8654fb694
tile2-lapack.test  12dde1fca486352887b315fab12f384079f28578ae7f1348bef571e0b770dc24
```

The DGEMM benchmark source is identical in both trees. Its SHA256 is
`26d72066a96b670aa95272d663d89a714fc2191d53916560b9c7882a1fd8d48a`.

Native Netlib differential checks:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib ./lapack/gonum -run '^Test(DgesvdNetlib.*|DbdsqrNetlib.*)$' -count=3
```

The native oracle is Homebrew reference LAPACK/BLAS, not Accelerate/OpenBLAS:

- Homebrew resolves to `Cellar/lapack/3.12.1`.
- The runtime version test reports ILAVER=3.12.0.
- Source review is pinned to Reference-LAPACK v3.12.1.
