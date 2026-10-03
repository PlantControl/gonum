# Bottom-up SVD rotation follow-up

## Summary

ARM64-experimental Dlasr changes speed up Left and Right variable rotations
and all nine thin-SVD public cases. Small fallback and dispatch costs remain.

## Scope

Start point: `511627624ef848acf7faba7e253cb897471b47b5`. The work uses
`go-optimisation`, `gonum-simd` and dedicated Sol numerical/code-generation
reviews.

The earlier thin wide SVD n=256 profile:

- Dlasr: 34.22% cumulative CPU, below Dbdsqr (38.05% cumulative). Do not add
  these nested percentages.
- Of the 660 ms flat samples of Dlasr, about 450 ms are in
  Left/Variable/Backward and 210 ms in the Right fallback.
- The old blocked Right helper adds another 490 ms cumulative.

Thus we optimize rotations and do not rewrite SVD.

Public path: mat.SVD.Factorize -> Dgesvd -> Dbdsqr -> Dlasr. Dbdsqr applies
left/right variable-pivot rotations in the two directions to shrinking active
blocks. These blocks often have a leading dimension from the larger matrix.

## Changes

The ARM64-experimental changes stay inside Dlasr, after its original
validation.

Left/Variable:

- Uses the existing f64.RotUnitary NEON kernel for n>=16. A single rotation
  (m=2) requires n>=32.
- All-identity calls return without matrix writes.
- Active A and coefficients must be disjoint. Coefficients must be finite in
  [-1,1]. Other calls use the old loop.

Right/Variable:

- Keeps the old m,n>=64 and all-active gate. It then carries four independent
  rows through the original rotation sequence.
- Two-row and one-row tails handle awkward dimensions.
- Steady-state A traffic goes from two loads/two stores to one load/one store
  per row/rotation, plus endpoint loads/stores. This does not include
  coefficient traffic.
- Overlap and nonfinite or out-of-range coefficients use the old blocked
  fallback, with its pre-existing alias behavior (not a new sequential alias
  contract).
- After the blocked path declines, m>=32/n>=16 sequential Right calls use a
  separate non-inlined copy of the original loop. This isolates its hot code
  from dispatcher growth. Smaller calls keep the in-function loop.

Other properties:

- Alias checks conservatively cover the bounded A storage span, including
  padding, not only logically active elements.
- No new allocation, public API, workspace, parallelism or experimental API.
- Top/Bottom pivots, AMD64, default, safe and noasm keep their existing
  implementations.
- Right uses scalar multiply-plus-FMA sequencing, not vector SIMD. It stays
  behind the measured experimental ARM64 gate.
- The coefficient guard does not test that c*c+s*s equals one.

Race builds keep the original Right implementation, because instrumentation
changes compiler scalar fusion. Left SIMD stays enabled and public numerical
regressions still run under race. Only tests that assert selection of the
optimized Right helper are excluded.

## Numerical review

The source-review boundary is Dlasr, its new helpers and existing
f64.RotUnitary. It is not every translated routine that Dgesvd can reach.
Reference: [Reference-LAPACK v3.12.1, SRC/dlasr.f](https://github.com/Reference-LAPACK/lapack/blob/6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca/SRC/dlasr.f).

- All 12 Side/Pivot/Direct branches were source-reviewed. The existing
  finite/mixed-identity Netlib differential suite exercises them.
- Left/Right Variable have regression tests. Top/Bottom source expressions do
  not change.
- Existing Go deviations: row-major storage/lda>=n, typed arguments and
  panics, slice validation and the pre-existing all-active Right row-block
  schedule.
- Dlasr has no workspace query or convergence status.
- Existing quick-return and invalid-argument tests stay in force.
- This is not a claim of complete transitive SVD parity or Netlib
  endorsement.

Persistent tests cover:

- Cutoff/tail boundaries, compact/padded storage and offsets.
- Exact guard/padding preservation.
- Identity, signed-zero and nonfinite inputs.
- Coefficient/A aliases and normalized extreme values.
- Rejection before writes.

Independent Dlasr, Dbdsqr and Dgesvd Netlib tests run separately from timing.
The linked Homebrew package is lapack 3.12.1. Live ILAVER reports 3.12.0.
This is reference Netlib, not Accelerate or optimized OpenBLAS.

### Issues found and resolved

- **Unguarded RotUnitary reuse** can turn an unnormalized overflow result
  into NaN. Whole-call coefficient/alias guards keep the previous path. A
  first bit-exact extreme reference also failed the untouched baseline: the
  compiler fused a different term. The new test now uses scaled forward error
  and nonfinite classification. No existing tolerance was relaxed.
- **First Right carry prototype.** It failed the existing 1e-14
  block-boundary regression. Direction-specific explicit math.FMA
  expressions now keep the fused term of the baseline ARM64 loops. The
  existing test does not change.
- **Fallback layout regression.** The first ten-round timing found a 10-13%
  m=63/n=65 fallback regression. The hot instructions and registers did not
  change, but the added dispatch moved them. This is consistent with a
  code-layout effect. Isolation of the sequential loop removed the
  regression. Broader screens also found 64x31 and set the m>=32/n>=16
  isolation boundary. We added no padding or instruction-alignment hacks.
  The two cases stay in the benchmarks.
- **Race build.** The candidate failed the unchanged Right block-boundary
  test under race; the baseline passed. The Right-only race opt-out keeps the
  original instrumented arithmetic and does not weaken the test. Non-race
  instruction identity is checked separately after this build-tag change.

## Generated code

Left (native disassembly):

- The four-column NEON loop has four vector loads/stores, eight multiplies
  and two adds/subtracts.
- No hot-loop calls, bounds checks or spills.
- The 128-byte wrapper keeps one RotUnitary call per active rotation.
- Bulk arithmetic is unfused and the scalar tail uses FMA, so bitwise
  equivalence is not claimed.

Right:

- The direction-specific arithmetic helpers inline. Four carries stay in FP
  registers.
- The 16-byte-frame kernel has no hot-loop calls, divisions, allocations or
  spills.
- Bounds branches remain on row loads/stores. We do not claim to remove them.

Sequential fallback:

- 16-byte frame.
- Its arithmetic/FMA order, per-rotation coefficient reads, identity checks
  and bounds checks match the original scalar Right loop.
- No hot calls, divisions or spills.
- It adds one call per qualifying Dlasr call, not per matrix element.

## Setup

Checkpoints:

- Final clean common-harness checkpoint:
  `3d49adb8ad9b2e71cc84c2d8769d1c67d1fbffec`. It differs from the published
  start point only in persistent tests and benchmark coverage. Compared
  production baseline code does not change.
- Measured implementation: `a261c3530e22875d324364d0653b06993ecbd2a5`.
- Race-only build selection repair: `c7d22d61c985e9459c42c4274fd62bf268090cf9`.
  The complete non-race machine-code sections match before and after this
  repair.
- Final benchmark source SHA-256:
  `05c9b253f6ef3fe012b852314569fec42b0b024d35d8e8ae5770fca39943fac5`.
- The baseline mat binary comes from the earlier test-only checkpoint. It has
  identical mat benchmark and production source.
- The first acceptance used `7f087432d485dff49c853c301a4d4b7b8f76e834` and
  pre-isolation candidate binaries. We keep those results separate and do not
  pool them with the final rerun.

Host and toolchain:

- Apple M1 Pro, darwin/arm64, macOS 26.6.2 (25G83), AC power. No affinity or
  frequency lock. Normal desktop background activity. The host is not
  isolated.
- Native builds: Go 1.27.1, GOEXPERIMENT=simd, explicit -pgo=off, empty
  GOFLAGS. GOGC/GOMEMLIMIT are unset.
- LAPACK timing binaries include the netlib tag. Public mat binaries do not.
- Compatibility: Go 1.26.4.
- [Go 1.27 SIMD notes](https://go.dev/doc/go1.27#simd) and installed API
  documentation were read again. Experimental adoption stays reversible.

Method:

- The go-optimisation compare_benchmarks.py runner alternates prebuilt B/C
  and C/B. It keeps binary hashes/environment and rejects failures or
  mismatched benchmarks.
- Acceptance: ten independent rounds, 150 ms per case, GOMAXPROCS=1 unless
  stated.
- The final exhaustive Right/Variable cohort uses ten 100 ms rounds. Other
  final cohorts use ten 150 ms rounds unless specified.
- No compilation, test suites or profiles overlap timing.
- Three-round 100 ms screens are diagnostic, not acceptance data.
- benchstat: x/perf v0.0.0-20260312031701-16a31bc5fbd0.

Fixtures:

- Kernel fixtures repeatedly apply normalized rotations to finite, nonzero
  matrices, with post-run finite checks.
- Public SVD benchmarks include real Factorize costs and reuse the SVD
  receiver.
- Native Netlib comparisons use the native layout of each backend. They
  include restoring A and cgo overhead, not layout conversion. They are not
  wrapper-inclusive API parity benchmarks.

Raw logs, codegen, runner metadata and binaries are in the temporary
directory `/tmp/gonum-dlasr-followup.CGDjvd`. Persistent benchmarks, tests
and source checkpoints permit reruns.

## Final acceptance results

Percentages are time changes, not throughput changes. Intervals are 95%
benchstat intervals. Per-case p-values are unadjusted; many comparisons
increase false-positive risk. Initial and final samples are not pooled.

### Left variable rotations

Ten 150 ms rounds. All 12 dense eligible cases improve 14.10-51.72%, p<0.001.
All 26 cases keep zero B/op and allocations. Selected Forward/Backward
medians:

| m x n / lda | Baseline ns/op | Candidate ns/op | Change |
| --- | ---: | ---: | ---: |
| 2 x 32 / 32 | 26.41 / 26.75 | 22.69 / 22.67 | -14.10% / -15.25% |
| 4 x 16 / 16 | 41.94 / 42.25 | 35.41 / 35.44 | -15.56% / -16.11% |
| 32 x 16 / 16 | 393.6 / 392.7 | 310.1 / 309.8 | -21.19% / -21.11% |
| 32 x 32 / 35 | 703.8 / 702.2 | 470.4 / 484.1 | -33.17% / -31.06% |
| 32 x 256 / 256 | 5497 / 5494 | 2654 / 2653 | -51.72% / -51.71% |

- At 32x256/lda259, sparse cases improve 22.96%/22.71%. Identity-only calls
  improve 16.83%/16.51%. All p<0.001. Candidate intervals are 0-4%.
- Regression: single-rotation n=15/16/17/31 fallbacks add about
  0.56-0.76 ns (2.14-4.10%, p<=0.017).
- Regression: sparse m=4/n=17 adds 0.64-0.93 ns (3.19-4.70%, p<0.001).

The dispatch is not zero-cost.

### Right variable rotations and fallback repair

Ten 100 ms rounds cover all 142 persistent Right/Variable cases. These
include tiny inputs, 31/32/33 and 63/64/65 boundaries, 79/80/81 row tails,
compact/padded storage, dense/sparse/identity patterns and the two
directions.

- All 32 carry-eligible dense cases improve. Forward: 22.74-28.79%
  (p<=0.002). Backward: 2.10-12.27% (p<=0.022).
- Candidate intervals are 0-7%.
- All cases keep zero B/op and allocations.

| m x n / lda | Forward baseline -> candidate us/op | Backward baseline -> candidate us/op |
| --- | ---: | ---: |
| 64 x 64 / 64 | 2.829 -> 2.152 | 2.827 -> 2.564 |
| 65 x 65 / 68 | 2.910 -> 2.240 | 2.910 -> 2.751 |
| 80 x 65 / 68 | 3.657 -> 2.673 | 3.653 -> 3.220 |
| 256 x 64 / 67 | 11.210 -> 8.213 | 11.211 -> 9.980 |
| 256 x 256 / 256 | 46.72 -> 33.27 | 46.49 -> 42.97 |

Fallback boundary cases:

- The first 10-13% 63x65 regression is gone.
- 63x65 Forward compact/padded: inconclusive (2.735 -> 2.745/2.744 us,
  p=0.269/0.054).
- 63x65 Backward: +0.20%/+0.33% remains (2.733 -> 2.739/2.742 us,
  p<=0.002).
- 64x31 Forward: -0.65% (1.315 -> 1.307 us, p=0.013).
- 64x31 Backward: +0.23% remains (1.307 -> 1.310 us, p<0.001).

Other costs (reported with the gains):

- Identity calls keep about 0.8-1.9 ns overhead (0.40-3.37%, p<=0.018).
- Tiny n=1 quick returns add about 0.58-0.61 ns. This is 12-13% of a
  baseline of about 4.7 ns.
- Other significant small dense fallback increases are about 0.1-4 ns (up to
  5.53% on the 7x2 case).
- Medium fallbacks increase at most 0.48% in this cohort.
- Sparse cases have no statistically significant slowdown. Some improve by
  up to 9%, but they use the sequential fallback, not the new carry
  algorithm.

### Public SVD

Ten 150 ms rounds. All nine thin-SVD cases improve, p<=0.019. Candidate
intervals are 0-5%. Shapes are square n x n, tall 2n x n and wide n x 2n.

| n | Shape | Baseline ms/op | Candidate ms/op | Change |
| ---: | --- | ---: | ---: | ---: |
| 32 | square | 0.1826 | 0.1738 | -4.81% |
| 32 | tall | 0.2744 | 0.2647 | -3.54% |
| 32 | wide | 0.2334 | 0.2253 | -3.45% |
| 128 | square | 6.372 | 5.487 | -13.89% |
| 128 | tall | 10.98 | 10.09 | -8.13% |
| 128 | wide | 9.040 | 8.178 | -9.54% |
| 256 | square | 55.56 | 49.00 | -11.80% |
| 256 | tall | 86.63 | 79.97 | -7.69% |
| 256 | wide | 80.34 | 73.72 | -8.24% |

- Eight values-only controls are inconclusive.
- Wide n=128 improves 0.16% (p=0.023) on an unchanged path, not attributed
  to rotations.
- Bytes/op do not change in effect. Reported differences round to
  +/-0.00%.
- Allocation medians do not change, except thin/wide n=256: four -> three.
  The new kernels removed no allocation. Short benchmark iteration counts can
  change setup amortization.

### Worker-count and unrelated controls

GOMAXPROCS=4, ten 150 ms rounds:

- All six thin-SVD n=128/256 gains are confirmed, p<0.001. Candidate
  intervals are 0-1%.
- Square/tall/wide time changes: -13.85%/-8.30%/-9.95% at n=128 and
  -12.71%/-9.25%/-9.50% at n=256.
- Other worker counts can give different percentages.

GOMAXPROCS=1 controls:

- All 15 QR/LQ/LU/Cholesky factorization controls are inconclusive
  (p=0.075-0.971). They include sizes 32/128/256 and applicable
  square/tall/wide shapes.
- No statistically significant slowdown in this final control cohort.

### Native reference Netlib

Ten 150 ms rounds. All five Gonum Dgesvd cases improve 2.88-15.94%,
p<=0.002. All five unchanged Netlib timing controls are inconclusive. All ten
cases report zero B/op and allocations. Native-layout thin-vector medians:

| m x n | Gonum before ms | Gonum after ms | Netlib after ms |
| --- | ---: | ---: | ---: |
| 32 x 24 | 0.1064 | 0.1034 | 0.07344 |
| 24 x 32 | 0.10104 | 0.09636 | 0.08071 |
| 96 x 64 | 1.315 | 1.224 | 1.090 |
| 64 x 96 | 1.219 | 1.077 | 1.060 |
| 256 x 256 | 52.90 | 44.47 | 89.66 |

The 256-square Gonum fixture takes about half the time of Netlib. Gonum
still takes about 2-41% more time on the four smaller fixtures.

## Validation

- Full default Go 1.26.4 and experimental Go 1.27.1 suites pass.
- Affected f64/BLAS/LAPACK/mat suites pass with safe, noasm and bounds
  separately. After the final build-tag repair, they also pass with the
  combined opt-out configuration.
- Race-enabled Dlasr tests pass three times.
- The same common public regressions pass on the clean baseline.
- Native Dlasr/Dbdsqr/Dgesvd Netlib checks pass three times on experimental
  and default release builds.
- GODEBUG=simd=0 Dlasr/bidiagonal/SVD/public-mat checks pass. This setting
  controls portable emulation, not all architecture-specific leaves.
- Existing tolerances do not change.
- Vet, formatting, import policy, copyright and diff checks pass.
- Linker duplicate-library/rpath warnings were benign.

Default build:

- New helper symbols and calls are removed.
- Default Right blocked/sequential instruction bodies match the baseline.
- We do not claim whole default Dlasr instruction identity. The compiler
  changed one FMA association in the source-unchanged Left/Top/Backward loop.
- Default all-layout Netlib invariants and existing tests pass.
- Floating-point bitwise identity across builds is not guaranteed.

Race-only repair: full Mach-O __TEXT,__text dumps have identical SHA-256
hashes before and after. LAPACK
`1e0a7e405e2c798da4d9de2458b7013686cd5daca52140c1e6ad5279969f75b0`, mat
`4825a001c38f6606b5b01c30651ca08854ecf2d143baca7870723962441f5c10`.
These hashes exclude headers from the otool text-section dump. The
whole-file binary hashes below include them. Thus the race guard does not
make the non-race timings invalid.

## Limits

- Results are not a general claim for all matrices, SVD jobs or
  architectures.
- Inconclusive results do not show exact equivalence.
- The Netlib comparison does not show a general SVD win. It does not compare
  Accelerate, optimized OpenBLAS or optimized vendor BLAS.
- Native AMD64, other CPUs, PGO and all SVD jobs are not measured.
- No cross-compilation was done, as requested.

## Remaining work

A separate three-second thin-wide n=256 profile, after validation, has 3.30 s
of CPU samples:

- Dlasr is 28.48% cumulative below Dbdsqr (34.24%). Do not add these nested
  values.
- Right carry is 12.42% flat, sequential Right 6.06% flat, and RotUnitary
  8.79% cumulative.
- DotUnitary is now 21.21% flat/26.67% cumulative. Of its 0.88 sampled
  seconds, 0.79 come through dgemmSerialNotTrans.

The dot-product/DGEMM path is the next measured bottom-up target. It is not
an accepted optimization yet.

## Reproduce

Build separate baseline and candidate binaries at the checkpoints above:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib -c ./lapack/gonum -o /tmp/lapack.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -c ./mat -o /tmp/mat.test
```

Run compare_benchmarks.py of the skill:

- Pass separate --baseline/--candidate and a new --output.
- Pass --rounds 10, --cpu 1 and the given --benchtime. Set GOMAXPROCS=1.

Selectors:

```text
Left, 150ms:   ^BenchmarkDlasrVariable$
Right, 100ms:  ^BenchmarkDlasr$/side=R/pivot=V/
SVD, 150ms:    ^BenchmarkFactorization$/SVD/
Netlib, 150ms: ^BenchmarkDgesvdNetlibKernels$
Controls:     ^BenchmarkFactorization$/(QR|LQ|LU|Cholesky)$/
P4 SVD:       ^BenchmarkFactorization$/SVD/kind=thin/shape=./n=(128|256)$
```

For P4, use --cpu 4 and GOMAXPROCS=4. Run benchstat on baseline.txt and
candidate.txt. Keep its per-case results and the runner metadata.

Oracle:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib ./lapack/gonum -run '^Test(Dlasr.*|DbdsqrNetlib.*|DgesvdNetlib.*)$' -count=3
```

Final binary SHA-256 values (LAPACK baseline/candidate, mat baseline/candidate):

```text
2b0f2ffa4e7ce7f7c5f9d7bc21eef89ee5a82687f75299678ba41228f79b0a0c
201f56aa55c1b8b2e6745095c692c5625a13f822a6894074cfb27c84e59ca41d
8ac176b8b64fa5ed1807bba4fabe0a97625978a693d76c362aaad9332c6d1c78
5bfcc6555c5ba6462671a00f4e38eba923ba5a665ee4b16759f3e0a1547b0027
```
