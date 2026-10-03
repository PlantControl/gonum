# Level 1 follow-up: strided index scans and single-precision norms

## Summary

- Snrm2 and Scnrm2 take 85.82% to 89.35% less time; strided Idamax/Isamax
  take 70.38% and 63.98% less. The gains do not need experimental SIMD.
- Reference BLAS still leads for strided index scans at n=4096.

## Setup

- Base: `792803f0ed5ebe046f74620a7d95840b7ff1c592`.
- Candidate: `856a28593860b8ecaac50537134852c1dc8a7a6d`.
- Native host: Apple M1 Pro, darwin/arm64.
- Primary toolchain: Go 1.27.1, `GOEXPERIMENT=simd`, `GOMAXPROCS=1`.
- Fallback toolchain: installed Go 1.26.4, without experimental SIMD.

This report follows [the complete Level 1 reference comparison](RESULTS_LEVEL1_NETLIB.md).
It targets strided Idamax/Isamax, Snrm2, Scnrm2, and the prior n=256 Scnrm2
regression. It does not add an Accelerate backend or claim AMD64 performance.
There was no cross-compilation.

## Implementation

### Norms

The new norm algorithms are shared scalar Go, not new SIMD intrinsics.

- They widen each float32 component to float64 before squaring.
- They accumulate four independent sums.
- Every finite float32 square is exact in float64. Nonzero squares have an
  exponent range of approximately -298 through 256.
- Even the maximum addressable slice length cannot overflow the float64 sum.
- Thus there are no per-element scaling divisions and no intermediate
  underflow, without a magnitude-specific fast-path gate.

### Rounding

- Summation still rounds. Output is not guaranteed correctly rounded for every
  vector.
- Tests compare against a 256-bit reference over the actual rounded float32
  inputs.
- A finite-input norm can overflow when it is converted back to float32.

### Special values

- Snrm2 keeps the real-norm NaN-over-Inf behavior through ordinary
  floating-point propagation.
- Scnrm2 uses widened accumulation at n>=32, for contiguous and strided
  vectors. If the sum is NaN, it retries the original scaled recurrence. This
  keeps the complex-norm Inf-over-NaN behavior.
- Smaller complex vectors keep their original recurrence.

### Dznrm2 dispatch

- The existing ARM64 SIMD path of Dznrm2 keeps its original contiguous
  dispatch.
- A separate generated strided hook enables the complex64 fast path. Its
  complex128 counterpart is an inlineable stub that returns false. Thus
  strided Dznrm2 does not call a rejected fast path.
- The single-precision BLAS sources stay generated from their owning masters.

### Index scans

- Strided Idamax/Isamax screen four magnitudes at a time.
- Blocks that cannot improve the current maximum skip index updates.
- Other blocks keep ordered strict comparisons, earliest ties, and the existing
  first/later-NaN behavior.
- Small strided scans keep a separate original loop. A shared loop with the
  unrolled tail first made small Isamax slower. Repeated boundary benchmarks
  found this, and the separate loop removed it.

### Code inspection

- Native code shows scalar float32-to-float64 conversions, independent
  double-precision multiply-add chains, and a final square root.
- The contiguous real norm has no bounds-check calls or per-element divisions.
- Strided kernels keep bounds checks.
- The new norm algorithms add no assembly and no unsafe memory access.

## Method

### Timing

- All binaries were built before timing.
- Benchmarks ran serially, six samples per case, with alternating base/change
  order.
- The native backend is Homebrew Reference-LAPACK libblas, not optimized vendor
  BLAS.
- The batched harness amortizes CGo and reports time per BLAS call. Residual
  CGo and public argument-check costs are included.

### Coverage

- The revision/native suite covers the four target routines, plus Dznrm2 as a
  dispatch guard. It uses n=4/16/31/32/33/256/4096/65536 and increments 1 and 2.
- Pattern tests also cover rare and frequent index updates and wider strides.
- The persistent pattern harness also supports random data and strides 3 and
  257.
- Default-toolchain and real LAPACK consumer checks are separate
  within-toolchain comparisons.

### Sample times

- Final native/revision and index-pattern samples: 40ms.
- Default-toolchain samples: 30ms.
- Dgeev consumer samples: 60ms.
- The native comparison uses the 40ms samples of the final complex dispatch
  recheck in both toolchains.
- Each side uses the same inputs and batched-call normalization.
- The desktop machine is not CPU-isolated. The tables show medians and
  benchstat significance, not fastest observations.

### Legacy caveat

- These are legacy, manually orchestrated measurements. The report has no
  self-contained runner metadata, such as binary hashes, per-process sample
  order, PGO state, or the complete runtime environment. The numbers and
  qualifiers below stay as recorded.
- For new acceptance work, use the `go-optimisation` prebuilt-binary comparison
  runner. Keep its metadata and raw per-run output.

## Results

- The primary sweep has 160 cases, six samples per case, 960 timings per
  revision. It includes Gonum and native backends.
- After the original Dznrm2 dispatch was restored, the two complex routines
  were retested with six samples in the primary and default toolchains.
- The tables use these final complex measurements, and the primary
  measurements of the unchanged real/index kernels. The two runs are not
  pooled.

Median Gonum time per call, Go 1.27.1 SIMD configuration:

| Routine | n | inc | Base | Change | Time reduction |
|---|---:|---:|---:|---:|---:|
| Snrm2 | 256 | 1 | 604.55 ns | 64.39 ns | 89.35% |
| Snrm2 | 4096 | 1 | 10.281 µs | 1.254 µs | 87.80% |
| Scnrm2 | 256 | 1 | 1073.0 ns | 145.2 ns | 86.47% |
| Scnrm2 | 4096 | 1 | 17.888 µs | 2.536 µs | 85.82% |
| Idamax | 4096 | 2 | 5.093 µs | 1.509 µs | 70.38% |
| Isamax | 4096 | 2 | 5.100 µs | 1.837 µs | 63.98% |

- All differences above have p=0.002, n=6.
- At n=4096, inc=2: Snrm2 improved 87.61%, and Scnrm2 improved 85.95%.
- The new n=256 Scnrm2 time is also far below the approximately 940 ns
  recorded before the previous patch. The statistical comparison uses the
  immediate base revision, not that older measurement.

### Compared with native Reference BLAS

At n=4096, the final norms take roughly one quarter of Reference BLAS time.
This is not an Accelerate or OpenBLAS comparison.

| Routine | Gonum inc=1 (µs) | Netlib inc=1 (µs) | Go/native inc=1 | Go/native inc=2 |
|---|---:|---:|---:|---:|
| Snrm2 | 1.254 | 5.167 | 0.24 | 0.25 |
| Scnrm2 | 2.536 | 10.350 | 0.25 | 0.24 |
| Idamax | 1.454 | 1.386 | 1.05 | 1.07 |
| Isamax | 1.778 | 1.392 | 1.28 | 1.33 |
| Dznrm2 | 6.319 | 10.363 | 0.61 | 1.73 |

- A ratio below 1 favors Gonum.
- The strided index gap is much smaller, but Reference BLAS still leads at
  n=4096.
- At n=256, inc=2, Idamax takes 100.2 ns, and the reference takes 108.1 ns.
  Which side is faster thus depends on size.

### Index patterns

At n=4096, inc=2:

- Rare-update index inputs improve 70.48% for float64 and 64.15% for float32.
- Monotone inputs improve 19.84% and 22.06%.

Wider strides:

- inc=17: gains of 6.40% to 48.82%.
- inc=257: gains of 8.51% to 47.23%.
- These are specific measured strides. They do not guarantee gains for all
  memory layouts.
- The initial 25–43% regression of tiny Isamax was removed before acceptance.

### Fallback toolchain

With Go 1.26.4 at n=4096:

- Snrm2 improves 75.86% (inc=1).
- Scnrm2 improves 75.05% (inc=1).
- Strided Idamax improves 68.29%, and strided Isamax improves 59.34% (both
  inc=2).

### Consumers

- Dgeev on AntisymRandom50/100 and Circulant50/100 shows no practical overall
  change.
- Two cases are statistically unchanged. Two improve only 0.24–0.36%.
- This is not evidence of a large eigenvalue or SVD speedup.
- The Gonum LAPACK implementation is double precision. Do not attribute these
  single-precision norm gains to its real SVD.

## Limits

The final code is not faster in every tiny case:

- Scnrm2 at n=4 adds approximately 0.3–0.5 ns in the SIMD build and
  0.3–0.7 ns in the default build (up to 5.47%).
- Small default Dznrm2 strided cases add about 0.3–0.7 ns.
- The initial 7.41% contiguous default Dznrm2 regression is resolved: 11.90
  versus 11.94 ns at n=4 is statistically unchanged (p=0.725).
- Other small index/unchanged unitary timing differences are below 1%.
- These kernel benchmarks showed no Go allocations. The counters are per outer
  batch and do not measure native heap allocations.

The changes stay, with these small overheads documented. The reasons are the
larger single-precision gains, the supported fallback behavior, and the
numerical tests. Possible future work: the remaining index gap, or optimized
vendor BLAS. This report does not claim either as complete.

## Reproduce

Build matching binaries from the base and candidate with the same benchmark
files. Then alternate their execution. For example:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c -tags netlib \
  -o /tmp/level1-followup.test ./blas/gonum
GOMAXPROCS=1 /tmp/level1-followup.test -test.run '^$' \
  -test.bench '^BenchmarkLevel1Netlib$/routine=(Snrm2|Scnrm2|Idamax|Isamax|Dznrm2)$' \
  -test.benchtime=40ms -test.count=6 > level1-followup.txt
benchstat -col /implementation level1-followup.txt
```

- The single-binary `-test.count=6` command above reproduces the within-binary
  Gonum-versus-native sweep.
- It does not reproduce the base-versus-candidate alternation of the revision
  comparison.
- For a new revision comparison, build matching binaries at both named commits
  with the same harness.
- Use `go-optimisation/scripts/compare_benchmarks.py` to alternate them.
- Keep the binary hashes, run order, environment metadata, and raw samples.

## Validation

Correctness checks cover high-precision magnitude/overflow cases, zero, NaN/Inf
priority, n=31/32/33 and 255/256/257 boundaries, positive strides, padding
sentinels, index ties, offsets, and monotone/random inputs.

Passed:

- full Go 1.27.1 SIMD and installed Go 1.26.4 suites
- three native differential/ABI repetitions
- affected safe/noasm packages
- BLAS/f32 bounds checks
- focused race tests
- generated-source inspection
- formatting, import policy, copyright, and whitespace checks

No tolerance in an existing conformance test was relaxed.
