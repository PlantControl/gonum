# AMD64 patch integration, 2026-09-04

The two Samsung-drive patches matched base
`e925697fb16c99cbbe52cf4a489f1e6f503923db`. Every bundle SHA256 checksum
passed. They stay separate commits, `d224925a` (AMD64 assembly
correctness) and `33a826fc` (SIMD tuning), followed by the review corrections
below.

The [supplied AMD64 report](RESULTS_AMD64.md) measures the imported tuning,
not these later corrections. Its raw samples were not supplied. Production
dispatch does not change. The assembly correctness repair is active on AMD64,
but candidate speedups do not mean BLAS, SVD or application speedups.

## Native ARM64 results

Selected medians compare final SIMD candidates with base SIMD, not assembly.
Every table improvement has benchstat p=0.002, n=6.

| Kernel/size | Base ns/op | Final ns/op | Time change |
|---|---:|---:|---:|
| f64/L2NormUnitary/4096 | 29271 | 1024 | -96.50% |
| f64/L2DistanceUnitary/4096 | 29239 | 1462 | -95.00% |
| f64/L2NormInc/4096 | 29317 | 3826 | -86.95% |
| f64/GemvN/64 | 1598 | 844.3 | -47.17% |
| f64/GemvT/64 | 2057 | 843.5 | -58.99% |
| f64/Ger/64 | 2056 | 921.3 | -55.19% |
| f32/Ger/64 | 1222 | 495.8 | -59.42% |
| c64/AxpyUnitary/4096 | 8608 | 2792 | -67.56% |
| c64/DotcUnitary/4096 | 5740 | 2796 | -51.30% |
| c128/DotcUnitary/4096 | 8915 | 6700 | -24.85% |

The equal-weight kernel geomean fell 25.77%, which is not an application
speedup. Per-case significance uses alpha=0.05, with no multiple-comparison
correction. Short-call differences include complex AXPY
dependency-guard costs of approximately 2–3%, and smaller changes in unchanged
functions.

### Unresolved CumSum regression

The unchanged f64 CumSum candidate regressed:

- n=31: 26.71 to 37.61 ns/op (+40.76%)
- n=4096: 3110 to 4461 ns/op (+43.45%)

An isolated six-round 200ms rerun confirmed it (p=0.002). Its source and
normalized SIMD128 instruction stream did not change. Binary layout is a
hypothesis, not a known cause. Resolve this before you promote that candidate.
The current production routine is not affected.

### Setup

Apple M1 Pro, macOS 26.6.2, darwin/arm64, Go 1.27.1, `GOEXPERIMENT=simd`,
128-bit vectors, `GOMAXPROCS=1`. The baseline and candidate binaries were
built before timing. The same unchanged comparison harness ran six 50ms
samples per case and reversed the binary order each round. No concurrent test
or compilation workload ran. CPU affinity was not pinned. All 1,368
measurements across 114 cases and two revisions reported zero allocations.

## Review corrections

- Added runnable complex64/complex128 dot-product regression tests. Separate
  component sums overflowed before cancellation and returned a NaN component,
  not zero. Non-finite unitary results now retry sequential complex
  multiplication and summation, including contiguous increment calls.
- Kept direct portable widening/staging for targets without the AMD64 widening
  leaf. The imported fallback made ARM64 Ddot 52–157% slower. Final Ddot
  timings are statistically inconclusive versus base.
- Without the AMD64 leaf, complex128 unitary dot and scale reuse the portable
  strided SIMD core. This removes the measured shuffle-fallback regressions,
  with no duplicate ARM64 candidate implementation.
- Matrix benchmark baselines are named `current`, because ARM64 does not use
  AMD64 assembly.

## Validation

- Full default Go 1.26.4 and SIMD-enabled Go 1.27.1 suites passed.
- Affected native ARM64 packages passed with hardware and emulated SIMD.
- Race, safe and noasm tests passed for internal assembly and BLAS.
- AMD64 numeric and comparison binaries passed under Rosetta with
  `GODEBUG=simd=0` and `simd=128`, including the assembly alignment
  regressions. This is translated correctness evidence, not native AMD64
  performance evidence.
- Generated code contains ARM64 vector FMUL/FADD in the norm fast path, and
  AMD64 VCVTPS2PD with vector arithmetic in the widened dot leaf. Bounds checks
  stay in generated loops, so more compiler optimization is possible.
- Formatting/import grouping, repository import/copyright policies and diff
  checks passed.

Local integration logs and raw samples are in `/tmp/gonum-patches.C7GiDm/`:
`checked-base.txt`, `checked-final.txt`, `arm64-final.txt`,
`scan-{base,final}.txt` and test logs. These temporary files are not
committed. The [comparison instructions](README.md) give the runnable kernel
harness. Before you change dispatch, remeasure the integrated revision on
native AMD64, with AVX256 and AVX512.
