# AMD64 tail and stride follow-up

## Summary

- The new candidates improve the experimental SIMD kernels on AMD64. AMD64
  production assembly stays selected. No native ARM64 performance claim.
- On ARM64, the candidates stay comparison candidates, not production dispatch.

## Setup

- Date: 2026-09-05.
- Base: fetched `codex/arm64-simd-blas` commit
  `12e546269777a957f63c7ed7ea963e98e817998b`.
- Go 1.27.1 on an Intel Core i7-11850H.
- Six interleaved samples per case, prebuilt binaries, CPU2 affinity,
  `GOMAXPROCS=1`, same benchmark harness on both revisions.

## Results

| Native width | Fetched SIMD wins vs ASM | New candidate wins vs ASM | New vs fetched SIMD: faster / slower / inconclusive |
|---|---:|---:|---:|
| 512 bits | 33/114 | 37/114 | 52 / 5 / 57 |
| 256 bits | 25/114 | 27/114 | 52 / 5 / 57 |

How to read the counts:

- A win needs benchstat's per-case 5% significance test.
- "Inconclusive" does not prove equivalence.
- Counts are unweighted, with no multiple-comparison correction, and do not
  show application throughput.

Selected 512-bit medians (ns/op, lower is better):

| Kernel | Fetched SIMD | New candidate | ASM |
|---|---:|---:|---:|
| f32 AXPY, n31 | 16.06 | 10.56 | 14.29 |
| c64 conjugated dot, n31 | 76.09 | 38.96 | 13.40 |
| c128 strided real scale, n4096 | 8052.50 | 2004.50 | 1561.00 |
| f32 strided widened dot, n4096 | 6667.00 | 2709.50 | 1960.50 |
| f64 cumulative sum, n4096 | 3827.00 | 2091.00 | 2284.50 |
| f64 cumulative product, n4096 | 3801.00 | 2128.00 | 2053.00 |

- The cumulative-product comparison with ASM is inconclusive.
- Prefix gains come from fixed scalar blocks. These blocks keep the original
  block/carry arithmetic order and remove staging. They are not pure
  vector-arithmetic gains.

## Why the gains occur

The [SIMD article supplied for this investigation](https://mitchellh.com/writing/everyone-should-know-simd)
divides cost into load, compute, reduction and tail. The tuned ASM baseline
already uses SIMD where useful. These changes give more benefit than wider
vectors alone:

- no scratch packing
- an inlined short float32 tail leaf
- narrower complex64 short-dot dispatch at 256 bits
- exact-address sparse arithmetic

## Extended coverage

Each width has 448 boundary cases and 252 stride cases.

| Width | Boundary cases improved | Stride cases improved |
|---|---:|---:|
| 512 bits | 128 | 205 |
| 256 bits | 110 | 206 |

- All significant stride regressions in this sweep use unit increments.
- The separate f64 matrix sweep runs unit-stride blocked paths. It does not
  prove consumer gains from the new sparse leaves.
- All 21,336 timed measurements reported zero allocations.

## Retained costs

- Some short exact-size complex64 AXPY calls pay about 1–2 ns of helper setup.
  Useful tails saved about 12–18 ns in the crossover review.
- Sparse ASM often keeps a large lead, because it uses fewer bounds/address
  operations.
- The final manifest has five significant regressions per width:
  - 512 bits: f32 DotUnitary n4096 (+6.63%), Ger n8/n64 (+3.89%/+2.71%),
    f64 AxpyUnitary n31 (+3.23%), and L1NormInc n4096 (+7.11%).
  - 256 bits: f32 Ger n8 (+6.42%), f64 AddConst n31 (+2.52%), AxpyUnitary n31
    (+2.73%), L2DistanceUnitary n4096 (+2.34%), and ScalUnitaryTo n31 (+7.13%).
- L1NormInc differs between the manifest and stride sweeps. Unchanged routines
  also show layout-sensitive timing shifts. The full report has the raw
  evidence.

## Correctness repairs

- Go 1.27.1 native partial-load APIs can overread short slices. A guard-page
  regression reproduces a fault in the fetched f32 Ger implementation. Full
  bounded chunks plus scalar remainders replace these unsafe tails.
- Exceptional reductions first retry the established accumulation order, then
  use sequential recovery. Regrouping can otherwise overflow a result that was
  finite. Compensated float64 norms do not change.
- New native helpers that need AVX2 check for it explicitly. This includes the
  case where portable SIMD selects 128 bits on an AVX-only CPU.
- CPU-feature-disabled tests verify fallback selection. Other architectures
  keep their established algorithms.
- Cumulative-product timing uses reciprocal factors with finite normal
  prefixes. The earlier input overflowed at element 1985, so its timings were
  rejected.

## Validation

Passed:

- full experimental test suite
- numeric/BLAS tests at emulated/128/256/512 widths
- AVX2-disabled fallback checks
- race, default, safe, noasm and Go 1.24 default configurations
- ARM64 cross-builds
- numerical, overlap, stride and guard-page regressions
- formatting and repository policy checks

## Limits

- The workstation used its powersave governor. It did not isolate SMT sibling
  CPU10.
- Main cases use 50 ms samples. Boundary/stride cases use 20 ms. Treat small
  changes with caution.

## Patch bundle

- The bundle has raw logs, benchstat summaries, source/binary hashes,
  rejected screens and the full report.
- The incremental patch targets 12e54626.
- A complete alternative targets e925697f. It includes the intervening fetched
  source and earlier complex64 AXPY ASM fixes.
- Apply only the patch that matches the base of the clean checkout.

## ARM64 integration check

### Import

- Integrated on 2026-09-05 from the incremental 12e54626 patch.
- All 34 kernel/test source hashes match the AMD64-tested manifest.
- The import did not change any numerical implementation.
- The historical [BLAS report](RESULTS_BLAS.md) also records the original
  CumProd timing caveat.

### Setup

- Native Apple M1 Pro, Go 1.27.1, `GOEXPERIMENT=simd`, `GODEBUG=simd=128`,
  `GOMAXPROCS=1`.
- Prebuilt binaries, six interleaved samples against 12e54626.
- Both revisions used the same corrected benchmark harness.
- Compilation and validation stopped before timing.
- Public BLAS/SVD samples used 50 ms. The 114 candidate cases used 30 ms.

### Results

- None of the 22 selected public BLAS cases or four SVD cases changed
  significantly.
- Candidate-only sweep: 3 faster, 8 slower, 103 inconclusive. These use
  per-case 5% tests without multiple-comparison correction.
- Largest candidate regression: c128 DotuInc n31, 56.58 to 60.89 ns (+7.60%,
  p=0.002).
- Its n4096 case and the related unitary cases also regressed about 4.6–5.7%.
- No specific hardware cause for the timing shifts was found.
- With corrected finite inputs, ARM64 CumProd n4096 measured 2.663 versus
  2.720 µs (p=0.855). This does not prove a difference.
- Candidate and public BLAS allocations stayed zero. SVD allocation counts did
  not change.

### Validation

Passed:

- full Go 1.27.1 SIMD and Go 1.26.4 default suites
- affected emulation, safe and noasm tests, including LAPACK/mat
- numeric/BLAS race and bounds checks
- formatting, import policy, copyright and diff checks

Native AMD64 timings and Linux guard-page execution come from the supplied
bundle. They are not new measurements on this ARM64 host. No new
cross-compilation was done.
