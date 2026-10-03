Native SIMD tuning against AMD64 assembly, 2026-09-04 US/Central.

This report came with the AMD64 patch bundle, not from a rerun on the
integration host. Integration later added a non-finite
complex-dot retry and kept faster portable fallbacks. Remeasure the integrated
revision on AMD64 before you use these timings as current. See
[ARM64 integration validation](RESULTS_IMPORT.md).

## Summary

- SIMD now beats assembly in 34 of 114 kernel/size cases, loses 68 and is
  statistically inconclusive in 12.
- At the larger sizes, SIMD wins 30 of 57 cases, loses 23 and is inconclusive
  in 4.
- These counts use per-case benchstat tests at alpha=0.05, with no
  multiple-comparison correction. The preceding branch benchmark had 16 wins
  overall.
- Against base SIMD, 42 cases improved significantly, 68 were inconclusive and
  4 regressed. The equally weighted geometric mean of kernel times fell 44.15%.
  This is not an application speedup.
- Production AMD64 BLAS dispatch stays on assembly.

## Results

Representative medians, in ns/op. All highlighted improvements versus assembly
below have p=0.002, n=6. GemvN is inconclusive.

| Kernel/size | ASM | Base SIMD | Tuned SIMD | Speedup vs ASM | Time change vs base SIMD |
|---|---:|---:|---:|---:|---:|
| f64/L2NormUnitary/n=4096 | 3614.5 | 28543.0 | 320.4 | 11.28x | -98.88% |
| f64/L2DistanceUnitary/n=4096 | 4088.5 | 28805.0 | 438.2 | 9.33x | -98.48% |
| c128/DotcUnitary/n=4096 | 2776.0 | 8096.5 | 1140.0 | 2.44x | -85.92% |
| c128/AxpyUnitary/n=4096 | 2268.0 | 14280.0 | 1295.5 | 1.75x | -90.93% |
| f32/DdotUnitary/n=4096 | 898.1 | 14279.0 | 515.1 | 1.74x | -96.39% |
| f64/Ger/n=64 | 576.6 | 1024.5 | 350.3 | 1.65x | -65.81% |
| f64/GemvT/n=64 | 522.2 | 998.8 | 333.3 | 1.57x | -66.63% |
| f32/Ger/n=64 | 318.4 | 750.1 | 212.2 | 1.50x | -71.72% |
| f64/GemvN/n=64 | 506.1 | 1077.5 | 524.1 | 0.97x | -51.36% |

## Setup

- Base: e925697fb16c99cbbe52cf4a489f1e6f503923db on codex/arm64-simd-blas.
  Tuning branch: codex/simd-beat-asm.
- Intel Core i7-11850H, Linux amd64, Go 1.27.1, GOEXPERIMENT=simd,
  GOAMD64=v1, 512-bit runtime vectors, GOMAXPROCS=1, CPU2 affinity.
- Six 50ms samples per case, with prebuilt binaries. Alternate rounds reverse
  the assembly/new/base order.
- One current binary contains assembly and tuned candidates. Base SIMD comes
  from the unmodified e925697f source export.
- The governor stays at powersave. The SMT sibling was not isolated.
- All 2052 timed measurements completed with zero bytes/allocations per call.
- Timings compare kernels on this CPU only. They are not native ARM64 or
  AVX256 performance evidence.

## Implementation changes

- Contiguous complex arithmetic works on interleaved vectors. It does not
  deinterleave each element into scratch arrays. AMD64 complex128 leaves avoid
  the missing portable shuffle and Go 1.27.1 FromArch stack copies.
- The mixed-precision unitary dot uses vector widening and independent
  accumulators in AMD64 leaves. Disassembly shows VCVTPS2PD with register
  arithmetic and no conversion bridge spills in the hot loop.
- Matrix kernels share input loads across four rows and avoid per-row
  dispatch/setup. Rectangular benchmarks include padding and dimensions from
  7x9 through 512x512. Wide matrices are faster. Tiny and tall-narrow shapes
  can still favor assembly. Large Gemv cases are approximately equal or gain
  a few percent.
- Ordinary L2 magnitudes use direct vector sums of squares. Overflow,
  nonfinite values and sums below a conservative underflow-error threshold
  retry the scaled recurrence. Extreme/subnormal tests use a 256-bit arithmetic
  reference. The headline norm timings use ordinary magnitudes.
- Assembly fixes: AXPY counter underflow after a one-element alignment prefix,
  and valid 4-byte complex64 alignment in both assembly AXPY routines.
- Fixed sequential write dependencies for zero increments and overlapping SIMD
  index streams.

The first full screen found short-call regressions from a horizontal-reduction
helper and an inlined contiguous-dot branch. We discarded the helper and moved
dot dispatch outside the generic loop. The final results include these
corrections. The first 100ms screens in the parent directory are investigation
evidence, not the final result.

## Limits

Four final comparisons against base SIMD are still slower: c128 AxpyInc n31
+4.40%, n4096 +5.62%, c128 AxpyIncTo n31 +5.34% and f32 AxpyIncTo n4096
+7.09%.

- Complex increment paths now have dependency/alias guards that correctness
  requires.
- The f32 AxpyIncTo source is the same as base. This run alone does not show
  why its timing is different.
- General-stride staging and small-call overhead are follow-up work in Ergo
  FQ6C3R.
- This report claims no universal assembly replacement or cutoff.

## Validation

These checks passed after the corrections:

- `GOEXPERIMENT=simd go test ./...` — full suite passed, including unfiltered SIMD equivalence/allocation tests.
- All f32/f64/c64/c128 tests and the shared comparison suite passed at `GODEBUG=simd=0`, `128`, `256`, `512`.
- ARM64 cross-builds of all four affected numeric test binaries. No native
  ARM64 performance claim.
- Default, safe and noasm internal-assembly/BLAS tests. Go1.24.0
  compatibility checks.
- Race tests of the four numeric packages and simdbench.
- gofmt/goimports, repository import and copyright checks, and git diff
  --check.

## Toolchain

The SIMD API and compiler are experimental. Toolchain release status was
checked against the [Go release history](https://go.dev/doc/devel/release).
The installed Go1.27.1 source is the API authority. On a future Go version,
recheck missing operations and generated code before you keep
architecture-specific workarounds.

The source AMD64 machine keeps the raw samples in
`benchmark-results/simd-tuned/final/`, with the repeated-sample runner and
machine metadata. This repository and the imported bundle do not include it.
The [comparison instructions](README.md) build the same candidate/assembly
harness.
