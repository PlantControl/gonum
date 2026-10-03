# ARM64 adoption of the September AMD64 bundle

## Summary

The import is usable on native ARM64 with the conformance adjustments below.
Two kernels are kept, DDOT and four-row DGER. Both use Go archsimd NEON, and
the compiler already emits the necessary instructions. No handwritten assembly
is kept. AMD64 production routing does not change. Not every AMD64
optimization has an ARM64 counterpart, and not every BLAS routine is newly
faster.

## Retained native results

Times are medians from the final ten-round cohorts, against the unchanged
import. All reported active-path reductions have p<0.01. These measurements
are for the Apple M1 Pro only, not AMD64 or other ARM64 hosts.

| Operation | Import | Retained Go | Time change |
| --- | ---: | ---: | ---: |
| DotUnitary n=16, offset=0 | 9.800ns | 4.706ns | -51.97% |
| DotUnitary n=64, offset=0 | 35.985ns | 9.077ns | -74.78% |
| DotUnitary n=256, offset=0 | 140.45ns | 30.69ns | -78.15% |
| DotUnitary n=4096, offset=0 | 2237.0ns | 621.9ns | -72.20% |
| BLAS Ddot n=16, offset=0 | 11.130ns | 5.790ns | -47.97% |
| BLAS Ddot n=256, offset=0 | 141.45ns | 31.75ns | -77.55% |
| BLAS Ddot n=4096, offset=0 | 2233.5ns | 624.8ns | -72.03% |
| mat.Inner small/small | 93.39ns | 91.15ns | -2.39% |
| mat.Inner medium/medium | 5.690us | 1.574us | -72.34% |
| mat.Inner large/large | 555.0us | 156.9us | -71.73% |
| mat.Inner large/small | 7.795us | 7.506us | -3.71% |
| Dger 4x2, pad=0 | 19.88ns | 10.30ns | -48.18% |
| Dger 8x8, pad=0 | 39.31ns | 18.39ns | -53.22% |
| Dger 64x16, pad=0 | 374.0ns | 159.1ns | -57.45% |
| Dger 64x64, pad=3 | 920.2ns | 725.6ns | -21.15% |
| Dger 64x512, pad=0 | 7.128us | 5.766us | -19.10% |
| Dger 64x512, pad=3 | 7.231us | 6.804us | -5.91% |

### DDOT

- Every final measured case reports zero allocations.
- DotUnitary n=0/1/7/8/15 and public Ddot n=7/8/15 show no significant
  difference in either tested offset. This does not prove equivalence.
- Across all measured active lengths 16–4096, leaf reductions are
  49.48–78.18% and public Ddot reductions are 46.52–77.93%.

### DGER

The 24 active shape/padding cases improve 5.91–57.45%. Retained tiny fallback
costs:

- 3x7 pad0: 20.56→20.87ns (+1.51%, p=0.001)
- 3x7 pad3: 20.60→20.89ns (+1.41%, p=0.008)
- 4x1 pad0: 19.77→20.02ns (+1.26%, p=0.014)

The fourth tiny control is inconclusive. The guarded route is kept for its
larger gains. It is not universally faster. The three strided
controls measured 11.97–12.97% lower times, but they execute the unchanged
scalar path. This work claims no new strided algorithm or causal explanation.

### Confirmation run

A separate ten-round confirmation, in `ddot-confirm` and `ger-confirm`, uses
200ms per public-BLAS sample.

- Ddot n=16/128/256, offset=0, takes 48.86%/76.93%/78.23% less time
  (all p<0.001). Baseline confidence intervals narrowed to approximately 1%.
- DGER 64x512 improves 20.40% with pad0 and 5.98% with pad3 (both p<0.001).
- All four tiny DGER controls confirm a 0.25–0.29ns cost, 1.21–1.39%
  (p≤0.017). We explicitly accept this small fallback regression for the
  measured active-path improvements.

## Baseline and scope

- Imported baseline: `53702365864f9d09bad60c5fe54891d8777cc406`. Its tree
  matches bundle source `9a1169e8268c8581fa1022a8422aa272903552d9`
  (`747704973603960872bf348263b68083e479cc7c`).
- All delivered SHA256 checks passed before import.
- Native host: Apple M1 Pro, darwin/arm64, stock Go 1.27.1 and
  `GOEXPERIMENT=simd`.
- This host does not execute AMD64-specific assembly. The imported report is
  separate evidence.

The work first fixes native conformance failures. Then it evaluates bounded
ARM64 candidates against the unchanged import. The baseline is the existing
portable SIMD and archsimd kernels, not scalar code selected to exaggerate
gains. Handwritten NEON needs generated-code evidence and repeated native leaf
and consumer measurements. Unsupported build modes keep their existing
implementations.

## Import validation

- Default Go toolchain and Go 1.24: affected kernel, BLAS, LAPACK and mat tests pass.
- Go 1.27.1 SIMD with combined `safe noasm bounds` tags: the same scope passes.
- Full native SIMD: fails only in f64, at `TestGemvTSmallIEEEAndPadding`
  (Inf versus NaN) and `TestSIMDPositiveStridePanics` (writes before panic).
- Portable emulation and race tests reproduce those two f64 failures.
- Native Netlib: LAPACK passes. BLAS fails its old Sdsdot zero-length expectation.
- Changed Go files are gofmt-clean. The imported diff passes whitespace checks.

These are baseline results, not acceptance of the later adoption changes. The
bundle separately discloses AMD64 checkptr level-2 allocation assertion
failures. Ordinary and level-1 results do not cancel those failures.

## Conformance fixes

- Sdsdot now agrees directly with Reference-LAPACK 3.12.1 for zero length. It
  returns the supplied bias. The old oracle test expected Gonum to disagree
  with Netlib. We removed that special case, not the oracle comparison.
- The small transposed GEMV reference separates ARM64 scalar FMA tails from
  separately rounded vector products. The production arithmetic and the AMD64
  reference do not change.
- Oversized-stride tests still require a bounds panic. They no longer require
  partial mutations of invalid inputs before that panic. Go can move bounds
  checks within an unrolled block.
- Valid-input numerical, padding, aliasing and access tests stay in place.

## Candidate selection

- **Four-row DGER:** Reuse y loads across rows, not AXPY for each row. The
  first scope is disjoint unit-stride input. This does not cover Dgetf2's
  strided, shared-matrix input and is not a claimed LU improvement.
- **DDOT:** The current slice intrinsic loop emits much per-load bounds and
  address work. Compare a checked pointer-loop archsimd implementation with a
  whole-kernel NEON assembly leaf, and keep its exact FMA/reduction order. A
  baseline CPU profile of `mat.BenchmarkInnerMedMed` gives 92.67% of sampled
  CPU cumulatively to DotUnitary and its inlined loads. Profile timing is not
  performance acceptance evidence.

## Method

- All acceptance timings use the skill's alternating prebuilt-binary runner.
- `GOMAXPROCS=1`, stock Go 1.27.1 with SIMD, default native NEON width, no
  PGO input and no GOFLAGS override.
- Builds, tests and worker CPU work were stopped before timing.
- Both revisions use identical deterministic benchmark fixtures.
- Each case has ten samples per revision: 100ms per leaf/public-BLAS sample
  and 200ms for matrix-inner-product samples.
- Benchstat is `golang.org/x/perf v0.0.0-20260312031701-16a31bc5fbd0`, with
  its default Mann-Whitney comparison and 95% confidence intervals.
- These are per-case tests, not a multiplicity-adjusted aggregate
  application-throughput estimate.

The unchanged baseline worktree is at the imported commit above. Its only
additions are identical public DDOT/DGER benchmark files. Generation of all
single-precision sources reproduces the baseline exactly.

Raw observations, run order, settings and binary hashes are under
`/tmp/gonum-arm64-adopt.79xQZz/`, in `dot-final`, `ddot-final`, `ger-final`
and `inner-final`. Each directory contains `baseline.txt`, `candidate.txt`,
`metadata.json` and per-run stdout/stderr. The retained Go benchmark names
reproduce the benchmark commands without those temporary binaries.

## Rejected alternatives

A real NEON assembly dot leaf passed exact arithmetic, tail, offset, overlap,
ABI and Darwin guard-page checks. It used two post-increment four-vector loads
and four FMLA operations per eight elements.

We **rejected** it. In ten-round native comparisons against a checked Go
pointer-loop control, it took 4.6–100.4% more time. This applied to all
tested active lengths, 16–4096, aligned and offset. The Go
control emits eight vector loads and four FMLAs with no hot-loop bounds
checks. Fewer assembly instructions did not give faster execution. ABI and
load-scheduling costs were not separately isolated.

The temporary evidence directory keeps the rejected source/control checkout
and binaries. The repository does not keep the assembly, its race-only
fallback or its comparison-only source files.

Early wrapper costs:

- The first dot wrapper added approximately 0.62ns on short inputs. The
  original short path, restored before long-path validation, removed the
  significant cost.
- The first DGER entry called its helper even for tiny rejected shapes, which
  cost approximately 2.3ns. The final entry checks unit strides and cheap
  size bounds before the helper call.
- Both rejected measurements are in `dot-import-v-neon`, `dot-go-v-neon` and
  `ger-import-v-candidate`. They are not final-source results.

## Toolchain boundary

Release notes and current proposal pages were checked on 2026-09-08. SIMD
stays behind the experiment in this work. SVE and default-enablement proposals
are not evidence that a new API or hardware capability is available. Use the
installed Go 1.27.1 source to find exact intrinsic signatures.

- <https://go.dev/doc/go1.27>
- <https://github.com/golang/go/issues/78902>
- <https://github.com/golang/go/issues/73787>
- <https://github.com/golang/go/issues/79781>
- <https://github.com/golang/go/issues/78979>
- <https://github.com/golang/go/issues/76175>

## Reproduce

Use identical harnesses in both revisions:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c ./internal/asm/f64 -o f64.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c ./blas/gonum -o blas.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c ./mat -o mat.test
# Run each prebuilt binary in alternating baseline/candidate order, GOMAXPROCS=1:
# -test.run=^$ -test.bench=^BenchmarkDotUnitaryLengths$ -test.benchtime=100ms -test.benchmem
# -test.run=^$ -test.bench=^BenchmarkDdotUnitaryLengths$ -test.benchtime=100ms -test.benchmem
# -test.run=^$ -test.bench='^BenchmarkDgerARM64SIMD(StridedControl)?$' -test.benchtime=100ms -test.benchmem
# -test.run=^$ -test.bench='^BenchmarkInner(SmSm|MedMed|LgLg|LgSm)$' -test.benchtime=200ms -test.benchmem
```

## Final validation

These checks pass:

- the full native Go 1.27.1 SIMD test suite
- default-toolchain and Go 1.24 tests for internal assembly helpers, BLAS,
  LAPACK/gonum and mat
- Go 1.27.1 SIMD tests with combined `safe noasm bounds` tags
- native Reference-LAPACK 3.12.1 oracle tests for BLAS/gonum and LAPACK/gonum
- focused race tests for DDOT, DGER, imported GEMVT arithmetic and stride
  panics
- portable SIMD emulation for f64, BLAS/gonum, LAPACK/gonum and mat tests
- focused DDOT checkptr level-2 tests
- final f64/BLAS vet, changed-file goimports, repository import/copyright
  policy and whitespace checks

Persistent tests include exact FMA/reduction behavior, tails, padding, alias
fallbacks, invalid geometry and Darwin guard pages.

The DDOT checkptr pass does not resolve the bundle's unrelated whole-suite
checkptr allocation assertions. No native AMD64 execution or cross-compilation
was done. Preservation of AMD64 routing is a source-review result.

## Remaining opportunities

- Single-precision GER is a separate candidate. The double-precision timings
  do not automatically promote it.
- Fixed-stride widened dots can get an interleaved NEON-load experiment later.
  Arbitrary NEON gathers and short-call setup need their own evidence.
- On future Go releases, recheck generated code and crossovers before you
  remove guards or change APIs. This work makes no GA release promise.
