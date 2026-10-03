# Gonum SIMD versus AMD64 assembly — measured checkpoint

## Summary

This patch makes Go SIMD candidates faster on the native Intel i7-11850H,
but not faster than assembly in every routine or shape. The optimization goal
stays open. No unmeasured case is classified as
theoretically impossible. AMD64 production BLAS dispatch does not change.

Raw evidence comes with the Samsung patch bundle. Evidence paths below are in
its `evidence/` directory. Binary copies are in `measured-binaries/`.

## Results

| Comparison | Cases | Prior SIMD ASM wins | Selected ASM wins, median | Significant ASM wins | Significant regressions of at least 5% versus prior SIMD |
| --- | ---: | ---: | ---: | ---: | ---: |
| manifest-512 | 114 | 48 | 87 | 85 | 1 |
| manifest-256 | 114 | 46 | 86 | 83 | 0 |
| consumers-512 | 402 | 59 | 209 | 205 | 6 |
| consumers-256 | 402 | 43 | 209 | 205 | 1 |

### Kernel manifest

The manifest covers all 57 BLAS assembly entry points at two sizes per width.
33 routines win all four tested size/width medians. 31 win all four
significantly. `selected-v2-acceptance/routine-coverage.csv` and JSON give the
full per-routine result, with every remaining loss. These counts are coverage
summaries, not a workload-weighted speedup estimate.

### Consumer cases

The 402 consumer cases execute real BLAS methods with matched temporary
method-local overlays. They include triangular products, norms, rank-one
updates, transposed and nontransposed matrix-vector products, complex dots,
widened float32 dots and complex scaling. Every variant gets identical finite
inputs, mutation resets and fixture checks.

The 68 widened-dot, 72 complex-scale, 34 complex cutoff, 32 contiguous GER and
34 small-GemvT cases intentionally cover important boundaries. Their frequency
is not an application workload model.

### Production routing

- SIMD candidates are directly callable for measurement.
  `build-consumers-v2.py` reproduces real consumer candidate routing.
- For positive-stride `Zdscal`, unchanged production uses scalar Go. The
  comparator labeled assembly routes that loop to `DscalInc` assembly.
  Separate `zdscal-production-*` timings keep the actual scalar production
  comparison.
- The Go 1.27 AMD64 `internal/math32.Sqrt` intrinsic change affects
  production. `sqrt-production-*` samples measure it separately against
  the prior assembly.

### Production sqrt

| Width | Go intrinsic | Former ASM route | Median time change |
| --- | ---: | ---: | ---: |
| width512 | 1.208 ns | 2.176 ns | -44.5% |
| width256 | 1.2 ns | 2.178 ns | -44.9% |

### Regressions

Measured significant regressions of at least 5% against prior SIMD. A negative
assembly delta still means faster than assembly.

| Suite and case | Prior ns | Selected ns | ASM ns | Change vs prior | Change vs ASM |
| --- | ---: | ---: | ---: | ---: | ---: |
| consumers-256: `BenchmarkSIMDComplexScaleConsumers/Zdscal/n=1/inc=1` | 7.379 | 8.011 | 4.999 | +8.6% | +60.3% |
| consumers-512: `BenchmarkSIMDComplexScaleConsumers/Zdscal/n=1/inc=1` | 7.702 | 8.692 | 4.962 | +12.9% | +75.2% |
| consumers-512: `BenchmarkSIMDGerShapes/m=16/n=32/inc=1` | 65.31 | 70.25 | 71.97 | +7.6% | -2.4% |
| consumers-512: `BenchmarkSIMDWidenedDotConsumers/Dsdot/n=32/inc=1` | 11.57 | 12.55 | 10.88 | +8.5% | +15.4% |
| consumers-512: `BenchmarkSIMDWidenedDotConsumers/Sdsdot/n=32/inc=1` | 12.04 | 13.66 | 11.13 | +13.5% | +22.6% |
| consumers-512: `BenchmarkSIMDWidenedDotConsumers/Dsdot/n=64/inc=1` | 14.62 | 16.09 | 18.24 | +10.1% | -11.8% |
| consumers-512: `BenchmarkSIMDWidenedDotConsumers/Sdsdot/n=64/inc=1` | 15.48 | 17.07 | 19.3 | +10.2% | -11.6% |
| manifest-512: `BenchmarkCurrentVsSIMD/f64/AddConst/n=31/implementation=kernel` | 9.073 | 9.739 | 17.27 | +7.3% | -43.6% |
| scal-width-512: `BenchmarkSIMDBoundaries/f64/ScalUnitaryTo/n=64/stride=1/implementation=kernel` | 10.5 | 11.25 | 10.82 | +7.2% | +3.9% |

## Changes

Accepted:

- direct short reduction entries
- bounded native tails and strides
- exact gathered GER tiles with reuse across rows
- a wide contiguous GER path
- native small GEMV paths
- complex short-dot and scaling improvements
- the corrected explicit AVX2 admission for widened dots
- 512-bit c64 dot dispatch uses its faster native256 path through 511 elements
- at width256, ScalUnitaryTo also uses the measured native256 loop at long
  lengths

Counts come from the integrated binary, not a sum of isolated prototype wins.

Rejected:

- seeded short float32 reductions
- packed short float64 Sum
- a larger GER tile that spilled registers
- a ScalUnitaryTo unroll rollback
- complex FMA changes that altered exceptional-value behavior

The evidence keeps their source, disassembly, raw samples and failure reasons.
Wider vectors, fewer apparent operations or more unrolling alone proved nothing.

## Remaining gaps

The gaps are mostly in short reductions, some short strided operations and
awkward matrix shapes. Source and generated-code work found repeated
eligibility checks, stack and register moves, lane extraction and packing,
bounds/address work and shape branches. Existing assembly also pipelines
independent row products effectively. These are observed mechanisms and
hypotheses, not proof that each loss has one cause.

The AddConst entry and both native256/native512 lowerings are
instruction-identical after relocation normalization, in the final and prior
binaries. Thus its regression is a placement or surrounding-state effect, not
a changed arithmetic body. The exact cause is not isolated.
Cumulative source bodies also showed placement-sensitive shifts in earlier
passes. Cached startup eligibility and small-GemvT row pipelining are separate,
unaccepted follow-ups.

## Setup

- Base: `93a1976af7d80c677589f2e9d1f76ad2ed152d87` on
  `origin/codex/arm64-simd-blas`.
- Prior SIMD comparator: local commit
  `81b40a257c34b14b82646ca7163271d800b7240f`, with the earlier accepted
  tuning.
- The bundle SOURCE.json pins the delivered commit and verified tree.
- Final measured source hashes: `selected-v2-final-source-sha256.json`. Raw
  samples and binary hashes: `selected-v2-acceptance/`.
- Linux AMD64, Go 1.27.1, `GOEXPERIMENT=simd`, `GOAMD64=v1`,
  `GOMAXPROCS=1`, CPU 2. 512-bit and 256-bit SIMD widths, each selected
  separately.
- ARM64 was cross-built only. The branch name is not ARM64 performance
  evidence.
- Existing handwritten assembly already uses SIMD instructions.

## Method

- Each main comparison uses six alternating samples with 50 ms timed work per
  case. The scalar sqrt comparison uses 150 ms.
- `simdbenchclean` clears upper AVX state before both implementations.
- The quiet runner pauses and restores authorized background menu helpers. No
  compilation, tests or profiling run during timing.
- The evidence keeps `benchstat` output and exact two-sided rank-test
  probabilities.
- Significance is per case, not adjusted for multiple comparisons. Marginal
  differences are not universal claims.
- All 19,320 recorded observations report zero bytes and zero allocations per
  operation. The raw records and allocation-check.json keep that proof.

## Validation

The integrated source passed 17 validation stages: kernel suites at
0/128/256/512, independently disabled AVX2 configurations, FMA-disabled
coverage, race, checkptr correctness, the full repository SIMD suite,
default/safe/noasm paths, Go 1.24 kernel and BLAS compatibility, ARM64
full-repository compilation and diff checks.

- Full BLAS candidate-overlay suites passed at 512 and 256.
- All 402 named consumer fixtures were counted and passed in each of eight
  width/feature configurations.
- Formatting, import policy and copyright checks passed.
- Numerical checks keep cancellation, overflow/NaN classification, alias,
  exact-tail and protected-gap behavior, with no relaxed tolerances.
- Normal allocation checks stay on. Only checkptr2 excludes allocation
  assertions: its instrumentation intentionally moves some unsafe
  conversions to the heap. Its correctness checks still run.

Limits:

- Feature overrides on this capable processor validate routing, not execution
  on physically AVX-only hardware. The evidence also keeps the necessary
  generated-code admission inspection.
- Width512 with the AVX512 hardware bundle disabled correctly panics during Go
  SIMD initialization. The supported width256 configuration tests that
  fallback.
- Exact test names and child-case counts corrected an earlier Dger
  test-selection mistake.
- The final benchmark preflight found and corrected an overestimated count in
  the separate Zdscal production selection.
- The evidence keeps failed first attempts with explanations.

## Export

The export script fetches origin again and requires the exact base. It replays
the patch series and the combined diff in separate clean worktrees to
identical trees. It verifies preservation of all 56 original dirty source files. It writes a new Samsung bundle and rereads every copied file against
SHA256SUMS.

The previous USB bundle is kept. The base already contains the prior c64
assembly alignment and empty-tail repair `d224925a`. Its reference patch is
under `already-in-upstream`. Do not apply it again. The script pushes nothing to
a remote.

## Reproduce

Apply the patch. Then run the kernel comparison:

```sh
GOEXPERIMENT=simd GOMAXPROCS=1 GODEBUG=simd=512 go test -tags simdbenchclean ./internal/asm/simdbench -run '^$' -bench '^BenchmarkCurrentVsSIMD$' -benchtime=100ms -count=6
```

Repeat at `simd=256` on a quiet, pinned native CPU.

For matched real consumers:

- Regenerate the overlays with `build-consumers-v2.py --candidate-root <patched-checkout> --base-root <prior-81b40a25-checkout> --output <overlay-directory>`.
- Compile each variant with the commands in `final-build/status.json`. Adapt
  them to local paths.

The bundle also contains the exact measured Linux AMD64 binaries. They
reproduce the recorded executable without a newer toolchain.
