# Portable Go SIMD candidates

## Summary

- SIMD candidates live beside the routines that they can replace, in
  `internal/asm/{f32,f64,c64,c128}/simd.go`. They cover every BLAS-related
  AMD64 assembly entry point.
- All architectures share the portable operations. Portable GEMM tiles live in
  `blas/gonum/{d,s}gemm_simd.go`.
- Small AMD64 leaves do complex permutations and widened dot products, where
  the portable API or compiler conversions need scalar staging.
- Other targets keep portable fallbacks.
- This package keeps only the coverage manifest, equivalence tests and
  comparison benchmarks.

## Production status

With Go 1.27 SIMD on, measured ARM64 paths use portable kernels for real GEMM,
selected contiguous GEMV shapes and sufficiently long float64 norms. All other
candidates are for comparison only. AMD64 production keeps assembly dispatch,
with the separately validated assembly correctness repairs.

The [BLAS integration results](RESULTS_BLAS.md) give dispatch boundaries,
accuracy checks and consumer measurements.

`current` is the existing production entry point. Most AMD64 entries use
assembly. The routine inventory identifies the remaining Go paths.

## Reproduce

Use Go 1.27.1 or newer. Run the same-binary equivalence tests and benchmarks:

```sh
GOEXPERIMENT=simd go test ./internal/asm/simdbench
GOMAXPROCS=1 GOEXPERIMENT=simd go test ./internal/asm/simdbench \
  -run '^$' -bench '^BenchmarkCurrentVsSIMD$' -benchmem -count=10 \
  | tee simd.txt
benchstat -col /implementation simd.txt
```

If necessary, install `benchstat` with
`go install golang.org/x/perf/cmd/benchstat@latest`.

Do not change the `go` directive in `go.mod`. The Go 1.27 toolchain and
`GOEXPERIMENT=simd` satisfy the source build constraints.

On Windows PowerShell, run the full comparison:

```powershell
$env:GOTOOLCHAIN = "go1.27.1"
$env:GOEXPERIMENT = "simd"
$env:GOAMD64 = "v1"
$env:GOMAXPROCS = "1"
go test ./internal/asm/simdbench
go test ./internal/asm/simdbench -run '^$' `
  -bench '^BenchmarkCurrentVsSIMD$' -benchmem -benchtime=200ms -count=10 -timeout=0 `
  | Tee-Object -FilePath amd64-simd.txt
benchstat -col /implementation amd64-simd.txt
```

Record `git rev-parse HEAD`, `go version`, CPU model, OS, selected vector
width and processor affinity with the results. On a hybrid Intel CPU, use the
same P-core affinity for each run. `GOAMD64=v1` still lets portable SIMD
select wider supported vectors at runtime.

## Result documents (Issue 6 follow-up)

- [RESULTS.md](RESULTS.md): measured ARM64 results, AMD64 instruction findings.
- [RESULTS_AMD64.md](RESULTS_AMD64.md): native AMD64 tuning against assembly on an AVX512 host.
- [RESULTS_IMPORT.md](RESULTS_IMPORT.md): integration review, ARM64 measurements, correctness repairs, portable fallback validation, remaining regressions.
- [RESULTS_BLAS.md](RESULTS_BLAS.md): prefix-scan regression fix, real BLAS and SVD calls.
- [UPSTREAM.md](UPSTREAM.md): Go compiler experiments, changes to recheck with future releases.
- [RESULTS_TAIL_STRIDE.md](RESULTS_TAIL_STRIDE.md): AMD64 candidate tuning, compatibility repairs.
- [RESULTS_SVD.md](RESULTS_SVD.md): native SVD rotation, independent Netlib comparisons, shared LAPACK cache-blocking improvements.
- [RESULTS_LEVEL1_NETLIB.md](RESULTS_LEVEL1_NETLIB.md): all 46 public Level 1 routines, scalar and SIMD changes.
- [RESULTS_LEVEL1_NORMS_STRIDES.md](RESULTS_LEVEL1_NORMS_STRIDES.md): single-precision norm gap, strided maximum-index scans.
- [RESULTS_BOTTOM_UP.md](RESULTS_BOTTOM_UP.md): bottom-up triangular solve without repeated single-element AXPY dispatch.
- [RESULTS_STRIDED_GEMV.md](RESULTS_STRIDED_GEMV.md): reflector workspace strides through Dlarft, public QR gains.
- [RESULTS_ROWWISE_GEMV.md](RESULTS_ROWWISE_GEMV.md): RowWise GEMV with shared input loads across ordered row reductions, public LQ gains.
- [RESULTS_ALL_ASM.md](RESULTS_ALL_ASM.md): first complete native AMD64 checkpoint. All 57 BLAS assembly entries, 402 matched real BLAS consumer cases, remaining losses, separate square-root production change.
- [RESULTS_GO1271.md](RESULTS_GO1271.md): stock Go 1.27.1 investigations, retained candidates, correctness repairs, measured costs.
- [RESULTS_ISSUE10.md](RESULTS_ISSUE10.md): ARM64 downstream regression traced to executable layout under Go 1.26.4, not the DGEMM threshold.

SIMD candidates do not change AMD64 production BLAS dispatch.

## Scratch buffers and norms

The [initial AMD64 results](https://github.com/jamestjsp/gonum/issues/6#issuecomment-5541048813)
found oversized scratch buffers, unnecessary staging for contiguous matrix rows
and benchmark inputs that decay.

Most scratch now uses `make([]T, width)`. Go 1.27 specializes the width before
escape analysis, so the scratch goes on the stack for the selected vector size.
This sets no fixed ceiling on future vector widths. The allocation regression
test must pass on each target.

The first scratch-only change did not make robust L2 norms faster. The later
tuning:

- Ordinary magnitudes use sums of squares.
- Extreme or non-finite results retry the scaled recurrence.
- Float64 norms compensate summation error and, on fused backends, product
  error. The uncompensated candidate failed an existing SVD tolerance together
  with GEMV promotion.

## Kernel structure

- Real contiguous increment operations use their unitary candidates.
- Dot and sum candidates use four independent accumulators. Explicit load
  spans remove redundant bounds checks.
- On AMD64, native leaves avoid component scratch arrays for several strided
  float32 and complex operations. Float64 sparse updates use scalar unrolling
  where it is faster.
- Fixed scalar prefix blocks keep the existing arithmetic order without
  intermediate staging.
- Other architectures keep their established portable paths.

These are comparison candidates. Their memory and address overhead against
assembly is still significant.

## AVX/SSE mixing

AMD64 code inspection found legacy SSE scalar moves between AVX vector
operations in staging loops. Now, integer memory views move lane bits into and
out of unsigned scratch. `BitsToFloat32`/`BitsToFloat64` then reinterpret the
bits for vector arithmetic. Unlike scalar `math.Float64bits` calls, these views
keep integer moves through Go 1.27 optimization. Complex alpha broadcasts are
outside the vector loop.

[Intel documents penalties for AVX/SSE mixing](https://www.intel.com/content/dam/develop/external/us/en/documents/11mc12-avoiding-2bavx-sse-2btransition-2bpenalties-2brh-2bfinal-809104.pdf).
Windows/AMD64 compiler output shows that these instructions are gone. Their
effect on the office timings still needs native measurement.

## Clean assembly comparisons

For controlled AMD64 assembly comparisons, build the benchmark binary with
`-tags simdbenchclean`. When AVX is available, the manifest, boundary and
stride benchmarks then execute `VZEROUPPER` immediately before every current
and candidate call. Both measurements include the same state-preparation cost.
Kernels and production dispatch do not change.

Keep these results separate from ordinary benchmark runs. Use the same
benchmark harness and tag for both revisions. Instruction-state diagnostics
reproduced an approximately fivefold slow state in the unchanged short assembly
norm. One clear before the trial did not make timings stable.

The optional `simdasmstate` tag gives explicit native/clean/dirty controls on
Linux AMD64.

## Open work

- Portable mixed-precision and prefix fallbacks still mix scalar arithmetic
  with vectors.
- The AMD64 Ddot leaf avoids scalar widening. Ordinary norms avoid the scaled
  recurrence.
- Efficient portable widening, reduction and scan operations are still
  possible improvements.
- With future Go releases, recheck generated instructions before you keep a
  source workaround.

## Benchmark inputs

In-place scaling and division benchmarks use unit-magnitude factors to prevent
subnormal decay. Numerical equivalence tests keep the original factors. Old
timings for the seven affected in-place `Div`, `Scal` and `Dscal` cases are
not directly comparable with corrected timings.

The `BenchmarkSIMDBoundaries` and `BenchmarkSIMDStrides` sweeps cover uneven
lengths and increments 1, 2, 3, 7, 16 and 63. Cumulative-product timing uses
bounded alternating reciprocal factors, so prefixes stay normal and finite.

To compare source revisions, use the same corrected benchmark harness on both
checkouts.

## Manifest checks

The checked manifest fails when:

- an AMD64 assembly symbol is added or removed
- its package-local candidate is absent
- a candidate does not reach a `simd` operation
- its benchmark or equivalence runner is missing

Equivalence tests cover empty inputs and vector boundaries. Allocation tests
cover all 57 current and candidate entry points.

## Native AMD64 follow-up

AMD64 production dispatch does not change.

- Contiguous complex kernels use interleaved vectors.
- Matrix kernels share loads across rows.
- Ordinary L2 norms use vector sums of squares, with a scaled fallback for
  extreme magnitudes.
- The widening and complex-shuffle AMD64 leaves avoid Go 1.27.1 `FromArch`
  stack copies in hot loops.

Recheck these workarounds when the compiler or the portable API changes. Small
calls, arbitrary strides and matrix shapes have different crossovers. Measure
native timings before you change production dispatch.

## Stable SIMD migration

Recheck these items when portable SIMD becomes stable: package path, build
constraint, vector-width contract, generated code and benchmark crossovers. Do
this before you change dispatch. The removal of `goexperiment.simd` is intentionally a small
boundary change. Do not assume that it is the only migration Go will require.
