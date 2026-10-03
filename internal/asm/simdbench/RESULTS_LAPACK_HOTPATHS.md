# BLAS-to-LAPACK hot-path performance pass

## Summary

- Shared-panel GEMM guard: faster LU and upper Cholesky at n=256; public
  Cholesky n=128 is slower (+3.6%).
- Forward blocked TRSM: faster 16-RHS LU and Cholesky solves; some fallback
  STRSM cases are slightly slower.
- One-RHS solves stay much slower than Reference-LAPACK.

## Setup

- Required workflow: `go-optimisation`, with `gonum-simd` and the scoped LAPACK
  numerical-review gates.
- The user added `go-optimisation` during this pass. Later acceptance uses its
  prebuilt-binary comparison runner. The runner records binary hashes, sample
  order, runtime settings, and matching benchmark selection.
- When checked, there was no repository PGO profile or GOFLAGS override.
- Starting implementation: `856a28593860b8ecaac50537134852c1dc8a7a6d` on
  `codex/arm64-simd-blas`.
- Common `mat` APIs are the workload proxy. This is not usage telemetry. It
  does not claim that every LAPACK routine is optimized.
- Native host: Apple M1 Pro, darwin/arm64.
- SIMD measurements: Go 1.27.1, `GOEXPERIMENT=simd`, initially `GOMAXPROCS=1`.
- Default-path checks: Go 1.26.4.
- This host gives no AMD64 performance data. AMD64 production dispatch does
  not change; this pass did not benchmark it or cross-compile.

Method:

1. Prebuild test binaries before timing.
2. Run baseline and candidate serially, and interleave repeated samples.
3. Record allocation counts.
4. Keep public API costs separate from reusable-workspace LAPACK kernels.

Reference:

- Native comparisons use Homebrew Reference-LAPACK, not OpenBLAS or Accelerate.
- Source review uses v3.12.1 commit
  `6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca`.
- The installed formula is 3.12.1_1, but ILAVER reports 3.12.0. Source identity
  and runtime identity are different.

## Consumer call ledger

| Public workload | LAPACK path | Shared lower-level work |
| --- | --- | --- |
| QR / tall solve | Dgeqrf, Dtrcon; Dormqr, Dtrtrs | Dlarf/Dlarfb, GEMV/GER, GEMM/TRMM/TRSM |
| LQ / wide solve | Dgelqf, Dorglq, Dtrcon; Dormlq, Dtrtrs | Reflectors and GEMM/TRMM/TRSM |
| LU / square solve | Dgetrf, Dgecon; Dgetrs | Panel GER, trailing GEMM, TRSM |
| Cholesky / SPD solve | Dpotrf, Dpocon; Dpotrs | SYRK, GEMM, TRSM; triangular vector solves |
| SVD | Dgesvd: QR/LQ or direct Dgebrd, then Dbdsqr | Reflectors, GEMM, GEMV/GER, rotations |
| Symmetric eigen | Dsyev: Dsytrd, then Dsterf or Dorgtr/Dsteqr | Symmetric matrix/vector updates, reflectors, rotations |
| General eigen | Dgeev: Dgebal, Dgehrd, Dhseqr, Dtrevc3 | Hessenberg/Schur updates, GEMM/GEMV, triangular solves |

- Factorization APIs query the preferred workspace.
- QR, LU, and Cholesky also estimate condition numbers. Public API
  measurements include these costs.
- SVD solves use matrix multiplication and singular-value scaling, not a LAPACK
  solve routine.
- `Dense.Solve` chooses LU, QR, or LQ by matrix shape.

The ledger helps to set priorities. It is not a full transitive
numerical-parity audit. Each change needs persistent regression tests, affected
conformance tests, and suitable independent reconstruction/residual or native
checks.

## Benchmarks

Persistent entry points:

- `mat`: `BenchmarkFactorization` and `BenchmarkFactorizationSolve`.
  - Sizes 32/128/256, shape and vector-job variants, and one/16 right-hand
    sides.
  - PCG fixtures and one untimed warm-up make steady-state reuse explicit.
- `blas/gonum`: `BenchmarkDtrsmSizes` and `BenchmarkDsyrkSizes` cover small and
  block-boundary dimensions. `BenchmarkDgemmSharedBacking` and
  `BenchmarkSgemmSharedBacking` isolate LU/upper-Cholesky panel layouts.
- `lapack/gonum`, tag `netlib`: `BenchmarkDgeqrfNetlib`,
  `BenchmarkDgetrfNetlib`, `BenchmarkDpotrfNetlib`, and the existing
  `BenchmarkDgesvdNetlibKernels`.
  - These keep reusable-workspace kernel costs separate from public API
    overhead.
  - Input resets are included equally on both sides. Layout conversion and
    workspace queries are excluded.
  - Native allocation counts cover Go, not C.

## Profile and priorities

The first target was GEMM eligibility for shared-backing panels:

- A three-second CPU profile of public LU at 256x256 gave 30.6% of samples
  cumulatively to `dgemmSerialNotNot`.
- The existing SIMD guard rejected entire overlapping slice suffixes, even when
  their active matrix entries were disjoint.
- The repair checks only active row intervals and keeps rejection of genuine
  overlap. It reuses the existing SIMD kernel and does not change LAPACK
  algorithms.
- Strided AXPY and triangular solves stay large independent costs.

Longer baseline profiles (three-second benchmark targets, `GOMAXPROCS=1`) set
the next priorities:

| Workload | Cumulative hotspot share | Implication |
| --- | --- | --- |
| LU factorization, n=256 | 30.6% non-transposed GEMM | Enable disjoint shared-storage panels first |
| Tall QR, 512x256 | 43.0% GEMM with transposed B | Separate transpose-aware kernel opportunity |
| Cholesky, n=256 | 25.0% SYRK; 23.2% TRSV | Symmetric updates and condition estimation both matter |
| LU solve, n=256, 16 RHS | 99.0% TRSM | Test blocked triangular solves next |

These are sampled CPU shares for specific inputs, not universal workload
weights.

## Shared-panel SIMD results

- Benchmark-only checkpoint: `fefd7949` (same implementation as `856a2859`).
  Shared-panel candidate: `dee9b2d4`.
- The initial manual six-sample screen used 100 ms for broad native/public
  cohorts and 150 ms for shared-panel kernels.
- After the skill audit, both committed trees were rebuilt with Go 1.27.1 SIMD
  and explicit `-pgo=off`.
- The skill runner alternated ten rounds: 400 ms for LU/Cholesky and 300 ms for
  shared GEMM. It recorded binary hashes, raw per-process output, runtime
  environment and order.
- The runner results replace the initial public and kernel percentages below.
- Raw local evidence: `/tmp/gonum-lapack-hotpaths.NcL3gt/`.

| Workload, one worker | Before | Active-row guard | Time change |
| --- | ---: | ---: | ---: |
| Dgetrf, n=128, reusable workspace | 276.9 us | 251.8 us | -9.1% |
| Dgetrf, n=256, reusable workspace | 1.681 ms | 1.454 ms | -13.5% |
| Dpotrf Upper, n=256 | 783.7 us | 692.3 us | -11.7% |
| Public LU, n=128, audited | 360.4 us | 339.9 us | -5.7% |
| Public LU, n=256, audited | 2.013 ms | 1.813 ms | -10.0% |
| Public Cholesky, n=256, audited | 1.237 ms | 1.182 ms | -4.5% |
| Public Cholesky, n=128, audited | 256.1 us | 265.3 us | +3.6% |

- All tabulated differences have p<0.05. Native rows use six manual samples.
  Public rows use ten runner samples.
- The n=128 Cholesky slowdown repeated. It is a reported tradeoff.
- That size does not execute the changed GEMM path, so its cause is not known.
- In this cohort, the larger LU/Cholesky gains outweigh this small-size cost.
  This is not true for every possible workload.

Shared-panel DGEMM/SGEMM (audited):

- One worker: runtimes decreased 27.6-61.1% (ten rounds, all p<0.001).
- Four workers, LU-shaped n=256: DGEMM decreased 23.3% and SGEMM 36.7% (ten
  rounds, p<0.001).
- One-worker cases allocate nothing. The parallel cases keep about 2 KiB and
  20 allocations per operation in both revisions.
- These recorded runner results replace the outliers and the overbroad
  zero-allocation claim of the initial screen.
- The generated kernel still emits ARM64 vector FMLA instructions. Only the
  dispatch geometry changed, not the arithmetic loop or the existing bounds
  checks.

Other workloads:

- QR and most SVD/public solve cases were statistically neutral.
- The broad screen suggested general-eigen gains, but it had large baseline
  outliers. This pass does not claim these gains.
- Lower Cholesky does not change. At n=256 it is slower than this
  Reference-LAPACK runtime: 1.668 ms versus 1.147 ms.
- At n=256 on these fixtures, Gonum is faster than the native reference:
  - Upper Cholesky: 0.692 versus 1.783 ms.
  - LU kernel: 1.454 versus 2.207 ms.
- These comparisons say nothing about optimized Accelerate/OpenBLAS
  performance.
- Runner directories: `audit-mat`, `audit-gemm`, and `audit-gemm-p4` in the
  evidence directory.
- Public API comparison, SHA-256:
  - baseline binary `d719facb5ec154b65d988d72a18787f93d458679e600fc6788013a9f8b739068`
  - candidate binary `55b80a875faabc6982087900cb9d8c9a3eb9d4eb5242b8339067d2e817d19966`
- Runtime controls: GOMAXPROCS as labeled. GODEBUG, GOGC and GOMEMLIMIT unset.

## Dedicated skill audit

Three independent Sol reviewers checked measurement quality, numerical
behavior, and code generation/dispatch. The audit:

- corrected parallel-allocation reporting
- added self-contained comparison metadata and legacy-provenance notes
- strengthened shared-panel exceptional-value and native solve tests

### TRSM prototype repair

- The uncommitted blocked-TRSM prototype failed a persistent finite-to-infinity
  regression in both precisions for Upper/NoTrans.
- Backward panel traversal also reverses Lower/Trans accumulation in panels.
- The revised candidate blocks only Lower/NoTrans and Upper/Trans/ConjTrans. It
  keeps the original backward paths.
- The original Lower/Trans test fixture did not discriminate. It was corrected
  to isolate descending-versus-ascending order in one panel.
- No prototype timing is accepted as production or consumer performance
  evidence.

### Native solve gates

- The gates now compare Gonum and Netlib solutions directly. They also check
  independent residuals.
- Selected cases: n=128/129/192/256, RHS widths 1/16/17/64, both
  transpose/triangle choices, RHS scales 1e-200/1e200.
- Exponent-normalized residuals prevent overflow of the tolerance scale.
- These are bounded numerical gates, not a full LAPACK parity audit.
- The final extreme-scale solve cases use 16 RHS at n=192, so they run the
  blocked path. A separate one-RHS case keeps fallback coverage.

### Forward blocked TRSM scope

- Forward blocked DTRSM/STRSM applies only to left-sided, alpha=1 solves with
  at least 128 rows and 16 RHS.
- It uses 64-row diagonal solves and serial GEMM updates.
- Backward substitution, other scales, small cases, and unsupported builds
  keep the old implementation.
- No workspace allocation or public API was added.

### Emulation gate

- The audit found that blocking under `GODEBUG=simd=0` sent updates through
  emulated GEMM.
- A three-round diagnostic screen roughly doubled n=256, 16-RHS public solve
  times. A separate CPU profile gave 63% cumulatively to
  `dgemmSerialSIMD@simd0`.
- This was not accepted as a production result.
- The revised gate checks `simd.Emulated()` and keeps the original solve path
  under emulation. It does not change existing GEMM emulation behavior.

## Forward triangular-solve results

- Baseline: `4fb7419037b6f7c1fba1f75decc94434db4c18df`.
  Candidate: `4d5e6224539a98917d4b27ed5fd038e853e3871d`.
- Both include the shared-panel repair and identical benchmark code.
- Test-only extreme-RHS coverage was strengthened after the candidate binary
  build. No benchmark or production source changed.
- All binaries use Go 1.27.1, `GOEXPERIMENT=simd`, `-pgo=off`.
- The skill runner collected ten interleaved rounds, with `GOMAXPROCS=1`.
- 100 ms per BLAS case, 150 ms per public solve case.
- BLAS timings exclude benchmark input resets.
- Public solves are prefactored. These are solve times, not
  factorization-plus-solve times.

| Public solve, 16 RHS | Before | Forward blocking | Time change |
| --- | ---: | ---: | ---: |
| LU, n=128 | 112.29 us | 94.42 us | -15.91% |
| Cholesky, n=128 | 112.50 us | 94.90 us | -15.65% |
| LU, n=256 | 443.5 us | 338.4 us | -23.69% |
| Cholesky, n=256 | 444.4 us | 339.0 us | -23.72% |
| LQ, n=128 | 909.5 us | 892.1 us | -1.92% |
| LQ, n=256 | 3.895 ms | 3.791 ms | -2.67% |

- All rows have p<0.01 with ten samples per revision.
- QR and all measured one-RHS consumer cases are statistically inconclusive.
  This does not prove them equivalent.
- LU and Cholesky solves stay zero-allocation. QR/LQ keep 32 B and two
  allocations.

### BLAS kernels

The 104-case public BLAS cohort covers both precisions, all four
transpose/triangle combinations, 127/128/129-row and 15/16/17-RHS dispatch
boundaries, and 256/512-row cases with 16/64 RHS.

- Enabled forward DTRSM cases reduce time by 28.75-58.86%, and STRSM by
  31.02-64.45% (all p<0.001).
- Both precisions stay zero-allocation.
- These are per-case kernel results, not an application throughput multiplier.
- Fallback cases include small statistically significant slowdowns. The largest
  in the broad run is STRSM Lower/Trans at 127x15 (+3.52%).
- A change in source dispatch can move generated code, even when a case does
  not enter the new helper. This is a hypothesis, not a diagnosed cause.

### Fallback STRSM cost

A focused one-worker repeat at 400 ms confirms the Lower/Trans, 15-RHS STRSM
cost:

- 127/128/129 rows regress 3.49/3.42/3.13% respectively (ten rounds, all
  p<=0.001, `final-fallback-repeat`).
- This is an accepted, disclosed tradeoff against the much larger
  enabled-path and consumer gains. Not all calls improve.
- The measured solve cohort showed no public API consumer regression.
- Recalibrate these dispatch decisions on other ARM64 CPUs and after toolchain
  changes. Do not extrapolate an M1 result to AMD64.
- Local runner directories: `final-trsm` and `final-consumers`.
- BLAS binary SHA-256:
  - baseline `9395e6111828fa0413127d55102053947f2a707e87c9f1abee3600c432b8902e`
  - candidate `b3c8a9ed85136d6e18ce631add1f391392513b1436850738c8c9051c71c2a4f3`

### Four workers

With `GOMAXPROCS=4`, the 128/256-row, 16-RHS cohort (ten rounds at 100 ms,
all p<0.001, results in `final-trsm-p4`):

- Forward gains stay: 32.6-49.8% for DTRSM and 33.7-54.5% for STRSM.
- These helpers use serial GEMM on purpose. Four available workers do not
  mean a parallel triangular solve.
- Allocation counts stay zero.
- DTRSM Lower/Trans at 128x16 regresses 1.73% in this run. It stays on the
  original arithmetic path.

### Emulation

The final emulation check (`final-emulation`, ten rounds at 200 ms,
`GOMAXPROCS=1 GODEBUG=simd=0`):

- No statistically significant change in LU/Cholesky 128/256, 16-RHS solve
  times (all p>=0.85; zero allocations, unchanged).
- LU n=256 is 443.2 us on both revisions.
- This removes the earlier diagnostic regression of roughly 2x.
- It is not a claim of native SIMD speed under emulation.

Binary SHA-256:

- Public solve baseline
  `3d4130294b08fcc0755b309038df14478f5f5c2234a74680ec68c154889131c0`
- Public solve candidate
  `b18fe05633c43c165e2141101811e2b6e0bbbe4ce415cfcf90b7576088be92e7`
- Native solve baseline
  `e034590031dc19949e7c523a434322df67a42597e38fb75f43320ec128e29ab8`
- Native solve candidate
  `42f3b319e1e5067923333d41f55f241cfed3bbac7ed2e4239fe630362beeeee7`

### Native solves

The native solve cohort (`final-native`, ten rounds at 100 ms) independently
confirms the gains across both DGETRS transpose and DPOTRS triangle choices
(all p<0.001):

- 16 RHS: 15.1-24.2% less Gonum time.
- 64 RHS: 9.4-14.9% less Gonum time.
- n=256, 16 RHS: DGETRS NoTrans takes 339.1 us; Reference-LAPACK takes
  476.5 us.
- n=256, 16 RHS: DPOTRS Upper takes 339.0 us; Reference-LAPACK takes 642.6 us.
- Both sides stay at zero Go allocations. Native-reference controls are
  largely unchanged.

### One-RHS solves

One-RHS solves are a separate, major remaining bottleneck. At n=256:

- DGETRS NoTrans: 273.8 us; Reference-LAPACK: 30.20 us.
- DPOTRS Upper: 273.6 us; Reference-LAPACK: 40.97 us.

This pass does not improve them. They do many one-element TRSM updates. Do a
dedicated vector-solve dispatch/profile investigation first, with transpose,
layout, scaling and numerical gates. Then do instruction-level tuning.

### Factorization control

The final factorization control (`final-factorizations`, ten rounds at 200 ms)
compares the same solve baseline/candidate for LU/Cholesky at 128 and 256:

- No material additional change.
- Three cases are statistically inconclusive.
- LU n=128 changes -0.20% (p=0.029).
- Allocations do not change.

## Next priorities

1. the one-RHS solve path
2. transposed-B GEMM for QR/LQ reflector updates
3. SYRK/TRSV for Cholesky and condition estimation

- Backward TRSM needs an order-preserving algorithm before it is reconsidered.
- The earlier n=128 Cholesky factorization regression stays as disclosed
  above. This solve-only pass does not fix it.
- This pass does not show a general SVD speedup.

## Final validation

Passed:

- full `go test ./...` with Go 1.26.4 and Go 1.27.1 SIMD
- BLAS/LAPACK/mat `noasm` and `bounds` suites
- focused safe-build, race and emulation checks for blocked DTRSM/STRSM and
  shared GEMM regressions
- strengthened native DGETRS/DPOTRS tests, including 16-RHS extreme scales
  that enter blocked dispatch
- earlier shared-panel gates for native factorization reconstruction and SVD
  conformance
- `go generate ./blas/gonum` leaves generated files unchanged.
- Import/copyright policy checks and whitespace checks pass.
- Final binary inspection shows ARM64 D2/S4 vector FMLA updates in the reused
  GEMM kernels.
- No mallocgc or makeslice call appears in the inspected TRSM/GEMM symbols.
- Existing bounds exits remain. This does not claim that all bounds checks or
  spills are gone.
- A final independent Sol review found no correctness or build-compatibility
  blockers in the corrected dispatch.

## Reproduce

- Analysis uses `golang.org/x/perf/cmd/benchstat` at
  `v0.0.0-20260312031701-16a31bc5fbd0`.
- Host: macOS 26.6.2 (25G83), Apple M1 Pro. No affinity or frequency lock.
- GOMAXPROCS is 1, and GODEBUG, GOGC and GOMEMLIMIT are unset. Exceptions: the
  explicitly labeled worker/emulation runs.
- Builds, tests and profiles ran outside acceptance timing windows.
- Raw evidence is temporary, not a permanent repository artifact. The
  persistent benchmarks and exact revisions above support reproduction.

Build each tree before timing, with matching benchmark files:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c -pgo=off ./mat -o mat.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c -pgo=off ./blas/gonum -o blas.test
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c -pgo=off -tags netlib ./lapack/gonum -o lapack.test
GOMAXPROCS=1 ./mat.test -test.run '^$' -test.bench '^BenchmarkFactorization' -test.benchtime=400ms -test.benchmem
GOMAXPROCS=1 ./blas.test -test.run '^$' -test.bench '^Benchmark[DS]gemmSharedBacking$' -test.benchtime=400ms -test.benchmem
GOMAXPROCS=1 ./lapack.test -test.run '^$' -test.bench '^BenchmarkD(geqrf|getrf|potrf)Netlib$' -test.benchtime=400ms -test.benchmem
GOMAXPROCS=1 ./mat.test -test.run '^$' -test.bench '^BenchmarkFactorization$/LU$/n=256$' -test.benchtime=3s -test.cpuprofile=lu.cpu
GOTOOLCHAIN=go1.27.1 go tool pprof -top mat.test lu.cpu
```

- Use the skill comparison runner with ten alternating baseline/candidate
  rounds.
- Compare its recorded files with `benchstat`.
- Repeat selected kernels with `GOMAXPROCS=4`.
- Repeat solve consumers with `GODEBUG=simd=0`.

Native tags need the optional darwin/cgo Homebrew Reference-LAPACK
installation. The `mat` and BLAS manifests run without it, also on AMD64.
AMD64 gets no new automatic GEMM dispatch from this ARM64-validated change.

Workflow per milestone:

1. Set up persistent factorization/solve and BLAS3 cohorts. Profile the current
   implementation. Compare representative kernels with Reference-LAPACK.
2. Change a measured shared bottleneck. Validate kernel boundaries, supported
   layouts, numerical behavior, and end-to-end consumers against the same base.
3. Run fallback and full-suite gates. Publish measured results and remaining
   priorities. Verify each milestone on the remote branch.

Unresolved questions: none.
