# Dgges Schur vectors and real sorting versus reference Netlib

Status: measured baseline, not a production optimization. 2026-09-06.

## Scope and method

Production source: `0d23bc3f388b99360d940cb3bb54b9289334ce1a`,
verified against fetched `origin/codex/arm64-simd-blas`.
The accompanying changes add benchmark fixtures, numerical tests, and a native
Netlib bridge only; the existing three Dgges benchmarks now reject non-convergence.

The shared fixture is a synthetic dense regular pencil, not a captured controlsys
workload or a Hamiltonian/DARE benchmark. Distinct conjugate pairs have real parts
near -2, +0.5, +2, -0.5 and imaginary magnitude near 0.2. Common orthogonal left
and right transformations densify A and B while preserving the known spectrum.
Half the eigenvalues lie in the open left half-plane and half inside the unit disk.

Modes:

- `form`: no Schur vectors, no sorting; still computes the generalized Schur form.
- `vectors`: both Schur-vector matrices, no sorting.
- `left`: both vectors, select negative real part.
- `unit`: both vectors, select magnitude strictly less than one.

Both implementations use the same mathematical inputs in their native layouts:
Gonum row-major, Netlib column-major. Fixture generation, layout conversion,
workspace queries and caller-owned allocation are outside timing. Both timed
loops restore A and B with contiguous copies, call Dgges, and check success and
selected dimension. Timings therefore include input restoration, not just the
factorization. The C bridge calls LAPACK_dgges directly, not the allocating and
transposing LAPACKE row-major convenience entry point.

This is native Apple M1 Pro, darwin/arm64 v8.0, macOS 26.6.2 (25G83), AC power
at 80%, Go 1.27.1, GOEXPERIMENT=simd, -pgo=off, empty GOFLAGS, GOMAXPROCS=1
and benchmark CPU=1. No frequency/affinity pinning. Normal desktop background
activity remains. Builds, tests and profiles did not overlap timed cohorts.

The linked reference is Homebrew LAPACK 3.12.1 with runtime ILAVER=3.12.0 and
reference libblas, not Accelerate or OpenBLAS. The measured binary links
`/opt/homebrew/opt/lapack/lib/liblapack.3.dylib` and `libblas.3.dylib`.
Binary SHA256:
`fcbda8e4f810239d937fe0d9ca2ec2d4a391bc587be712f3fcc0edc93e30fbd7`.

Ten 100 ms rounds alternate Gonum/Netlib then Netlib/Gonum. Benchstat is
x/perf `v0.0.0-20260312031701-16a31bc5fbd0`. Each row has ten samples per
backend; uncertainty is benchstat's 95% median interval. The raw samples are
retained in [samples.txt](results/dgges-netlib/samples.txt). They are grouped
by backend in lexical round-file order; execution order was alternating.

## Time results

Percentages below are **Netlib time relative to Gonum**, not improvements made
by this commit. Negative means Netlib uses less time.

| Mode | n | Gonum | Netlib | Netlib time difference |
| --- | ---: | ---: | ---: | ---: |
| form | 32 | 196.5 us | 159.1 us | -19.04% |
| form | 64 | 1.102 ms | 0.870 ms | -21.04% |
| form | 128 | 7.742 ms | 7.461 ms | -3.64% |
| form | 256 | 116.1 ms | 112.9 ms | -2.72% |
| vectors | 32 | 294.0 us | 224.2 us | -23.72% |
| vectors | 64 | 1.742 ms | 1.328 ms | -23.76% |
| vectors | 128 | 15.93 ms | 11.41 ms | -28.38% |
| vectors | 256 | 239.9 ms | 143.2 ms | -40.30% |
| left | 32 | 420.5 us | 297.8 us | -29.18% |
| left | 64 | 2.489 ms | 2.373 ms | -4.65% |
| left | 128 | 20.53 ms | 17.12 ms | -16.60% |
| left | 256 | 265.7 ms | 155.9 ms | -41.31% |
| unit | 32 | 784.1 us | 512.2 us | -34.67% |
| unit | 64 | 4.378 ms | 2.720 ms | -37.87% |
| unit | 128 | 32.37 ms | 19.29 ms | -40.40% |
| unit | 256 | 355.3 ms | 192.5 ms | -45.83% |

All differences have p<0.001 except form n=128/256 (p=0.002).
Median intervals are at most +/-3% in this cohort. Multiple comparisons and one
synthetic pencil per size limit generalization; there is no application-speedup
or native AMD64 claim.

The 100 ms cohort permits one iteration per sample at n=256. A separate ten-round
500 ms confirmation, not pooled with the original, gives:

| n=256 mode | Gonum | Netlib | Netlib time difference |
| --- | ---: | ---: | ---: |
| vectors | 239.6 ms +/-3% | 143.4 ms +/-0% rounded | -40.16%, p<0.001 |
| unit | 349.0 ms +/-2% | 193.0 ms +/-1% | -44.70%, p<0.001 |

Raw confirmation samples: [recheck.txt](results/dgges-netlib/recheck.txt).

## Allocations and profiles

Gonum form/vectors have zero measured B/op and allocs/op. Sorting exposes
previously unmeasured scratch allocation:

| n | left: bytes / allocations | unit: bytes / allocations |
| ---: | ---: | ---: |
| 32 | 23,552 / 144 | 94,208 / 576 |
| 64 | 105,984 / 648 | 376,832 / 2,304 |
| 128 | 423,936 / 2,592 | 1,507,328 / 9,216 |
| 256 | 1,507,328 / 9,216 | 6,029,312 / 36,864 |

Netlib reports zero Go allocations/op throughout. This is not a measurement of
C/Fortran heap activity. The first form n=32 case has a small noisy Go B/op signal
(median 3.5 bytes, rounded zero allocs/op); do not interpret it as a native
allocation or make a zero-total-memory claim.

Separate five-second n=256 CPU profiles, not used for timing comparisons:

- Vectors (5.67 s sampled CPU): Drot 58.38% flat / 60.67% cumulative;
  Dgghrd 59.96% cumulative; double QZ sweep 32.63% flat / 33.69% cumulative.
- The Drot samples are overwhelmingly the general strided loop at
  `blas/gonum/level1float64.go:560-565`, not the contiguous SIMD helper.
  Dgghrd's strided Q, A, B and Z column updates at lines 120, 127, 128, 131
  account for 0.75, 0.97, 0.64 and 0.91 seconds respectively.
- Unit sorting (5.39 s sampled CPU): Drot 41.93% cumulative,
  double QZ sweep 21.52%, Dtgsen 27.64%, and applyDtgex2Transforms 19.85%.
  Cumulative figures overlap and must not be added.
- Sampled allocation objects attribute 99.08% to the Dtgex2 call chain:
  46.51% flat in Dtgex2, 33.49% in dtgex2SwapLarge, 19.07% in Dtgsy2.
  Profiles also include benchmark calibration/setup and runtime activity.
- Go compiler escape diagnostics confirm local scratch arrays in Dtgex2
  (`s, tt, li, ir` at line 81) move to the heap. This is an observed mechanism,
  not evidence that removing allocations alone closes the timing gap.

The next measured targets are (1) strided rotations beneath Dgghrd, including
Schur-vector accumulation and native-layout effects; (2) block-swap scratch
escapes and tiny/skinny matrix transforms beneath Dtgsen; then (3) QZ sweep
data movement. Any change needs leaf and full-Dgges before/after gates, preserving
block rejection, scaling, ordering, supported BLAS backends and workspace contracts.
No such production changes are included here.

## Validation and limits

- Native SIMD + Netlib: all Dgges tests pass, and the complete lapack/gonum
  test binary passes.
- Native SIMD + Netlib race: all Dgges tests pass.
- Go 1.24.0 minimum-version Dgges tests pass.
- Default Go 1.26.4: complete lapack/gonum and lapack/testlapack tests pass.
- Formatting and git diff checks pass.
- New fixture tests cover n=8/32/64/128/256 and all four modes. Each backend's
  unsorted spectrum must contain selected values after unselected values for
  both predicates. Sorted outputs must have exactly n/2 leading selected values
  without splitting conjugate pairs.
- Tests check finite values, homogeneous eigenvalue matching, canonical Schur
  structure, vector orthogonality, workspace queries and O(n^3) reconstruction
  residuals at the existing tolerances.
- This validates the benchmark cases, not complete branch-level Netlib parity.
  The new fixtures are regular and well separated from selection boundaries.
  Existing exceptional-input tests remain unchanged.
- Dgges remains float64 real input only; no Sgges/Cgges/Zgges, mat wrapper,
  release guarantee, or new production SIMD kernel is added.
- The shared Gonum benchmark is portable. The optional native oracle bridge
  currently requires darwin+cgo+netlib and the existing Homebrew library paths.
  No cross-compilation was performed.

## Reproduction

Build once, validate, then time the prebuilt binary with no concurrent build/test work:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib -c ./lapack/gonum -o /tmp/dgges.test
GOMAXPROCS=1 /tmp/dgges.test -test.run '^Test(Dgges|NetlibRuntimeVersion)' -test.v
```

Run the two commands below in alternating order for ten rounds, storing each
output separately. For the longer recheck, append
`/mode=(vectors|unit)$/n=256$` to the benchmark selector and use 500ms.

```sh
GOMAXPROCS=1 /tmp/dgges.test -test.run '^$' -test.bench '^BenchmarkDggesNetlibControl$/implementation=Gonum$' -test.benchtime 100ms -test.cpu 1 -test.benchmem
GOMAXPROCS=1 /tmp/dgges.test -test.run '^$' -test.bench '^BenchmarkDggesNetlibControl$/implementation=Netlib$' -test.benchtime 100ms -test.cpu 1 -test.benchmem
benchstat -col /implementation internal/asm/simdbench/results/dgges-netlib/samples.txt
benchstat -col /implementation internal/asm/simdbench/results/dgges-netlib/recheck.txt
```

Profile separately with the exact single-case selector, 5s benchtime and
`-test.cpuprofile`; add `-test.memprofile` for sorting. Read with
`go tool pprof -top` and `-sample_index=alloc_objects`. Confirm scratch escapes
with `go build -pgo=off -gcflags='plantcontrol.org/v1/gonum/lapack/gonum=-m=2' ./lapack/gonum`
under the same toolchain/experiment.

Original local evidence (not portable storage):
`/tmp/gonum-dgges-bench.5Un0qq`, including both CPU profiles, allocation profile,
compiler diagnostics, validation logs and per-round stdout/stderr.
