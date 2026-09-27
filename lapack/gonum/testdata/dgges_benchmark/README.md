# DGGES strengthening and reference benchmarks

This directory contains the reproducible comparison for the Go DGGES driver and
its block-reordering path. The baseline is
`a78cf83eff160df0e423b08af0c7642ade964150`; the candidate is the accompanying
working-tree implementation. No numerical algorithm or tolerance in the solver
was changed.

## Changes and scope

`Dtgsen` now owns scratch storage for the full reordering operation and passes it
through `dtgexc`, `dtgex2`, and `dtgsy2`. Previously, small local arrays escaped
through configurable BLAS calls on every adjacent swap. The scratch is cleared
before reuse. The exported signatures, minimum and preferred workspaces, BLAS
backend selection, numerical operations, and rejection/cleanup paths are
preserved. Standalone calls still allocate their own scratch. A standalone
`Dtgex2` now reserves 1536 bytes even for a 1-by-1 swap; the intended benefit is
amortizing scratch across the many swaps in a reordering operation.

The DGGES documentation now correctly states that selecting either eigenvalue
of a conjugate pair selects both. Existing benchmark invocations now fail when
the solver fails, rather than timing an unsuccessful result.

Oracle reconstruction checks now normalize before multiplication: the previous
absolute floor could accept an incorrect result on a tiny pencil. The ordinary
relative eigenvalue comparison remains in use for finite spectra. Singular
pencils use chordal distance between homogeneous eigenvalue pairs, which is
well-defined at infinity. Right-vector tests compare selected subspace
projectors, avoiding assumptions about basis signs or rotations.

This is a focused scratch-lifetime change with broader dynamic numerical
coverage, not a claim of complete source-level parity for every branch of the
transitive LAPACK stack. QZ iteration limits, exceptional shifts, and scaling
algorithms were not changed. No new mathematical solver mismatch was found in
the exercised cases.

## Environment and interpretation

Measurements use an Apple M1 Pro, darwin/arm64, Go 1.26.4, and the default pure-Go
BLAS/LAPACK backend. The oracle is Homebrew's `lapack/3.12.1` package, with
reference BLAS linked to reference LAPACK. Its runtime `LAPACKE_ilaver` and
library compatibility metadata report **3.12.0**; both identifiers are retained
rather than treating the package name as the runtime version. The Go source
cites LAPACK 3.12.1 commit `6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca`.

All timing processes use `GOMAXPROCS=1`, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1`. The linked reference BLAS
is not OpenBLAS or Accelerate. Results are specific to this hardware/backend and
do not establish performance on other architectures or optimized vendor BLAS.

Both implementations reuse queried workspaces and receive the same mathematical
inputs in their native layout: row-major for Go and column-major for Netlib.
Input restoration is timed equally. Layout conversion and workspace allocation
are excluded. Netlib runs through `LAPACKE_dgges_work`, avoiding the allocating
row-major convenience wrapper. The cgo call cost is included. Go's allocation
counters do not measure C allocations.

## Reproduction

On an Apple Silicon Mac with Go, Python 3.12 or later, and Homebrew LAPACK:

```sh
uv run python3 lapack/gonum/testdata/dgges_benchmark/run.py --output /tmp/dgges-results
uv run python3 lapack/gonum/testdata/dgges_benchmark/summarize.py /tmp/dgges-results
```

The runner extracts the exact baseline to a temporary directory and copies the
same benchmark harness and oracle bridge to it. It performs six samples of each
case at `100ms` per sample, alternating baseline/current process order. It records
the implementation patch, versions, threading settings, library links and
hashes. The summarizer retains every case and reports medians, paired ratios,
paired ranges, and 95% percentile-bootstrap intervals. Six short samples provide
exploratory uncertainty estimates; a small timing difference is not a durable
speedup claim.

The matrix covers sizes 10, 50, 100, and 200, with no vectors, right vectors, or
both vectors, and with/without ordering. Five pencil families yield 120 cases per
implementation:

- Random dense, with negative-real-part selection.
- The same dense distribution scaled by `1e-200`, with actual reordering.
- Singular triangular B with finite and infinite eigenvalues; unit-circle selection.
- Conjugate pairs at radii `1 ± 1e-7`, transformed by orthogonal equivalences.
- Reduced discrete Riccati pencils, also transformed by orthogonal equivalences.

The latter two use unit-circle selection, matching the policy used in
controlsys's discrete Riccati solver. Its descriptor pole path uses no vectors
and no sorting.

## Verification

The source diff was checked for unchanged arithmetic, argument validation,
workspace queries, movement/splitting branches, and failure returns in the four
changed reordering/Sylvester routines. Reusing memory is the only execution
change. The retained tests cover:

- Poisoned and reused scratch across all four adjacent block-size combinations.
- A bounded-allocation regression at sizes 20 and 100 with many selected blocks.
- Existing Netlib differential tests for DGGES, DHGEQZ, DTGSEN, DTGEX2, DTGEXC,
  DTGSY2 and DTGSYL, including rejected swaps, block splitting, scaling and
  minimum workspaces.
- All 120 benchmark configurations: status, selected dimension, homogeneous
  eigenvalues, Schur structure, orthogonality, reconstruction where both vectors
  are available, and selected right subspaces.
- A selection-at-infinity case where both libraries return a sorting failure;
  the partial decompositions still satisfy reconstruction checks.
- Scale-invariance of the residual check at `1e-300`, 1, and `1e300`.

Measured reconstruction errors in the expanded suite were at most `2.46e-14`
for Go and `2.18e-14` for Netlib, using maximum-entry relative residuals. The
controlsys package was tested with a temporary module replacement pointing at this
checkout; its files and dependency pin were not modified.

## Recorded results

Full raw samples and per-case medians are in `results/`; `summary.csv` retains
all 120 configurations. Representative right-vector, ordered results (median
milliseconds across six samples):

| Pencil | n | Baseline Go | Current Go | Netlib | Go allocations, before → after |
|---|---:|---:|---:|---:|---:|
| Dense | 100 | 11.137 | 11.048 | 6.644 | 3249 → 1 |
| Singular | 100 | 0.771 | 0.680 | 0.498 | 4488 → 1 |
| Riccati | 100 | 14.441 | 13.997 | 7.380 | 6064 → 1 |
| Dense | 200 | 77.107 | 76.577 | 49.130 | 13595 → 1 |
| Singular | 200 | 3.982 | 3.607 | 2.726 | 17956 → 1 |
| Riccati | 200 | 88.738 | 88.237 | 49.708 | 23389 → 1 |

The firm improvement is memory use: reordered calls use one 1536-byte scratch
allocation rather than allocating per swap. For the 200-by-200 Riccati case,
this replaces 3,824,832 bytes per call. The unsorted cases remain allocation-free.
The bounded-allocation test fails on the baseline with 220 allocations at n=20
and 5100 at n=100, and passes on the candidate.

Singular-pencil right-vector sorting shows a consistent reduction: paired timing
ratios are 0.879 at n=100 (bootstrap interval 0.872–0.886) and 0.906 at n=200
(0.902–0.910). Dense and Riccati differences are generally much smaller and their
intervals often include no change. Netlib remains faster in the representative
cases above. This change does not eliminate the QZ or matrix-update CPU cost.

Across all cases, the geometric mean of paired ratios is 0.9877 for sorted calls
and 1.0244 for unsorted calls. Small-case outliers include apparent regressions;
see the individual samples and the longer follow-up measurements rather than
interpreting these aggregate values as universal speedups.

Validation commands and outcomes:

```sh
go test ./lapack/gonum ./lapack/testlapack ./mat
# PASS

go test -tags netlib ./lapack/gonum \
  -run 'Test(Dgges|Dhgeqz|Dtgsen|Dtgex2|Dtgexc|Dtgsy2|Dtgsyl|PencilResidual)' -count=1 -v
# PASS

go test -tags 'netlib,bounds,noasm' ./lapack/gonum \
  -run 'Test(Dgges|Dhgeqz|Dtgsen|Dtgex2|Dtgexc|Dtgsy2|Dtgsyl|PencilResidual)' -count=1
# PASS

go test -tags safe ./lapack/gonum \
  -run 'Test(Dgges|Dhgeqz|Dtgsen|Dtgex2|Dtgexc|Dtgsy2|Dtgsyl)' -count=1
# PASS

go test -race ./lapack/gonum \
  -run 'Test(Dgges|Dhgeqz|Dtgsen|Dtgex2|Dtgexc|Dtgsy2|Dtgsyl)' -count=1
# PASS
```

Controlsys: 26 descriptor, generalized-pole, DARE, and CARE/DARE-consistency tests
passed using a temporary copy of its go.mod/go.sum with
`gonum.org/v1/gonum` replaced by this checkout. The baseline also passes the
expanded numerical oracle fixtures; this is not presented as a numerical fix
for a previously failing solver case. The candidate's independent failure and
scale checks and allocation regression are retained in source.

## Longer follow-up and limitations

`focused/` retains six one-second samples for five apparent regressions from the
initial run. Reproduce with the runner's repeated `--case` option and
`--benchtime 1s`; the exact case list is recorded in `focused/metadata.json`.

| Case | Baseline median | Current median | Paired ratio | Bootstrap interval |
|---|---:|---:|---:|---:|
| n=10 dense, no vectors, sorted | 52.60 µs | 54.08 µs | 1.043 | 0.866–1.134 |
| n=10 Riccati, both vectors, unsorted | 35.60 µs | 42.94 µs | 1.123 | 1.017–1.339 |
| n=10 Riccati, right vectors, unsorted | 24.74 µs | 26.26 µs | 1.092 | 0.867–1.175 |
| n=10 tiny, right vectors, unsorted | 29.25 µs | 29.54 µs | 0.987 | 0.921–1.140 |
| n=50 Riccati, no vectors, sorted | 2.271 ms | 2.487 ms | 1.096 | 1.088–1.122 |

The n=50 sorted slowdown persists across all six paired samples despite reducing
1719 allocations (279872 bytes) to one (1536 bytes). The small unsorted Riccati
case also remains slower, with substantial variation; it does not execute the
changed reordering path, and its timing difference has not been causally
isolated. The candidate is therefore an allocation and validation improvement,
not a uniform CPU-speed improvement. These measurements are retained as material
tradeoffs for review. Ratios are medians of paired sample ratios, which need not
equal the ratio of the two displayed medians.

Full-stack source parity, optimized vendor BLAS comparisons, and performance on
other architectures are not established by this work. Numerical tolerances and
QZ convergence policy remain unchanged. The optional oracle/benchmark bridge is
currently restricted to darwin+cgo and the existing Homebrew LAPACK paths; the
Go implementation and its normal regressions have no such dependency.
