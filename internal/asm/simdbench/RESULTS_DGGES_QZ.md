# Dgges QZ left-update indexing

## Summary

- Accepted: the left-only production delta. It bounds the six H/T row windows
  in `doQZSweepDouble` once per outer iteration, with one explicit shared width.
  Right-side H/T and Q/Z indexing does not change.
- Rejected: two broader row-window experiments. Compact n=256 sweep controls
  regressed.
- Dgges gains are modest. Allocations do not change.

## Baseline

Baseline: `57149f493cf53011dfea6befeded3f82fcf9416d`. Before the edits, we
fetched it and confirmed that it equals origin/codex/arm64-simd-blas.

- An isolated detached worktree gets only the identical new test/benchmark
  harnesses, not the proposed production change.
- The change adds no public API, new intrinsic, architecture dispatch, storage
  allocation, workspace contract or algorithm.

The earlier vector n=256 CPU profile had 5.48 s of samples:

- doQZSweepDouble was 46.53% flat / 47.81% cumulative.
- Line attribution showed repeated indexed left H/T and right H/T/Q/Z updates.
- Dgghrd's remaining A/B rotations are also expensive. But their outputs feed
  the next elimination, so they cannot be delayed like independent Q/Z
  accumulation. This milestone does not change them.

## Setup

Host and toolchain:

- Native Apple M1 Pro, darwin/arm64 v8.0, macOS 26.6.2 (25G83).
- AC attached at 80% battery, normal desktop background activity, no
  affinity/frequency pinning.
- Primary toolchain: Go 1.27.1, GOEXPERIMENT=simd, netlib build tag, -pgo=off,
  empty GOFLAGS, GOMAXPROCS=1, test.cpu=1.
- Baseline and candidate use the default Gonum BLAS.
- Oracle: Homebrew reference LAPACK package 3.12.1 (runtime ILAVER reports
  3.12.0), not Accelerate or OpenBLAS.

Method:

- Each primary experiment uses ten serial alternating baseline/candidate rounds,
  100 ms per case.
- Binaries are built before timing. Builds, profiles and tests do not overlap
  measurement.
- Benchstat is `golang.org/x/perf v0.0.0-20260312031701-16a31bc5fbd0`.
- Intervals are benchstat's 95% median confidence intervals.

Statistical limits:

- Many per-case tests can give incidental significance.
- Unweighted aggregates are not application speedups.
- An inconclusive change does not show equivalence.

### Harnesses

The persistent private-sweep harness covers:

- n=3/4/5/32/64/128/256;
- all four Q/Z on/off combinations;
- compact storage, and also n=256 stride=259.

It uses deterministic finite upper-Hessenberg H and triangular T. It resets
inputs and optional vectors inside timing, and checks finite outputs after
timing. Setup and buffer allocation are outside timing. It measures one forced
double sweep plus reset, not convergence or the public entry-point cost.

The unchanged DggesControl harness is the representative public caller:

- form: no vectors/sorting;
- vectors: both vectors/no sorting;
- left: both vectors, left-half-plane sorting;
- unit: both vectors, unit-disk sorting.

## Experiments and generated code

1. **Independent high/low row slices on the left, three-element windows on the
   right.** This removed some indexing/checks, but left some checks in the left
   loop. The symbol grew 3792 to 3904 bytes. Compact n=256 leaf none/q/z
   increased 3.17% / 1.50% / 1.87%. Dgges unit n=32 increased 2.92%
   (p=0.023). Dgges was mostly inconclusive, despite several leaf wins.
   Rejected.
2. **Explicit equal width on all six left slices, with the right windows.**
   This removed all indexed checks inside the left loop. Symbol size returned to
   3792 bytes. Dgges n=32-128 had several 1-3% gains. But compact n=256
   none/q/z still increased 2.15% / 0.96% / 1.52% (p<=0.004). Rejected.
3. **Only the explicit-width left change, with all original right indexing.**
   - All left-loop bounds checks are removed.
   - The ordered FMADDD/FMSUBD arithmetic stays the same.
   - There are no hot-loop calls, stack traffic or floating-point spills.
   - The stack frame stays 368 bytes. Symbol size is 3872 bytes (+80).
   - Right-side generated code returns to the baseline scalar indexed shape.

The right-window variants changed loads (including FLDP), checks, scheduling and
code layout together. Thus the experiments do not isolate one instruction as
the cause of the compact regression. No platform/stride threshold was added to
hide those cases.

## Left-only primary results

- All primary leaf cases stay at 0 B/op and 0 allocs/op.
- All Dgges allocation counts and bytes are the same as the baseline.
- The left-only cohort has no statistically significant adverse time result.

| Dgges case | Baseline | Left-only | Time change |
| --- | ---: | ---: | ---: |
| form n=32 | 196.6 us +/-6% | 192.7 us +/-0% | -2.01%, p<0.001 |
| form n=64 | 1.101 ms +/-3% | 1.075 ms +/-0% | -2.30%, p<0.001 |
| form n=128 | 7.720 ms +/-0% | 7.536 ms +/-1% | -2.39%, p<0.001 |
| form n=256 | 116.5 ms +/-1% | 114.6 ms +/-1% | -1.59%, p=0.001 |
| vectors n=64 | 1.655 ms +/-2% | 1.629 ms +/-0% | -1.62%, p<0.001 |
| vectors n=128 | 13.53 ms +/-0% | 13.35 ms +/-1% | -1.30%, p<0.001 |
| vectors n=256 | 169.8 ms +/-3% | 169.0 ms +/-0% | -0.48%, p=0.005 |
| left n=64 | 2.401 ms +/-6% | 2.374 ms +/-1% | -1.14%, p=0.003 |
| unit n=64 | 4.296 ms +/-2% | 4.255 ms +/-1% | -0.97%, p=0.003 |

Other Dgges cases:

- Vectors n=32: -1.60% (p=0.043).
- Left n=32 and unit n=128/256: inconclusive.
- Small sorting signals: left n=128 -0.72% (p=0.019), left n=256 -0.62%
  (p=0.035), unit n=32 -1.79% (p=0.043).
- Do not treat these marginal or sub-percent signals as broad application gains.

Representative sweep results:

| Sweep | n=64 | Compact n=256 | Padded n=256 |
| --- | ---: | ---: | ---: |
| no-vector | -5.54% | -1.72% | -7.35% |
| both-vector | -2.64% | -1.13% | -3.25% |

Compact q-only n=256 is inconclusive, not a proven win. The raw evidence
includes every mode, size, allocation metric and inconclusive case.

## Final confirmation

The Go optimization skill's interleaved caller gates and separate
numerical/code-generation reviews rejected the larger variants. They kept the
smaller, measured improvement.

A separate longer recheck uses ten alternating 500 ms rounds with the same SIMD
baseline/left-only binaries. All six changes have p < 0.001:

| Dgges case | Baseline | Left-only | Time change |
| --- | ---: | ---: | ---: |
| form n=64 | 1.100 ms | 1.075 ms | -2.25% |
| form n=128 | 7.718 ms | 7.547 ms | -2.21% |
| form n=256 | 116.0 ms | 114.6 ms | -1.20% |
| vectors n=64 | 1.653 ms | 1.630 ms | -1.38% |
| vectors n=128 | 13.52 ms | 13.36 ms | -1.17% |
| vectors n=256 | 170.0 ms | 169.1 ms | -0.52% |

- Median intervals round to +/-0%, except candidate vectors n=64 (+/-1%).
- These results confirm modest gains. The n=256 vector change is still
  sub-percent and does not materially close the reference-library gap.
- We keep the primary and recheck data separately. We do not pool them.

### Default toolchain cohort

Default Go 1.26.4, GOEXPERIMENT empty, no netlib tag, same -pgo=off/P=1
settings, ten alternating 100 ms rounds:

- Form n=32/64/128/256 improves 1.93% / 2.59% / 2.58% / 1.12%.
- Vectors n=64/128 improve 1.51% / 1.57%.
- Vectors n=32/256 and every sorting case are inconclusive.
- There is no significant adverse time result.
- Allocations and bytes do not change.

Compare within each toolchain cohort. Do not compare absolute timings across
toolchains.

## Numerical and compatibility evidence

Scoped source pin: Reference-LAPACK v3.12.1,
`6ec7f2bc4ecf4c4a93496aa2fa519575bc0e39ca`, SRC/dhgeqz.f, label 230.
We traced the reachable stack. This work reviews the changed row-index delta,
not every reachable translated routine. It does not establish full transitive
Dhgeqz/Dgges parity.

Index mapping:

- The explicit width is ilastm-j+1.
- Each row starts at row*ld+j and covers the same old inclusive j..ilastm
  columns.
- Local range index k maps exactly to old column j+k.
- H expressions/stores come before T expressions/stores, in the same order.

These do not change: floating-point association, exact-zero stores,
Householder/shift state, convergence counters, final Givens operations and
optional-vector paths. No numerical tolerances were relaxed.

Bounds:

- For an invalid direct private-helper call, two-stage slices can in theory
  extend to capacity.
- Valid public callers need full minimal backing (n-1)*ld+n. All six windows
  are inside that length.
- No valid public panic or failure-output contract changes.

### Tests

New persistent TestDoQZSweepDoubleStridedRanges covers:

- minimum-width bulges, n=3/4/5/8/33;
- full/interior active ranges, nonzero first updated row, different
  ilast/ilastm;
- four Q/Z combinations;
- distinct padded strides with minimum legal backing lengths;
- padding canaries and untouched leading/trailing envelopes;
- bitwise compact-versus-padded results.

Unexpected nonfinite values fail. This is same-kernel geometry coverage, not an
independent arithmetic oracle.

Existing native Dhgeqz/Dgges oracle suites cover optional-vector modes,
convergence/failure, real/complex blocks, scaling, active subranges, sorting
and deterministic Dgges benchmark pencils through n=256. Shared/internal tests
cover workspace behavior.

Passed for left-only production:

- Full Go 1.27.1 SIMD repository suite.
- Full SIMD+Netlib lapack/gonum test binary.
- Full default Go 1.26.4 lapack/gonum test binary.
- Go 1.24.0 lapack/gonum and mat tests.
- Go 1.27.1 SIMD safe and noasm lapack/gonum and mat tests.
- SIMD+Netlib race tests for Dhgeqz, Dgges and the direct sweep regression.
- Independent Sol numerical and generated-code reviews.

## Limits

- No native AMD64 measurement or cross compilation was done.
- This is generic Go code. The production change needs no experimental SIMD API.
- We did not run a new Netlib speed comparison for this small delta. Do not
  derive a new Netlib-relative speed claim from the before/after results.
- The remaining dependency-sensitive A/B rotations are a separate future target.
  This work does not change those rotations, QZ shifts or convergence policy.

## Raw evidence

Paired samples:

- Rejected broad slices: [baseline](results/dgges-qz/slices-baseline.txt),
  [candidate](results/dgges-qz/slices-candidate.txt).
- Rejected explicit-width plus right windows:
  [baseline](results/dgges-qz/width-baseline.txt),
  [candidate](results/dgges-qz/width-candidate.txt).
- Left-only primary: [baseline](results/dgges-qz/left-baseline.txt),
  [candidate](results/dgges-qz/left-candidate.txt).
- Longer recheck: [longer baseline](results/dgges-qz/recheck-baseline.txt),
  [longer candidate](results/dgges-qz/recheck-candidate.txt).
- Default toolchain: [default baseline](results/dgges-qz/default-baseline.txt),
  [default candidate](results/dgges-qz/default-candidate.txt).

Actual binary SHA-256 values:

| Binary | SHA-256 |
| --- | --- |
| SIMD baseline | `ecffb39e8a5a3e5695d1b9fc9bae1bf9deccea5bb7c4d0a4b95e679e57bff956` |
| Broad slices | `69e513ce7ba1bb96895eb525962058b7d2fac92d15875ab25fa2b12c38f96446` |
| Width plus right windows | `5dbdaed2f7fec6002d7c86717933b803fc53a65ee9dbb126d5cb0a865f229a62` |
| Left-only | `4ee81b7188963165bbc135a7537579a947d5f18508a68f7ff022a1a05f3af225` |
| Default baseline | `dcd05db79fc196dbc176be5b8ac4dfae8e1ae78137346097fd64179eacd1136d` |
| Default left-only | `7441aed0d14f87f5defc3e766c11d33eb703ed685afb0c2606d8a5278d39d8f5` |

- We hardened the new geometry test after the first build. The V2 and left-only
  binaries contain the hardened checks.
- Benchmark source is identical in all builds.
- The local worktree, raw rounds, runner metadata, diagnostics and test logs
  are in `/tmp/gonum-qz-tune.JEqPVt`.

## Reproduce

Build two isolated revisions with the same benchmark harness:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -pgo=off -tags netlib \
  -c -o /tmp/qz-candidate.test ./lapack/gonum
GOMAXPROCS=1 python3 /path/to/go-optimisation/scripts/compare_benchmarks.py \
  --baseline /tmp/qz-base.test --candidate /tmp/qz-candidate.test \
  --bench '^Benchmark(DhgeqzSweepDouble|DggesControl)$' \
  --rounds 10 --benchtime 100ms --cpu 1 --output /tmp/qz-comparison
benchstat internal/asm/simdbench/results/dgges-qz/left-baseline.txt \
  internal/asm/simdbench/results/dgges-qz/left-candidate.txt
```

- Longer caller recheck: select
  `^BenchmarkDggesControl$/mode=(form|vectors)$/n=(64|128|256)$` at 500 ms.
- Default cohort: build with `GOTOOLCHAIN=go1.26.4 GOEXPERIMENT=` and omit the
  netlib tag. Select `^BenchmarkDggesControl$` at 100 ms.
