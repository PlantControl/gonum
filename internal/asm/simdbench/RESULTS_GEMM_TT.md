# TT GEMM register blocking

## Summary

Scalar 2x4 register blocking makes left-side Dormlq faster on ARM64 (see
Caller below). Some tiny and awkward tile shapes are slightly slower (see
Limits below).

## Setup

- Baseline: `fec3d70fd8b8d6025a10571da104773da32b5e06`.
- Native host: Apple M1 Pro, darwin/arm64, Go 1.27.1, GOEXPERIMENT=simd,
  -pgo=off.
- Production activation stays behind the existing ARM64 experiment gate.
- There is no AMD64 performance claim or dispatch change.

## Consumer

- Left-side Dormlq calls row-wise Dlarfb. Its two TT products have shapes
  `(n,ib,m-i-ib)` and `(m-i-ib,n,ib)`.
- Ordinary QR factorization and Dgelqf are not the relevant TT consumers.
- A three-second baseline profile of Dormlq(Left), m=256/n=64/k=160, gave
  dgemmSerialTransTrans 86.16% of sampled CPU cumulative.
- Its inlined AxpyInc had 70.93% flat.

## Mechanism

- The candidate keeps a 2x4 output tile across the inner dimension.
- This is scalar register blocking, not a new SIMD intrinsic kernel.
- Both real precisions are generated from the same source.
- Each output keeps increasing-k accumulation order, alpha scaling before each
  term, and the exact-zero skip.
- Active C/A or C/B overlap rejects the helper. Tails use the previous kernel.
- The screening boundary is m>=2, n>=4, k>=16. The code checks it before the
  helper call.

Native disassembly shows:

- eight accumulators in F1-F8 and scalar FMADDD updates;
- no accumulator spills or calls in the full-tile inner loop;
- bounds checks for A and B accesses. Their removal is a future hypothesis to
  measure, not a result here.

## Method

- Independent detached baseline worktree with identical persistent benchmark
  harnesses.
- Prebuilt binaries, ten alternating baseline/candidate samples.
- No builds, tests or profiles ran at the same time as acceptance timing.
  Normal desktop activity continued.
- Main runs use GOMAXPROCS=1. Public TT dispatch was also checked at four
  workers.
- All accepted timing cohorts reported zero allocations.
- Durations: leaf 100ms/sample; caller and non-TT controls 200ms/sample.

## Results

### Caller

Time, not throughput; p<.001, n=10 each:

| Left Dormlq m/n/k | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| 128/16/95 | 206.3 us | 149.1 us | -27.71% |
| 256/32/127 | 1081.6 us | 664.8 us | -38.54% |
| 256/64/160 | 2.531 ms | 1.356 ms | -46.45% |

The caller fixture:

- factors its input outside timing;
- alternates Q/Q-transpose applications;
- resets every 16 calls outside timing;
- checks finite scaled Frobenius norm preservation;
- includes full and partial reflector blocks.

### Leaf

The final leaf controls include both precisions and padding 0/7.

- 2x4x16 activation case: 53-55% faster.
- 16x32x224 and 224x16x32: 53-55% faster.
- 224x64x32: 54-62% faster.
- 224x64x32 at four workers: 35.6-37.2% faster.

Before the guard was hoisted, a broader shape cohort showed 33-71% gains in
square, rectangular and tail-heavy cases. Those numbers are exploratory. They
do not replace the final guarded-candidate controls.

## Limits

- The first version made tiny fallback calls about 3% slower. The hoisted guard
  removed most of that. Final 4x4x4 controls were inconclusive in both
  precisions. A separate longer recheck found a 0.47% Sgemm cost in one case.
- The awkward 3x5x17 tile is 1.4-2.4% slower (roughly 3-6 ns) in both
  precisions. Most of its work is in the retained tails. We accept this small
  absolute cost for the much larger caller gains.
- One float32 k=15 padded fallback control is 0.93% slower. Other k=15 controls
  were inconclusive. The change is not a universal improvement.
- Existing medium NN, TN and NT controls were inconclusive: 147.1->147.2 us,
  147.0->147.2 us, and 115.8->115.8 us. These legacy controls use random finite
  inputs and repeated additive updates, unlike the reset caller fixture.
- These cohorts show no application-wide speedup, new parallel cutoff, or
  native AMD64 result.

## Reproduce

- Benchmarks: BenchmarkDgemmTT, BenchmarkSgemmTT, BenchmarkDormlqLeft.
- Runner: go-optimisation/scripts/compare_benchmarks.py.
- Analysis: benchstat.
- Local raw samples, binary hashes, codegen, profile and metadata:
  `/tmp/gonum-tt.yWX4lf/`.
- Final cohorts: `final-leaf`, `final-consumer`, `four-workers`,
  `non-tt-controls`.
- `tiny-recheck` is the longer tiny-call check.
- These temporary artifacts are not repository fixtures.

## Correctness

Persistent tests cover:

- both precisions;
- tile boundaries and tails;
- offset slices and padding;
- shared disjoint backing and active overlap rejection;
- exact scalar-order results;
- exceptional arithmetic, overflow/cancellation and signed zero;
- public dispatch.

Independent native Netlib tests exercise active tiles and
transpose/conjugate-transpose combinations with exact binary inputs.

Signed-zero difference against Netlib:

- An exact-cancellation case gives a different zero sign. The unchanged
  baseline shows the same difference.
- The cause: Netlib scales after the reduction. Gonum scales each term.
- The oracle allows only this zero-sign difference in active outputs. Nonzero
  outputs and padding stay exact.
- Scalar-order tests still check zero signs against the established Gonum
  behavior.

## Validation

These checks passed:

- full Go 1.27.1 SIMD suite;
- native Netlib BLAS suite;
- focused GEMM race tests;
- default-Go affected packages;
- combined safe/noasm/bounds affected packages;
- BLAS/testlapack vet, formatting and diff checks;
- minimum Go 1.24.0 affected-package tests;
- focused portable-emulation tests (`GODEBUG=simd=0`).

The generated Sgemm source was reviewed together with its generator mappings.
No cross-compilation was requested.

## Next

- Profile SYRK under Cholesky from the committed TT revision.
- Other TT opportunities are tail cost and bounds-check overhead.
- Do not combine them with this milestone without a new independent comparison.
