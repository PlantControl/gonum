# Go SIMD development findings

Checked 2026-09-04 (America/Chicago). These are compiler-development findings,
not Gonum benchmark results or a release commitment. The supported experiment
for this work is still Go 1.27.1 with `GOEXPERIMENT=simd`.

## Source snapshot

Live Go refs at inspection:

- `dev.simd`: `22fde194033d4d969811153efbc06d71f95ee819`.
- `master`: `714d9afb66994e987aa715e088279e75efcec315`.
- `release-branch.go1.27`: `2ee6421c51553e7164590445f96b123a858c1f4a`.

Inspect [master changes](https://go-review.googlesource.com/q/project:go+branch:master+simd)
and [dev.simd](https://go.googlesource.com/go/+/refs/heads/dev.simd). Much
newer SIMD performance work goes directly to master.

## Patterns to use now

- Keep a matrix output tile in vector accumulators across the inner dimension.
  Reuse each scalar broadcast across multiple strips. The experimental
  [SVE SGEMM kernel, CL 827812](https://go-review.googlesource.com/c/go/+/827812)
  (`6b8e07c91f38e60be2d8a5fe09a82af557a23fef`, open) uses sliced rows,
  `j+vl <= len(row)`, explicit `s[off:off+vl]` loads and incrementally
  advanced offsets. Its SVE predicate-hoisting trick is not a NEON
  optimization.
- Hoist invariant broadcasts, masks and shuffle constants explicitly.
  [SIMD loop-invariant-code motion, CL 803220](https://go-review.googlesource.com/c/go/+/803220)
  (`ecafbe48e206bd28e53385cb78e4448e1a5eb1b5`) is still open/WIP.
- Keep hot arithmetic in vectors. The Go 1.27.1
  [FromArch implementation](https://go.googlesource.com/go/+/refs/tags/go1.27.1/src/simd/tofrom_amd64.go)
  needs inlining to remove conversion code. Check generated instructions
  before you put a bridge inside a loop. A narrow whole-kernel leaf can avoid
  repeated stack copies.
- Select unrolling from measured register pressure and loop
  instructions. Extra accumulators can hide arithmetic latency, but can also
  cause spills. When you use fused arithmetic or change the reduction order,
  test floating-point cancellation, extreme magnitudes and fallback paths.

## Compiler and API changes to recheck

| Change | Status at inspection | Consequence |
|---|---|---|
| [CL 825146](https://go-review.googlesource.com/c/go/+/825146), `9cb51e7a659ee21720d79e84309b7add66f9b40d` | Open | VEX encodings for spills, moves, and scalar transfers address [AVX/SSE transition penalties](https://github.com/golang/go/issues/80835). Retest integer-staging workarounds after it ships. |
| [CL 824624](https://go-review.googlesource.com/c/go/+/824624), `fcc582225c8550b3663cce3b2e68376bc01ebc2b` | Open | Packed FMA231 accumulation can remove copies caused by FMA213's overwritten multiplicand. This does not also implement embedded memory broadcast. |
| [CL 827904](https://go-review.googlesource.com/c/go/+/827904), `d8d13647598ede1b01de5c607f1d98bcce8dee21` | Open | NEON broadcast lowering removes redundant zero-vector initialization and scalar insertion before DUP. |
| [CL 827424](https://go-review.googlesource.com/c/go/+/827424), `ba7fdba26bb5f9698a0ccd719c49d5ba2d1327a2` | Merged on master | Float archsimd `ReduceSum` uses pairwise adds and low/high-half reductions. These arithmetic trees can already be expressed with Go 1.27.1 intrinsics. |
| [CL 827464](https://go-review.googlesource.com/c/go/+/827464), `501c052c3e0220d2b1c3018ae2da9dbe1e049659` | Open | Exposes portable float `ReduceSum`; absent from Go 1.27.1. |
| [CL 817020](https://go-review.googlesource.com/c/go/+/817020), `0cc58e9616765a4c9211f96ea462ab091216bff4` | Open | Adds a NEON variant without hardware PMULL, avoiding whole-program portable SIMD emulation on affected ARM64 hosts. |
| [CL 768264](https://go-review.googlesource.com/c/go/+/768264), `ab1d580d957a77eaf35f4ef93017bd880c532013` | Open on dev.simd | Experiments with high AVX512 registers to avoid overlap with scalar SSE registers. |

### Missing operations

The installed Go 1.27.1 portable API has no float32-to-float64 widening, float
horizontal reductions or complex lane permutations. Keep the measured
widening/permutation leaves until a portable operation and its
generated code justify replacement. The new reduction API is not a
prefix-scan API, so it alone does not remove cumulative-sum staging.

## Compatibility gate

[Portable SIMD](https://github.com/golang/go/issues/78902) is accepted as an
experiment. [Default AMD64 enablement](https://github.com/golang/go/issues/78979)
and the [CPU-feature vet proposal](https://github.com/golang/go/issues/76175)
are on hold. [SVE](https://github.com/golang/go/issues/79781) has active
implementation work, but the Apple M1 benchmark host does not have it. Do not
remove experiment or CPU-feature guards for proposal status alone.

For each new release, verify the actual API and feature contracts. Then
compare the same source and benchmark harness on native ARM64 and AMD64, with
supported vector widths and emulation. Remove a workaround only when numerical
tests, generated-code inspection and repeated timings support it.
