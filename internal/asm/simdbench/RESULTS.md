# SIMD optimization follow-up, 2026-09-04

This report compares the improved portable candidates with the previous
portable candidates. It does **not** measure gains over AMD64 assembly or an
end-to-end BLAS/SVD speedup. Production dispatch does not change.

## Results

All rows below have p=0.002, n=6 and zero allocations.

| Candidate | Size | Previous | Improved | Time change |
| --- | ---: | ---: | ---: | ---: |
| f32 Ger | 64×64 | 6.089 µs | 1.221 µs | -79.95% |
| f64 GemvN | 64×64 | 5.396 µs | 1.578 µs | -70.77% |
| f64 GemvT | 64×64 | 9.386 µs | 2.058 µs | -78.07% |
| f64 Ger | 64×64 | 9.202 µs | 2.057 µs | -77.65% |
| f64 DotUnitary | 4096 | 2.408 µs | 1.103 µs | -54.19% |
| f64 Sum | 4096 | 1.893 µs | 565.9 ns | -70.10% |
| f64 CumSum | 4096 | 9.502 µs | 3.109 µs | -67.28% |
| c128 AxpyInc | 4096, inc=2 | 16.000 µs | 9.555 µs | -40.28% |

Short cases are also faster. f64 DotUnitary at n=31 went from 20.81ns to
15.11ns (-27.41%). CumSum went from 46.45ns to 26.69ns (-42.54%).

### Complex dots

The full screening run found a 6–7% ARM64 regression in large complex unitary
dots after integer staging. Exact-width input slices removed it. An independent
eight-sample, 100ms alternating rerun of the final complex dot loops measured:

| Candidate, n=4096 | Previous | Improved | Time change |
| --- | ---: | ---: | ---: |
| c128 DotuUnitary | 9.560 µs | 8.911 µs | -6.79%, p<0.001 |
| c128 DotcUnitary | 8.926 µs | 8.909 µs | -0.18%, p=0.004 |
| c64 DotuUnitary | 5.746 µs | 5.735 µs | -0.19%, p=0.004 |
| c64 DotcUnitary | 5.750 µs | 5.738 µs | -0.22%, p=0.010 |

Treat changes below 1% as no change. Final complex dot short cases are 7–17%
faster.

### Limits

- L2 norm edits caused short-case regressions of approximately 5–6%. We
  discarded them and kept the previous implementations.
- Staging still significantly limits some general-stride and mixed-precision
  cases.

## Setup

- Host: Apple M1 Pro, macOS 26.6.2 (25G83), darwin/arm64.
- Go 1.27.1, GOEXPERIMENT=simd, GOMAXPROCS=1; 128-bit vectors.
- Baseline kernels: 2ed36cad, with the corrected comparison harness commit
  f900637c3357e28324fc8b2cf5e46752e37b1b96.
- Candidate: the source changes in this report.
- Binaries built before timing. No timed compilation.
- Six samples per case at 50ms. Alternate samples reverse the
  baseline/candidate order.
- Both binaries use identical stable benchmark inputs. Analysis uses
  benchstat.
- The full comparison covers 57 symbols at two sizes.

## Reproduce

Build each checkout:

```sh
GOTOOLCHAIN=go1.27.1 GOEXPERIMENT=simd go test -c -o simdbench.test ./internal/asm/simdbench
```

For each sample, run the old and new binaries in alternating order:

```sh
GOMAXPROCS=1 ./simdbench.test -test.run '^$' \
  -test.bench 'BenchmarkCurrentVsSIMD/.*/.*/.*/implementation=simd$' \
  -test.benchmem -test.benchtime=50ms -test.count=1
```

## AMD64 evidence

Windows/AMD64 Go 1.27.1 compiler listings show:

- The selected AVX256 real strided loops use integer MOVL/MOVQ lane transfers,
  not legacy MOVSS/MOVSD.
- Complex AXPY builds alpha vectors before the loop. The multiply-add helper
  inlines.
- The inspected scalar SSE moves are gone from those vector loops.
- The f64 Sum AVX256 clone has a 40-byte nosplit frame, not the previous
  264-byte frame.
- Exact-width scratch is compiler-specialized stack storage. Persistent
  allocation checks cover every candidate.

Scalar tails, reductions, mixed-precision conversion and prefix arithmetic
need more work.

The AMD64 benefit is not measured here. Run the corrected
[Windows comparison](README.md) on the issue's i7-1270P. Use the original
affinity and flags. Compare current assembly with the new candidates. Keep
assembly as the production path until native evidence supports a change.
