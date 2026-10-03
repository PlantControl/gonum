# Issue 10: ARM64 downstream regression

Investigated 2026-09-12 on Apple M1 Pro (darwin/arm64), GOMAXPROCS=8.
[Issue #10](https://github.com/jamestjsp/gonum/issues/10) reports downstream
controlsys regressions between v0.17.7-fork and v0.19.0-fork.

## Summary

- The regressions come from executable layout, not from an incorrect
  parallel-work threshold. Do not revert the threshold to fix them.
- With Go 1.26.4, the first large increase is at
  `ebe982daa841625254b141121265738410ba42ee`, the worker-aware DGEMM dispatch
  change. The preceding commit `7fd5fefe` does not show it.
- A relink with `-ldflags=-funcalign=64` removes almost all of the three
  largest regressions. The code and arithmetic stay the same.
- With default Go 1.27.1 builds, none of the seven reported cases has a
  statistically significant slowdown.
- No production numerical code changed.

## Evidence for a layout cause

- The MatLog_N50 trace has no Dgemv calls, setup included. Its GEMM shapes are
  51×51×51 (setup), 50×50×50 and 36×36×64. All stay serial in both releases,
  because each has fewer than four output tiles.
- The triangular solves include a 64×36 panel with lda=ldb=100 and a 100×50
  solve with lda=100, ldb=50.
- Dtrsm source is the same in both releases. Normalized disassembly has the
  same 888 instructions and the same hot-body operations. The operands that
  differ are addresses of panic strings and types in cold paths.
- Dtrsm starts at 0x100127980 in the baseline and 0x100128410 in the
  candidate. Its position modulo 64 changes from 0 to 16. Other unchanged
  functions also move, so Dtrsm alone is not identified as the cause.
- The global alignment flag is an experiment, not a proposed build requirement.
  The issue still reproduces with the older compiler's default layout.

## Go 1.26.4 reproduction

All values are medians of ten interleaved samples per binary. The release
comparison uses 300ms per case.

| Benchmark | Go 1.26.4 old release | New release | Change |
| --- | ---: | ---: | ---: |
| MatLog_N50 | 335.2 us | 377.7 us | +12.67% |
| D2C_ZOH_N50 | 413.1 us | 453.2 us | +9.69% |
| Stabsep_N100 | 9.030 ms | 9.626 ms | +6.60% |
| Reduce | 46.47 us | 49.19 us | +5.85% |
| DiscretizeZOH | 21.35 us | 22.58 us | +5.74% |
| Modsep_N50 | 1.453 ms | 1.535 ms | +5.63% |

- All six comparisons have p≤0.005.
- The FRD timing regression did not reproduce (p=0.579). Its median
  allocation count rose from 60 to 61.
- Current master also shows the large regressions with Go 1.26.4.

## Go 1.26.4 bisection

Bisection uses 200ms per case.

| Go 1.26.4 revision/build | MatLog_N50 | D2C_ZOH_N50 | Stabsep_N100 |
| --- | ---: | ---: | ---: |
| Old release `f43007ad` | 327.7 us | 409.2 us | 8.857 ms |
| Merge `7fd5fefe` | 330.1 us | 411.9 us | 8.846 ms |
| Worker-aware gate `ebe982da` | 370.1 us | 450.3 us | 9.398 ms |
| Darwin calibration `777029ff` | 369.5 us | 450.2 us | 9.452 ms |
| New release `a78cf83e` | 369.7 us | 451.1 us | 9.421 ms |
| Old release, function alignment 64 | 327.1 us | 408.2 us | 8.892 ms |
| New release, function alignment 64 | 329.2 us | 410.6 us | 8.874 ms |

The bisection finds the trigger in these executables. It does not show that a
threshold reversion is the correct repair. A GEMM reversion loses the confirmed
simulation and regulator gains.

## Go 1.27.1 results

Both release endpoints and current master were rebuilt with Go 1.27.1. The
comparison also has two ablation builds: the new release with the old Dgemv,
and the new release with the old dgemm.go. These builds are experiments, not
proposed source changes.

| Benchmark | Old release | New release | Current master |
| --- | ---: | ---: | ---: |
| MatLog_N50 | 335.4 us | 340.0 us | 338.7 us |
| D2C_ZOH_N50 | 416.1 us | 418.7 us | 419.1 us |
| Stabsep_N100 | 8.672 ms | 8.697 ms | 8.674 ms |
| Simulate_DCMotor | 44.44 us | 38.12 us | 37.99 us |
| Reg_N100_M5_P5 | 80.56 us | 61.20 us | 60.60 us |

- The new release keeps the simulation gain (-14.22%) and the regulator gain
  (-24.02%).
- Allocations fall from 45 to 11 and from 55 to 25, the same as with
  Go 1.26.4. These changes come from Gonum, not from the new compiler.
- Some smaller allocation increases remain. See the full tables.
- No significant slowdown is not proof of equivalence. Smaller effects and
  other workloads are still possible.

## Go 1.27.1 allocator check

Go 1.27 adds size-specialized allocation calls for small objects. See the
[release notes](https://go.dev/doc/go1.27#runtime). One paired comparison uses
the same new-release source, with the feature on (default) and off
(`GOEXPERIMENT=nosizespecializedmalloc`):

- Reduce is 1.88% slower with the feature off (p<0.001).
- MatLog and D2C are 1.25% and 1.41% faster with the feature off (p<0.001).
- Stabsep is inconclusive (p=0.579).

The feature does not explain why the large regressions went away. A toggle
also changes emitted code and layout, so these are whole-program effects. No
persistent GOEXPERIMENT override was set.

## Setup

- Desktop host with no affinity or power-state control.
- Native default Go BLAS/LAPACK, no build tags, no PGO.
- All binaries in a timed comparison were prebuilt. No builds, tests or
  profiles ran at the same time.
- Ten rounds per comparison, with alternating run order.
- Benchstat: `golang.org/x/perf v0.0.0-20260312031701-16a31bc5fbd0`.
- controlsys fixture: `54d5d6b119c8b4351db0cd0bc0ebb4122bde94d4`. Each
  comparison replaces only its Gonum dependency.
- Gonum endpoints (commits, not annotated tags):
  `f43007ad8a2d208bc8f47c338e85f05381d9643b` and
  `a78cf83eff160df0e423b08af0c7642ade964150`. Current master:
  `1c42629c5cb53235d9ae074bb1be6da6471771da`.

## Limits

- Results from the two toolchain sessions are descriptive. Comparisons inside
  one session are interleaved.
- An interrupted Go 1.26.4 ablation run was discarded.
- No other platforms or worker counts were measured.
- This investigation does not replace the original 236-case suite. It does not
  show that the release is free of regressions on all workloads.

## Evidence files

In [results/issue10](results/issue10):

- `go1.26.4/benchstat.txt`: release and master comparison, 10×300ms.
- `go1.27.1/benchstat.txt`: release, master and ablation comparison, 10×300ms.
- `allocator-go1.27.1/benchstat.txt`: allocator on and off, 10×300ms.
- `bisect-go1.26.4/benchstat.txt`: five commits and two alignment builds,
  10×200ms. It has all uncertainty and allocation results.
- `matlog-trace.txt`: call-shape trace of one MatLog iteration, setup included.
  Its timing is not performance evidence.
- `code-comparison.txt`: normalized instruction counts and function addresses.
  The diff counts include cold panic-address code.

The raw samples, CPU profiles and the run harness are not kept. The harness
needed a local controlsys checkout and a `replace` directive for the Gonum
dependency.

## Kernel benchmark

`BenchmarkDtrsmSmallRectangular` in `blas/gonum/dtrsmbench_test.go` covers the
traced solve shapes with a bounded, known solution. It times the RHS restore
and the public solve, and it checks the fixture outside the timer. It guards
the kernel. It does not replace the downstream executable for layout studies.
