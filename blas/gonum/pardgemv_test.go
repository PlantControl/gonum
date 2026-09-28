// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math/rand/v2"
	"runtime"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/floats"
)

// TestDgemvParallel exercises the row-block parallel Dgemv path against the
// serial GemvN/GemvT kernels for both NoTrans and Trans cases. Sizes span
// below the threshold (small fallback) and well above it (parallel split).
func TestDgemvParallel(t *testing.T) {
	rnd := rand.New(rand.NewPCG(2, 3))
	cases := []struct {
		name string
		m, n int
	}{
		{"tiny", 7, 5},
		{"belowThresh", 100, 100},
		{"squareMed", 512, 512},
		{"atThresh", 1024, 1024},
		{"tallSkinny", 2000, 80},
		{"shortFat", 80, 2000},
		{"tallSkinnyLarge", 20000, 60},
		{"shortFatLarge", 60, 20000},
		{"oddSplitLarge", 1031, 1049},
		{"oddSplit", 333, 777},
	}
	for _, alpha := range []float64{1, -2.5, 0.75} {
		for _, beta := range []float64{0, 1, -0.5} {
			for _, c := range cases {
				for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
					name := fmt.Sprintf("%s/a=%v/b=%v/t=%v", c.name, alpha, beta, trans)
					t.Run(name, func(t *testing.T) {
						checkDgemvParallel(t, rnd, trans, c.m, c.n, alpha, beta)
					})
				}
			}
		}
	}
}

func checkDgemvParallel(t *testing.T, rnd *rand.Rand, tA blas.Transpose, m, n int, alpha, beta float64) {
	t.Helper()
	lda := n
	a := randmat(m, n, lda, rnd)

	xLen := n
	yLen := m
	if tA != blas.NoTrans {
		xLen = m
		yLen = n
	}
	x := make([]float64, xLen)
	for i := range x {
		x[i] = rnd.NormFloat64()
	}
	yWant := make([]float64, yLen)
	for i := range yWant {
		yWant[i] = rnd.NormFloat64()
	}
	yGot := make([]float64, yLen)
	copy(yGot, yWant)

	// Reference: force the serial kernel by calling dgemvSerial directly.
	dgemvSerial(tA != blas.NoTrans, m, n, alpha, a, lda, x, beta, yWant)

	// Under test: full Dgemv path; for sizes below threshold this falls back
	// to the same serial kernel, for sizes above it dispatches dgemvParallel.
	Implementation{}.Dgemv(tA, m, n, alpha, a, lda, x, 1, beta, yGot, 1)

	if !floats.EqualApprox(yGot, yWant, 1e-12) {
		t.Errorf("mismatch m=%d n=%d alpha=%v beta=%v trans=%v", m, n, alpha, beta, tA)
	}
}

// Below the threshold Dgemv must stay on the serial kernel, which does not
// allocate; the parallel split allocates for its goroutines and join.
// testing.AllocsPerRun is not usable here: it forces GOMAXPROCS=1, which
// makes dgemvParallel fall back to the serial kernel on its own.
func TestDgemvSerialBelowThresholdDoesNotAllocate(t *testing.T) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(max(2, runtime.GOMAXPROCS(0))))
	rnd := rand.New(rand.NewPCG(4, 5))
	const n, runs = 500, 50
	a := randmat(n, n, n, rnd)
	x := make([]float64, n)
	y := make([]float64, n)
	for _, tA := range []blas.Transpose{blas.NoTrans, blas.Trans} {
		Implementation{}.Dgemv(tA, n, n, 1.5, a, n, x, 1, 0.5, y, 1)
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		for range runs {
			Implementation{}.Dgemv(tA, n, n, 1.5, a, n, x, 1, 0.5, y, 1)
		}
		runtime.ReadMemStats(&after)
		if allocs := after.Mallocs - before.Mallocs; allocs >= runs {
			t.Errorf("Dgemv %c %d×%d: %d allocs over %d calls, want the non-allocating serial kernel", tA, n, n, allocs, runs)
		}
	}
}

// Benchmarks for the parallel Dgemv path at sizes around and above the
// threshold. Each iteration writes y in place from fresh inputs so the
// kernel is exercised end-to-end.
func benchmarkDgemvSquare(b *testing.B, sz int, tA blas.Transpose) {
	rnd := rand.New(rand.NewPCG(uint64(sz), 99))
	a := randmat(sz, sz, sz, rnd)
	x := make([]float64, sz)
	for i := range x {
		x[i] = rnd.NormFloat64()
	}
	y := make([]float64, sz)
	for i := range y {
		y[i] = rnd.NormFloat64()
	}
	impl := Implementation{}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		impl.Dgemv(tA, sz, sz, 1.5, a, sz, x, 1, 0.5, y, 1)
	}
}

func BenchmarkDgemvParallelNoTrans1024(b *testing.B) { benchmarkDgemvSquare(b, 1024, blas.NoTrans) }
func BenchmarkDgemvParallelNoTrans2048(b *testing.B) { benchmarkDgemvSquare(b, 2048, blas.NoTrans) }
func BenchmarkDgemvParallelNoTrans4096(b *testing.B) { benchmarkDgemvSquare(b, 4096, blas.NoTrans) }
func BenchmarkDgemvParallelTrans1024(b *testing.B)   { benchmarkDgemvSquare(b, 1024, blas.Trans) }
func BenchmarkDgemvParallelTrans2048(b *testing.B)   { benchmarkDgemvSquare(b, 2048, blas.Trans) }
func BenchmarkDgemvParallelTrans4096(b *testing.B)   { benchmarkDgemvSquare(b, 4096, blas.Trans) }

// BenchmarkDgemvSplit times the serial kernel against the parallel split on
// the same square inputs, so the dispatch rule in dgemv_parallel.go can be
// checked on the target machine with -cpu.
func BenchmarkDgemvSplit(b *testing.B) {
	for _, sz := range []int{256, 500, 724, 1024, 1448, 2048, 4096} {
		rnd := rand.New(rand.NewPCG(uint64(sz), 7))
		a := randmat(sz, sz, sz, rnd)
		x := make([]float64, sz)
		y := make([]float64, sz)
		for i := range x {
			x[i] = rnd.NormFloat64()
			y[i] = rnd.NormFloat64()
		}
		for _, trans := range []bool{false, true} {
			for _, path := range []struct {
				name string
				run  func(bool, int, int, float64, []float64, int, []float64, float64, []float64)
			}{{"serial", dgemvSerial}, {"parallel", dgemvParallel}} {
				b.Run(fmt.Sprintf("n=%d/trans=%v/%s", sz, trans, path.name), func(b *testing.B) {
					for b.Loop() {
						path.run(trans, sz, sz, 1.5, a, sz, x, 0.5, y)
					}
				})
			}
		}
	}
}
