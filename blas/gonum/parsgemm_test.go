// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"
	"math/rand/v2"
	"runtime"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

func TestSgemmParallel(t *testing.T) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for i, test := range []struct {
		m     int
		n     int
		k     int
		alpha float32
	}{
		{m: 3, n: 4, k: 2, alpha: 2.5},
		{m: blockSize*2 + 5, n: 3, k: 2, alpha: 2.5},
		{m: 3, n: blockSize * 2, k: 2, alpha: 2.5},
		{m: 2, n: 3, k: blockSize*3 - 2, alpha: 2.5},
		{m: blockSize * minParBlock, n: 3, k: 2, alpha: 2.5},
		{m: 3, n: blockSize * minParBlock, k: 2, alpha: 2.5},
		{m: 2, n: 3, k: blockSize * minParBlock, alpha: 2.5},
		{m: blockSize*minParBlock + 1, n: blockSize * minParBlock, k: 3, alpha: 2.5},
		{m: 3, n: blockSize*minParBlock + 2, k: blockSize * 3, alpha: 2.5},
		{m: blockSize * minParBlock, n: 3, k: blockSize * minParBlock, alpha: 2.5},
		{m: blockSize * minParBlock, n: blockSize * minParBlock, k: blockSize * 3, alpha: 2.5},
		{m: blockSize + blockSize/2, n: blockSize + blockSize/2, k: blockSize + blockSize/2, alpha: 2.5},
	} {
		testMatchParallelSerial32(t, rnd, i, blas.NoTrans, blas.NoTrans, test.m, test.n, test.k, test.alpha)
		testMatchParallelSerial32(t, rnd, i, blas.Trans, blas.NoTrans, test.m, test.n, test.k, test.alpha)
		testMatchParallelSerial32(t, rnd, i, blas.NoTrans, blas.Trans, test.m, test.n, test.k, test.alpha)
		testMatchParallelSerial32(t, rnd, i, blas.Trans, blas.Trans, test.m, test.n, test.k, test.alpha)
	}
}

func testMatchParallelSerial32(t *testing.T, rnd *rand.Rand, i int, tA, tB blas.Transpose, m, n, k int, alpha float32) {
	var (
		rowA, colA int
		rowB, colB int
	)
	if tA == blas.NoTrans {
		rowA = m
		colA = k
	} else {
		rowA = k
		colA = m
	}
	if tB == blas.NoTrans {
		rowB = k
		colB = n
	} else {
		rowB = n
		colB = k
	}

	lda := colA
	a := randmat32(rowA, colA, lda, rnd)
	aCopy := make([]float32, len(a))
	copy(aCopy, a)

	ldb := colB
	b := randmat32(rowB, colB, ldb, rnd)
	bCopy := make([]float32, len(b))
	copy(bCopy, b)

	ldc := n
	c := randmat32(m, n, ldc, rnd)
	want := make([]float32, len(c))
	copy(want, c)

	sgemmSerial(tA == blas.Trans, tB == blas.Trans, m, n, k, a, lda, b, ldb, want, ldc, alpha)
	sgemmParallel(tA == blas.Trans, tB == blas.Trans, m, n, k, a, lda, b, ldb, c, ldc, alpha)

	if !equal32(a, aCopy) {
		t.Errorf("Case %v: a changed during call to sgemmParallel", i)
	}
	if !equal32(b, bCopy) {
		t.Errorf("Case %v: b changed during call to sgemmParallel", i)
	}
	if !equalApprox32(c, want, 5e-4) {
		t.Errorf("Case %v: answer not equal parallel and serial", i)
	}
}

func randmat32(r, c, stride int, rnd *rand.Rand) []float32 {
	data := make([]float32, r*stride+c)
	for i := range data {
		data[i] = float32(rnd.NormFloat64())
	}
	return data
}

func equal32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		if v != b[i] {
			return false
		}
	}
	return true
}

// equalApprox32 reports whether a and b match to within the given tolerance,
// elementwise (relative above magnitude 1, absolute below). The parallel path
// accumulates blocks in a different order than the serial path, so results
// differ by float32 rounding noise that grows with k; genuine dispatch or
// blocking bugs produce O(1) errors, far above this bound.
func equalApprox32(a, b []float32, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i, v := range a {
		diff := math.Abs(float64(v) - float64(b[i]))
		scale := math.Max(math.Abs(float64(v)), math.Abs(float64(b[i])))
		if scale > 1 {
			diff /= scale
		}
		if diff > tol {
			return false
		}
	}
	return true
}

// TestSgemmDispatchGate pins the serial/parallel dispatch decision (see
// sgemmParallelWorkerCount) at fixed worker counts, mirroring
// TestDgemmDispatchGate. The gate ports the worker-aware Dgemm design from
// gonum PR #4: parallel only when there are enough (m,n) blocks to fan out,
// at least two workers can run, and each occupiable worker receives at least
// sgemmMinParFLOPsPerWorker multiply-adds. The per-worker floor was
// calibrated with BenchmarkSgemmCrossover; the Darwin arm64 floor reuses the
// Dgemm default/Darwin ratio and has not been measured on that platform.
func TestSgemmDispatchGate(t *testing.T) {
	for _, test := range []struct {
		m, n, k               int
		workers               int
		wantSerial            bool
		wantSerialDarwinARM64 bool
	}{
		{100, 100, 100, 4, false, false}, // 1e6 ops, 4 blocks
		{100, 100, 100, 8, false, false},
		{128, 128, 128, 4, false, false},
		{200, 200, 200, 8, false, false},
		{1000, 100, 10, 4, false, false}, // skinny, 32 blocks, 1e6 ops
		{1000, 100, 10, 8, true, false},
		{128, 128, 8, 4, true, false},
		{128, 128, 8, 32, true, false},
		{60, 60, 60, 8, true, true}, // 1 (m,n) block: block-count gate
		{64, 64, 64, 8, true, true},
		{80, 80, 80, 4, true, false},
		{100, 100, 100, 1, true, true},
		{1024, 1024, 1024, 1, true, true},
	} {
		old := runtime.GOMAXPROCS(test.workers)
		got := sgemmParallelWorkerCount(test.m, test.n, test.k) == 0
		runtime.GOMAXPROCS(old)
		want := test.wantSerial
		if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
			want = test.wantSerialDarwinARM64
		}
		if got != want {
			t.Errorf("sgemmParallelWorkerCount(%d,%d,%d) at GOMAXPROCS=%d: got serial=%v, want %v",
				test.m, test.n, test.k, test.workers, got, want)
		}
	}
}
