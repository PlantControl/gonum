// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math/rand/v2"
	"runtime"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/floats"
)

func TestDgemmParallel(t *testing.T) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for i, test := range []struct {
		m     int
		n     int
		k     int
		alpha float64
		tA    blas.Transpose
		tB    blas.Transpose
	}{
		{
			m:     3,
			n:     4,
			k:     2,
			alpha: 2.5,
			tA:    blas.NoTrans,
			tB:    blas.NoTrans,
		},
		{
			m:     blockSize*2 + 5,
			n:     3,
			k:     2,
			alpha: 2.5,
			tA:    blas.NoTrans,
			tB:    blas.NoTrans,
		},
		{
			m:     3,
			n:     blockSize * 2,
			k:     2,
			alpha: 2.5,
			tA:    blas.NoTrans,
			tB:    blas.NoTrans,
		},
		{
			m:     2,
			n:     3,
			k:     blockSize*3 - 2,
			alpha: 2.5,
			tA:    blas.NoTrans,
			tB:    blas.NoTrans,
		},
		{
			m:     blockSize * minParBlock,
			n:     3,
			k:     2,
			alpha: 2.5,
			tA:    blas.NoTrans,
			tB:    blas.NoTrans,
		},
		{
			m:     3,
			n:     blockSize * minParBlock,
			k:     2,
			alpha: 2.5,
			tA:    blas.NoTrans,
			tB:    blas.NoTrans,
		},
		{
			m:     2,
			n:     3,
			k:     blockSize * minParBlock,
			alpha: 2.5,
			tA:    blas.NoTrans,
			tB:    blas.NoTrans,
		},
		{
			m:     blockSize*minParBlock + 1,
			n:     blockSize * minParBlock,
			k:     3,
			alpha: 2.5,
			tA:    blas.NoTrans,
			tB:    blas.NoTrans,
		},
		{
			m:     3,
			n:     blockSize*minParBlock + 2,
			k:     blockSize * 3,
			alpha: 2.5,
			tA:    blas.NoTrans,
			tB:    blas.NoTrans,
		},
		{
			m:     blockSize * minParBlock,
			n:     3,
			k:     blockSize * minParBlock,
			alpha: 2.5,
			tA:    blas.NoTrans,
			tB:    blas.NoTrans,
		},
		{
			m:     blockSize * minParBlock,
			n:     blockSize * minParBlock,
			k:     blockSize * 3,
			alpha: 2.5,
			tA:    blas.NoTrans,
			tB:    blas.NoTrans,
		},
		{
			m:     blockSize + blockSize/2,
			n:     blockSize + blockSize/2,
			k:     blockSize + blockSize/2,
			alpha: 2.5,
			tA:    blas.NoTrans,
			tB:    blas.NoTrans,
		},
	} {
		testMatchParallelSerial(t, rnd, i, blas.NoTrans, blas.NoTrans, test.m, test.n, test.k, test.alpha)
		testMatchParallelSerial(t, rnd, i, blas.Trans, blas.NoTrans, test.m, test.n, test.k, test.alpha)
		testMatchParallelSerial(t, rnd, i, blas.NoTrans, blas.Trans, test.m, test.n, test.k, test.alpha)
		testMatchParallelSerial(t, rnd, i, blas.Trans, blas.Trans, test.m, test.n, test.k, test.alpha)
	}
}

func testMatchParallelSerial(t *testing.T, rnd *rand.Rand, i int, tA, tB blas.Transpose, m, n, k int, alpha float64) {
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
	a := randmat(rowA, colA, lda, rnd)
	aCopy := make([]float64, len(a))
	copy(aCopy, a)

	ldb := colB
	b := randmat(rowB, colB, ldb, rnd)
	bCopy := make([]float64, len(b))
	copy(bCopy, b)

	ldc := n
	c := randmat(m, n, ldc, rnd)
	want := make([]float64, len(c))
	copy(want, c)

	dgemmSerial(tA == blas.Trans, tB == blas.Trans, m, n, k, a, lda, b, ldb, want, ldc, alpha)
	dgemmParallel(tA == blas.Trans, tB == blas.Trans, m, n, k, a, lda, b, ldb, c, ldc, alpha)

	if !floats.Equal(a, aCopy) {
		t.Errorf("Case %v: a changed during call to dgemmParallel", i)
	}
	if !floats.Equal(b, bCopy) {
		t.Errorf("Case %v: b changed during call to dgemmParallel", i)
	}
	if !floats.EqualApprox(c, want, 1e-12) {
		t.Errorf("Case %v: answer not equal parallel and serial", i)
	}
}

func randmat(r, c, stride int, rnd *rand.Rand) []float64 {
	data := make([]float64, r*stride+c)
	for i := range data {
		data[i] = rnd.NormFloat64()
	}
	return data
}

// TestDgemmDispatchGate pins the serial/parallel dispatch decision (J10 gate,
// see dgemmParallelWorkerCount) at fixed worker counts. In particular,
// 100x100x100 on four workers must take the parallel path: with the earlier
// flat m*n*k < 1<<20 cutoff it went serial, a +59% regression on four-worker
// hosts (gonum PR #4 review). Darwin arm64 uses a lower per-worker floor to
// match the crossover of its pure-Go kernels.
func TestDgemmDispatchGate(t *testing.T) {
	for _, test := range []struct {
		m, n, k               int
		workers               int
		wantSerial            bool
		wantSerialDarwinARM64 bool
	}{
		{100, 100, 100, 4, false, false}, // 1e6 ops, 4 blocks: the reviewer's M1 shape
		{100, 100, 100, 8, false, false},
		{128, 128, 128, 4, false, false},
		{200, 200, 200, 8, false, false},
		{1000, 100, 10, 4, false, false}, // skinny, 32 blocks, 1e6 ops
		{1000, 100, 10, 8, true, false},
		{128, 128, 8, 4, true, false}, // pure-Go Darwin arm64 profits at this boundary
		{128, 128, 8, 32, true, false},
		{60, 60, 60, 8, true, true}, // 1 (m,n) block: block-count gate
		{64, 64, 64, 8, true, true},
		{80, 80, 80, 4, true, false}, // pure-Go Darwin arm64 crosses over below 2^17/worker
		{100, 100, 100, 1, true, true},
		{1024, 1024, 1024, 1, true, true},
	} {
		old := runtime.GOMAXPROCS(test.workers)
		got := dgemmParallelWorkerCount(test.m, test.n, test.k) == 0
		runtime.GOMAXPROCS(old)
		want := test.wantSerial
		if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
			want = test.wantSerialDarwinARM64
		}
		if got != want {
			t.Errorf("dgemmParallelWorkerCount(%d,%d,%d) at GOMAXPROCS=%d: got serial=%v, want %v",
				test.m, test.n, test.k, test.workers, got, want)
		}
	}
}
