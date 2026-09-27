// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"runtime"
	"sync"

	"plantcontrol.org/v1/gonum/internal/asm/f64"
)

// dgemvParallelThreshold is the M*N product at or above which Dgemv switches
// from a single-threaded GemvN/GemvT call to a row-block parallel dispatch.
//
// Chosen empirically on a 4-core box: for square matrices the parallel path
// pulls ahead around 256² (M*N ≈ 6.5e4) and is comfortably faster by 512²
// (M*N ≈ 2.6e5). 1e5 is a conservative midpoint that avoids scheduling
// overhead dominating very small problems while keeping the fast path live
// for everything in the medium/large regime DMCplus actually exercises.
const dgemvParallelThreshold = 100000

// dgemvParallel computes y = alpha*A*x + beta*y (or with Aᵀ when trans is
// true) by partitioning the output dimension across worker goroutines. It is
// safe only when incX == 1 && incY == 1; callers must dispatch to the serial
// kernel otherwise.
//
// For trans == false the m output rows of y are split into worker bands; each
// worker calls f64.GemvN on its row slice of A. For trans == true the n
// output entries of y are split; each worker calls f64.GemvT on the
// corresponding column slice of A (a[lo:] with the same lda gives a column
// window because rows still stride by lda).
//
// Output ranges are disjoint per worker so no synchronization on y is
// required beyond the WaitGroup.
func dgemvParallel(trans bool, m, n int, alpha float64, a []float64, lda int, x []float64, beta float64, y []float64) {
	workers := runtime.GOMAXPROCS(0)
	// outDim is the length of y that we partition across workers.
	outDim := m
	if trans {
		outDim = n
	}
	if workers < 2 || outDim < workers {
		dgemvSerial(trans, m, n, alpha, a, lda, x, beta, y)
		return
	}

	block := (outDim + workers - 1) / workers
	// Spawn workers-1 goroutines and run the final band on the calling
	// goroutine. This trims one goroutine creation per Dgemv (small but
	// measurable on memory-bound 4096² calls) and matches the standard
	// "leader+followers" pattern.
	var wg sync.WaitGroup
	runBand := func(lo, hi int) {
		if trans {
			// a[:, lo:hi] window with same lda; output is y[lo:hi] of length hi-lo.
			f64.GemvT(uintptr(m), uintptr(hi-lo), alpha,
				a[lo:], uintptr(lda),
				x, 1,
				beta, y[lo:hi], 1)
			return
		}
		// a[lo:hi, :] window with same lda; output is y[lo:hi].
		f64.GemvN(uintptr(hi-lo), uintptr(n), alpha,
			a[lo*lda:], uintptr(lda),
			x, 1,
			beta, y[lo:hi], 1)
	}
	lastLo := 0
	for lo := 0; lo < outDim; lo += block {
		hi := lo + block
		if hi > outDim {
			hi = outDim
		}
		if hi == outDim {
			lastLo = lo
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			runBand(lo, hi)
		}(lo, hi)
	}
	runBand(lastLo, outDim)
	wg.Wait()
}

// dgemvSerial is the unit-increment serial fallback used by dgemvParallel
// when the workload is too small to benefit from parallelism. It mirrors the
// post-validation portion of Dgemv for incX == incY == 1.
func dgemvSerial(trans bool, m, n int, alpha float64, a []float64, lda int, x []float64, beta float64, y []float64) {
	if trans {
		f64.GemvT(uintptr(m), uintptr(n), alpha, a, uintptr(lda), x, 1, beta, y, 1)
		return
	}
	f64.GemvN(uintptr(m), uintptr(n), alpha, a, uintptr(lda), x, 1, beta, y, 1)
}
