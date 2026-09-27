// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"testing"

	"plantcontrol.org/v1/gonum/blas/blas64"
	blasgonum "plantcontrol.org/v1/gonum/blas/gonum"
	"plantcontrol.org/v1/gonum/lapack"
)

func TestRecheckDggesSelectionAfterFailure(t *testing.T) {
	calls := 0
	selector := func(alphar, _, _ float64) bool {
		calls++
		return alphar < 0
	}
	sdim, ok := recheckDggesSelection(false, selector,
		[]float64{-1, 2}, []float64{0, 0}, []float64{1, 1})
	if ok {
		t.Fatal("prior reordering failure was lost")
	}
	if calls != 2 || sdim != 1 {
		t.Fatalf("calls=%d sdim=%d, want calls=2 sdim=1", calls, sdim)
	}
}

func TestDggesReorderingAllocations(t *testing.T) {
	if _, ok := blas64.Implementation().(blasgonum.Implementation); !ok {
		t.Skip("allocation bound applies to the Go BLAS backend")
	}
	for _, n := range []int{20, 100} {
		a, b := make([]float64, n*n), make([]float64, n*n)
		ar, ai, beta := make([]float64, n), make([]float64, n), make([]float64, n)
		bw := make([]bool, n)
		selector := func(ar, _, beta float64) bool { return beta != 0 && ar < 0 }
		query := make([]float64, 1)
		Implementation{}.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortSelected, selector, n, nil, n, nil, n, nil, nil, nil, nil, 1, nil, 1, query, -1, nil)
		work := make([]float64, int(query[0]))
		allocs := testing.AllocsPerRun(2, func() {
			clear(a)
			clear(b)
			// Every negative eigenvalue must move past preceding positive eigenvalues.
			for i := range n {
				a[i*n+i] = float64(i + 1)
				if i%2 == 1 {
					a[i*n+i] = -a[i*n+i]
				}
				b[i*n+i] = 1
			}
			sdim, ok := Implementation{}.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortSelected, selector, n, a, n, b, n, ar, ai, beta, nil, 1, nil, 1, work, len(work), bw)
			if !ok || sdim != n/2 {
				t.Fatalf("n=%d: sdim=%d, ok=%t", n, sdim, ok)
			}
		})
		// Allow compiler variation, but not an allocation per adjacent block swap.
		if allocs > 8 {
			t.Errorf("n=%d: %g allocations; reordering must reuse scratch", n, allocs)
		}
	}
}
