// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

// TestDggesMinimumWorkSorted checks that Dgges with eigenvalue sorting
// completes with lwork = max(8n, 6n+16), the minimum DGGES accepts, although
// its optimal size, which queries blocked subroutines, is larger than Netlib's.
func TestDggesMinimumWorkSorted(t *testing.T) {
	rnd := rand.New(rand.NewPCG(83, 83))
	sel := func(ar, ai, b float64) bool { return ar < 0 }
	for _, n := range []int{1, 2, 3, 5, 8, 13, 20, 40} {
		for _, jobv := range []lapack.SchurComp{lapack.SchurNone, lapack.SchurHess} {
			t.Run(fmt.Sprintf("n=%d/jobv=%c", n, jobv), func(t *testing.T) {
				a, b := make([]float64, n*n), make([]float64, n*n)
				for i := range a {
					a[i], b[i] = rnd.NormFloat64(), rnd.NormFloat64()
				}
				vsl, vsr := make([]float64, n*n), make([]float64, n*n)
				ar, ai, beta := make([]float64, n), make([]float64, n), make([]float64, n)
				lwork := max(8*n, 6*n+16)
				work := make([]float64, lwork)
				sdim, ok := Implementation{}.Dgges(jobv, jobv, lapack.SortSelected, sel, n,
					a, n, b, n, ar, ai, beta, vsl, n, vsr, n, work, lwork, make([]bool, n))
				if !ok {
					t.Fatalf("Dgges failed (sdim=%d)", sdim)
				}
				for j := range sdim {
					if !sel(ar[j], ai[j], beta[j]) {
						t.Errorf("eigenvalue %d=(%g,%g)/%g in leading block not selected", j, ar[j], ai[j], beta[j])
					}
				}
			})
		}
	}
}
