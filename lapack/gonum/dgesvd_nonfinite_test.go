// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"testing"
	"time"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
)

// guardValue runs f and returns its panic value, or "hang" if f does not
// return within a few seconds.
func guardValue(f func()) any {
	ch := make(chan any, 1)
	go func() { ch <- panicValue(f) }()
	select {
	case r := <-ch:
		return r
	case <-time.After(5 * time.Second):
		return "hang"
	}
}

func TestFmaxFmin(t *testing.T) {
	t.Parallel()
	nan := math.NaN()
	for _, c := range []struct{ a, b, max, min float64 }{
		{1, 2, 2, 1},
		{2, 1, 2, 1},
		{nan, 1, 1, 1},
		{1, nan, 1, 1},
		{math.Inf(-1), nan, math.Inf(-1), math.Inf(-1)},
	} {
		if got := fmax(c.a, c.b); got != c.max {
			t.Errorf("fmax(%v, %v) = %v, want %v", c.a, c.b, got, c.max)
		}
		if got := fmin(c.a, c.b); got != c.min {
			t.Errorf("fmin(%v, %v) = %v, want %v", c.a, c.b, got, c.min)
		}
	}
	if !math.IsNaN(fmax(nan, nan)) || !math.IsNaN(fmin(nan, nan)) {
		t.Error("fmax/fmin of two NaNs is not NaN")
	}
}

var nonFiniteValues = []float64{math.NaN(), math.Inf(1), math.Inf(-1)}

// TestDgesvdNonFinite checks that Dgesvd neither panics nor hangs when A
// contains NaN or ±Inf. Reference LAPACK 3.12 DGESVD returned INFO = 0 with NaN
// in S, or INFO > 0, for every case here.
func TestDgesvdNonFinite(t *testing.T) {
	t.Parallel()
	jobs := []lapack.SVDJob{lapack.SVDAll, lapack.SVDStore, lapack.SVDNone}
	for _, mn := range [][2]int{{1, 1}, {2, 2}, {3, 3}, {5, 5}, {3, 5}, {5, 3}, {1, 4}, {4, 1}, {12, 4}, {4, 12}} {
		m, n := mn[0], mn[1]
		for _, p := range spreadPositions(m*n, 9) {
			for _, v := range nonFiniteValues {
				for _, jobU := range jobs {
					for _, jobVT := range jobs {
						name := fmt.Sprintf("m=%d,n=%d,pos=%d,v=%g,jobU=%c,jobVT=%c", m, n, p, v, jobU, jobVT)
						a := make([]float64, m*n)
						for i := range a {
							a[i] = float64((i*7)%5) - 1.5
						}
						a[p] = v
						s := make([]float64, min(m, n))
						u := make([]float64, m*m)
						vt := make([]float64, n*n)
						work := make([]float64, 1)
						impl := Implementation{}
						impl.Dgesvd(jobU, jobVT, m, n, a, n, s, u, m, vt, n, work, -1)
						work = make([]float64, int(work[0]))
						var ok bool
						r := guardValue(func() {
							ok = impl.Dgesvd(jobU, jobVT, m, n, a, n, s, u, m, vt, n, work, len(work))
						})
						if r != nil {
							t.Errorf("%s: unexpected panic %v", name, r)
							continue
						}
						if ok && !hasNaN(s) {
							t.Errorf("%s: ok with no NaN in s", name)
						}
					}
				}
			}
		}
	}
}

// TestDbdsqrNonFinite checks that Dbdsqr neither panics nor hangs on a
// bidiagonal matrix containing NaN or ±Inf. Before thresh ignored NaN, as
// gfortran's MAX does in reference DBDSQR, a NaN thresh blocked deflation and
// Dbdsqr looped forever.
func TestDbdsqrNonFinite(t *testing.T) {
	t.Parallel()
	for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower} {
		for _, n := range []int{1, 2, 3, 5} {
			for p := range 2*n - 1 {
				for _, v := range nonFiniteValues {
					for _, nc := range [][2]int{{0, 0}, {n, 0}, {0, n}, {n, n}} {
						ncvt, nru := nc[0], nc[1]
						name := fmt.Sprintf("uplo=%c,n=%d,pos=%d,v=%g,ncvt=%d,nru=%d", uplo, n, p, v, ncvt, nru)
						d, e := nonFiniteBidiagonal(n, p, v, false)
						ldvt := max(1, ncvt)
						vt := make([]float64, n*ldvt)
						for i := range min(n, ncvt) {
							vt[i*ldvt+i] = 1
						}
						u := make([]float64, max(1, nru)*n)
						for i := range min(n, nru) {
							u[i*n+i] = 1
						}
						work := make([]float64, 4*n)
						r := guardValue(func() {
							Implementation{}.Dbdsqr(uplo, n, ncvt, nru, 0, d, e, vt, ldvt, u, n, nil, 1, work)
						})
						if r != nil {
							t.Errorf("%s: unexpected panic %v", name, r)
						}
					}
				}
			}
		}
	}

	// Reference DBDSQR returns INFO = 2 here.
	d := []float64{1, math.NaN(), 2}
	e := []float64{1, 1}
	vt := make([]float64, 9)
	u := make([]float64, 9)
	for i := range 3 {
		vt[i*3+i] = 1
		u[i*3+i] = 1
	}
	if (Implementation{}).Dbdsqr(blas.Upper, 3, 3, 3, 0, d, e, vt, 3, u, 3, nil, 1, make([]float64, 12)) {
		t.Error("Dbdsqr: ok for NaN diagonal with vectors, reference INFO = 2")
	}
}

// TestDlasq1NonFinite checks that Dlasq1 does not panic when d or e contains
// NaN or ±Inf. Before sigmx ignored NaN, as gfortran's MAX does in reference
// DLASQ1, Dlascl panicked with a NaN cfrom. Reference DLASQ1 returned INFO = 0
// for every case here.
func TestDlasq1NonFinite(t *testing.T) {
	t.Parallel()
	for _, n := range []int{1, 2, 3, 5} {
		for p := range 2*n - 1 {
			for _, v := range nonFiniteValues {
				for _, diag := range []bool{false, true} {
					name := fmt.Sprintf("n=%d,pos=%d,v=%g,diag=%t", n, p, v, diag)
					d, e := nonFiniteBidiagonal(n, p, v, diag)
					var info int
					r := guardValue(func() { info = Implementation{}.Dlasq1(n, d, e, make([]float64, 4*n)) })
					if r != nil {
						t.Errorf("%s: unexpected panic %v", name, r)
						continue
					}
					if info != 0 {
						t.Errorf("%s: info=%d, reference INFO = 0", name, info)
					}
				}
			}
		}
	}
}

// nonFiniteBidiagonal returns d and e of an n×n bidiagonal matrix with v at
// position p of the concatenation of d and e. If diag is true, e is zero
// apart from v.
func nonFiniteBidiagonal(n, p int, v float64, diag bool) (d, e []float64) {
	d = make([]float64, n)
	e = make([]float64, max(1, n-1))
	for i := range d {
		d[i] = float64(i%3) + 1
	}
	if !diag {
		for i := range n - 1 {
			e[i] = 0.5 * float64(i%2+1)
		}
	}
	if p < n {
		d[p] = v
	} else {
		e[p-n] = v
	}
	return d, e
}

// spreadPositions returns up to k indices in [0, nn), spread evenly and
// including both ends.
func spreadPositions(nn, k int) []int {
	if nn <= k {
		pos := make([]int, nn)
		for i := range pos {
			pos[i] = i
		}
		return pos
	}
	pos := make([]int, k)
	for i := range pos {
		pos[i] = i * (nn - 1) / (k - 1)
	}
	return pos
}
