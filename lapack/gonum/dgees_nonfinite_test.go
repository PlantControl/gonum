// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

func hasNaN(s ...[]float64) bool {
	for _, v := range s {
		if slices.ContainsFunc(v, math.IsNaN) {
			return true
		}
	}
	return false
}

// positions returns up to k element positions of an n×n matrix, spread
// evenly and including both ends.
func positions(n, k int) []int {
	if n*n <= k {
		pos := make([]int, n*n)
		for p := range pos {
			pos[p] = p
		}
		return pos
	}
	pos := make([]int, k)
	for i := range pos {
		pos[i] = i * (n*n - 1) / (k - 1)
	}
	return pos
}

type nonFiniteCase struct {
	n, pos int
	v      float64
	compz  lapack.SchurComp
	opt    bool
}

// nonFiniteCases enumerates every combination for small n and a single
// representative case for larger n, where a NaN drives QR to its iteration
// limit and each call costs tens to hundreds of milliseconds.
func nonFiniteCases(small []int, k int, compzs []lapack.SchurComp, large []nonFiniteCase) []nonFiniteCase {
	var cases []nonFiniteCase
	for _, n := range small {
		for _, p := range positions(n, k) {
			for _, v := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
				for _, compz := range compzs {
					for _, opt := range []bool{false, true} {
						cases = append(cases, nonFiniteCase{n, p, v, compz, opt})
					}
				}
			}
		}
	}
	return append(cases, large...)
}

// TestDgeesNonFinite checks that Dgees does not panic when A contains NaN or
// ±Inf. Reference LAPACK 3.12 DGEES returns INFO = 0 or INFO > 0 for such
// input, never a negative INFO, so the non-finite value must surface as
// ok == false or as NaN in T, wr or wi.
func TestDgeesNonFinite(t *testing.T) {
	t.Parallel()
	selctg := func(wr, _ float64) bool { return wr > 0 }
	cases := nonFiniteCases([]int{1, 2, 3, 5}, 9, []lapack.SchurComp{lapack.SchurNone, lapack.SchurHess}, []nonFiniteCase{
		{20, 210, math.NaN(), lapack.SchurHess, true},
		{60, 1830, math.NaN(), lapack.SchurHess, false},
	})
	for _, c := range cases {
		n := c.n
		sort := lapack.SortNone
		if c.opt {
			sort = lapack.SortSelected
		}
		name := fmt.Sprintf("n=%d,pos=%d,v=%g,jobvs=%c,sort=%c", n, c.pos, c.v, c.compz, sort)
		a := nonFiniteTestMatrix(n)
		a[c.pos] = c.v
		wr, wi := make([]float64, n), make([]float64, n)
		vs := make([]float64, n*n)
		bwork := make([]bool, n)
		work := make([]float64, 1)
		impl := Implementation{}
		impl.Dgees(c.compz, sort, selctg, n, a, n, wr, wi, vs, max(1, n), work, -1, bwork)
		work = make([]float64, int(work[0]))
		var ok bool
		r := panicValue(func() {
			_, ok = impl.Dgees(c.compz, sort, selctg, n, a, n, wr, wi, vs, max(1, n), work, len(work), bwork)
		})
		if r != nil {
			t.Errorf("%s: unexpected panic %v", name, r)
			continue
		}
		if ok && !hasNaN(a, wr, wi) {
			t.Errorf("%s: ok with no NaN in T, wr or wi", name)
		}
	}
}

// TestDhseqrNonFinite drives Dhseqr into its Dlahqr failure fallback, which
// calls Dlaqr04 on a padded copy for n < 49 and in place for 49 <= n <= 75.
func TestDhseqrNonFinite(t *testing.T) {
	t.Parallel()
	cases := nonFiniteCases([]int{3, 5}, 9, []lapack.SchurComp{lapack.SchurNone, lapack.SchurHess, lapack.SchurOrig}, []nonFiniteCase{
		{20, 210, math.NaN(), lapack.SchurOrig, true},
		{49, 1200, math.NaN(), lapack.SchurOrig, true},
	})
	for _, c := range cases {
		n := c.n
		job := lapack.EigenvaluesOnly
		if c.opt {
			job = lapack.EigenvaluesAndSchur
		}
		if i, j := c.pos/n, c.pos%n; i > j+1 {
			continue
		}
		name := fmt.Sprintf("n=%d,pos=%d,v=%g,compz=%c,job=%c", n, c.pos, c.v, c.compz, job)
		h := nonFiniteTestMatrix(n)
		for r := 2; r < n; r++ {
			clear(h[r*n : r*n+r-1])
		}
		h[c.pos] = c.v
		z := make([]float64, n*n)
		for k := range n {
			z[k*n+k] = 1
		}
		wr, wi := make([]float64, n), make([]float64, n)
		work := make([]float64, 1)
		impl := Implementation{}
		impl.Dhseqr(job, c.compz, n, 0, n-1, h, n, wr, wi, z, n, work, -1)
		work = make([]float64, int(work[0]))
		var unconverged int
		r := panicValue(func() {
			unconverged = impl.Dhseqr(job, c.compz, n, 0, n-1, h, n, wr, wi, z, n, work, len(work))
		})
		if r != nil {
			t.Errorf("%s: unexpected panic %v", name, r)
			continue
		}
		if unconverged < 0 || n < unconverged {
			t.Errorf("%s: unconverged=%d out of range", name, unconverged)
		}
		if unconverged == 0 && !hasNaN(h, wr, wi) {
			t.Errorf("%s: converged with no NaN in H, wr or wi", name)
		}
	}
}

func TestDlaqr1NaNShifts(t *testing.T) {
	t.Parallel()
	nan := math.NaN()
	for _, n := range []int{2, 3} {
		h := nonFiniteTestMatrix(n)
		for _, s := range [][4]float64{{nan, 0, 1, 0}, {1, nan, 1, 0}, {1, 1, nan, -1}, {nan, nan, nan, nan}} {
			v := make([]float64, n)
			if r := panicValue(func() { Implementation{}.Dlaqr1(n, h, n, s[0], s[1], s[2], s[3], v) }); r != nil {
				t.Errorf("n=%d,shifts=%v: unexpected panic %v", n, s, r)
			}
		}
		if r := panicValue(func() { Implementation{}.Dlaqr1(n, h, n, 1, 1, 2, -1, make([]float64, n)) }); r != badShifts {
			t.Errorf("n=%d: mismatched finite shifts: got panic %v, want %q", n, r, badShifts)
		}
	}
}
