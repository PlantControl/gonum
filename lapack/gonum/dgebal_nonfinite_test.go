// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

func nonFiniteTestMatrix(n int) []float64 {
	a := make([]float64, n*n)
	for i := range a {
		a[i] = float64((i*7)%5) - 1.5
	}
	return a
}

func panicValue(f func()) (r any) {
	defer func() { r = recover() }()
	f()
	return nil
}

func TestDgebalNaN(t *testing.T) {
	t.Parallel()
	for _, n := range []int{1, 2, 5} {
		for pos := range n * n {
			for _, job := range []lapack.BalanceJob{lapack.Scale, lapack.PermuteScale} {
				name := fmt.Sprintf("n=%d,pos=%d,job=%c", n, pos, job)
				a := nonFiniteTestMatrix(n)
				a[pos] = math.NaN()
				r := panicValue(func() {
					Implementation{}.Dgebal(job, n, a, n, make([]float64, n))
				})
				switch {
				case r == nil && job == lapack.Scale:
					t.Errorf("%s: no panic for NaN in scaled block", name)
				case r != nil && r != nanA:
					t.Errorf("%s: unexpected panic %v", name, r)
				}
			}
		}
	}
}

func TestDgebalInf(t *testing.T) {
	t.Parallel()
	for _, n := range []int{2, 5} {
		for pos := range n * n {
			for _, v := range []float64{math.Inf(1), math.Inf(-1)} {
				a := nonFiniteTestMatrix(n)
				a[pos] = v
				if r := panicValue(func() {
					Implementation{}.Dgebal(lapack.PermuteScale, n, a, n, make([]float64, n))
				}); r != nil {
					t.Errorf("n=%d,pos=%d,v=%g: unexpected panic %v", n, pos, v, r)
				}
			}
		}
	}
}

func TestDgeevNonFinite(t *testing.T) {
	t.Parallel()
	for _, n := range []int{1, 2, 5} {
		for pos := range n * n {
			for _, v := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
				for _, jobvr := range []lapack.RightEVJob{lapack.RightEVNone, lapack.RightEVCompute} {
					a := nonFiniteTestMatrix(n)
					a[pos] = v
					work := make([]float64, 1)
					impl := Implementation{}
					impl.Dgeev(lapack.LeftEVNone, jobvr, n, a, n, make([]float64, n), make([]float64, n), nil, 1, make([]float64, n*n), n, work, -1)
					work = make([]float64, int(work[0]))
					r := panicValue(func() {
						impl.Dgeev(lapack.LeftEVNone, jobvr, n, a, n, make([]float64, n), make([]float64, n), nil, 1, make([]float64, n*n), n, work, len(work))
					})
					if r != nonFiniteA {
						t.Errorf("n=%d,pos=%d,v=%g,jobvr=%c: got panic %v, want %q", n, pos, v, jobvr, r, nonFiniteA)
					}
				}
			}
		}
	}
}
