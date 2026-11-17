// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lp

import (
	"math"
	"math/rand/v2"
	"testing"
)

// pivotProblem returns a feasible problem with a slack column for every
// row, whose other columns hold a nonzero with probability density.
func pivotProblem(rnd *rand.Rand, m, n int, density float64) (c []float64, a *cscMatrix, b, lower, upper []float64) {
	a = &cscMatrix{m: m, n: n + m, start: []int{0}}
	x := make([]float64, n)
	b = make([]float64, m)
	for j := range n {
		lower = append(lower, 0)
		upper = append(upper, 1+4*rnd.Float64())
		if rnd.IntN(8) == 0 {
			upper[j] = math.Inf(1)
		}
		c = append(c, rnd.NormFloat64())
		x[j] = rnd.Float64()
		for i := range m {
			if rnd.Float64() < density {
				v := float64(1+rnd.IntN(9)) * float64(1-2*rnd.IntN(2))
				a.index = append(a.index, i)
				a.value = append(a.value, v)
				b[i] += v * x[j]
			}
		}
		a.start = append(a.start, len(a.index))
	}
	for i := range m {
		a.index = append(a.index, i)
		a.value = append(a.value, 1)
		a.start = append(a.start, len(a.index))
		lower = append(lower, 0)
		upper = append(upper, math.Inf(1))
		c = append(c, 0)
		b[i] += rnd.Float64()
	}
	return c, a, b, lower, upper
}

// started returns the state of a solve at the basis of slack columns.
func started(t *testing.T, c []float64, a *cscMatrix, b, lower, upper []float64) *bounded {
	t.Helper()
	s := newBounded(newFactor, c, a, b, lower, upper, 1e-9)
	if !s.warm(s.slackBasis()) {
		t.Fatal("slack basis is singular")
	}
	return s
}

// TestPivotRow checks that the pivot row agrees to the last bit whether it
// is formed from the rows of A or from its columns, and that both ways are
// taken.
func TestPivotRow(t *testing.T) {
	t.Parallel()
	rnd := rand.New(rand.NewPCG(21, 21))
	byRows, byColumns := 0, 0
	for trial := range 40 {
		density := 0.05
		if trial%2 == 1 {
			density = 0.6
		}
		m := 10 + rnd.IntN(30)
		c, a, b, lower, upper := pivotProblem(rnd, m, 3*m, density)
		s := started(t, c, a, b, lower, upper)
		for range 3 {
			for r := range s.m {
				s.pivotRow(r)
				nz := 0
				for _, v := range s.rho {
					if v != 0 {
						nz++
					}
				}
				if float64(nz) > rowDensity*float64(s.m) {
					byColumns++
				} else {
					byRows++
				}
				for j := range s.total {
					if s.status[j] == basic || s.lower[j] == s.upper[j] {
						continue
					}
					want := 0.0
					for k := s.start[j]; k < s.start[j+1]; k++ {
						want += s.rho[s.index[k]] * s.value[k]
					}
					if s.arow[j] != want {
						t.Fatalf("trial %d row %d column %d: got %v, want %v", trial, r, j, s.arow[j], want)
					}
				}
			}
			// Some pivots fill the basis inverse.
			s.limit = s.iterations + m/2
			_ = s.run()
			if err := s.refactor(); err != nil {
				t.Fatal(err)
			}
		}
	}
	if byRows == 0 || byColumns == 0 {
		t.Fatalf("%d rows formed from rows of A, %d from columns", byRows, byColumns)
	}
}

// TestReduce exchanges basis columns that no pricing rule would choose and
// checks the updated multipliers and reduced costs against those computed
// from the new basis.
func TestReduce(t *testing.T) {
	t.Parallel()
	rnd := rand.New(rand.NewPCG(22, 22))
	pivots := 0
	for range 40 {
		m := 10 + rnd.IntN(30)
		c, a, b, lower, upper := pivotProblem(rnd, m, 3*m, 0.05+0.4*rnd.Float64())
		s := started(t, c, a, b, lower, upper)
		s.devex = make([]float64, s.total)
		for j := range s.devex {
			s.devex[j] = 1
		}
		s.reprice()
		for range 2 * m {
			row := rnd.IntN(m)
			s.pivotRow(row)
			q, big := -1, 0.1
			for j := range s.total {
				if s.status[j] == basic || s.lower[j] == s.upper[j] {
					continue
				}
				if v := math.Abs(s.arow[j]); v > big {
					q, big = j, v
				}
			}
			if q < 0 {
				continue
			}
			s.ftran(q)
			if !s.stable(row) {
				continue
			}
			leave := s.head[row]
			s.status[leave] = atLower
			s.reduce(q, leave, row)
			s.head[row], s.status[q] = q, basic
			if err := s.pivoted(row); err != nil {
				t.Fatal(err)
			}
			pivots++
			y := append([]float64(nil), s.y...)
			s.btran()
			for i := range y {
				if math.Abs(y[i]-s.y[i]) > 1e-9*math.Max(1, math.Abs(s.y[i])) {
					t.Fatalf("multiplier %d: updated %v, fresh %v", i, y[i], s.y[i])
				}
			}
			for j := range s.total {
				if s.status[j] == basic || s.lower[j] == s.upper[j] {
					continue
				}
				want, tol := s.reducedCost(j)
				if math.Abs(s.d[j]-want) > tol {
					t.Fatalf("reduced cost %d: updated %v, fresh %v", j, s.d[j], want)
				}
				if s.devex[j] < 1 {
					t.Fatalf("reference weight %d is %v", j, s.devex[j])
				}
			}
			if s.dtol[leave] < s.tol {
				t.Fatalf("tolerance of the leaving variable is %v", s.dtol[leave])
			}
			copy(s.y, y)
		}
	}
	if pivots < 500 {
		t.Fatalf("only %d pivots", pivots)
	}
}

// TestSparseLUUpdateDue checks that a factorization is asked for once the
// etas outweigh it, and after refactorEvery updates whatever their size.
func TestSparseLUUpdateDue(t *testing.T) {
	const m = 200
	identity := &columns{start: make([]int, m+1), index: make([]int, m), value: make([]float64, m)}
	head := make([]int, m)
	for i := range m {
		identity.start[i+1], identity.index[i], identity.value[i], head[i] = i+1, i, 1, i
	}
	for _, test := range []struct {
		fill, want int
	}{
		{fill: 1, want: refactorEvery},
		// A pass over k etas costs 200k, and the factorization
		// factorWeight*600: k*200k >= 4800 + 100k(k+1) first holds at 8.
		{fill: m, want: 8},
	} {
		f := newSparseLU(m)
		if err := f.factorize(identity, head); err != nil {
			t.Fatal(err)
		}
		got := 0
		for k := 1; k <= refactorEvery; k++ {
			alpha := make([]float64, m)
			for i := range test.fill {
				alpha[i] = 1
			}
			if f.update(0, alpha) {
				got = k
				break
			}
		}
		if got != test.want {
			t.Errorf("etas of %d entries: factorization due after %d updates, want %d", test.fill, got, test.want)
		}
	}
}
