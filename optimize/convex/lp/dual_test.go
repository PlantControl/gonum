// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lp

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/floats/scalar"
	"plantcontrol.org/v1/gonum/mat"
)

// takesDualPath reports whether BoundedSimplexFrom would re-optimize start
// with the dual simplex method.
func takesDualPath(fk factorKind, start *Basis, c []float64, a mat.Matrix, b, lower, upper []float64) bool {
	s := newBounded(fk.new, c, a, b, lower, upper, 1e-9)
	return s.warm(start) && !s.primalFeasible() && s.dualStart()
}

// perturb returns a copy of an LP with optimum x after one of: bounds
// tightened past x, rows appended, or the right-hand side shifted. Any may
// make it infeasible.
func perturb(rnd *rand.Rand, x []float64, a *mat.Dense, b, lower, upper []float64) (*mat.Dense, []float64, []float64, []float64) {
	m, n := a.Dims()
	lower, upper = append([]float64(nil), lower...), append([]float64(nil), upper...)
	b = append([]float64(nil), b...)
	switch rnd.IntN(3) {
	case 0:
		for range 1 + rnd.IntN(3) {
			j := rnd.IntN(n)
			step := 0.2 * rnd.Float64()
			if rnd.IntN(2) == 0 {
				upper[j] = math.Max(lower[j], x[j]-step)
			} else {
				lower[j] = math.Min(upper[j], x[j]+step)
			}
		}
	case 1:
		k := 1 + rnd.IntN(3)
		wide := mat.NewDense(m+k, n, nil)
		wide.Slice(0, m, 0, n).(*mat.Dense).Copy(a)
		for i := m; i < m+k; i++ {
			rhs := 0.0
			for j := range n {
				if rnd.IntN(2) == 0 {
					v := rnd.NormFloat64()
					wide.Set(i, j, v)
					rhs += v * x[j]
				}
			}
			b = append(b, rhs+0.1*rnd.NormFloat64())
		}
		a = wide
	default:
		for i := range b {
			b[i] += 0.05 * rnd.NormFloat64()
		}
	}
	return a, b, lower, upper
}

func TestBoundedSimplexDualWarm(t *testing.T) {
	t.Parallel()
	for _, fk := range factorKinds {
		t.Run(fk.name, func(t *testing.T) {
			rnd := rand.New(rand.NewPCG(13, 13))
			var dual, fewer, infeasible int
			for range 3000 {
				n := rnd.IntN(25) + 2
				m := rnd.IntN(n) + 1
				c, a, b, lower, upper := boundedProblem(rnd, m, n)
				first, err := fk.solve(nil, c, a, b, lower, upper, 0)
				if err != nil {
					continue
				}
				a2, b2, lower2, upper2 := perturb(rnd, first.X, a, b, lower, upper)
				cold, errCold := fk.solve(nil, c, a2, b2, lower2, upper2, 0)
				warm, errWarm := fk.solve(first.Basis, c, a2, b2, lower2, upper2, 0)
				if errors.Is(errCold, ErrSingular) {
					continue
				}
				if errCold != errWarm {
					t.Fatalf("cold %v, warm %v", errCold, errWarm)
				}
				if takesDualPath(fk, first.Basis, c, a2, b2, lower2, upper2) {
					dual++
					if errCold == nil && warm.Iterations < cold.Iterations {
						fewer++
					}
					if errCold == ErrInfeasible {
						infeasible++
					}
				}
				if errCold != nil {
					continue
				}
				if !scalar.EqualWithinAbsOrRel(cold.F, warm.F, 1e-7, 1e-7) {
					t.Fatalf("cold %v, warm %v", cold.F, warm.F)
				}
				if e := checkKKT(c, a2, b2, lower2, upper2, warm, 1e-7); e != nil {
					t.Fatal(e)
				}
			}
			t.Logf("dual path %d, fewer pivots %d, infeasible %d", dual, fewer, infeasible)
			if dual < 1000 || infeasible < 100 || 4*fewer < 3*(dual-infeasible) {
				t.Fatalf("dual path %d, fewer pivots %d, infeasible %d", dual, fewer, infeasible)
			}
		})
	}
}

// BenchmarkBoundedSimplexBoundChange re-solves a dispatch LP after capping
// one unit's output in every hour, as when unit commitment forces a unit
// down, cold and from the previous optimal basis.
func BenchmarkBoundedSimplexBoundChange(b *testing.B) {
	for _, hours := range []int{8, 24, 48} {
		c, a, rhs, lower, upper := sparseDispatchLP(30, hours)
		first, err := BoundedSimplex(c, a, rhs, lower, upper, 0)
		if err != nil {
			b.Fatal(err)
		}
		capped := append([]float64(nil), upper...)
		for t := range hours {
			for u := 0; u < 30; u += 7 {
				j := u*hours + t
				capped[j] = math.Min(capped[j], 0.5*first.X[j])
			}
		}
		for _, warm := range []bool{false, true} {
			start := first.Basis
			if !warm {
				start = nil
			}
			b.Run(fmt.Sprintf("rows=%d/warm=%v", a.m, warm), func(b *testing.B) {
				var it int
				for b.Loop() {
					res, err := BoundedSimplexFrom(start, c, a, rhs, lower, capped, 0)
					if err != nil {
						b.Fatal(err)
					}
					it = res.Iterations
				}
				b.ReportMetric(float64(it), "pivots")
			})
		}
	}
}
