// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lp

import (
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/floats/scalar"
	"plantcontrol.org/v1/gonum/mat"
)

type factorKind struct {
	name string
	new  func(m, n, nnz int) basisFactor
}

var factorKinds = []factorKind{
	{"dense", func(m, _, _ int) basisFactor { return newDenseLU(m) }},
	{"sparse", func(m, _, _ int) basisFactor { return newSparseLU(m) }},
	{"auto", newFactor},
}

func (fk factorKind) solve(start *Basis, c []float64, a mat.Matrix, b, lower, upper []float64, tol float64) (BoundedResult, error) {
	m, n := a.Dims()
	return boundedSimplex(fk.new, start, c, a, b, lower, upper, tol, iterationFactor*(n+m+1))
}

// checkKKT verifies primal feasibility and the reduced-cost signs that
// certify optimality, independently of how the result was found.
func checkKKT(c []float64, a mat.Matrix, b, lower, upper []float64, res BoundedResult, tol float64) error {
	m, n := a.Dims()
	scale := 1.0
	for _, v := range b {
		scale = math.Max(scale, math.Abs(v))
	}
	for j, x := range res.X {
		scale = math.Max(scale, math.Abs(x))
		if x < lower[j]-tol*scale || x > upper[j]+tol*scale {
			return fmt.Errorf("x[%d] = %v outside [%v, %v]", j, x, lower[j], upper[j])
		}
	}
	for i := range m {
		sum := 0.0
		for j := range n {
			sum += a.At(i, j) * res.X[j]
		}
		if math.Abs(sum-b[i]) > tol*scale {
			return fmt.Errorf("row %d: %v, want %v", i, sum, b[i])
		}
	}
	dscale := 1.0
	for _, y := range res.Dual {
		dscale = math.Max(dscale, math.Abs(y))
	}
	for j := range n {
		d := c[j]
		for i := range m {
			d -= a.At(i, j) * res.Dual[i]
		}
		atLo := res.X[j] <= lower[j]+tol*scale
		atUp := res.X[j] >= upper[j]-tol*scale
		dt := tol * dscale * 10
		switch {
		case atLo && atUp:
		case atLo && d < -dt:
			return fmt.Errorf("reduced cost %v < 0 at lower bound, column %d", d, j)
		case atUp && d > dt:
			return fmt.Errorf("reduced cost %v > 0 at upper bound, column %d", d, j)
		case !atLo && !atUp && math.Abs(d) > dt:
			return fmt.Errorf("reduced cost %v of interior column %d", d, j)
		}
	}
	f := 0.0
	for j := range n {
		f += c[j] * res.X[j]
	}
	if !scalar.EqualWithinAbsOrRel(f, res.F, 1e-9, 1e-9) {
		return fmt.Errorf("objective %v, reported %v", f, res.F)
	}
	return nil
}

func TestBoundedSimplexSmall(t *testing.T) {
	t.Parallel()
	for _, fk := range factorKinds {
		t.Run(fk.name, func(t *testing.T) {
			inf := math.Inf(1)
			for _, test := range []struct {
				name         string
				c            []float64
				a            *mat.Dense
				b            []float64
				lower, upper []float64
				f            float64
				err          error
			}{
				{"standard", []float64{-1, -2, 0, 0}, mat.NewDense(2, 4, []float64{-1, 2, 1, 0, 3, 1, 0, 1}), []float64{4, 9},
					[]float64{0, 0, 0, 0}, []float64{inf, inf, inf, inf}, -8, nil},
				{"upper bounds bind", []float64{-1, -1}, mat.NewDense(1, 2, []float64{1, -1}), []float64{0},
					[]float64{0, 0}, []float64{3, 5}, -6, nil},
				{"free variable", []float64{1, 0}, mat.NewDense(1, 2, []float64{1, 1}), []float64{2},
					[]float64{-inf, 0}, []float64{inf, 7}, -5, nil},
				{"infeasible", []float64{1}, mat.NewDense(1, 1, []float64{1}), []float64{5},
					[]float64{0}, []float64{3}, 0, ErrInfeasible},
				{"unbounded", []float64{-1, 0}, mat.NewDense(1, 2, []float64{1, -1}), []float64{0},
					[]float64{0, 0}, []float64{inf, inf}, 0, ErrUnbounded},
				{"fixed", []float64{1, 1}, mat.NewDense(1, 2, []float64{1, 1}), []float64{3},
					[]float64{1, 0}, []float64{1, 9}, 3, nil},
			} {
				res, err := fk.solve(nil, test.c, test.a, test.b, test.lower, test.upper, 0)
				if !errors.Is(err, test.err) {
					t.Errorf("%s: error %v, want %v", test.name, err, test.err)
					continue
				}
				if err != nil {
					continue
				}
				if math.Abs(res.F-test.f) > 1e-9 {
					t.Errorf("%s: objective %v, want %v", test.name, res.F, test.f)
				}
				if err := checkKKT(test.c, test.a, test.b, test.lower, test.upper, res, 1e-9); err != nil {
					t.Errorf("%s: %v", test.name, err)
				}
			}
		})
	}
}

// TestBoundedSimplexMatchesSimplex compares the two solvers on the random
// standard-form problems of TestSimplex.
func TestBoundedSimplexMatchesSimplex(t *testing.T) {
	t.Parallel()
	rnd := rand.New(rand.NewPCG(2, 2))
	for _, cfg := range []struct {
		count, maxN int
		pZero       float64
	}{{4000, 10, 0.2}, {4000, 10, 0}, {100, 60, 0.3}} {
		for range cfg.count {
			n := rnd.IntN(cfg.maxN) + 2
			m := rnd.IntN(n-1) + 1
			value := func() float64 {
				if rnd.Float64() < cfg.pZero {
					return 0
				}
				return rnd.NormFloat64()
			}
			a := mat.NewDense(m, n, nil)
			for i := range m {
				for j := range n {
					a.Set(i, j, value())
				}
			}
			b, c := make([]float64, m), make([]float64, n)
			for i := range b {
				b[i] = value()
			}
			for j := range c {
				c[j] = value()
			}
			lower, upper := make([]float64, n), make([]float64, n)
			for j := range upper {
				upper[j] = math.Inf(1)
			}
			want, _, errWant := Simplex(c, a, b, 0, nil)
			res, err := BoundedSimplex(c, a, b, lower, upper, 0)
			switch {
			case errWant == ErrSingular || errWant == ErrZeroRow || errWant == ErrZeroColumn:
			case err != errWant:
				t.Fatalf("m=%d n=%d: error %v, Simplex %v", m, n, err, errWant)
			case err == nil:
				if e := checkKKT(c, a, b, lower, upper, res, 1e-8); e != nil {
					t.Fatalf("m=%d n=%d: %v", m, n, e)
				}
				if !scalar.EqualWithinAbsOrRel(res.F, want, 1e-8, 1e-8) {
					t.Fatalf("m=%d n=%d: objective %v, Simplex %v", m, n, res.F, want)
				}
			}
		}
	}
}

// boundedProblem is feasible by construction: b = A x0 for x0 within the
// bounds. Some bounds are infinite, some equal, and costs may vanish.
func boundedProblem(rnd *rand.Rand, m, n int) (c []float64, a *mat.Dense, b, lower, upper []float64) {
	a = mat.NewDense(m, n, nil)
	lower, upper = make([]float64, n), make([]float64, n)
	x0 := make([]float64, n)
	c = make([]float64, n)
	for j := range n {
		switch rnd.IntN(5) {
		case 0:
			lower[j], upper[j] = math.Inf(-1), math.Inf(1)
		case 1:
			lower[j], upper[j] = 0, math.Inf(1)
		case 2:
			lower[j] = rnd.NormFloat64()
			upper[j] = lower[j]
		default:
			lower[j] = rnd.NormFloat64()
			upper[j] = lower[j] + 3*rnd.Float64()
		}
		lo, hi := math.Max(lower[j], -2), math.Min(upper[j], 2)
		if lo > hi {
			lo, hi = lower[j], lower[j]
		}
		x0[j] = lo + (hi-lo)*rnd.Float64()
		if rnd.IntN(4) > 0 {
			c[j] = rnd.NormFloat64()
		}
		// Free columns with cost make most instances unbounded; keep them
		// costless half the time.
		if math.IsInf(lower[j], -1) && rnd.IntN(2) == 0 {
			c[j] = 0
		}
	}
	b = make([]float64, m)
	for i := range m {
		for j := range n {
			if rnd.IntN(3) > 0 {
				v := rnd.NormFloat64()
				a.Set(i, j, v)
				b[i] += v * x0[j]
			}
		}
	}
	return c, a, b, lower, upper
}

func TestBoundedSimplexKKT(t *testing.T) {
	t.Parallel()
	for _, fk := range factorKinds {
		t.Run(fk.name, func(t *testing.T) {
			rnd := rand.New(rand.NewPCG(3, 3))
			solved := 0
			for range 3000 {
				n := rnd.IntN(30) + 1
				m := rnd.IntN(n) + 1
				c, a, b, lower, upper := boundedProblem(rnd, m, n)
				res, err := fk.solve(nil, c, a, b, lower, upper, 0)
				switch {
				case err == nil:
					solved++
					if e := checkKKT(c, a, b, lower, upper, res, 1e-8); e != nil {
						t.Fatalf("m=%d n=%d: %v", m, n, e)
					}
				case errors.Is(err, ErrInfeasible):
					t.Fatalf("m=%d n=%d: feasible by construction", m, n)
				case errors.Is(err, ErrUnbounded), errors.Is(err, ErrSingular):
				default:
					t.Fatalf("m=%d n=%d: %v", m, n, err)
				}
			}
			if solved < 1000 {
				t.Fatalf("only %d of 3000 problems solved", solved)
			}
		})
	}
}

func TestBoundedSimplexDegenerate(t *testing.T) {
	t.Parallel()
	for _, fk := range factorKinds {
		t.Run(fk.name, func(t *testing.T) {
			// Assignment polytopes are highly degenerate.
			rnd := rand.New(rand.NewPCG(4, 4))
			for size := 2; size <= 12; size++ {
				n := size * size
				a := mat.NewDense(2*size, n, nil)
				b := make([]float64, 2*size)
				c := make([]float64, n)
				lower, upper := make([]float64, n), make([]float64, n)
				for i := range size {
					b[i], b[size+i] = 1, 1
					for j := range size {
						a.Set(i, i*size+j, 1)
						a.Set(size+j, i*size+j, 1)
						c[i*size+j] = float64(rnd.IntN(4))
						upper[i*size+j] = 1
					}
				}
				res, err := fk.solve(nil, c, a, b, lower, upper, 0)
				if err != nil {
					t.Fatalf("size %d: %v", size, err)
				}
				if e := checkKKT(c, a, b, lower, upper, res, 1e-9); e != nil {
					t.Fatalf("size %d: %v", size, e)
				}
			}
		})
	}
}

// zeroRowsLP is a random sparse LP in which most rows have a zero
// right-hand side and the coefficients spread over four magnitudes, so
// long runs of degenerate pivots meet ties between pivots of very
// different size. If open, about half the columns have no upper bound, and
// most problems are unbounded.
func zeroRowsLP(seed uint64, m, n int, open bool) (c []float64, a *cscMatrix, b, lower, upper []float64) {
	type entry struct {
		i int
		v float64
	}
	rnd := rand.New(rand.NewPCG(seed, 7))
	cols := make([][]entry, 0, n+m)
	for range n {
		var col []entry
		for i := range m {
			if rnd.Float64() < 4/float64(m) {
				v := math.Pow(10, -4*rnd.Float64())
				if rnd.IntN(2) == 0 {
					v = -v
				}
				col = append(col, entry{i, v})
			}
		}
		if len(col) == 0 {
			col = append(col, entry{rnd.IntN(m), 1})
		}
		cols = append(cols, col)
		c = append(c, -rnd.Float64())
		lower = append(lower, 0)
		upper = append(upper, 1+rnd.Float64())
		if open && rnd.IntN(2) == 0 {
			upper[len(upper)-1] = math.Inf(1)
			if rnd.IntN(10) < 7 {
				c[len(c)-1] = rnd.Float64()
			}
		}
	}
	b = make([]float64, m)
	for i := range m {
		if rnd.IntN(10) == 0 {
			b[i] = rnd.Float64()
		}
		if rnd.IntN(8) == 0 {
			continue
		}
		cols = append(cols, []entry{{i, 1}})
		c = append(c, 0)
		lower = append(lower, 0)
		upper = append(upper, math.Inf(1))
	}
	a = &cscMatrix{m: m, n: len(cols), start: []int{0}}
	for _, col := range cols {
		for _, e := range col {
			a.index = append(a.index, e.i)
			a.value = append(a.value, e.v)
		}
		a.start = append(a.start, len(a.index))
	}
	return c, a, b, lower, upper
}

// With these seeds Bland's rule used to take the smallest index among tied
// rows whatever its pivot, and the sparse factorization then failed with
// ErrSingular.
func TestBoundedSimplexBlandPivot(t *testing.T) {
	t.Parallel()
	for _, seed := range []uint64{5, 28} {
		c, a, b, lower, upper := zeroRowsLP(seed, 80, 300, false)
		var want float64
		for k, fk := range factorKinds {
			res, err := fk.solve(nil, c, a, b, lower, upper, 0)
			if err != nil {
				t.Fatalf("seed %d %s: %v", seed, fk.name, err)
			}
			if e := checkKKT(c, a, b, lower, upper, res, 1e-7); e != nil {
				t.Fatalf("seed %d %s: %v", seed, fk.name, e)
			}
			if k == 0 {
				want = res.F
			} else if !scalar.EqualWithinAbsOrRel(res.F, want, 1e-8, 1e-8) {
				t.Errorf("seed %d %s: objective %v, dense %v", seed, fk.name, res.F, want)
			}
		}
	}
}

// The verdicts and objectives are those of HiGHS without presolve at
// tolerances of 1e-9. On each of these the primal simplex method from the
// artificial basis failed with ErrSingular or ErrIterationLimit, or, for
// seed 74, returned an optimum of -2.5e17.
func TestBoundedSimplexZeroRows(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		m, n int
		open bool
		seed uint64
		err  error
		f    float64
	}{
		{40, 120, false, 169, nil, -59.05114276428671},
		{80, 300, false, 97, nil, -198.17994043201563},
		{150, 600, false, 16, nil, -357.15887093514783},
		{150, 600, false, 19, nil, -348.0467155951103},
		{80, 300, false, 118, ErrInfeasible, 0},
		{150, 600, false, 5, ErrInfeasible, 0},
		{150, 600, false, 13, ErrInfeasible, 0},
		{20, 60, true, 74, ErrUnbounded, 0},
		{20, 60, true, 88, ErrUnbounded, 0},
		{150, 600, true, 0, ErrUnbounded, 0},
	} {
		c, a, b, lower, upper := zeroRowsLP(test.seed, test.m, test.n, test.open)
		for _, fk := range factorKinds {
			name := fmt.Sprintf("%dx%d open=%t seed %d %s", test.m, test.n, test.open, test.seed, fk.name)
			res, err := fk.solve(nil, c, a, b, lower, upper, 0)
			if !errors.Is(err, test.err) || (err == nil) != (test.err == nil) {
				t.Errorf("%s: got %v, want %v", name, err, test.err)
				continue
			}
			if err != nil {
				continue
			}
			if e := checkKKT(c, a, b, lower, upper, res, 1e-7); e != nil {
				t.Errorf("%s: %v", name, e)
			}
			if !scalar.EqualWithinAbsOrRel(res.F, test.f, 1e-7, 1e-7) {
				t.Errorf("%s: objective %v, want %v", name, res.F, test.f)
			}
		}
	}
}

func TestCertify(t *testing.T) {
	t.Parallel()
	// x₀ + x₁ = b with 0 ≤ x ≤ 1, and a third column with no upper bound.
	a := mat.NewDense(1, 3, []float64{1, 1, 0})
	lower := []float64{0, 0, 0}
	upper := []float64{1, 1, math.Inf(1)}
	for _, test := range []struct {
		b    float64
		y    float64
		want bool
	}{
		{3, 1, true},
		{3, -2, true},
		{-0.5, 1, true},
		{2, 1, false},
		{2 + 1e-12, 1, false},
		{1, 1, false},
	} {
		s := newBounded(newFactor, make([]float64, 3), a, []float64{test.b}, lower, upper, 1e-9)
		if got := s.certify([]float64{test.y}); got != test.want {
			t.Errorf("b = %v, y = %v: certified %t, want %t", test.b, test.y, got, test.want)
		}
	}
	// A multiplier that reaches the unbounded column proves nothing.
	a.Set(0, 2, 1)
	s := newBounded(newFactor, make([]float64, 3), a, []float64{3}, lower, upper, 1e-9)
	if s.certify([]float64{1}) {
		t.Error("certified through a column with no upper bound")
	}
}

func TestBlandRatioSkipsSmallPivot(t *testing.T) {
	t.Parallel()
	// Two basic variables at their lower bounds tie at a zero step.
	for _, test := range []struct {
		alpha []float64
		want  int
	}{
		{[]float64{0.5, 1}, 0},
		{[]float64{1e-8, 1}, 1},
		{[]float64{1, 1e-8}, 0},
	} {
		inf := math.Inf(1)
		s := &bounded{
			tol:   1e-9,
			bland: true,
			alpha: test.alpha,
			head:  []int{0, 1},
			x:     []float64{0, 0, 0},
			lower: []float64{0, 0, 0},
			upper: []float64{inf, inf, inf},
		}
		theta, row, flip := s.ratio(2, 1)
		if theta != 0 || row != test.want || flip {
			t.Errorf("alpha %v: step %v, row %d, flip %v; want row %d", test.alpha, theta, row, flip, test.want)
		}
	}
}

func TestBoundedSimplexWarm(t *testing.T) {
	t.Parallel()
	for _, fk := range factorKinds {
		t.Run(fk.name, func(t *testing.T) {
			rnd := rand.New(rand.NewPCG(6, 6))
			warmed, total := 0, 0
			for range 500 {
				n := rnd.IntN(25) + 2
				m := rnd.IntN(n) + 1
				c, a, b, lower, upper := boundedProblem(rnd, m, n)
				first, err := fk.solve(nil, c, a, b, lower, upper, 0)
				if err != nil {
					continue
				}
				// New costs, then one appended column.
				for j := range c {
					c[j] += 0.3 * rnd.NormFloat64()
				}
				wide := mat.NewDense(m, n+1, nil)
				wide.Slice(0, m, 0, n).(*mat.Dense).Copy(a)
				for i := range m {
					wide.Set(i, n, rnd.NormFloat64())
				}
				c = append(c, rnd.NormFloat64())
				lower = append(lower, 0)
				upper = append(upper, 1)
				cold, errCold := fk.solve(nil, c, wide, b, lower, upper, 0)
				warm, errWarm := fk.solve(first.Basis, c, wide, b, lower, upper, 0)
				if (errCold == nil) != (errWarm == nil) {
					t.Fatalf("cold %v, warm %v", errCold, errWarm)
				}
				if errCold != nil {
					continue
				}
				total++
				if warm.Iterations < cold.Iterations {
					warmed++
				}
				if !scalar.EqualWithinAbsOrRel(cold.F, warm.F, 1e-8, 1e-8) {
					t.Fatalf("cold %v, warm %v", cold.F, warm.F)
				}
				if e := checkKKT(c, wide, b, lower, upper, warm, 1e-8); e != nil {
					t.Fatal(e)
				}
			}
			if warmed*2 < total {
				t.Fatalf("warm start saved pivots in only %d of %d solves", warmed, total)
			}
		})
	}
}

// TestBoundedSimplexBudget checks that one call stays within its pivot
// budget over the whole cascade of a warm start, the dual simplex method
// from the slack basis and the primal method from scratch, and that a
// budget that runs out ends in ErrIterationLimit rather than another
// verdict.
func TestBoundedSimplexBudget(t *testing.T) {
	t.Parallel()
	for _, fk := range factorKinds {
		t.Run(fk.name, func(t *testing.T) {
			rnd := rand.New(rand.NewPCG(7, 7))
			limited, solved := 0, 0
			for range 150 {
				n := rnd.IntN(25) + 5
				m := rnd.IntN(n) + 1
				c, a, b, lower, upper := boundedProblem(rnd, m, n)
				first, err := fk.solve(nil, c, a, b, lower, upper, 0)
				if err != nil {
					continue
				}
				// The old basis is neither primal nor dual feasible for
				// reversed costs and narrowed boxes.
				for j := range c {
					c[j] = -c[j] + 0.5*rnd.NormFloat64()
					if w := upper[j] - lower[j]; w > 0 && !math.IsInf(w, 1) {
						upper[j] = lower[j] + 0.4*w
					}
				}
				want, errWant := fk.solve(first.Basis, c, a, b, lower, upper, 0)
				for _, budget := range []int{0, 1, 2, 3, 5, 8, 13, 21, 34, 55} {
					got, err := boundedSimplex(fk.new, first.Basis, c, a, b, lower, upper, 0, budget)
					if got.Iterations > budget {
						t.Fatalf("budget %d: %d pivots", budget, got.Iterations)
					}
					switch {
					case errors.Is(err, ErrIterationLimit):
						limited++
					case (err == nil) != (errWant == nil) || err != nil && !errors.Is(err, errWant):
						t.Fatalf("budget %d: %v, want %v", budget, err, errWant)
					case err == nil:
						solved++
						if !scalar.EqualWithinAbsOrRel(got.F, want.F, 1e-8, 1e-8) {
							t.Fatalf("budget %d: F = %v, want %v", budget, got.F, want.F)
						}
					}
				}
			}
			if limited == 0 || solved == 0 {
				t.Fatalf("%d limited, %d solved: the budgets do not bind", limited, solved)
			}
		})
	}
}

// dispatchLP is a ramp-coupled economic dispatch: units × hours of output
// with capacity bounds, ramp rows, and one demand row per hour.
func dispatchLP(units, hours int) (c []float64, a *mat.Dense, b, lower, upper []float64) {
	c, csc, b, lower, upper := sparseDispatchLP(units, hours)
	a = mat.NewDense(csc.m, csc.n, nil)
	for j := range csc.n {
		csc.DoColNonZero(j, a.Set)
	}
	return c, a, b, lower, upper
}

func sparseDispatchLP(units, hours int) (c []float64, a *cscMatrix, b, lower, upper []float64) {
	rnd := rand.New(rand.NewPCG(5, uint64(hours)))
	n := units*hours + 2*units*(hours-1)
	m := hours + 2*units*(hours-1)
	cols := make([][]struct {
		i int
		v float64
	}, n)
	set := func(i, j int, v float64) {
		cols[j] = append(cols[j], struct {
			i int
			v float64
		}{i, v})
	}
	b = make([]float64, m)
	c = make([]float64, n)
	lower, upper = make([]float64, n), make([]float64, n)
	for j := range upper {
		upper[j] = math.Inf(1)
	}
	p := func(u, t int) int { return u*hours + t }
	slack, row := units*hours, 0
	for t := range hours {
		for u := range units {
			set(row, p(u, t), 1)
			upper[p(u, t)] = 30
		}
		b[row] = 300 + 200*rnd.Float64()
		row++
	}
	for u := range units {
		cost := 10 + 40*rnd.Float64()
		for t := range hours {
			c[p(u, t)] = cost
		}
		for t := 1; t < hours; t++ {
			set(row, p(u, t), 1)
			set(row, p(u, t-1), -1)
			set(row, slack, 1)
			b[row] = 10
			slack++
			row++
			set(row, p(u, t-1), 1)
			set(row, p(u, t), -1)
			set(row, slack, 1)
			b[row] = 10
			slack++
			row++
		}
	}
	a = &cscMatrix{m: m, n: n, start: []int{0}}
	for _, col := range cols {
		slices.SortFunc(col, func(x, y struct {
			i int
			v float64
		}) int {
			return x.i - y.i
		})
		for _, e := range col {
			a.index = append(a.index, e.i)
			a.value = append(a.value, e.v)
		}
		a.start = append(a.start, len(a.index))
	}
	return c, a, b, lower, upper
}

// cscMatrix is a compressed sparse column matrix implementing
// mat.ColNonZeroDoer.
type cscMatrix struct {
	m, n         int
	start, index []int
	value        []float64
}

func (a *cscMatrix) Dims() (int, int) { return a.m, a.n }

func (a *cscMatrix) At(i, j int) float64 {
	for k := a.start[j]; k < a.start[j+1]; k++ {
		if a.index[k] == i {
			return a.value[k]
		}
	}
	return 0
}

func (a *cscMatrix) T() mat.Matrix { return mat.Transpose{Matrix: a} }

func (a *cscMatrix) DoColNonZero(j int, fn func(i, j int, v float64)) {
	for k := a.start[j]; k < a.start[j+1]; k++ {
		fn(a.index[k], j, a.value[k])
	}
}

func TestBoundedSimplexDispatch(t *testing.T) {
	t.Parallel()
	for _, fk := range factorKinds {
		t.Run(fk.name, func(t *testing.T) {
			c, a, b, lower, upper := dispatchLP(30, 8)
			res, err := fk.solve(nil, c, a, b, lower, upper, 0)
			if err != nil {
				t.Fatal(err)
			}
			if e := checkKKT(c, a, b, lower, upper, res, 1e-8); e != nil {
				t.Fatal(e)
			}
		})
	}
}

func BenchmarkBoundedSimplexDispatch(b *testing.B) {
	for _, hours := range []int{2, 4, 8, 16, 24, 48, 96, 168} {
		c, a, rhs, lower, upper := sparseDispatchLP(30, hours)
		for _, fk := range factorKinds {
			if fk.name == "dense" && a.m > 1500 {
				continue
			}
			b.Run(fmt.Sprintf("rows=%d/%s", a.m, fk.name), func(b *testing.B) {
				var it int
				for b.Loop() {
					res, err := fk.solve(nil, c, a, rhs, lower, upper, 0)
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

// TestBoundedSimplexWarmStall replays a pooling relaxation warm started
// from its parent node's basis. Dual phase 1 leaves one reduced cost of the
// wrong sign at 1.3 times its tolerance; refusing it sent the solve through
// the zero-cost dual to a feasible point and a primal method that stalled on
// degenerate pivots, 211,424 pivots without a budget across the cascade and
// 47,424 with one. Shifting that cost lets the dual method finish from the
// warm basis.
func TestBoundedSimplexWarmStall(t *testing.T) {
	t.Parallel()
	lp := loadPoolLP(t, "testdata/pool_warm_stall.json.gz")
	res, err := BoundedSimplexFrom(lp.start, lp.c, lp.a, lp.b, lp.lower, lp.upper, 1e-9)
	if err != nil {
		t.Fatal(err)
	}
	if want := -275.64033179; !scalar.EqualWithinAbsOrRel(res.F, want, 1e-8, 1e-8) {
		t.Errorf("F = %v, want %v", res.F, want)
	}
	if e := checkKKT(lp.c, lp.a, lp.b, lp.lower, lp.upper, res, 1e-7); e != nil {
		t.Error(e)
	}
	if limit := lp.a.n + lp.a.m + 1; res.Iterations > limit {
		t.Errorf("%d pivots, want at most %d", res.Iterations, limit)
	}
}

// TestBoundedSimplexPoolDegenerate solves a pooling node relaxation, warm and
// cold, on which the primal method meets long runs of degenerate pivots:
// after the dual method on the warm start, in the cleanup of its cost
// perturbation, and in both phases from the artificial basis. Under Bland's
// rule alone each attempt ran to its limit, 229,150 pivots without a
// verdict. HiGHS 1.15 finds -422.36912701.
func TestBoundedSimplexPoolDegenerate(t *testing.T) {
	t.Parallel()
	lp := loadPoolLP(t, "testdata/pool_node_degenerate.json.gz")
	for _, start := range []*Basis{lp.start, nil} {
		res, err := BoundedSimplexFrom(start, lp.c, lp.a, lp.b, lp.lower, lp.upper, 1e-9)
		if err != nil {
			t.Fatalf("warm=%t: %v", start != nil, err)
		}
		if want := -422.36912701; !scalar.EqualWithinAbsOrRel(res.F, want, 1e-8, 1e-8) {
			t.Errorf("warm=%t: F = %v, want %v", start != nil, res.F, want)
		}
		if e := checkKKT(lp.c, lp.a, lp.b, lp.lower, lp.upper, res, 1e-7); e != nil {
			t.Errorf("warm=%t: %v", start != nil, e)
		}
		if limit := 4 * (lp.a.n + lp.a.m + 1); res.Iterations > limit {
			t.Errorf("warm=%t: %d pivots, want at most %d", start != nil, res.Iterations, limit)
		}
	}
}

// TestPhaseOneStopsFeasible checks that the first primal phase from the
// artificial basis stops once the artificial variables are zero. On this
// pooling relaxation it went on pivoting at a phase objective of 1e-16 to
// the iteration limit, 229,150 pivots.
func TestPhaseOneStopsFeasible(t *testing.T) {
	t.Parallel()
	lp := loadPoolLP(t, "testdata/pool_node_degenerate.json.gz")
	s := newBounded(newFactor, lp.c, lp.a, lp.b, lp.lower, lp.upper, 1e-9)
	if err := s.refactor(); err != nil {
		t.Fatal(err)
	}
	if err := s.run(); err != nil {
		t.Fatal(err)
	}
	if !s.fresh || !s.artificialsClear() {
		t.Error("phase 1 returned with an artificial variable above the tolerance")
	}
	if limit := lp.a.n + lp.a.m + 1; s.iterations > limit {
		t.Errorf("%d pivots, want at most %d", s.iterations, limit)
	}
}

// TestExpand checks that expand widens the finite bounds of basic structural
// variables only, and that unexpand restores every bound exactly and puts
// nonbasic variables back on them.
func TestExpand(t *testing.T) {
	t.Parallel()
	lp := loadPoolLP(t, "testdata/pool_node_degenerate.json.gz")
	s := newBounded(newFactor, lp.c, lp.a, lp.b, lp.lower, lp.upper, 1e-9)
	if !s.warm(lp.start) {
		t.Fatal("warm basis singular")
	}
	lower := append([]float64(nil), s.lower...)
	upper := append([]float64(nil), s.upper...)
	if !s.expand() {
		t.Fatal("nothing expanded")
	}
	if s.expand() {
		t.Error("second expand of the same basis widened bounds")
	}
	for j := range s.total {
		widened := s.lower[j] != lower[j] || s.upper[j] != upper[j]
		want := j < s.n && s.status[j] == basic && lower[j] != upper[j]
		if widened != want {
			t.Fatalf("variable %d (status %d): widened %t, want %t", j, s.status[j], widened, want)
		}
		if widened && (s.lower[j] > lower[j] || s.upper[j] < upper[j]) {
			t.Fatalf("variable %d narrowed: [%v, %v] from [%v, %v]", j, s.lower[j], s.upper[j], lower[j], upper[j])
		}
	}
	// Move one expanded basic variable to a nonbasic bound, as a pivot would.
	moved := -1
	for i, j := range s.head {
		if j < s.n && s.expanded[j] && !math.IsInf(s.lower[j], -1) {
			s.head[i] = s.n + i
			s.status[s.n+i] = basic
			s.x[j], s.status[j] = s.lower[j], atLower
			moved = j
			break
		}
	}
	if moved < 0 {
		t.Fatal("no basic variable with a finite lower bound")
	}
	s.unexpand(false)
	if !slices.Equal(s.lower, lower) || !slices.Equal(s.upper, upper) {
		t.Error("bounds not restored")
	}
	if s.x[moved] != lower[moved] {
		t.Errorf("nonbasic x = %v, want its bound %v", s.x[moved], lower[moved])
	}
}

// poolLP is a pooling relaxation dumped with the basis it was started from.
type poolLP struct {
	c, b, lower, upper []float64
	a                  *cscMatrix
	start              *Basis
}

func loadPoolLP(t *testing.T, path string) poolLP {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var lp struct {
		Rows         int
		Start, Index []int
		Value, C, B  []float64
		Lower, Upper []any
		Head, Status []int
	}
	if err := json.NewDecoder(z).Decode(&lp); err != nil {
		t.Fatal(err)
	}
	bound := func(v []any) []float64 {
		out := make([]float64, len(v))
		for i, x := range v {
			switch x {
			case "inf":
				out[i] = math.Inf(1)
			case "-inf":
				out[i] = math.Inf(-1)
			default:
				out[i] = x.(float64)
			}
		}
		return out
	}
	return poolLP{
		c: lp.C, b: lp.B, lower: bound(lp.Lower), upper: bound(lp.Upper),
		a:     &cscMatrix{m: lp.Rows, n: len(lp.Start) - 1, start: lp.Start, index: lp.Index, value: lp.Value},
		start: &Basis{head: lp.Head, status: lp.Status},
	}
}

// TestDualStartShift checks that dualStart shifts the cost of a variable
// with one finite bound whose reduced cost has the wrong sign at the level of
// the tolerance, that dual takes the shift back out, that a larger error is
// refused without a shift, and that a warm solve through the shift is
// optimal for the true costs.
func TestDualStartShift(t *testing.T) {
	t.Parallel()
	// x0 - x1 = -1 with x0 in [0, 10] basic at -1 and x1 >= 0 at zero: the
	// basis is primal infeasible, and x1's reduced cost is -eps.
	a := mat.NewDense(1, 2, []float64{1, -1})
	b := []float64{-1}
	lower := []float64{0, 0}
	upper := []float64{10, math.Inf(1)}
	start := &Basis{head: []int{0}, status: []int{basic, atLower}}
	for _, test := range []struct {
		eps     float64
		shifted bool
	}{
		{5e-9, true},
		{1e-3, false},
	} {
		c := []float64{1, -1 - test.eps}
		s := newBounded(newFactor, c, a, b, lower, upper, 1e-9)
		if !s.warm(start) {
			t.Fatalf("eps=%g: warm basis singular", test.eps)
		}
		if got := s.dualStart(); got != test.shifted {
			t.Fatalf("eps=%g: dualStart = %v, want %v", test.eps, got, test.shifted)
		}
		if !test.shifted {
			if s.shifted != nil || !slices.Equal(s.cost, s.c) {
				t.Errorf("eps=%g: refused start shifted costs %v", test.eps, s.cost)
			}
		} else {
			if s.shifted == nil || !s.shifted[1] || s.cost[1] == s.c[1] || s.d[1] != 0 {
				t.Errorf("eps=%g: x1 not shifted: cost %v, d %v", test.eps, s.cost, s.d[1])
			}
			if err := s.dual(); err != nil {
				t.Fatalf("eps=%g: dual: %v", test.eps, err)
			}
			if s.shifted != nil || !slices.Equal(s.cost, s.c) {
				t.Errorf("eps=%g: dual left costs %v, want %v", test.eps, s.cost, s.c)
			}
		}
		res, err := BoundedSimplexFrom(start, c, a, b, lower, upper, 1e-9)
		if err != nil {
			t.Fatalf("eps=%g: %v", test.eps, err)
		}
		// x0 at its upper bound 10 and x1 = 11.
		if want := 10 + 11*c[1]; !scalar.EqualWithinAbsOrRel(res.F, want, 1e-12, 1e-12) {
			t.Errorf("eps=%g: F = %v, want %v", test.eps, res.F, want)
		}
		if e := checkKKT(c, a, b, lower, upper, res, 1e-9); e != nil {
			t.Errorf("eps=%g: %v", test.eps, e)
		}
	}
}

// TestBoundedSimplexWarmChain replays the master LPs of a Dantzig–Wolfe
// unit commitment dispatch, each warm-started from the previous basis as
// columns are appended. Degenerate convexity rows made this cycle.
func TestBoundedSimplexWarmChain(t *testing.T) {
	t.Parallel()
	for _, fk := range factorKinds {
		t.Run(fk.name, func(t *testing.T) {
			f, err := os.Open("testdata/dw_master_chain.json.gz")
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			z, err := gzip.NewReader(f)
			if err != nil {
				t.Fatal(err)
			}
			var chain []struct {
				M, N                  int
				C, A, B, Lower, Upper []float64
			}
			if err := json.NewDecoder(z).Decode(&chain); err != nil {
				t.Fatal(err)
			}
			var basis *Basis
			for round, lp := range chain {
				for j := range lp.Upper {
					if lp.Upper[j] >= 1e300 {
						lp.Upper[j] = math.Inf(1)
					}
					if lp.Lower[j] <= -1e300 {
						lp.Lower[j] = math.Inf(-1)
					}
				}
				a := mat.NewDense(lp.M, lp.N, lp.A)
				cold, errCold := fk.solve(nil, lp.C, a, lp.B, lp.Lower, lp.Upper, 0)
				if errCold != nil {
					t.Fatalf("round %d cold: %v", round+1, errCold)
				}
				warm, err := fk.solve(basis, lp.C, a, lp.B, lp.Lower, lp.Upper, 0)
				if err != nil {
					t.Fatalf("round %d warm: %v", round+1, err)
				}
				if !scalar.EqualWithinAbsOrRel(cold.F, warm.F, 1e-7, 1e-9) {
					t.Fatalf("round %d: cold %v, warm %v", round+1, cold.F, warm.F)
				}
				if e := checkKKT(lp.C, a, lp.B, lp.Lower, lp.Upper, warm, 1e-7); e != nil {
					t.Fatalf("round %d: %v", round+1, e)
				}
				basis = warm.Basis
			}
		})
	}
}

func BenchmarkBoundedSimplexRandom(b *testing.B) {
	for _, size := range []int{5, 15, 40} {
		rnd := rand.New(rand.NewPCG(9, uint64(size)))
		type problem struct {
			c, b, lower, upper []float64
			a                  *mat.Dense
		}
		var problems []problem
		for range 50 {
			n := size + rnd.IntN(size)
			m := 1 + rnd.IntN(n)
			c, a, rhs, lower, upper := boundedProblem(rnd, m, n)
			problems = append(problems, problem{c, rhs, lower, upper, a})
		}
		for _, fk := range factorKinds {
			b.Run(fmt.Sprintf("n=%d/%s", size, fk.name), func(b *testing.B) {
				for b.Loop() {
					for _, p := range problems {
						fk.solve(nil, p.c, p.a, p.b, p.lower, p.upper, 0)
					}
				}
			})
		}
	}
}
