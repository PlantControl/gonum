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
	"testing"

	"plantcontrol.org/v1/gonum/floats/scalar"
	"plantcontrol.org/v1/gonum/mat"
)

type factorKind struct {
	name string
	new  func(m int) basisFactor
}

var factorKinds = []factorKind{
	{"dense", func(m int) basisFactor { return newDenseLU(m) }},
	{"sparse", func(m int) basisFactor { return newSparseLU(m) }},
}

func (fk factorKind) solve(start *Basis, c []float64, a mat.Matrix, b, lower, upper []float64, tol float64) (BoundedResult, error) {
	return boundedSimplex(fk.new, start, c, a, b, lower, upper, tol)
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

// dispatchLP is a ramp-coupled economic dispatch: units × hours of output
// with capacity bounds, ramp rows, and one demand row per hour.
func dispatchLP(units, hours int) (c []float64, a *mat.Dense, b, lower, upper []float64) {
	rnd := rand.New(rand.NewPCG(5, uint64(hours)))
	n := units*hours + 2*units*(hours-1)
	m := hours + 2*units*(hours-1)
	a = mat.NewDense(m, n, nil)
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
			a.Set(row, p(u, t), 1)
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
			a.Set(row, p(u, t), 1)
			a.Set(row, p(u, t-1), -1)
			a.Set(row, slack, 1)
			b[row] = 10
			slack++
			row++
			a.Set(row, p(u, t-1), 1)
			a.Set(row, p(u, t), -1)
			a.Set(row, slack, 1)
			b[row] = 10
			slack++
			row++
		}
	}
	return c, a, b, lower, upper
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
	for _, hours := range []int{2, 4, 8, 16, 24, 48} {
		c, a, rhs, lower, upper := dispatchLP(30, hours)
		m, _ := a.Dims()
		b.Run(fmt.Sprintf("rows=%d", m), func(b *testing.B) {
			var it int
			for b.Loop() {
				res, err := BoundedSimplex(c, a, rhs, lower, upper, 0)
				if err != nil {
					b.Fatal(err)
				}
				it = res.Iterations
			}
			b.ReportMetric(float64(it), "pivots")
		})
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
