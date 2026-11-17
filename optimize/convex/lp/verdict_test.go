// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lp

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

const (
	plantOptimal = iota
	plantInfeasible
	plantUnbounded
)

// plantedLP is a random sparse LP whose verdict is known by construction.
// x0 is feasible unless the problem is infeasible, and f is the optimal
// objective if it has one.
type plantedLP struct {
	c            []float64
	a            *cscMatrix
	b            []float64
	lower, upper []float64
	x0           []float64
	f            float64
}

// planted builds an LP around a point x0 and row multipliers y0. The costs
// are Aᵀy0 plus a reduced cost of the sign that x0's bound allows, so x0 is
// optimal. An infeasible problem adds a row over boxed columns with a
// right-hand side beyond their reach; an unbounded one adds a column that
// cancels a few columns with no upper bound and whose cost falls along
// them. Coefficients spread over 10^spread, and each row is scaled by up to
// 10^rowScale, which changes neither the verdict nor the objective.
func planted(rnd *rand.Rand, m, n, kind int, spread, rowScale float64) plantedLP {
	type entry struct {
		i int
		v float64
	}
	inf := math.Inf(1)
	var cols [][]entry
	var lower, upper, x0 []float64
	mag := func() float64 {
		v := math.Pow(10, -spread*rnd.Float64())
		if rnd.IntN(2) == 0 {
			v = -v
		}
		return v
	}
	for range n {
		var col []entry
		for i := range m {
			if rnd.Float64() < 3/float64(m) {
				col = append(col, entry{i, mag()})
			}
		}
		if len(col) == 0 {
			col = append(col, entry{rnd.IntN(m), mag()})
		}
		cols = append(cols, col)
		var lo, up, x float64
		switch rnd.IntN(10) {
		case 0, 1, 2, 3:
			lo, up = 0, 1+rnd.Float64()
		case 4, 5, 6:
			lo, up = 0, inf
		case 7:
			lo, up = -inf, rnd.Float64()
		case 8:
			lo, up = -inf, inf
		default:
			lo = rnd.NormFloat64()
			up = lo
		}
		switch t := rnd.IntN(10); {
		case t < 5 && !math.IsInf(lo, -1):
			x = lo
		case t < 7 && !math.IsInf(up, 1):
			x = up
		case !math.IsInf(lo, -1) && !math.IsInf(up, 1):
			x = lo + rnd.Float64()*(up-lo)
		case !math.IsInf(lo, -1):
			x = lo + rnd.ExpFloat64()
		case !math.IsInf(up, 1):
			x = up - rnd.ExpFloat64()
		default:
			x = rnd.NormFloat64()
		}
		lower, upper, x0 = append(lower, lo), append(upper, up), append(x0, x)
	}
	for i := range m {
		if rnd.IntN(8) == 0 {
			continue
		}
		cols = append(cols, []entry{{i, 1}})
		lower, upper = append(lower, 0), append(upper, inf)
		x := 0.0
		if rnd.IntN(3) == 0 {
			x = rnd.ExpFloat64()
		}
		x0 = append(x0, x)
	}
	y0 := make([]float64, m)
	for i := range y0 {
		if rnd.IntN(3) != 0 {
			y0[i] = rnd.NormFloat64()
		}
	}
	c := make([]float64, len(cols))
	for j, col := range cols {
		for _, e := range col {
			c[j] += e.v * y0[e.i]
		}
		if rnd.IntN(3) == 0 {
			continue
		}
		d := rnd.Float64()
		switch {
		case lower[j] == upper[j]:
			c[j] += rnd.NormFloat64()
		case x0[j] == lower[j]:
			c[j] += d
		case x0[j] == upper[j]:
			c[j] -= d
		}
	}
	rows := m
	b := make([]float64, m)
	for j, col := range cols {
		for _, e := range col {
			b[e.i] += e.v * x0[j]
		}
	}
	f := 0.0
	for j := range c {
		f += c[j] * x0[j]
	}
	switch kind {
	case plantInfeasible:
		var row []entry
		hi, size := 0.0, 0.0
		for j := range cols {
			if math.IsInf(lower[j], -1) || math.IsInf(upper[j], 1) || len(row) >= 4 || rnd.IntN(3) != 0 {
				continue
			}
			w := mag()
			row = append(row, entry{j, w})
			hi += math.Max(w*lower[j], w*upper[j])
			size += math.Abs(w*lower[j]) + math.Abs(w*upper[j])
		}
		if len(row) == 0 {
			cols = append(cols, nil)
			lower, upper, c = append(lower, 0), append(upper, 1), append(c, rnd.NormFloat64())
			row = append(row, entry{len(cols) - 1, 1})
			hi, size = 1, 1
		}
		for _, e := range row {
			cols[e.i] = append(cols[e.i], entry{rows, e.v})
		}
		delta := []float64{1e-3, 0.1, 1}[rnd.IntN(3)]
		b = append(b, hi+delta*(1+size))
		rows++
	case plantUnbounded:
		acc := map[int]float64{}
		cd := 0.0
		picked := 0
		for j := range cols {
			if !math.IsInf(upper[j], 1) || picked >= 3 || rnd.IntN(2) != 0 {
				continue
			}
			picked++
			d := 0.5 + 1.5*rnd.Float64()
			cd += c[j] * d
			for _, e := range cols[j] {
				acc[e.i] -= e.v * d
			}
		}
		var col []entry
		for i, v := range acc {
			if v != 0 {
				col = append(col, entry{i, v})
			}
		}
		delta := []float64{1e-3, 0.1, 1}[rnd.IntN(3)]
		cols = append(cols, col)
		lower, upper = append(lower, 0), append(upper, inf)
		x0 = append(x0, 0)
		c = append(c, -cd-delta*(1+math.Abs(cd)))
	}
	scale := make([]float64, rows)
	for i := range scale {
		scale[i] = math.Pow(10, rowScale*rnd.Float64())
		b[i] *= scale[i]
	}
	a := &cscMatrix{m: rows, n: len(cols), start: []int{0}}
	for _, col := range cols {
		slices.SortFunc(col, func(p, q entry) int { return p.i - q.i })
		for _, e := range col {
			a.index = append(a.index, e.i)
			a.value = append(a.value, e.v*scale[e.i])
		}
		a.start = append(a.start, len(a.index))
	}
	return plantedLP{c, a, b, lower, upper, x0, f}
}

// check returns an error if res and err contradict what is known of p. A
// solver that gives up contradicts nothing.
func (p plantedLP) check(kind int, res BoundedResult, err error) error {
	switch {
	case errors.Is(err, ErrInfeasible):
		if kind != plantInfeasible {
			return errors.New("ErrInfeasible for a problem with a feasible point")
		}
	case errors.Is(err, ErrUnbounded):
		if kind != plantUnbounded {
			return errors.New("ErrUnbounded for a problem with no feasible ray")
		}
	case err == nil:
		if kind != plantOptimal {
			return fmt.Errorf("optimum %v for a problem that has none", res.F)
		}
		const tol = 1e-6
		r := slices.Clone(p.b)
		size := make([]float64, p.a.m)
		for i, v := range p.b {
			size[i] = math.Abs(v)
		}
		for j, x := range res.X {
			if x < p.lower[j]-tol*math.Max(1, math.Abs(p.lower[j])) || x > p.upper[j]+tol*math.Max(1, math.Abs(p.upper[j])) {
				return fmt.Errorf("x[%d] = %v outside [%v, %v]", j, x, p.lower[j], p.upper[j])
			}
			for k := p.a.start[j]; k < p.a.start[j+1]; k++ {
				t := p.a.value[k] * x
				r[p.a.index[k]] -= t
				size[p.a.index[k]] += math.Abs(t)
			}
		}
		for i := range r {
			if math.Abs(r[i]) > tol*math.Max(1, size[i]) {
				return fmt.Errorf("row %d: residual %v", i, r[i])
			}
		}
		if math.Abs(res.F-p.f) > tol*(1+math.Abs(p.f)) {
			return fmt.Errorf("objective %v, want %v", res.F, p.f)
		}
	}
	return nil
}

// child tightens a few bounds of p around x0 without moving it off a bound
// it rests on, and appends a row with a slack that x0 satisfies, as a
// branch-and-bound node does to its parent. x0 stays optimal.
func (p plantedLP) child(rnd *rand.Rand) plantedLP {
	n := p.a.n
	lower, upper := slices.Clone(p.lower), slices.Clone(p.upper)
	for range 4 {
		j := rnd.IntN(n)
		if x := p.x0[j]; x > lower[j] && x < upper[j] {
			if rnd.IntN(2) == 0 {
				lower[j] = x - rnd.Float64()*math.Min(x-lower[j], 1)
			} else {
				upper[j] = x + rnd.Float64()*math.Min(upper[j]-x, 1)
			}
		}
	}
	row := map[int]float64{}
	for range 4 {
		row[rnd.IntN(n)] = rnd.NormFloat64()
	}
	a := &cscMatrix{m: p.a.m + 1, n: n + 1, start: []int{0}}
	act := 0.0
	for j := range n {
		a.index = append(a.index, p.a.index[p.a.start[j]:p.a.start[j+1]]...)
		a.value = append(a.value, p.a.value[p.a.start[j]:p.a.start[j+1]]...)
		if v, ok := row[j]; ok {
			a.index = append(a.index, p.a.m)
			a.value = append(a.value, v)
			act += v * p.x0[j]
		}
		a.start = append(a.start, len(a.index))
	}
	a.index = append(a.index, p.a.m)
	a.value = append(a.value, 1)
	a.start = append(a.start, len(a.index))
	slack := 0.0
	if rnd.IntN(2) == 0 {
		slack = rnd.Float64()
	}
	return plantedLP{
		c:     append(slices.Clone(p.c), 0),
		a:     a,
		b:     append(slices.Clone(p.b), act+slack),
		lower: append(lower, 0),
		upper: append(upper, math.Inf(1)),
		x0:    append(slices.Clone(p.x0), slack),
		f:     p.f,
	}
}

// primal solves p by the primal simplex method alone, the last method
// BoundedSimplex tries.
func (p plantedLP) primal() (BoundedResult, error) {
	s := newBounded(newFactor, p.c, p.a, p.b, p.lower, p.upper, 1e-9)
	if err := s.solve(); err != nil {
		return BoundedResult{}, err
	}
	n := p.a.n
	res := BoundedResult{X: s.x[:n:n], Dual: s.y}
	for j := range n {
		res.F += p.c[j] * s.x[j]
	}
	return res, nil
}

// No verdict may contradict the construction of the problem, by any route:
// from scratch, by the primal method alone, or from the basis of a parent
// problem. Problems with an optimum must be solved.
func TestBoundedSimplexPlanted(t *testing.T) {
	t.Parallel()
	seeds := 60
	if testing.Short() {
		seeds = 10
	}
	for _, cfg := range []struct{ spread, rowScale float64 }{{0, 0}, {4, 0}, {4, 3}, {0, 6}} {
		for _, size := range [][2]int{{3, 5}, {8, 20}, {20, 60}} {
			for kind := plantOptimal; kind <= plantUnbounded; kind++ {
				for seed := range seeds {
					rnd := rand.New(rand.NewPCG(uint64(seed), uint64(1000*size[0]+kind)))
					p := planted(rnd, size[0], size[1], kind, cfg.spread, cfg.rowScale)
					name := fmt.Sprintf("spread %g scale %g %dx%d kind %d seed %d", cfg.spread, cfg.rowScale, size[0], size[1], kind, seed)
					res, err := BoundedSimplex(p.c, p.a, p.b, p.lower, p.upper, 0)
					if e := p.check(kind, res, err); e != nil {
						t.Errorf("%s: %v", name, e)
					}
					if kind == plantOptimal && err != nil {
						t.Errorf("%s: %v", name, err)
					}
					pres, perr := p.primal()
					if e := p.check(kind, pres, perr); e != nil {
						t.Errorf("%s: primal: %v", name, e)
					}
					if kind != plantOptimal || err != nil {
						continue
					}
					q := p.child(rnd)
					res, err = BoundedSimplexFrom(res.Basis, q.c, q.a, q.b, q.lower, q.upper, 0)
					if err != nil {
						t.Errorf("%s: child: %v", name, err)
					} else if e := q.check(kind, res, err); e != nil {
						t.Errorf("%s: child: %v", name, e)
					}
				}
			}
		}
	}
}

// On this infeasible problem the primal method met a ratio test with no
// limit in its first phase, which is bounded below, on stale factors, and
// returned ErrUnbounded.
func TestBoundedSimplexPhase1Ray(t *testing.T) {
	t.Parallel()
	rnd := rand.New(rand.NewPCG(511, 8000+plantInfeasible))
	p := planted(rnd, 8, 20, plantInfeasible, 4, 6)
	res, err := p.primal()
	if e := p.check(plantInfeasible, res, err); e != nil {
		t.Error(e)
	}
}

// One row holds coefficients of 1e6 and 1e-4. The multiplier that the dual
// simplex method puts on it is 1e-6, so the small coefficient's entry of
// yᵀA is 1e-10, and so is the entry of the primal ratio test's column in
// that row. Taking either for zero made the first problem infeasible and
// the second unbounded.
func TestBoundedSimplexWideRow(t *testing.T) {
	t.Parallel()
	inf := math.Inf(1)
	a := mat.NewDense(2, 3, []float64{
		1e6, 1e-4, 0,
		0, 1, -1,
	})
	lower := []float64{0, 0, 0}
	upper := []float64{1, inf, inf}
	for _, test := range []struct {
		c, b []float64
		f    float64
	}{
		// x₀ = 1 leaves 1 for the second column to supply.
		{c: []float64{0, 1, 0}, b: []float64{1e6 + 1, 0}, f: 1e4},
		// The second column can rise until x₀ reaches zero.
		{c: []float64{0, 0, -1}, b: []float64{1, 0}, f: -1e4},
	} {
		for _, fk := range factorKinds {
			res, err := fk.solve(nil, test.c, a, test.b, lower, upper, 0)
			if err != nil {
				t.Errorf("%s: c = %v: %v", fk.name, test.c, err)
				continue
			}
			if e := checkKKT(test.c, a, test.b, lower, upper, res, 1e-7); e != nil {
				t.Errorf("%s: c = %v: %v", fk.name, test.c, e)
			}
			if math.Abs(res.F-test.f) > 1e-6*math.Abs(test.f) {
				t.Errorf("%s: c = %v: objective %v, want %v", fk.name, test.c, res.F, test.f)
			}
		}
	}
}

func TestCertifyOpenColumn(t *testing.T) {
	t.Parallel()
	// 1e6 x₀ + 1e-4 x₁ = 1e6 + 1 with 0 ≤ x₀ ≤ 1 and x₁ ≥ 0, which x₁ = 1e4
	// satisfies. With y = 1e-6 the entry of yᵀA on x₁ is 1e-10.
	inf := math.Inf(1)
	a := mat.NewDense(1, 2, []float64{1e6, 1e-4})
	b := []float64{1e6 + 1}
	lower := []float64{0, 0}
	s := newBounded(newFactor, make([]float64, 2), a, b, lower, []float64{1, inf}, 1e-9)
	if s.certify([]float64{1e-6}) {
		t.Error("certified through a column with no upper bound")
	}
	if s.weak != ErrInfeasible {
		t.Errorf("weak verdict %v, want ErrInfeasible", s.weak)
	}
	// Bounded at 1e3, the column cannot close the gap.
	s = newBounded(newFactor, make([]float64, 2), a, b, lower, []float64{1, 1e3}, 1e-9)
	if !s.certify([]float64{1e-6}) {
		t.Error("not certified with every column bounded")
	}
	// A gap of 1e-12 would take x₁ = 1e10 to close.
	a.Set(0, 1, 1e-10)
	s = newBounded(newFactor, make([]float64, 2), a, []float64{1e6 + 1e6}, lower, []float64{1, inf}, 1e-9)
	if !s.certify([]float64{1e-6}) {
		t.Error("not certified with the open column 1e10 away")
	}
}

func TestRecedes(t *testing.T) {
	t.Parallel()
	// x₀ - x₁ + x₂ = 0 with x₀, x₁ ≥ 0 and 0 ≤ x₂ ≤ 1, at x = (0, room, room)
	// for a cost of -x₁.
	inf := math.Inf(1)
	a := mat.NewDense(1, 3, []float64{1, -1, 1})
	c := []float64{0, -1, 0}
	lower := []float64{0, 0, 0}
	upper := []float64{inf, inf, 1}
	for _, test := range []struct {
		room float64
		d    []float64
		want bool
	}{
		{0.5, []float64{1, 1, 0}, true},
		{0.5, []float64{1, 1, 0, 0}, true},
		// The cost does not fall.
		{0.5, []float64{1, 0, -1}, false},
		// A x changes.
		{0.5, []float64{1, 1, 1e-3}, false},
		// x₂ falls to its bound after a step of 5e5.
		{0.5, []float64{1 + 1e-6, 1, -1e-6}, false},
		// It would take a step of 5e9.
		{0.5, []float64{1 + 1e-10, 1, -1e-10}, true},
		// With no room it stops at once.
		{0, []float64{1 + 1e-10, 1, -1e-10}, false},
	} {
		s := newBounded(newFactor, c, a, []float64{0}, lower, upper, 1e-9)
		s.x[1], s.x[2] = test.room, test.room
		d := make([]float64, s.total)
		copy(d, test.d)
		if got := s.recedes(d); got != test.want {
			t.Errorf("room %v, d = %v: got %t, want %t", test.room, test.d, got, test.want)
		}
	}
}

// On these unbounded problems the primal method from the artificial basis
// took pivots a millionth of a millionth of their column, updated the
// factors on them, and ended with a basis it could not factorize. A pivot
// that small is now followed by a fresh factorization.
func TestBoundedSimplexSmallPrimalPivot(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		seed uint64
		m, n int
	}{{74, 20, 60}, {187, 40, 120}} {
		c, a, b, lower, upper := zeroRowsLP(test.seed, test.m, test.n, true)
		s := newBounded(newFactor, c, a, b, lower, upper, 1e-9)
		if err := s.solve(); !errors.Is(err, ErrUnbounded) {
			t.Errorf("seed %d: primal method returned %v, want ErrUnbounded", test.seed, err)
		}
	}
}
