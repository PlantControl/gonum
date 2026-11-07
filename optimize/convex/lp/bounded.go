// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lp

import (
	"errors"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// ErrIterationLimit reports that BoundedSimplex stopped before optimality.
var ErrIterationLimit = errors.New("lp: iteration limit reached")

// BoundedResult is an optimal solution from BoundedSimplex.
type BoundedResult struct {
	// F is the optimal objective cᵀx.
	F float64
	// X is the optimal point.
	X []float64
	// Dual holds the equality-row multipliers y, so that c - Aᵀy is
	// nonnegative where x is at its lower bound, nonpositive where x is at
	// its upper bound, and zero where x is strictly between its bounds.
	Dual []float64
	// Iterations counts simplex pivots and bound flips over both phases.
	Iterations int
	// Basis is the optimal basis, for a warm start with BoundedSimplexFrom.
	Basis *Basis
}

// Basis records an optimal basis of BoundedSimplex.
type Basis struct {
	head   []int // per row: the basic variable, or -1-i for row i's artificial
	status []int // per variable
}

// BoundedSimplex solves the linear program
//
//	minimize	cᵀ x
//	s.t.		A x = b,
//			lower ≤ x ≤ upper,
//
// where a bound may be infinite. It is a revised primal simplex method for
// bounded variables (Bixby, "Implementing the simplex method: the initial
// basis", ORSA J. Computing 4(3), 1992; Chvátal, Linear Programming, ch. 8).
// Phase 1 minimizes the sum of one artificial variable per row. Each pivot
// reuses a dense LU factorization of an earlier basis through product-form
// eta updates, refactorizing periodically, so a pivot costs O(m²) rather
// than a fresh O(m³) factorization. Pricing is Dantzig's rule; the ratio
// test is Harris's two-pass test. After a run of degenerate pivots it
// switches to Bland's rule until the objective moves, which prevents
// cycling.
//
// tol is the optimality and feasibility tolerance; zero selects 1e-9.
// BoundedSimplex returns ErrInfeasible, ErrUnbounded, ErrSingular if a
// basis cannot be factorized, or ErrIterationLimit. It panics if the input
// sizes do not match or a lower bound exceeds its upper bound.
func BoundedSimplex(c []float64, A mat.Matrix, b, lower, upper []float64, tol float64) (BoundedResult, error) {
	return BoundedSimplexFrom(nil, c, A, b, lower, upper, tol)
}

// BoundedSimplexFrom is BoundedSimplex started from the basis of an earlier
// solve with the same rows. The earlier problem may have had fewer columns;
// columns appended since start nonbasic at a finite bound, or at zero if
// free. Costs and right-hand sides may change. If the basis is singular or
// no longer primal feasible, BoundedSimplexFrom solves from scratch, so the
// result never depends on whether the warm start was usable. A nil start
// is the same as BoundedSimplex.
func BoundedSimplexFrom(start *Basis, c []float64, A mat.Matrix, b, lower, upper []float64, tol float64) (BoundedResult, error) {
	return boundedSimplex(newFactor, start, c, A, b, lower, upper, tol)
}

func boundedSimplex(factor func(m int) basisFactor, start *Basis, c []float64, A mat.Matrix, b, lower, upper []float64, tol float64) (BoundedResult, error) {
	m, n := A.Dims()
	if len(c) != n || len(b) != m || len(lower) != n || len(upper) != n {
		panic(badShape)
	}
	for j := range n {
		if lower[j] > upper[j] {
			panic("lp: lower bound exceeds upper bound")
		}
	}
	if tol == 0 {
		tol = 1e-9
	}
	if start != nil && (len(start.head) != m || len(start.status) > n) {
		panic(badShape)
	}
	s := newBounded(factor, c, A, b, lower, upper, tol)
	var err error
	if start == nil || !s.warm(start) {
		s = newBounded(factor, c, A, b, lower, upper, tol)
		err = s.solve()
	} else {
		err = s.run()
	}
	if err != nil {
		return BoundedResult{Iterations: s.iterations}, err
	}
	res := BoundedResult{X: s.x[:n:n], Dual: s.y, Iterations: s.iterations, Basis: s.basis()}
	for j := range n {
		res.F += c[j] * s.x[j]
	}
	return res, nil
}

const (
	refactorEvery   = 64
	blandAfter      = 50
	pivotTol        = 1e-9
	iterationFactor = 50
)

const (
	atLower = iota
	atUpper
	atZero // free nonbasic variable held at zero
	basic
)

type bounded struct {
	m, n, total int
	tol         float64
	columns     // [A | artificial identity]

	b            []float64
	cost         []float64 // phase cost
	c            []float64 // original cost, padded with zeros
	lower, upper []float64
	x            []float64
	status       []int
	head         []int // head[i] is the variable basic in row i

	lu         basisFactor
	y          []float64
	alpha      []float64
	work       []float64
	bland      bool
	objScale   float64 // magnitude of the phase objective, for progress tests
	stall      int
	limit      int
	iterations int
}

func newBounded(factor func(m int) basisFactor, c []float64, A mat.Matrix, b, lower, upper []float64, tol float64) *bounded {
	m, n := A.Dims()
	s := &bounded{m: m, n: n, total: n + m, tol: tol}
	s.start = make([]int, 0, n+m+1)
	for j := range n {
		s.start = append(s.start, len(s.index))
		for i := range m {
			if v := A.At(i, j); v != 0 {
				s.index = append(s.index, i)
				s.value = append(s.value, v)
			}
		}
	}
	s.start = append(s.start, len(s.index))
	s.b = append([]float64(nil), b...)
	s.c = make([]float64, n+m)
	copy(s.c, c)
	s.cost = make([]float64, n+m)
	s.lower = make([]float64, n+m)
	s.upper = make([]float64, n+m)
	copy(s.lower, lower)
	copy(s.upper, upper)
	s.x = make([]float64, n+m)
	s.status = make([]int, n+m)
	for j := range n {
		switch {
		case !math.IsInf(lower[j], -1):
			s.x[j], s.status[j] = lower[j], atLower
		case !math.IsInf(upper[j], 1):
			s.x[j], s.status[j] = upper[j], atUpper
		default:
			s.x[j], s.status[j] = 0, atZero
		}
	}
	// Artificial i carries the row residual with a sign that makes it
	// nonnegative, so the starting basis is a signed identity.
	residual := append([]float64(nil), b...)
	for j := range n {
		if s.x[j] != 0 {
			for k := s.start[j]; k < s.start[j+1]; k++ {
				residual[s.index[k]] -= s.value[k] * s.x[j]
			}
		}
	}
	s.head = make([]int, m)
	for i := range m {
		j := n + i
		s.index = append(s.index, i)
		sign := 1.0
		if residual[i] < 0 {
			sign = -1
		}
		s.value = append(s.value, sign)
		s.start = append(s.start, len(s.index))
		s.x[j] = math.Abs(residual[i])
		s.lower[j], s.upper[j] = 0, math.Inf(1)
		s.cost[j] = 1
		s.status[j] = basic
		s.head[i] = j
	}
	s.y = make([]float64, m)
	s.alpha = make([]float64, m)
	s.lu = factor(m)
	s.work = make([]float64, m)
	s.limit = iterationFactor * (n + m + 1)
	return s
}

func (s *bounded) solve() error {
	if err := s.refactor(); err != nil {
		return err
	}
	if err := s.run(); err != nil {
		return err
	}
	infeasibility := 0.0
	scale := 1.0
	for i := range s.m {
		infeasibility += s.x[s.n+i]
		scale = math.Max(scale, math.Abs(s.b[i]))
	}
	if infeasibility > s.tol*scale*float64(max(s.m, 1)) {
		return ErrInfeasible
	}
	// Phase 2: artificials are fixed at zero; basic ones may still leave.
	for i := range s.m {
		j := s.n + i
		s.upper[j] = 0
		if s.status[j] != basic {
			s.x[j], s.status[j] = 0, atLower
		}
	}
	copy(s.cost, s.c)
	s.bland, s.stall = false, 0
	return s.run()
}

// warm installs a previous basis for phase 2 and reports whether it is
// nonsingular and primal feasible.
func (s *bounded) warm(start *Basis) bool {
	n := s.n
	for i := range s.m {
		j := n + i
		s.upper[j], s.x[j], s.status[j] = 0, 0, atLower
		s.value[s.start[j]] = 1
	}
	for j, st := range start.status {
		if st == basic {
			continue
		}
		switch {
		case st == atUpper && !math.IsInf(s.upper[j], 1):
			s.x[j], s.status[j] = s.upper[j], atUpper
		case !math.IsInf(s.lower[j], -1):
			s.x[j], s.status[j] = s.lower[j], atLower
		case !math.IsInf(s.upper[j], 1):
			s.x[j], s.status[j] = s.upper[j], atUpper
		default:
			s.x[j], s.status[j] = 0, atZero
		}
	}
	for i, h := range start.head {
		j := h
		if h < 0 {
			j = n + (-1 - h)
		}
		if s.status[j] == basic {
			return false
		}
		s.head[i] = j
		s.status[j] = basic
	}
	for j := range start.status {
		if start.status[j] == basic && s.status[j] != basic {
			return false
		}
	}
	if s.refactor() != nil {
		return false
	}
	// The ratio test relaxes bounds by tol only, so a basic value further
	// outside its bounds would block every step.
	for _, j := range s.head {
		if s.x[j] < s.lower[j]-s.tol || s.x[j] > s.upper[j]+s.tol {
			return false
		}
	}
	copy(s.cost, s.c)
	return true
}

func (s *bounded) basis() *Basis {
	b := &Basis{head: make([]int, s.m), status: append([]int(nil), s.status[:s.n]...)}
	for i, j := range s.head {
		if j >= s.n {
			j = -1 - (j - s.n)
		}
		b.head[i] = j
	}
	return b
}

// run iterates the simplex method on the current phase cost.
func (s *bounded) run() error {
	s.objScale = 0
	for j := range s.total {
		s.objScale += math.Abs(s.cost[j] * s.x[j])
	}
	for {
		if s.iterations >= s.limit {
			return ErrIterationLimit
		}
		s.btran()
		q, dir, gain := s.price()
		if q < 0 {
			return nil
		}
		s.ftran(q)
		theta, row, flip := s.ratio(q, dir)
		if math.IsInf(theta, 1) {
			return ErrUnbounded
		}
		s.iterations++
		// A step that barely changes the objective counts as degenerate,
		// so near-duplicate columns cannot alternate forever.
		if theta*gain <= s.tol*math.Max(1, s.objScale) {
			if s.stall++; s.stall >= blandAfter {
				s.bland = true
			}
		} else {
			s.stall, s.bland = 0, false
		}
		step := float64(dir) * theta
		for i, a := range s.alpha {
			s.x[s.head[i]] -= step * a
		}
		s.x[q] += step
		if flip {
			if dir > 0 {
				s.x[q], s.status[q] = s.upper[q], atUpper
			} else {
				s.x[q], s.status[q] = s.lower[q], atLower
			}
			continue
		}
		leave := s.head[row]
		if float64(dir)*s.alpha[row] > 0 {
			s.x[leave], s.status[leave] = s.lower[leave], atLower
		} else {
			s.x[leave], s.status[leave] = s.upper[leave], atUpper
		}
		s.head[row] = q
		s.status[q] = basic
		if s.lu.update(row, s.alpha) {
			if err := s.refactor(); err != nil {
				return err
			}
		}
	}
}

// refactor factorizes the current basis and recomputes basic values from
// the nonbasic ones to remove accumulated drift.
func (s *bounded) refactor() error {
	if err := s.lu.factorize(&s.columns, s.head); err != nil {
		return err
	}
	if s.m == 0 {
		return nil
	}
	r := s.work
	copy(r, s.b)
	for j := range s.total {
		if s.status[j] != basic && s.x[j] != 0 {
			for k := s.start[j]; k < s.start[j+1]; k++ {
				r[s.index[k]] -= s.value[k] * s.x[j]
			}
		}
	}
	s.lu.ftran(r)
	for i, j := range s.head {
		s.x[j] = r[i]
	}
	return nil
}

// btran sets y to solve Bᵀy = c_B.
func (s *bounded) btran() {
	for i, j := range s.head {
		s.y[i] = s.cost[j]
	}
	s.lu.btran(s.y)
}

// ftran sets alpha to solve B alpha = a_q.
func (s *bounded) ftran(q int) {
	z := s.alpha
	clear(z)
	for k := s.start[q]; k < s.start[q+1]; k++ {
		z[s.index[k]] = s.value[k]
	}
	s.lu.ftran(z)
}

// price returns an entering variable and its direction (+1 increase, -1
// decrease), or -1 at optimality.
func (s *bounded) price() (int, int, float64) {
	best, q, dir := 0.0, -1, 0
	for j := range s.total {
		st := s.status[j]
		if st == basic || s.lower[j] == s.upper[j] {
			continue
		}
		// The tolerance scales with the terms of d, whose rounding error
		// grows with them.
		d, size := s.cost[j], math.Abs(s.cost[j])
		for k := s.start[j]; k < s.start[j+1]; k++ {
			t := s.y[s.index[k]] * s.value[k]
			d -= t
			size += math.Abs(t)
		}
		tol := s.tol * math.Max(1, size)
		var gain float64
		var way int
		switch {
		case d < -tol && (st == atLower || st == atZero):
			gain, way = -d, 1
		case d > tol && (st == atUpper || st == atZero):
			gain, way = d, -1
		default:
			continue
		}
		if s.bland {
			return j, way, gain
		}
		if gain > best {
			best, q, dir = gain, j, way
		}
	}
	return q, dir, best
}

// ratio returns the step length, the leaving row, and whether the entering
// variable instead moves to its opposite bound.
func (s *bounded) ratio(q, dir int) (float64, int, bool) {
	limit := func(i int, slack float64) float64 {
		delta := -float64(dir) * s.alpha[i]
		j := s.head[i]
		switch {
		case delta < -pivotTol && !math.IsInf(s.lower[j], -1):
			return (s.x[j] - s.lower[j] + slack) / -delta
		case delta > pivotTol && !math.IsInf(s.upper[j], 1):
			return (s.upper[j] - s.x[j] + slack) / delta
		}
		return math.Inf(1)
	}
	// Harris pass 1: the largest step with bounds relaxed by tol.
	relaxed := math.Inf(1)
	for i := range s.m {
		relaxed = math.Min(relaxed, limit(i, s.tol))
	}
	flip := s.upper[q] - s.lower[q]
	if math.IsInf(flip, 1) && math.IsInf(relaxed, 1) {
		return relaxed, -1, false
	}
	if flip <= relaxed {
		return flip, -1, true
	}
	if math.IsInf(relaxed, 1) {
		return relaxed, -1, false
	}
	// Pass 2: among rows within that step, the largest pivot; Bland's rule
	// takes the smallest variable index instead.
	row, size := -1, 0.0
	for i := range s.m {
		if limit(i, 0) > relaxed {
			continue
		}
		a := math.Abs(s.alpha[i])
		if s.bland {
			if row < 0 || s.head[i] < s.head[row] {
				row, size = i, a
			}
		} else if a > size {
			row, size = i, a
		}
	}
	if row < 0 {
		return math.Inf(1), -1, false
	}
	return math.Max(limit(row, 0), 0), row, false
}
