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
	// Iterations counts primal simplex pivots and bound flips over both
	// phases, and dual simplex pivots, whose bound flips are not counted.
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
// where a bound may be infinite. It is a revised simplex method for bounded
// variables that starts from the basis of slack columns, one per row whose
// only nonzero is in that row, with an artificial variable fixed at zero
// where a row has none. From there the dual simplex method runs first
// (Koberstein, "The dual simplex method, techniques for a fast and stable
// implementation", PhD thesis, Paderborn, 2005): if the basis is not dual
// feasible, a phase 1 finds one that is by solving the problem with a zero
// right-hand side and each variable boxed around zero on the sides where
// it has no finite bound (Koberstein and Suhl, "Progress in the dual
// simplex method for large scale LP problems: practical dual phase 1
// algorithms", 2007). A run of degenerate dual pivots perturbs the costs;
// the primal simplex method removes what the perturbation leaves. If the dual method gives up, the
// primal method solves from the artificial basis in two phases (Chvátal,
// Linear Programming, ch. 8), with Dantzig pricing, Harris's two-pass ratio
// test and, after a run of degenerate pivots, Bland's rule among pivots
// that are not small. That limits cycling but does not rule it out; a
// problem that still cycles ends in ErrIterationLimit.
//
// Each pivot reuses an LU factorization of an earlier basis through
// product-form eta updates, refactorizing periodically. The factorization
// is sparse, with Markowitz pivot selection, unless A is dense, so the cost
// of a pivot follows the nonzeros of the factors rather than m².
//
// A result is returned only for a basis that is optimal on a fresh
// factorization and satisfies A x = b. ErrInfeasible comes with row
// multipliers y for which yᵀA x cannot reach yᵀb within the bounds, and
// ErrUnbounded with a feasible point and a direction of falling cost that
// leaves A x unchanged and keeps the finite bounds. In both proofs a
// coefficient counts as zero only at rounding level; a small one is passed
// over only if it would take a move of more than 1e9 for it to matter.
// When a proof holds only with every entry below 1e-9 taken for zero, the
// other methods are tried first, and its verdict is returned if none of
// them reaches one.
//
// If A implements mat.ColNonZeroDoer, only its nonzero elements are read,
// so a sparse A with many columns need not be stored densely.
//
// tol is the optimality and feasibility tolerance; zero selects 1e-9.
// BoundedSimplex returns ErrInfeasible, ErrUnbounded, ErrIterationLimit, or
// ErrSingular if a basis cannot be factorized or no method reaches a
// verdict it can check. It panics if the input sizes do not match or a
// lower bound exceeds its upper bound.
func BoundedSimplex(c []float64, A mat.Matrix, b, lower, upper []float64, tol float64) (BoundedResult, error) {
	return BoundedSimplexFrom(nil, c, A, b, lower, upper, tol)
}

// BoundedSimplexFrom is BoundedSimplex started from the basis of an earlier
// solve. Costs, right-hand sides and bounds may change. The earlier problem
// may have had fewer columns, which start nonbasic at a finite bound or at
// zero if free, and fewer rows, appended after its rows, which start with
// their artificial variable basic.
//
// If the basis is still primal feasible, the primal simplex method continues
// from it. Otherwise, if it is dual feasible, as after tightening bounds or
// appending rows to an optimal basis, the dual simplex method restores
// primal feasibility. If neither holds, the dual phase 1 starts from it. If
// the basis is singular, or no verdict is reached from it,
// BoundedSimplexFrom solves from scratch, so the result never depends on
// whether the warm start was usable. A nil start is the same as
// BoundedSimplex.
func BoundedSimplexFrom(start *Basis, c []float64, A mat.Matrix, b, lower, upper []float64, tol float64) (BoundedResult, error) {
	return boundedSimplex(newFactor, start, c, A, b, lower, upper, tol)
}

func boundedSimplex(factor func(m, n, nnz int) basisFactor, start *Basis, c []float64, A mat.Matrix, b, lower, upper []float64, tol float64) (BoundedResult, error) {
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
	if start != nil && (len(start.head) > m || len(start.status) > n) {
		panic(badShape)
	}
	s := newBounded(factor, c, A, b, lower, upper, tol)
	restart := func() {
		spent, weak := s.iterations, s.weak
		*s = *newBounded(factor, c, A, b, lower, upper, tol)
		s.iterations, s.weak = spent, weak
		s.limit += spent
	}
	err := errUnsettled
	if start != nil {
		if s.warm(start) {
			err = s.settle()
		}
		if unsettled(err) {
			restart()
		}
	}
	if unsettled(err) {
		// From the basis of slack columns the dual simplex method needs no
		// artificial variables to move.
		if s.warm(s.slackBasis()) {
			s.dualFirst = true
			s.limit = s.iterations + dualFactor*(n+m+1)
			err = s.settle()
		}
	}
	if unsettled(err) {
		restart()
		err = s.solve()
	}
	if unsettled(err) && s.weak != nil {
		err = s.weak
	}
	if err != nil {
		if errors.Is(err, errUnsettled) {
			err = ErrSingular
		}
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
	// dualFactor bounds the dual simplex attempt from the slack basis, so
	// that a failure leaves most of the pivots to the primal method.
	dualFactor   = 10
	settleRounds = 20
	// stablePivot is the smallest dual simplex pivot accepted, as a
	// fraction of the largest entry of its column.
	stablePivot = 1e-7
	// costShift scales the cost perturbation that separates tied dual
	// ratios.
	costShift = 1e-6
	// rowDensity is the fraction of nonzeros in a row of the basis inverse
	// below which the pivot row is formed from the rows of A.
	rowDensity = 0.5
	// devexReset is the reference weight above which primal pricing starts
	// a new reference framework.
	devexReset = 1e7
	// roundTol is the size, against the terms of a sum, below which the
	// sum counts as rounding error when a verdict is checked.
	roundTol = 1e-12
)

// errUnsettled reports that a method gave up without a verdict, so another
// should be tried.
var errUnsettled = errors.New("lp: no verdict")

// unsettled reports whether err leaves the problem undecided.
func unsettled(err error) bool {
	return err != nil && !errors.Is(err, ErrInfeasible) && !errors.Is(err, ErrUnbounded)
}

// blandPivot is the smallest pivot Bland's rule takes, as a fraction of the
// largest among the rows that tie. Taking the smallest index whatever its
// pivot can choose one at rounding level, after which the basis cannot be
// factorized. A larger fraction leaves fewer such bases but lets more
// degenerate problems cycle to the iteration limit.
const blandPivot = 1e-5

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
	fresh      bool   // no update since the last factorization
	barred     []bool // per row: its pivot was unstable in this basis
	bars       []int
	shifted    []bool    // per variable: its cost is perturbed
	ray        []float64 // a direction of falling cost found by phase 1
	weak       error     // the verdict of a proof that holds only to tolerance
	dualFirst  bool      // seek a dual feasible basis before any primal pivot
	seed       uint64
	rows       []int // ratio test candidates
	y          []float64
	alpha      []float64
	work       []float64
	rho        []float64 // a row of the basis inverse
	arow       []float64 // per variable: its entry of the pivot row
	d          []float64 // per nonbasic variable: its reduced cost, updated at each pivot
	dtol       []float64 // the tolerance for the sign of d
	devex      []float64 // per nonbasic variable: its reference weight in primal pricing
	byRow      *columns  // the columns of A by rows, built when first used
	bland      bool
	objScale   float64 // magnitude of the phase objective, for progress tests
	stall      int
	limit      int
	iterations int
}

func newBounded(factor func(m, n, nnz int) basisFactor, c []float64, A mat.Matrix, b, lower, upper []float64, tol float64) *bounded {
	m, n := A.Dims()
	s := &bounded{m: m, n: n, total: n + m, tol: tol}
	s.start = make([]int, 0, n+m+1)
	add := func(i, _ int, v float64) {
		if v != 0 {
			s.index = append(s.index, i)
			s.value = append(s.value, v)
		}
	}
	cols, sparse := A.(mat.ColNonZeroDoer)
	for j := range n {
		s.start = append(s.start, len(s.index))
		if sparse {
			cols.DoColNonZero(j, add)
			continue
		}
		for i := range m {
			add(i, j, A.At(i, j))
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
	s.lu = factor(m, n, s.start[n])
	s.work = make([]float64, m)
	s.rho = make([]float64, m)
	s.arow = make([]float64, n+m)
	s.d = make([]float64, n+m)
	s.dtol = make([]float64, n+m)
	s.barred = make([]bool, m)
	s.limit = iterationFactor * (n + m + 1)
	return s
}

// solve is the primal simplex method in two phases from the artificial
// basis.
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
		if !s.fresh {
			if err := s.refactor(); err != nil {
				return err
			}
		}
		s.btran()
		if s.certify(s.y) {
			return ErrInfeasible
		}
		return errUnsettled
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
	if err := s.run(); err != nil {
		return err
	}
	return s.settle()
}

// settle brings an installed phase 2 basis to optimality: by the primal
// simplex method if it is primal feasible, by the dual if it is dual
// feasible, and otherwise after a dual phase 1. It returns nil only for a
// basis that is optimal on a fresh factorization.
func (s *bounded) settle() error {
	started, reached := false, false
	for range settleRounds {
		if !s.fresh {
			if err := s.refactor(); err != nil {
				return err
			}
		}
		if s.within(2*s.tol) && s.optimal() {
			return s.residual()
		}
		if s.ray != nil && s.within(s.tol) && s.residual() == nil {
			if s.recedes(s.ray) {
				return ErrUnbounded
			}
			// The ray keeps the bounds only to the tolerance of phase 1.
			if s.weak == nil {
				s.weak = ErrUnbounded
			}
			s.ray = nil
		}
		switch {
		case s.within(s.tol) && (started || !s.dualFirst):
			s.bland, s.stall = false, 0
			if err := s.run(); err != nil {
				return err
			}
		case s.dualStart():
			if err := s.dual(); err != nil {
				return err
			}
		case !started:
			started = true
			if err := s.phase1(); err != nil {
				return err
			}
		case !reached:
			// No dual feasible basis was found. A feasible point decides
			// it: with the ray of phase 1 the problem is unbounded, and
			// without one the primal method goes on from there.
			reached = true
			if err := s.reach(); err != nil {
				return err
			}
		default:
			return errUnsettled
		}
	}
	return errUnsettled
}

// falls reports whether x, a point of the phase 1 problem, has a negative
// cost. Such a point is a direction that keeps A x and every finite bound
// of the original problem, so from a feasible point the objective falls
// without limit along it.
func (s *bounded) falls() bool {
	if !s.fresh {
		if s.refactor() != nil {
			return false
		}
	}
	if !s.within(s.tol) || s.residual() != nil {
		return false
	}
	f, size := 0.0, 0.0
	for j := range s.n {
		f += s.c[j] * s.x[j]
		size += math.Abs(s.c[j] * s.x[j])
	}
	return f < -math.Max(1e3*s.tol, 1e-9)*math.Max(1, size)
}

// reach moves an installed basis to a primal feasible one by the dual
// simplex method with zero costs, for which every basis is dual feasible.
func (s *bounded) reach() error {
	c := s.c
	s.c = make([]float64, s.total)
	clear(s.cost)
	err := errUnsettled
	if s.dualStart() {
		err = s.dual()
	}
	s.c = c
	copy(s.cost, c)
	return err
}

// optimal reports whether no nonbasic variable prices out, and leaves the
// multipliers of the true costs in y.
func (s *bounded) optimal() bool {
	s.bland = false
	s.btran()
	q, _, _ := s.price()
	return q < 0
}

// residual checks A x = b, which holds after a factorization unless the
// basis is too ill-conditioned for its factors to mean anything.
func (s *bounded) residual() error {
	r, size := s.work, s.alpha
	for i := range r {
		r[i], size[i] = s.b[i], math.Abs(s.b[i])
	}
	for j := range s.total {
		if s.x[j] == 0 {
			continue
		}
		for k := s.start[j]; k < s.start[j+1]; k++ {
			t := s.value[k] * s.x[j]
			r[s.index[k]] -= t
			size[s.index[k]] += math.Abs(t)
		}
	}
	tol := math.Max(1e3*s.tol, 1e-9)
	for i := range r {
		if !(math.Abs(r[i]) <= tol*math.Max(1, size[i])) {
			return errUnsettled
		}
	}
	return nil
}

// certify reports whether y proves the problem infeasible: over the bounds,
// yᵀA x cannot reach yᵀb. A column with no bound on the side that would
// help is passed over only if closing the gap would move it by more than
// 1/pivotTol, or if its entry of yᵀA is at rounding level against its
// terms. An entry that is merely small does not qualify: on such a column
// it moves yᵀA x as far as the gap asks. A proof that holds only with
// every entry below the pivot tolerance taken for zero is recorded as
// weak.
func (s *bounded) certify(y []float64) bool {
	rhs, size := 0.0, 0.0
	for i, v := range y {
		rhs += v * s.b[i]
		size += math.Abs(v * s.b[i])
	}
	// lo and hi bound yᵀA x over the finite bounds; openLo and openHi are
	// the largest entries on columns that could carry it past them.
	lo, hi, openLo, openHi := 0.0, 0.0, 0.0, 0.0
	for j := range s.n {
		z, terms := 0.0, 0.0
		for k := s.start[j]; k < s.start[j+1]; k++ {
			t := y[s.index[k]] * s.value[k]
			z += t
			terms += math.Abs(t)
		}
		if math.Abs(z) <= roundTol*terms {
			continue
		}
		l, u := s.lower[j], s.upper[j]
		if z < 0 {
			l, u = u, l
		}
		if math.IsInf(l, 0) {
			openLo = math.Max(openLo, math.Abs(z))
		} else {
			lo += z * l
			size += math.Abs(z * l)
		}
		if math.IsInf(u, 0) {
			openHi = math.Max(openHi, math.Abs(z))
		} else {
			hi += z * u
			size += math.Abs(z * u)
		}
	}
	margin := s.tol * math.Max(1, size)
	for _, side := range [][2]float64{{rhs - hi, openHi}, {lo - rhs, openLo}} {
		gap, open := side[0], side[1]
		if gap <= margin {
			continue
		}
		if open <= pivotTol*gap {
			return true
		}
		// With open columns taken for zero below the pivot tolerance
		// alone, the proof holds only if they stay within 1/pivotTol of
		// the gap.
		if open <= pivotTol && s.weak == nil {
			s.weak = ErrInfeasible
		}
	}
	return false
}

// slackBasis is the basis of one column per row whose only nonzero is in
// that row, the widest by its bounds, and of the row's artificial variable
// where there is none.
func (s *bounded) slackBasis() *Basis {
	b := &Basis{head: make([]int, s.m), status: make([]int, s.n)}
	for i := range b.head {
		b.head[i] = -1 - i
	}
	for j := range s.n {
		if s.start[j+1]-s.start[j] != 1 || s.lower[j] == s.upper[j] {
			continue
		}
		i := s.index[s.start[j]]
		if k := b.head[i]; k >= 0 {
			if s.upper[k]-s.lower[k] >= s.upper[j]-s.lower[j] {
				continue
			}
			b.status[k] = atLower
		}
		b.head[i], b.status[j] = j, basic
	}
	return b
}

// phase1 moves an installed basis to a dual feasible one by the dual
// simplex method on the problem with a zero right-hand side whose
// variables are boxed around zero on the sides where they have no finite
// bound to rest on (Koberstein and Suhl, 2007). Its optimal basis is dual
// feasible for the original problem if any basis is.
func (s *bounded) phase1() error {
	lower := append([]float64(nil), s.lower...)
	upper := append([]float64(nil), s.upper...)
	b := append([]float64(nil), s.b...)
	clear(s.b)
	for j := range s.n {
		lo, up := !math.IsInf(lower[j], -1), !math.IsInf(upper[j], 1)
		switch {
		case lo && up:
			s.lower[j], s.upper[j] = 0, 0
		case lo:
			s.lower[j], s.upper[j] = 0, 1
		case up:
			s.lower[j], s.upper[j] = -1, 0
		default:
			s.lower[j], s.upper[j] = -1000, 1000
		}
		if s.status[j] != basic {
			s.x[j], s.status[j] = s.lower[j], atLower
		}
	}
	err := errUnsettled
	if s.dualStart() {
		err = s.dual()
	}
	if err == nil {
		if s.falls() {
			s.ray = append([]float64(nil), s.x...)
		}
	}
	copy(s.lower, lower)
	copy(s.upper, upper)
	copy(s.b, b)
	for j := range s.n {
		if s.status[j] == basic {
			continue
		}
		switch {
		case !math.IsInf(lower[j], -1):
			s.x[j], s.status[j] = lower[j], atLower
		case !math.IsInf(upper[j], 1):
			s.x[j], s.status[j] = upper[j], atUpper
		default:
			s.x[j], s.status[j] = 0, atZero
		}
	}
	if err != nil {
		if errors.Is(err, ErrInfeasible) {
			err = errUnsettled
		}
		return err
	}
	if s.fresh {
		s.place()
		return nil
	}
	return s.refactor()
}

// warm installs a previous basis for phase 2 and reports whether it is
// nonsingular. Rows beyond those of start keep their artificial basic.
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
	for i := range s.m {
		j := n + i
		if i < len(start.head) {
			j = start.head[i]
			if j < 0 {
				j = n + (-1 - j)
			}
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
	copy(s.cost, s.c)
	return true
}

// within reports whether the basic values are within their bounds relaxed
// by tol. The ratio test relaxes bounds by s.tol only, so a basic value
// further outside would block every primal step.
func (s *bounded) within(tol float64) bool {
	for _, j := range s.head {
		if s.x[j] < s.lower[j]-tol || s.x[j] > s.upper[j]+tol {
			return false
		}
	}
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

// run iterates the primal simplex method on the current phase cost. The
// entering variable has the largest reduced cost against its devex
// reference weight (Forrest and Goldfarb, "Steepest-edge simplex algorithms
// for linear programming", Math. Programming 57, 1992, section 5). Reduced
// costs, multipliers and weights are updated from the pivot row, and the
// first two recomputed after each factorization; the reduced cost of the
// variable chosen is always recomputed from the multipliers, and run
// returns only when a pricing of every column from fresh multipliers finds
// none.
func (s *bounded) run() error {
	s.objScale = 0
	for j := range s.total {
		s.objScale += math.Abs(s.cost[j] * s.x[j])
	}
	if s.devex == nil {
		s.devex = make([]float64, s.total)
	}
	for j := range s.devex {
		s.devex[j] = 1
	}
	// priced: y, d and dtol belong to the current basis. exact: they were
	// computed from it, with no update since.
	priced, exact := false, false
	for {
		if s.iterations >= s.limit {
			return ErrIterationLimit
		}
		if !priced {
			s.reprice()
			priced, exact = true, true
		}
		q, dir := s.enter()
		if q < 0 {
			if exact {
				return nil
			}
			priced = false
			continue
		}
		if !exact {
			// Updated values chose q; the multipliers decide.
			s.d[q], s.dtol[q] = s.reducedCost(q)
			if d := float64(dir) * s.d[q]; d >= -s.dtol[q] {
				continue
			}
		}
		gain := math.Abs(s.d[q])
		s.ftran(q)
		theta, row, flip := s.ratio(q, dir)
		if math.IsInf(theta, 1) {
			// Confirm the ray on a fresh factorization and against the
			// problem data. If only stale factors showed it, go on.
			stale := !s.fresh
			if stale {
				if err := s.refactor(); err != nil {
					return err
				}
			}
			if s.descends(q, dir) {
				return ErrUnbounded
			}
			// descends used y.
			priced = false
			if stale {
				continue
			}
			// The ratio test passed over a row that stops the step.
			s.reprice()
			priced, exact = true, true
			s.ftran(q)
			if theta, row = s.blocker(dir); row < 0 {
				return errUnsettled
			}
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
		if !s.stable(row) {
			// An update on a pivot this small against its column would
			// multiply the error in every later solve; factorize instead.
			s.head[row] = q
			s.status[q] = basic
			s.unbar()
			if err := s.refactor(); err != nil {
				return err
			}
			priced = false
			continue
		}
		s.reduce(q, leave, row)
		exact = false
		s.head[row] = q
		s.status[q] = basic
		if err := s.pivoted(row); err != nil {
			return err
		}
		priced = !s.fresh
	}
}

// enter returns the entering variable and its direction (+1 increase, -1
// decrease) by the reduced costs in d, or -1 if none prices out: the first
// under Bland's rule, and otherwise the one with the largest reduced cost
// against its reference weight.
func (s *bounded) enter() (q, dir int) {
	best := 0.0
	q = -1
	for j := range s.total {
		st := s.status[j]
		if st == basic || s.lower[j] == s.upper[j] {
			continue
		}
		d, tol := s.d[j], s.dtol[j]
		var way int
		switch {
		case d < -tol && (st == atLower || st == atZero):
			way = 1
		case d > tol && (st == atUpper || st == atZero):
			way = -1
		default:
			continue
		}
		if s.bland {
			return j, way
		}
		if score := d * d / s.devex[j]; score > best {
			best, q, dir = score, j, way
		}
	}
	return q, dir
}

// reduce updates the multipliers, the reduced costs and the reference
// weights for the pivot in which q, whose column is in alpha, replaces the
// basic variable of row, which is leave.
func (s *bounded) reduce(q, leave, row int) {
	s.pivotRow(row)
	pivot := s.alpha[row]
	step := s.d[q] / pivot
	wq := s.devex[q]
	for j := range s.total {
		if s.status[j] == basic || s.lower[j] == s.upper[j] {
			continue
		}
		a := s.arow[j]
		if a == 0 {
			continue
		}
		s.d[j] -= step * a
		if w := a / pivot; w*w*wq > s.devex[j] {
			s.devex[j] = w * w * wq
		}
	}
	for i, v := range s.rho {
		s.y[i] += step * v
	}
	s.d[q] = 0
	s.d[leave], s.dtol[leave] = s.reducedCost(leave)
	s.devex[leave] = math.Max(wq/(pivot*pivot), 1)
	if s.devex[leave] > devexReset {
		for j := range s.devex {
			s.devex[j] = 1
		}
	}
}

// descends reports whether moving nonbasic q in direction dir from the
// current point, with alpha solving B alpha = a_q, is a ray of the problem
// along which the cost falls. With the costs of the first phase it never
// is, since that phase is bounded below by zero.
func (s *bounded) descends(q, dir int) bool {
	for i := range s.m {
		if s.cost[s.n+i] != 0 {
			return false
		}
	}
	if !s.within(s.tol) || s.residual() != nil {
		return false
	}
	s.ftran(q) // residual used alpha
	d := make([]float64, s.total)
	d[q] = float64(dir)
	for i, a := range s.alpha {
		d[s.head[i]] = -float64(dir) * a
	}
	return s.recedes(d)
}

// blocker returns the row whose basic variable first reaches a bound as the
// entering variable moves in direction dir, among the entries of alpha above
// rounding level, and the step to it. The ratio test ignores entries below
// the pivot tolerance, which is right unless the variable has almost no
// room: then a small entry still ends the step at a moderate length.
func (s *bounded) blocker(dir int) (float64, int) {
	big := 1.0
	for _, a := range s.alpha {
		big = math.Max(big, math.Abs(a))
	}
	theta, row := math.Inf(1), -1
	for i, a := range s.alpha {
		if math.Abs(a) <= roundTol*big {
			continue
		}
		j := s.head[i]
		room := s.upper[j] - s.x[j]
		if float64(dir)*a > 0 {
			room = s.x[j] - s.lower[j]
		}
		if t := math.Max(room, 0) / math.Abs(a); t < theta {
			theta, row = t, i
		}
	}
	return theta, row
}

// recedes reports whether the objective falls without limit along d from
// the current point, which must be feasible: d keeps every finite bound
// over a step that moves its largest entry by 1/pivotTol, leaves A x
// unchanged to the accuracy asked of a solution, and lowers the cost. The
// bound test is more than a sign test: a small entry towards a bound with
// little room left stops the step early. Entries at rounding level against
// the largest count as zero.
func (s *bounded) recedes(d []float64) bool {
	big := 0.0
	for _, v := range d {
		big = math.Max(big, math.Abs(v))
	}
	r, terms, scale := s.work, s.y, s.alpha
	clear(r)
	clear(terms)
	clear(scale)
	fall, size := 0.0, 0.0
	for j, v := range d {
		if v == 0 {
			continue
		}
		room := s.upper[j] - s.x[j]
		if v < 0 {
			room = s.x[j] - s.lower[j]
		}
		if math.Abs(v) > big*math.Max(pivotTol*(math.Max(room, 0)+s.tol), roundTol) {
			return false
		}
		for k := s.start[j]; k < s.start[j+1]; k++ {
			i, t := s.index[k], s.value[k]*v
			r[i] += t
			terms[i] += math.Abs(t)
			scale[i] = math.Max(scale[i], math.Abs(s.value[k]))
		}
		fall += s.c[j] * v
		size += math.Abs(s.c[j] * v)
	}
	tol := math.Max(1e3*s.tol, 1e-9)
	for i := range r {
		// The second term covers entries of d that are rounding error.
		if !(math.Abs(r[i]) <= tol*terms[i]+roundTol*scale[i]*big) {
			return false
		}
	}
	return fall < -s.tol*math.Max(big, size)
}

func (s *bounded) unbar() {
	for _, i := range s.bars {
		s.barred[i] = false
	}
	s.bars = s.bars[:0]
}

// pivoted records that alpha replaced the basis column of row.
func (s *bounded) pivoted(row int) error {
	s.unbar()
	s.fresh = false
	if s.lu.update(row, s.alpha) {
		return s.refactor()
	}
	return nil
}

// refactor factorizes the current basis and recomputes basic values from
// the nonbasic ones to remove accumulated drift.
func (s *bounded) refactor() error {
	if err := s.lu.factorize(&s.columns, s.head); err != nil {
		return err
	}
	s.fresh = true
	s.place()
	return nil
}

// place recomputes the basic values from the nonbasic ones.
func (s *bounded) place() {
	if s.m == 0 {
		return
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

// reprice sets y to the multipliers of the phase costs, and d and dtol to
// the reduced cost of each nonbasic variable that is not fixed and the
// tolerance for its sign.
func (s *bounded) reprice() {
	s.btran()
	for j := range s.total {
		if s.status[j] == basic || s.lower[j] == s.upper[j] {
			continue
		}
		s.d[j], s.dtol[j] = s.reducedCost(j)
	}
}

// pivotRow sets arow to row r of B⁻¹[A | artificials] and rho to row r of
// B⁻¹. The entries of basic and of fixed variables are not meaningful.
//
// A sparse rho is multiplied by the rows of A, which costs their nonzeros
// rather than all of A's. Both orders add the terms of an entry by
// increasing row when the columns are stored that way, so they agree to the
// last bit.
func (s *bounded) pivotRow(r int) {
	rho := s.rho
	clear(rho)
	rho[r] = 1
	s.lu.btran(rho)
	nz := 0
	for _, v := range rho {
		if v != 0 {
			nz++
		}
	}
	if float64(nz) > rowDensity*float64(s.m) {
		for j := range s.total {
			if s.status[j] == basic || s.lower[j] == s.upper[j] {
				continue
			}
			a := 0.0
			for k := s.start[j]; k < s.start[j+1]; k++ {
				a += rho[s.index[k]] * s.value[k]
			}
			s.arow[j] = a
		}
		return
	}
	if s.byRow == nil {
		s.byRow = s.transpose()
	}
	rows := s.byRow
	clear(s.arow[:s.n])
	for i, v := range rho {
		if v == 0 {
			s.arow[s.n+i] = 0
			continue
		}
		for k := rows.start[i]; k < rows.start[i+1]; k++ {
			s.arow[rows.index[k]] += v * rows.value[k]
		}
		s.arow[s.n+i] = v * s.value[s.start[s.n+i]]
	}
}

// transpose returns the n columns of A stored by rows.
func (s *bounded) transpose() *columns {
	nnz := s.start[s.n]
	t := &columns{start: make([]int, s.m+1), index: make([]int, nnz), value: make([]float64, nnz)}
	for _, i := range s.index[:nnz] {
		t.start[i+1]++
	}
	for i := range s.m {
		t.start[i+1] += t.start[i]
	}
	next := append([]int(nil), t.start[:s.m]...)
	for j := range s.n {
		for k := s.start[j]; k < s.start[j+1]; k++ {
			at := next[s.index[k]]
			next[s.index[k]]++
			t.index[at], t.value[at] = j, s.value[k]
		}
	}
	return t
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
	// Only rows with a nonzero pivot entry can limit the step.
	relaxed := math.Inf(1)
	s.rows = s.rows[:0]
	for i, a := range s.alpha {
		if math.Abs(a) <= pivotTol {
			continue
		}
		s.rows = append(s.rows, i)
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
	// takes the smallest variable index instead, among pivots that are not
	// small against the largest.
	row, size := -1, 0.0
	for _, i := range s.rows {
		if limit(i, 0) > relaxed {
			continue
		}
		if a := math.Abs(s.alpha[i]); a > size {
			row, size = i, a
		}
	}
	if s.bland && row >= 0 {
		for _, i := range s.rows {
			if s.head[i] < s.head[row] && math.Abs(s.alpha[i]) >= blandPivot*size && limit(i, 0) <= relaxed {
				row = i
			}
		}
	}
	if row < 0 {
		return math.Inf(1), -1, false
	}
	return math.Max(limit(row, 0), 0), row, false
}
