// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lp

import (
	"math"
	"slices"
)

// minWeight bounds the dual steepest-edge weights away from zero.
const minWeight = 1e-4

// reducedCost returns d_j = c_j - yᵀa_j and the tolerance for its sign,
// which scales with the terms of the sum as in price.
func (s *bounded) reducedCost(j int) (d, tol float64) {
	d, size := s.cost[j], math.Abs(s.cost[j])
	for k := s.start[j]; k < s.start[j+1]; k++ {
		t := s.y[s.index[k]] * s.value[k]
		d -= t
		size += math.Abs(t)
	}
	return d, s.tol * math.Max(1, size)
}

// dualStart prepares an installed phase 2 basis for the dual simplex
// method. Each nonbasic variable with two finite bounds moves to the bound
// its reduced cost favours; it reports false if a variable with at most one
// finite bound has a reduced cost of the wrong sign.
func (s *bounded) dualStart() bool {
	s.btran()
	for j := range s.total {
		st := s.status[j]
		if st == basic || s.lower[j] == s.upper[j] {
			continue
		}
		d, tol := s.reducedCost(j)
		lo, up := !math.IsInf(s.lower[j], -1), !math.IsInf(s.upper[j], 1)
		switch {
		case lo && up:
			if d >= 0 {
				s.x[j], s.status[j] = s.lower[j], atLower
			} else {
				s.x[j], s.status[j] = s.upper[j], atUpper
			}
		case st == atLower && d < -tol, st == atUpper && d > tol, st == atZero && math.Abs(d) > tol:
			return false
		}
	}
	return s.refactor() == nil
}

// candidate is an entering variable for the dual ratio test.
type candidate struct {
	j     int
	ratio float64 // dual step at which d_j reaches zero
	alpha float64 // |pivot row entry|
}

// dual runs the dual simplex method for bounded variables from a dual
// feasible basis until it is also primal feasible (Koberstein, "The dual
// simplex method, techniques for a fast and stable implementation", PhD
// thesis, Paderborn, 2005). The leaving row is chosen by dual steepest edge
// with unit initial weights; the ratio test passes breakpoints of boxed
// variables, flipping them to their other bound, while the primal
// infeasibility of the leaving row still decreases. A run of degenerate
// pivots perturbs the costs of the nonbasic variables, which the caller
// finds restored and must clean up after; once every cost is perturbed it
// uses Bland's rule without flips until the dual objective moves. It
// returns ErrInfeasible if a row cannot be made feasible and its row of
// the basis inverse certifies that, and errUnsettled if the certificate
// fails or only rows with unstable pivots are left.
func (s *bounded) dual() error {
	m := s.m
	weight := make([]float64, m)
	for i := range weight {
		weight[i] = 1
	}
	rho := make([]float64, m)
	tau := make([]float64, m)
	flip := make([]float64, m)
	var cand []candidate
	s.bland, s.stall = false, 0
	s.unbar()
	shake := false
	defer s.unshift()
	for {
		if s.iterations >= s.limit {
			return ErrIterationLimit
		}
		if shake {
			shake = false
			if !s.shift() {
				s.bland = true
			}
		}
		r, delta := -1, 0.0
		best, unstable := 0.0, false
		for i, j := range s.head {
			var inf float64
			switch {
			case s.x[j] < s.lower[j]-s.tol:
				inf = s.lower[j] - s.x[j]
			case s.x[j] > s.upper[j]+s.tol:
				inf = s.x[j] - s.upper[j]
			default:
				continue
			}
			if s.barred[i] {
				unstable = true
				continue
			}
			if s.bland {
				if r < 0 || j < s.head[r] {
					r, delta = i, inf
				}
				continue
			}
			if score := inf * inf / weight[i]; score > best {
				best, r, delta = score, i, inf
			}
		}
		if r < 0 {
			if unstable {
				return errUnsettled
			}
			s.bland, s.stall = false, 0
			return nil
		}
		leave := s.head[r]
		sign := 1.0 // x_r rises to its lower bound
		if s.x[leave] > s.upper[leave] {
			sign = -1
		}
		clear(rho)
		rho[r] = 1
		s.lu.btran(rho)
		s.btran()
		cand = cand[:0]
		for j := range s.total {
			st := s.status[j]
			if st == basic || s.lower[j] == s.upper[j] {
				continue
			}
			a := 0.0
			for k := s.start[j]; k < s.start[j+1]; k++ {
				a += rho[s.index[k]] * s.value[k]
			}
			a *= sign
			if math.Abs(a) <= pivotTol {
				continue
			}
			if (st == atLower && a > 0) || (st == atUpper && a < 0) {
				continue
			}
			d, _ := s.reducedCost(j)
			if st == atLower {
				d = math.Max(d, 0)
			} else {
				d = math.Max(-d, 0)
			}
			if st == atZero {
				d = 0
			}
			cand = append(cand, candidate{j, d / math.Abs(a), math.Abs(a)})
		}
		if len(cand) == 0 {
			// A dual ray: confirm it on a fresh factorization, then
			// against the problem data.
			if !s.fresh {
				if err := s.refactor(); err != nil {
					return err
				}
				continue
			}
			if s.certify(rho) {
				return ErrInfeasible
			}
			return errUnsettled
		}
		q, t, flips := s.dualRatio(cand, delta)
		if q < 0 {
			return errUnsettled
		}
		s.ftran(q)
		pivot := s.alpha[r]
		if math.Abs(pivot) <= pivotTol || !s.stable(r) {
			// The pivot row and column disagree, or the pivot is at
			// rounding level against its column: look again on fresh
			// factors, then leave the row until the basis changes.
			if !s.fresh {
				if err := s.refactor(); err != nil {
					return err
				}
				continue
			}
			s.barred[r] = true
			s.bars = append(s.bars, r)
			continue
		}
		// Primal update: flipped variables, then the entering variable.
		if len(flips) > 0 {
			clear(flip)
			for _, j := range flips {
				step := s.upper[j] - s.lower[j]
				if s.status[j] == atUpper {
					step = -step
					s.x[j], s.status[j] = s.lower[j], atLower
				} else {
					s.x[j], s.status[j] = s.upper[j], atUpper
				}
				for k := s.start[j]; k < s.start[j+1]; k++ {
					flip[s.index[k]] += s.value[k] * step
				}
			}
			s.lu.ftran(flip)
			for i, a := range flip {
				s.x[s.head[i]] -= a
			}
		}
		target := s.lower[leave]
		if sign < 0 {
			target = s.upper[leave]
		}
		theta := (s.x[leave] - target) / pivot
		for i, a := range s.alpha {
			s.x[s.head[i]] -= theta * a
		}
		s.x[q] += theta
		s.iterations++
		// A run of degenerate pivots first perturbs the costs, which
		// separates the tied ratios, and only then falls back on Bland's
		// rule.
		if t*delta <= s.tol*math.Max(1, s.objScale) {
			if s.stall++; s.stall >= blandAfter && !s.bland {
				s.stall, shake = 0, true
			}
		} else {
			s.stall, s.bland = 0, false
		}
		// Dual steepest-edge weights (Forrest and Goldfarb, 1992). The
		// weight of the leaving row is computed, not carried: an error in
		// it would spread to every other weight.
		copy(tau, rho)
		s.lu.ftran(tau)
		wr := 0.0
		for _, v := range rho {
			wr += v * v
		}
		for i, a := range s.alpha {
			if i == r {
				continue
			}
			k := a / pivot
			weight[i] = math.Max(weight[i]-2*k*tau[i]+k*k*wr, minWeight)
		}
		weight[r] = math.Max(wr/(pivot*pivot), minWeight)
		if sign > 0 {
			s.x[leave], s.status[leave] = target, atLower
		} else {
			s.x[leave], s.status[leave] = target, atUpper
		}
		s.head[r] = q
		s.status[q] = basic
		if err := s.pivoted(r); err != nil {
			return err
		}
	}
}

// stable reports whether the entry of alpha in row is large enough,
// against the rest of alpha, to pivot on.
func (s *bounded) stable(row int) bool {
	big := 0.0
	for _, a := range s.alpha {
		big = math.Max(big, math.Abs(a))
	}
	return math.Abs(s.alpha[row]) >= stablePivot*big
}

// shift perturbs the cost of each nonbasic variable not yet perturbed, away
// from zero reduced cost, and reports whether there was one. The basis
// stays dual feasible and ties between dual ratios separate.
func (s *bounded) shift() bool {
	if s.shifted == nil {
		s.shifted = make([]bool, s.n)
		s.seed = 0x9e3779b97f4a7c15
	}
	any := false
	for j := range s.n {
		st := s.status[j]
		if s.shifted[j] || (st != atLower && st != atUpper) || s.lower[j] == s.upper[j] {
			continue
		}
		s.seed ^= s.seed << 13
		s.seed ^= s.seed >> 7
		s.seed ^= s.seed << 17
		e := costShift * (1 + math.Abs(s.c[j])) * (0.5 + 0.5*float64(s.seed>>11)/(1<<53))
		if st == atUpper {
			e = -e
		}
		s.cost[j] += e
		s.shifted[j], any = true, true
	}
	return any
}

// unshift restores the true costs, leaving any reduced cost of the wrong
// sign to the primal simplex method.
func (s *bounded) unshift() {
	if s.shifted != nil {
		copy(s.cost, s.c)
		s.shifted = nil
	}
}

// dualRatio chooses the entering variable among the candidates for a
// leaving row whose primal infeasibility is delta. It returns the entering
// variable, the dual step, and the boxed variables whose breakpoints the
// step passes, which move to their other bound.
func (s *bounded) dualRatio(cand []candidate, delta float64) (int, float64, []int) {
	if s.bland {
		q, t := -1, math.Inf(1)
		for _, c := range cand {
			if c.ratio < t || (c.ratio == t && c.j < q) {
				q, t = c.j, c.ratio
			}
		}
		return q, t, nil
	}
	slices.SortFunc(cand, func(a, b candidate) int {
		switch {
		case a.ratio < b.ratio:
			return -1
		case a.ratio > b.ratio:
			return 1
		}
		return a.j - b.j
	})
	// Pass breakpoints while the slope of the dual objective stays positive.
	slope, k := delta, 0
	for ; k < len(cand)-1; k++ {
		j := cand[k].j
		if math.IsInf(s.lower[j], -1) || math.IsInf(s.upper[j], 1) {
			break
		}
		next := slope - cand[k].alpha*(s.upper[j]-s.lower[j])
		if next <= 0 {
			break
		}
		slope = next
	}
	// Harris: among the remaining candidates within the relaxed step, the
	// largest pivot.
	relaxed := math.Inf(1)
	for _, c := range cand[k:] {
		relaxed = math.Min(relaxed, c.ratio+s.tol/c.alpha)
	}
	q := k
	for i := k; i < len(cand) && cand[i].ratio <= relaxed; i++ {
		if cand[i].alpha > cand[q].alpha {
			q = i
		}
	}
	flips := make([]int, k)
	for i := range k {
		flips[i] = cand[i].j
	}
	return cand[q].j, cand[q].ratio, flips
}
