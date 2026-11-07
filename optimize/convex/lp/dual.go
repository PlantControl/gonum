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
// infeasibility of the leaving row still decreases. After a run of
// degenerate pivots it uses Bland's rule without flips until the dual
// objective moves. It returns ErrInfeasible if a row cannot be made
// feasible.
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
	rechecked := false
	s.bland, s.stall = false, 0
	for {
		if s.iterations >= s.limit {
			return ErrIterationLimit
		}
		r, delta := -1, 0.0
		best := 0.0
		for i, j := range s.head {
			var inf float64
			switch {
			case s.x[j] < s.lower[j]-s.tol*math.Max(1, math.Abs(s.lower[j])):
				inf = s.lower[j] - s.x[j]
			case s.x[j] > s.upper[j]+s.tol*math.Max(1, math.Abs(s.upper[j])):
				inf = s.x[j] - s.upper[j]
			default:
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
			// Confirm on a fresh factorization before reporting a dual ray.
			if !rechecked {
				rechecked = true
				if err := s.refactor(); err != nil {
					return err
				}
				continue
			}
			return ErrInfeasible
		}
		rechecked = false
		q, t, flips := s.dualRatio(cand, delta)
		if q < 0 {
			return ErrInfeasible
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
		s.ftran(q)
		pivot := s.alpha[r]
		if math.Abs(pivot) <= pivotTol {
			if err := s.refactor(); err != nil {
				return err
			}
			continue
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
		if t*delta <= s.tol*math.Max(1, s.objScale) {
			if s.stall++; s.stall >= blandAfter {
				s.bland = true
			}
		} else {
			s.stall, s.bland = 0, false
		}
		// Dual steepest-edge weights (Forrest and Goldfarb, 1992).
		copy(tau, rho)
		s.lu.ftran(tau)
		wr := weight[r]
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
		if s.lu.update(r, s.alpha) {
			if err := s.refactor(); err != nil {
				return err
			}
		}
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
