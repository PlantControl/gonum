// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
)

// TestDlarftForwardTrailingZeros checks the forward T factor when the first
// reflector extends past a shorter second one. Netlib DLARFT accumulates
// PREVLASTV from the second reflector on (I.GT.1, 1-based).
func TestDlarftForwardTrailingZeros(t *testing.T) {
	const n, k = 6, 3
	cols := [k][n]float64{
		{1, 0, 0, 1, 0, 0},
		{0, 1, 0, 0, 0, 0},
		{0, 0, 1, 1, 0, 0},
	}
	var tau [k]float64
	for i, v := range cols {
		var vv float64
		for _, x := range v {
			vv += x * x
		}
		tau[i] = 2 / vv
	}
	// H = H(0)·H(1)·H(2) with H(i) = I - tau[i]·v_i·v_iᵀ.
	var h [n * n]float64
	for i := range n {
		h[i*n+i] = 1
	}
	for c, v := range cols {
		var next [n * n]float64
		for i := range n {
			for j := range n {
				s := h[i*n+j]
				for l := range n {
					s -= h[i*n+l] * tau[c] * v[l] * v[j]
				}
				next[i*n+j] = s
			}
		}
		h = next
	}
	for _, store := range []lapack.StoreV{lapack.ColumnWise, lapack.RowWise} {
		var v []float64
		ldv := k
		if store == lapack.RowWise {
			ldv = n
		}
		v = make([]float64, n*k)
		for c := range k {
			for r := range n {
				if store == lapack.ColumnWise {
					v[r*ldv+c] = cols[c][r]
				} else {
					v[c*ldv+r] = cols[c][r]
				}
			}
		}
		tm := make([]float64, k*k)
		Implementation{}.Dlarft(lapack.Forward, store, n, k, v, ldv, tau[:], tm, k)
		var maxErr float64
		for i := range n {
			for j := range n {
				want := 0.0
				if i == j {
					want = 1
				}
				for a := range k {
					for b := range k {
						want -= cols[a][i] * tm[a*k+b] * cols[b][j]
					}
				}
				maxErr = math.Max(maxErr, math.Abs(want-h[i*n+j]))
			}
		}
		if maxErr > 1e-14 {
			t.Errorf("store=%c: |I - V·T·Vᵀ - H| = %g", store, maxErr)
		}
	}
}

// TestDormqrBlockedSparseReflectors compares blocked and unblocked Dormqr
// on the QR of a permutation, whose reflectors have ragged trailing zeros.
func TestDormqrBlockedSparseReflectors(t *testing.T) {
	const n = 160
	rnd := rand.New(rand.NewPCG(1, 2))
	perm := rnd.Perm(n)
	a := make([]float64, n*n)
	for i, p := range perm {
		a[i*n+p] = 1
	}
	impl := Implementation{}
	tau := make([]float64, n)
	work := make([]float64, 64*n+64*64)
	impl.Dgeqrf(n, n, a, n, tau, work, len(work))
	c0 := make([]float64, n*n)
	for i := range c0 {
		c0[i] = rnd.NormFloat64()
	}
	for _, side := range []blas.Side{blas.Left, blas.Right} {
		for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
			blocked := append([]float64(nil), c0...)
			impl.Dormqr(side, trans, n, n, n, a, n, tau, blocked, n, work, len(work))
			unblocked := append([]float64(nil), c0...)
			impl.Dorm2r(side, trans, n, n, n, a, n, tau, unblocked, n, work)
			var maxErr float64
			for i := range blocked {
				maxErr = math.Max(maxErr, math.Abs(blocked[i]-unblocked[i]))
			}
			if maxErr > 1e-12 {
				t.Errorf("side=%c trans=%c: blocked vs unblocked Dormqr differ by %g", side, trans, maxErr)
			}
		}
	}
}
