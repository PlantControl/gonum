// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

// TestDlanv2Netlib compares eigenvalues against reference LAPACK. The
// factorizations themselves may legitimately differ, for example when the
// real/complex classification of nearly equal eigenvalues flips, so only
// the eigenvalues are compared.
func TestDlanv2Netlib(t *testing.T) {
	const (
		big   = 1e300
		small = 1e-300
		eps   = dlamchE
	)
	cases := [][4]float64{
		{1e8, 1, -1, 1e8},
		{1, 1e-9, -1e-9, 1},
		{1e8, 1e16, -1, 1e8},
		{1e8, 1, 1, 1e8},
		{1, 1e-8, 1e-8, 1},
		{1, 1e-20, 1e-20, 1},
		{1e8, 1, -1, 1e8 + 1},
		{1e8, 1, 1, 1e8 + 2},
		{1, 1e-9, -1e-9, 1 + 2*eps},
		{1, 1e-9, 1e-9, 1 + 2*eps},
		{1, 1e-20, -1e-20, 1 + 2*eps},
		{1e8, 1e16, -1, 1e8 * (1 + 4*eps)},
		{1e8, 1e-16, -1e-16, -1e8},
		{1, 1e150, -1e-150, 1 + 4*eps},
		{big, big, -big, big * (1 + 4*eps)},
		{big, big, big, big * (1 + 4*eps)},
		{big, 1, -1, big * (1 + 4*eps)},
		{small, small, -small, 2 * small},
		{small, small, -small, small * (1 + 4*eps)},
		{small, small, small, small * (1 + 4*eps)},
		{1, big, -small, 1 + 4*eps},
		{0, math.MaxFloat64 / 4, -math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64},
		{0, 1, -1, math.SmallestNonzeroFloat64},
	}
	rnd := rand.New(rand.NewPCG(1, 1))
	for range 2000 {
		e := func() float64 { return math.Ldexp(rnd.NormFloat64(), rnd.IntN(1200)-600) }
		a := e()
		cases = append(cases, [4]float64{a, e(), e(), a * (1 + float64(rnd.IntN(9)-4)*eps)})
	}
	for _, m := range cases {
		a, b, c, d := m[0], m[1], m[2], m[3]
		_, _, _, _, g1r, g1i, g2r, g2i, _, _ := Implementation{}.Dlanv2(a, b, c, d)
		_, _, _, _, n1r, n1i, n2r, n2i, _, _ := netlib.Dlanv2(a, b, c, d)
		g1, g2 := complex(g1r, g1i), complex(g2r, g2i)
		n1, n2 := complex(n1r, n1i), complex(n2r, n2i)

		norm := math.Max(math.Max(math.Abs(a), math.Abs(b)), math.Max(math.Abs(c), math.Abs(d)))
		// Nearly repeated eigenvalues are only determined to about
		// sqrt(eps) relative to the matrix norm.
		tol := 16 * eps * norm
		if cmplx.Abs(n1-n2) <= math.Sqrt(eps)*norm {
			tol = 16 * math.Sqrt(eps) * norm
		}
		if (cmplx.Abs(g1-n1) > tol || cmplx.Abs(g2-n2) > tol) &&
			(cmplx.Abs(g1-n2) > tol || cmplx.Abs(g2-n1) > tol) {
			t.Errorf("Eigenvalues for [%v %v; %v %v]: got %v, %v; want Netlib %v, %v",
				a, b, c, d, g1, g2, n1, n2)
		}
	}
}
