// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/floats"
	"plantcontrol.org/v1/gonum/floats/scalar"
)

func TestGSVD(t *testing.T) {
	t.Parallel()

	const tol = 1e-10
	for _, test := range []struct {
		m, p, n int
	}{
		{5, 3, 5},
		{5, 3, 3},
		{3, 3, 5},
		{5, 5, 5},
		{5, 5, 3},
		{3, 5, 5},
		{150, 150, 150},
		{200, 150, 150},
		{150, 150, 200},
		{150, 200, 150},
		{200, 200, 150},
		{150, 200, 200},
	} {
		m := test.m
		p := test.p
		n := test.n
		t.Run(fmt.Sprintf("%v", test), func(t *testing.T) {
			t.Parallel()

			rnd := rand.New(rand.NewPCG(1, 1))
			for trial := 0; trial < 10; trial++ {
				a := NewDense(m, n, nil)
				for i := range a.mat.Data {
					a.mat.Data[i] = rnd.NormFloat64()
				}
				aCopy := DenseCopyOf(a)

				b := NewDense(p, n, nil)
				for i := range b.mat.Data {
					b.mat.Data[i] = rnd.NormFloat64()
				}
				bCopy := DenseCopyOf(b)

				// Test Full decomposition.
				var gsvd GSVD
				ok := gsvd.Factorize(a, b, GSVDU|GSVDV|GSVDQ)
				if !ok {
					t.Errorf("GSVD factorization failed")
				}
				if !Equal(a, aCopy) {
					t.Errorf("A changed during call to GSVD.Factorize with GSVDU|GSVDV|GSVDQ")
				}
				if !Equal(b, bCopy) {
					t.Errorf("B changed during call to GSVD.Factorize with GSVDU|GSVDV|GSVDQ")
				}
				c, s, sigma1, sigma2, zeroR, u, v, q := extractGSVD(&gsvd)
				var ansU, ansV, d1R, d2R Dense
				ansU.Product(u.T(), a, q)
				ansV.Product(v.T(), b, q)
				d1R.Mul(sigma1, zeroR)
				d2R.Mul(sigma2, zeroR)
				if !EqualApprox(&ansU, &d1R, tol) {
					t.Errorf("Answer mismatch with GSVDU|GSVDV|GSVDQ\nUᵀ * A * Q:\n% 0.2f\nΣ₁ * [ 0 R ]:\n% 0.2f",
						Formatted(&ansU), Formatted(&d1R))
				}
				if !EqualApprox(&ansV, &d2R, tol) {
					t.Errorf("Answer mismatch with GSVDU|GSVDV|GSVDQ\nVᵀ * B  *Q:\n% 0.2f\nΣ₂ * [ 0 R ]:\n% 0.2f",
						Formatted(&d2R), Formatted(&ansV))
				}

				// Check C^2 + S^2 = I.
				for i := range c {
					d := c[i]*c[i] + s[i]*s[i]
					if !scalar.EqualWithinAbsOrRel(d, 1, 1e-14, 1e-14) {
						t.Errorf("c_%d^2 + s_%d^2 != 1: got: %v", i, i, d)
					}
				}

				// Test None decomposition.
				ok = gsvd.Factorize(a, b, GSVDNone)
				if !ok {
					t.Errorf("GSVD factorization failed")
				}
				if !Equal(a, aCopy) {
					t.Errorf("A changed during call to GSVD with GSVDNone")
				}
				if !Equal(b, bCopy) {
					t.Errorf("B changed during call to GSVD with GSVDNone")
				}
				cNone := gsvd.ValuesA(nil)
				if !floats.EqualApprox(c, cNone, tol) {
					t.Errorf("Singular value mismatch between GSVDU|GSVDV|GSVDQ and GSVDNone decomposition")
				}
				sNone := gsvd.ValuesB(nil)
				if !floats.EqualApprox(s, sNone, tol) {
					t.Errorf("Singular value mismatch between GSVDU|GSVDV|GSVDQ and GSVDNone decomposition")
				}
			}
		})

	}
}

// Rank-deficient pairs from https://github.com/gonum/gonum/issues/2127.
func TestGSVDRankDeficient(t *testing.T) {
	t.Parallel()

	const tol = 1e-13
	for _, test := range []struct {
		m, p, n    int
		dupA, dupB bool // Set column 1 equal to column 0.
	}{
		{m: 4, p: 2, n: 5, dupA: true, dupB: true},
		{m: 4, p: 3, n: 3, dupB: true},
		{m: 3, p: 2, n: 7, dupA: true, dupB: true},
	} {
		rnd := rand.New(rand.NewPCG(1, 1))
		a := NewDense(test.m, test.n, nil)
		for i := range a.mat.Data {
			a.mat.Data[i] = rnd.NormFloat64()
		}
		b := NewDense(test.p, test.n, nil)
		for i := range b.mat.Data {
			b.mat.Data[i] = rnd.NormFloat64()
		}
		if test.dupA {
			a.SetCol(1, Col(nil, 0, a))
		}
		if test.dupB {
			b.SetCol(1, Col(nil, 0, b))
		}

		var gsvd GSVD
		if !gsvd.Factorize(a, b, GSVDU|GSVDV|GSVDQ) {
			t.Errorf("%+v: GSVD factorization failed", test)
			continue
		}
		_, _, sigma1, sigma2, zeroR, u, v, q := extractGSVD(&gsvd)
		var qtq Dense
		qtq.Mul(q.T(), q)
		if !EqualApprox(&qtq, eye(test.n), tol) {
			t.Errorf("%+v: Q not orthogonal", test)
		}
		var recA, recB Dense
		recA.Product(u, sigma1, zeroR, q.T())
		recB.Product(v, sigma2, zeroR, q.T())
		if !EqualApprox(&recA, a, tol) {
			t.Errorf("%+v: U*Σ₁*[ 0 R ]*Qᵀ != A", test)
		}
		if !EqualApprox(&recB, b, tol) {
			t.Errorf("%+v: V*Σ₂*[ 0 R ]*Qᵀ != B", test)
		}
	}
}

func extractGSVD(gsvd *GSVD) (c, s []float64, s1, s2, zR, u, v, q *Dense) {
	s1 = &Dense{}
	s2 = &Dense{}
	zR = &Dense{}
	u = &Dense{}
	v = &Dense{}
	q = &Dense{}
	gsvd.SigmaATo(s1)
	gsvd.SigmaBTo(s2)
	gsvd.ZeroRTo(zR)
	gsvd.UTo(u)
	gsvd.VTo(v)
	gsvd.QTo(q)
	c = gsvd.ValuesA(nil)
	s = gsvd.ValuesB(nil)
	return c, s, s1, s2, zR, u, v, q
}
