// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

type Dlanv2er interface {
	Dlanv2(a, b, c, d float64) (aa, bb, cc, dd float64, rt1r, rt1i, rt2r, rt2i float64, cs, sn float64)
}

func Dlanv2Test(t *testing.T, impl Dlanv2er) {
	rnd := rand.New(rand.NewPCG(1, 1))
	t.Run("UpperTriangular", func(t *testing.T) {
		for i := 0; i < 10; i++ {
			a := rnd.NormFloat64()
			b := rnd.NormFloat64()
			d := rnd.NormFloat64()
			dlanv2Test(t, impl, a, b, 0, d)
		}
	})
	t.Run("LowerTriangular", func(t *testing.T) {
		for i := 0; i < 10; i++ {
			a := rnd.NormFloat64()
			c := rnd.NormFloat64()
			d := rnd.NormFloat64()
			dlanv2Test(t, impl, a, 0, c, d)
		}
	})
	t.Run("StandardSchur", func(t *testing.T) {
		for i := 0; i < 10; i++ {
			a := rnd.NormFloat64()
			b := rnd.NormFloat64()
			c := rnd.NormFloat64()
			if math.Signbit(b) == math.Signbit(c) {
				c = -c
			}
			dlanv2Test(t, impl, a, b, c, a)
		}
	})
	t.Run("General", func(t *testing.T) {
		for i := 0; i < 100; i++ {
			a := rnd.NormFloat64()
			b := rnd.NormFloat64()
			c := rnd.NormFloat64()
			d := rnd.NormFloat64()
			dlanv2Test(t, impl, a, b, c, d)
		}

		// https://github.com/Reference-LAPACK/lapack/issues/263
		dlanv2Test(t, impl, 0, 1, -1, math.Nextafter(0, 1))
	})
	t.Run("BadlyScaled", func(t *testing.T) {
		const (
			big   = 1e300
			small = 1e-300
		)
		eps := dlamchE
		for _, test := range []struct {
			a, b, c, d float64
			// want holds the exact eigenvalues, if known.
			want []complex128
		}{
			// Equal diagonals in standard Schur form. A formula based on
			// tr^2 - 4*det loses the imaginary parts of these.
			{a: 1e8, b: 1, c: -1, d: 1e8, want: []complex128{complex(1e8, 1), complex(1e8, -1)}},
			{a: 1, b: 1e-9, c: -1e-9, d: 1, want: []complex128{complex(1, 1e-9), complex(1, -1e-9)}},
			{a: 1e8, b: 1e16, c: -1, d: 1e8, want: []complex128{complex(1e8, 1e8), complex(1e8, -1e8)}},
			{a: 1, b: -1e-20, c: 1e-20, d: 1, want: []complex128{complex(1, 1e-20), complex(1, -1e-20)}},
			{a: big, b: big, c: -big, d: big, want: []complex128{complex(big, big), complex(big, -big)}},
			{a: small, b: -small, c: small, d: small, want: []complex128{complex(small, small), complex(small, -small)}},

			// Equal diagonals, b*c > 0: real eigenvalues a ± sqrt(b*c).
			{a: 1e8, b: 1, c: 1, d: 1e8, want: []complex128{1e8 + 1, 1e8 - 1}},
			{a: 1, b: 1e-8, c: 1e-8, d: 1, want: []complex128{1 + 1e-8, 1 - 1e-8}},
			{a: 1, b: 1e-20, c: 1e-20, d: 1, want: []complex128{1, 1}},
			{a: 1, b: -4e-20, c: -1e-20, d: 1, want: []complex128{1, 1}},
			{a: 0, b: 4, c: 1, d: 0, want: []complex128{2, -2}},

			// Nearly equal diagonals reach the general branches.
			{a: 1e8, b: 1, c: -1, d: 1e8 + 1, want: []complex128{complex(1e8+0.5, math.Sqrt(3)/2), complex(1e8+0.5, -math.Sqrt(3)/2)}},
			{a: 1e8, b: 1, c: 1, d: 1e8 + 2, want: []complex128{1e8 + 1 + math.Sqrt2, 1e8 + 1 - math.Sqrt2}},
			{a: 1, b: 1e-9, c: -1e-9, d: 1 + 2*eps},
			{a: 1, b: 1e-9, c: 1e-9, d: 1 + 2*eps},
			{a: 1, b: 1e-20, c: -1e-20, d: 1 + 2*eps},
			{a: 1, b: 1e-20, c: 1e-20, d: 1 + 2*eps},
			{a: 1e8, b: 1e16, c: -1, d: 1e8 * (1 + 4*eps)},
			{a: 1e8, b: 1e-16, c: -1e-16, d: -1e8},
			{a: 1, b: 1e150, c: -1e-150, d: 1 + 4*eps},

			// Entries near overflow and underflow exercise the scaling
			// in the complex branch.
			{a: big, b: big, c: -big, d: big * (1 + 4*eps)},
			{a: big, b: big, c: big, d: big * (1 + 4*eps)},
			{a: big, b: 1, c: -1, d: big * (1 + 4*eps)},
			{a: small, b: small, c: -small, d: 2 * small},
			{a: small, b: small, c: -small, d: small * (1 + 4*eps)},
			{a: small, b: small, c: small, d: small * (1 + 4*eps)},
			{a: 1, b: big, c: -small, d: 1},
			{a: 1, b: big, c: -small, d: 1 + 4*eps},
			{a: 0, b: math.MaxFloat64 / 4, c: -math.SmallestNonzeroFloat64, d: math.SmallestNonzeroFloat64},
		} {
			rt1, rt2 := dlanv2Test(t, impl, test.a, test.b, test.c, test.d)
			if test.want == nil {
				continue
			}
			norm := math.Max(math.Max(math.Abs(test.a), math.Abs(test.b)), math.Max(math.Abs(test.c), math.Abs(test.d)))
			tol := dlanv2Tol * norm
			w1, w2 := test.want[0], test.want[1]
			if cmplx.Abs(rt1-w1) > tol || cmplx.Abs(rt2-w2) > tol {
				if cmplx.Abs(rt1-w2) > tol || cmplx.Abs(rt2-w1) > tol {
					t.Errorf("Unexpected eigenvalues for [%v %v; %v %v]: got %v, %v; want %v, %v",
						test.a, test.b, test.c, test.d, rt1, rt2, w1, w2)
				}
			}
		}
	})
	t.Run("RandomBadlyScaled", func(t *testing.T) {
		for range 1000 {
			e := func() float64 { return math.Ldexp(rnd.NormFloat64(), rnd.IntN(1200)-600) }
			a := e()
			d := a * (1 + float64(rnd.IntN(9)-4)*dlamchE)
			b := e()
			c := e()
			if rnd.IntN(2) == 0 && math.Signbit(b) == math.Signbit(c) {
				c = -c
			}
			dlanv2Test(t, impl, a, b, c, d)
		}
	})
}

// dlanv2Tol is the tolerance relative to the largest absolute entry of the
// input matrix for normwise backward errors and well-conditioned eigenvalues.
const dlanv2Tol = 16 * dlamchE

// dlanv2Test checks the standardized Schur factorization computed by Dlanv2
// and returns its eigenvalues. Equalities that hold exactly in exact
// arithmetic are checked exactly, so they must not depend on fused
// multiply-add or other evaluation differences. Results need not be
// bit-identical with and without fusion; to check the build without it run
//
//	go test -run Dlanv2 -gcflags='plantcontrol.org/v1/gonum/lapack/gonum=-d=fmahash=n' ./lapack/gonum
func dlanv2Test(t *testing.T, impl Dlanv2er, a, b, c, d float64) (rt1, rt2 complex128) {
	t.Helper()
	aa, bb, cc, dd, rt1r, rt1i, rt2r, rt2i, cs, sn := impl.Dlanv2(a, b, c, d)
	rt1 = complex(rt1r, rt1i)
	rt2 = complex(rt2r, rt2i)

	mat := fmt.Sprintf("[%v %v; %v %v]", a, b, c, d)
	for _, v := range []float64{aa, bb, cc, dd, rt1r, rt1i, rt2r, rt2i, cs, sn} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("Non-finite result for %v: got aa=%v bb=%v cc=%v dd=%v rt1=%v rt2=%v cs=%v sn=%v",
				mat, aa, bb, cc, dd, rt1, rt2, cs, sn)
			return rt1, rt2
		}
	}

	if rt1r != aa || rt2r != dd {
		t.Errorf("Real parts of eigenvalues for %v not equal to diagonal: got %v, %v; want %v, %v", mat, rt1r, rt2r, aa, dd)
	}
	if cc == 0 {
		if rt1i != 0 || rt2i != 0 {
			t.Errorf("Unexpected complex eigenvalues for %v: got %v, %v", mat, rt1, rt2)
		}
	} else {
		if aa != dd {
			t.Errorf("Diagonal elements not equal for %v: got [%v %v]", mat, aa, dd)
		}
		if bb == 0 || math.Signbit(bb) == math.Signbit(cc) {
			t.Errorf("Non-diagonal elements do not have opposite signs for %v: got [%v %v]", mat, bb, cc)
		}
		if rt1i <= 0 || rt2i != -rt1i {
			t.Errorf("Imaginary parts of eigenvalues for %v not a conjugate pair: got %v, %v", mat, rt1i, rt2i)
		}
		im := math.Sqrt(math.Abs(bb)) * math.Sqrt(math.Abs(cc))
		if math.Abs(rt1i-im) > dlanv2Tol*im {
			t.Errorf("Unexpected imaginary part of eigenvalue for %v: got %v, want %v", mat, rt1i, im)
		}
	}

	if math.Abs(cs*cs+sn*sn-1) > dlanv2Tol {
		t.Errorf("Unexpected unitary matrix for %v: got cs %v, sn %v", mat, cs, sn)
	}

	norm := math.Max(math.Max(math.Abs(a), math.Abs(b)), math.Max(math.Abs(c), math.Abs(d)))
	if norm == 0 {
		return rt1, rt2
	}
	tol := dlanv2Tol * norm

	// Re-compute the original matrix [a b; c d] from its factorization.
	gota := cs*(aa*cs-bb*sn) - sn*(cc*cs-dd*sn)
	gotb := cs*(aa*sn+bb*cs) - sn*(cc*sn+dd*cs)
	gotc := sn*(aa*cs-bb*sn) + cs*(cc*cs-dd*sn)
	gotd := sn*(aa*sn+bb*cs) + cs*(cc*sn+dd*cs)
	if math.Abs(gota-a) > tol ||
		math.Abs(gotb-b) > tol ||
		math.Abs(gotc-c) > tol ||
		math.Abs(gotd-d) > tol {
		t.Errorf("Unexpected factorization: got [%v %v; %v %v], want %v", gota, gotb, gotc, gotd, mat)
	}

	// The trace and determinant are well-conditioned functions of the
	// matrix entries even when the eigenvalues are not, so they bound
	// the eigenvalue error regardless of conditioning.
	s := complex(norm, 0)
	trace := a/norm + d/norm
	det := (a/norm)*(d/norm) - (b/norm)*(c/norm)
	if got := real(rt1/s + rt2/s); math.Abs(got-trace) > dlanv2Tol {
		t.Errorf("Eigenvalue sum for %v: got %v, want %v", mat, got*norm, trace*norm)
	}
	if got := (rt1 / s) * (rt2 / s); cmplx.Abs(got-complex(det, 0)) > dlanv2Tol {
		t.Errorf("Eigenvalue product for %v: got %v, want %v (relative to norm squared)", mat, got, det)
	}
	return rt1, rt2
}
