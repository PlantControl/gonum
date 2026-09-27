// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas/blas64"
)

type Dlagv2er interface {
	Dlagv2(a []float64, lda int, b []float64, ldb int) (csq, snq, csr, snr, csz, snz, scale1, scale2, alphar0, alphar1, alphai0, alphai1 float64)
}

func Dlagv2Test(t *testing.T, impl Dlagv2er) {
	// Test with real eigenvalues.
	testDlagv2Real(t, impl)

	// Test with complex eigenvalues.
	testDlagv2Complex(t, impl)
	testDlagv2Random(t, impl)
}

func testDlagv2Random(t *testing.T, impl Dlagv2er) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for k := 0; k < 1000; k++ {
		a := blas64.General{Rows: 2, Cols: 2, Stride: 2, Data: []float64{
			rnd.NormFloat64(), rnd.NormFloat64(), rnd.NormFloat64(), rnd.NormFloat64(),
		}}
		b := blas64.General{Rows: 2, Cols: 2, Stride: 2, Data: []float64{
			rnd.Float64() + 0.1, rnd.NormFloat64(), 0, rnd.Float64() + 0.1,
		}}
		aOrig := cloneGeneral(a)
		bOrig := cloneGeneral(b)
		cq, sq, _, _, cz, sz, _, _, _, _, _, _ := impl.Dlagv2(a.Data, 2, b.Data, 2)
		checkDlagv2Decomposition(t, aOrig, bOrig, a, b, cq, sq, cz, sz)
	}
}

func testDlagv2Real(t *testing.T, impl Dlagv2er) {
	// 2×2 upper triangular pencil (both A and B upper triangular).
	a := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			2, 1,
			0, 3,
		},
	}
	b := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 0.5,
			0, 1,
		},
	}

	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)
	csq, snq, _, _, csz, snz, scale1, scale2, alphar0, alphar1, alphai0, alphai1 := impl.Dlagv2(a.Data, a.Stride, b.Data, b.Stride)
	checkDlagv2Decomposition(t, aOrig, bOrig, a, b, csq, snq, csz, snz)

	// Check Q is orthogonal.
	if math.Abs(csq*csq+snq*snq-1) > 1e-10 {
		t.Errorf("Real case: Q not orthogonal: csq²+snq²=%v", csq*csq+snq*snq)
	}

	// Check Z is orthogonal.
	if math.Abs(csz*csz+snz*snz-1) > 1e-10 {
		t.Errorf("Real case: Z not orthogonal: csz²+snz²=%v", csz*csz+snz*snz)
	}

	// Check eigenvalues are real.
	if alphai0 != 0 || alphai1 != 0 {
		t.Errorf("Real case: imaginary parts non-zero: alphai0=%v, alphai1=%v", alphai0, alphai1)
	}

	// Check scale factors are positive.
	if scale1 <= 0 || scale2 <= 0 {
		t.Errorf("Real case: scale factors not positive: scale1=%v, scale2=%v", scale1, scale2)
	}

	_ = alphar0
	_ = alphar1
}

func testDlagv2Complex(t *testing.T, impl Dlagv2er) {
	// 2×2 pencil with complex eigenvalues.
	a := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 2,
			-2, 1,
		},
	}
	b := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 0,
			0, 1,
		},
	}

	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)
	csq, snq, _, _, csz, snz, scale1, scale2, alphar0, alphar1, alphai0, alphai1 := impl.Dlagv2(a.Data, a.Stride, b.Data, b.Stride)
	checkDlagv2Decomposition(t, aOrig, bOrig, a, b, csq, snq, csz, snz)

	// Check Q is orthogonal.
	if math.Abs(csq*csq+snq*snq-1) > 1e-10 {
		t.Errorf("Complex case: Q not orthogonal: csq²+snq²=%v", csq*csq+snq*snq)
	}

	// Check Z is orthogonal.
	if math.Abs(csz*csz+snz*snz-1) > 1e-10 {
		t.Errorf("Complex case: Z not orthogonal: csz²+snz²=%v", csz*csz+snz*snz)
	}

	// Complex eigenvalues should have conjugate pairs.
	if alphai0 != -alphai1 {
		t.Errorf("Complex case: eigenvalues not conjugate: alphai0=%v, alphai1=%v", alphai0, alphai1)
	}

	_ = alphar0
	_ = alphar1
	_ = scale1
	_ = scale2
}

func checkDlagv2Decomposition(t *testing.T, aOrig, bOrig, a, b blas64.General, cq, sq, cz, sz float64) {
	t.Helper()
	q := blas64.General{Rows: 2, Cols: 2, Stride: 2, Data: []float64{cq, -sq, sq, cq}}
	z := blas64.General{Rows: 2, Cols: 2, Stride: 2, Data: []float64{cz, -sz, sz, cz}}
	checkDtgex2Decomposition(t, "Dlagv2", aOrig, bOrig, a, b, q, z)
}
