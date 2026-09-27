// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

type Dtgsy2er interface {
	Dtgsy2(trans blas.Transpose, ijob, m, n int, a []float64, lda int,
		b []float64, ldb int, c []float64, ldc int, d []float64, ldd int,
		e []float64, lde int, f []float64, ldf int,
		rdsum, rdscal float64, iwork []int) (scale, rdsum2, rdscal2 float64, pq int, ok bool)
}

func Dtgsy2Test(t *testing.T, impl Dtgsy2er) {
	rnd := rand.New(rand.NewPCG(1, 1))

	for _, m := range []int{0, 1, 2, 3, 4, 5, 10} {
		for _, n := range []int{0, 1, 2, 3, 4, 5, 10} {
			for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
				for _, ijob := range []int{0} {
					for _, extra := range []int{0, 5} {
						testDtgsy2(t, impl, m, n, trans, ijob, extra, rnd)
					}
				}
			}
		}
	}

	testDtgsy2Special(t, impl)
	testDtgsy2DifAccumulatorDoesNotScaleSystem(t, impl)
	testDtgsy2TransIgnoresIjob(t, impl)
}

func testDtgsy2TransIgnoresIjob(t *testing.T, impl Dtgsy2er) {
	for _, ijob := range []int{-1, 3} {
		c := []float64{1}
		f := []float64{1}
		_, _, _, _, ok := impl.Dtgsy2(blas.Trans, ijob, 1, 1,
			[]float64{2}, 1, []float64{3}, 1, c, 1,
			[]float64{5}, 1, []float64{7}, 1, f, 1,
			0, 1, make([]int, 4))
		if !ok {
			t.Errorf("ijob=%d: transposed solve reported coincident eigenvalues", ijob)
		}
	}
}

func testDtgsy2DifAccumulatorDoesNotScaleSystem(t *testing.T, impl Dtgsy2er) {
	c := []float64{0}
	f := []float64{0}
	scale, _, _, pq, ok := impl.Dtgsy2(blas.NoTrans, 1, 1, 1,
		[]float64{1e-200}, 1, []float64{0}, 1, c, 1,
		[]float64{0}, 1, []float64{-1e-200}, 1, f, 1,
		0, 1, make([]int, 8))
	if !ok {
		t.Fatal("scaled identity operator reported coincident eigenvalues")
	}
	if scale != 1 {
		t.Fatalf("DIF accumulator changed equation scale to %v, want 1", scale)
	}
	if pq != 1 {
		t.Fatalf("PQ=%d, want one solved 2x2 subsystem", pq)
	}
}

func testDtgsy2(t *testing.T, impl Dtgsy2er, m, n int, trans blas.Transpose, ijob, extra int, rnd *rand.Rand) {
	const tol = 1e-10

	prefix := fmt.Sprintf("m=%v, n=%v, trans=%c, ijob=%d, extra=%v",
		m, n, trans, ijob, extra)

	// Generate random quasi-triangular matrices A (m×m) and B (n×n).
	a := randomSchur(m, m+extra, rnd)
	b := randomUpperTriangular(n, n+extra, rnd)

	// Generate random upper triangular matrices D (m×m) and E (n×n).
	d := randomUpperTriangular(m, m+extra, rnd)
	e := randomUpperTriangular(n, n+extra, rnd)

	// Generate random RHS matrices C, F (m×n).
	c := randomGeneral(m, n, n+extra, rnd)
	f := randomGeneral(m, n, n+extra, rnd)
	cOrig := cloneGeneral(c)
	fOrig := cloneGeneral(f)

	// Keep originals of A, B, D, E for residual check.
	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)
	dOrig := cloneGeneral(d)
	eOrig := cloneGeneral(e)

	// Allocate iwork.
	iwork := make([]int, m+n+6)

	// Call Dtgsy2.
	scale, _, _, _, ok := impl.Dtgsy2(trans, ijob, m, n,
		a.Data, a.Stride, b.Data, b.Stride, c.Data, c.Stride,
		d.Data, d.Stride, e.Data, e.Stride, f.Data, f.Stride,
		0, 1, iwork)

	// Check that A, B, D, E are not modified.
	if !generalEqual(a, aOrig) {
		t.Errorf("%v: A was modified", prefix)
	}
	if !generalEqual(b, bOrig) {
		t.Errorf("%v: B was modified", prefix)
	}
	if !generalEqual(d, dOrig) {
		t.Errorf("%v: D was modified", prefix)
	}
	if !generalEqual(e, eOrig) {
		t.Errorf("%v: E was modified", prefix)
	}

	// Quick return for empty matrices.
	if m == 0 || n == 0 {
		if scale != 1 {
			t.Errorf("%v: scale=%v, want 1 for empty matrix", prefix, scale)
		}
		if !ok {
			t.Errorf("%v: ok=false, want true for empty matrix", prefix)
		}
		return
	}

	// Verify the solution by computing the residual.
	// For trans='N':
	//   R = A*R - L*B - scale*C  (should be ≈ 0)
	//   R = D*R - L*E - scale*F  (should be ≈ 0)
	// For trans='T':
	//   R = A^T*R + D^T*L - scale*C  (should be ≈ 0)
	//   R = R*B^T + L*E^T + scale*F  (should be ≈ 0)
	resid := dtgsy2Residual(aOrig, bOrig, cOrig, dOrig, eOrig, fOrig, c, f, trans, scale)

	anorm := dlange(lapack.MaxColumnSum, m, m, aOrig.Data, aOrig.Stride)
	bnorm := dlange(lapack.MaxColumnSum, n, n, bOrig.Data, bOrig.Stride)
	dnorm := dlange(lapack.MaxColumnSum, m, m, dOrig.Data, dOrig.Stride)
	enorm := dlange(lapack.MaxColumnSum, n, n, eOrig.Data, eOrig.Stride)
	rnorm := dlange(lapack.MaxColumnSum, m, n, c.Data, c.Stride)
	lnorm := dlange(lapack.MaxColumnSum, m, n, f.Data, f.Stride)
	cnorm := dlange(lapack.MaxColumnSum, m, n, cOrig.Data, cOrig.Stride)
	fnorm := dlange(lapack.MaxColumnSum, m, n, fOrig.Data, fOrig.Stride)

	denom := (anorm+dnorm)*rnorm + (bnorm+enorm)*lnorm + cnorm + fnorm
	if denom == 0 {
		denom = 1
	}
	normRes := resid / (denom * float64(max(m, n)) * dlamchE)

	if normRes > tol/dlamchE {
		t.Errorf("%v: residual too large: %v, want <= %v (ok=%v)", prefix, normRes, tol/dlamchE, ok)
	}
}

func testDtgsy2Special(t *testing.T, impl Dtgsy2er) {
	const tol = 1e-10

	// Test case: simple 2×2 system with diagonal matrices.
	m, n := 2, 2
	a := blas64.General{Rows: m, Cols: m, Stride: m, Data: []float64{1, 0, 0, 2}}
	b := blas64.General{Rows: n, Cols: n, Stride: n, Data: []float64{3, 1, 0, 4}}
	d := blas64.General{Rows: m, Cols: m, Stride: m, Data: []float64{2, 0, 0, 1}}
	e := blas64.General{Rows: n, Cols: n, Stride: n, Data: []float64{1, 2, 0, 3}}
	c := blas64.General{Rows: m, Cols: n, Stride: n, Data: []float64{1, 2, 3, 4}}
	f := blas64.General{Rows: m, Cols: n, Stride: n, Data: []float64{5, 6, 7, 8}}

	cOrig := cloneGeneral(c)
	fOrig := cloneGeneral(f)

	iwork := make([]int, m+n+6)

	scale, _, _, _, ok := impl.Dtgsy2(blas.NoTrans, 0, m, n,
		a.Data, a.Stride, b.Data, b.Stride, c.Data, c.Stride,
		d.Data, d.Stride, e.Data, e.Stride, f.Data, f.Stride,
		0, 1, iwork)

	if !ok {
		t.Error("Special case: ok=false, expected true")
	}

	resid := dtgsy2Residual(a, b, cOrig, d, e, fOrig, c, f, blas.NoTrans, scale)

	anorm := dlange(lapack.MaxColumnSum, m, m, a.Data, a.Stride)
	bnorm := dlange(lapack.MaxColumnSum, n, n, b.Data, b.Stride)
	dnorm := dlange(lapack.MaxColumnSum, m, m, d.Data, d.Stride)
	enorm := dlange(lapack.MaxColumnSum, n, n, e.Data, e.Stride)
	rnorm := dlange(lapack.MaxColumnSum, m, n, c.Data, c.Stride)
	lnorm := dlange(lapack.MaxColumnSum, m, n, f.Data, f.Stride)
	cnorm := dlange(lapack.MaxColumnSum, m, n, cOrig.Data, cOrig.Stride)
	fnorm := dlange(lapack.MaxColumnSum, m, n, fOrig.Data, fOrig.Stride)

	denom := (anorm+dnorm)*rnorm + (bnorm+enorm)*lnorm + cnorm + fnorm
	if denom == 0 {
		denom = 1
	}
	normRes := resid / (denom * float64(max(m, n)) * dlamchE)

	if normRes > tol/dlamchE {
		t.Errorf("Special case: residual too large: %v", normRes)
	}
}

// dtgsy2Residual computes the combined residual for the generalized Sylvester equation.
func dtgsy2Residual(A, B, C, D, E, F, R, L blas64.General, trans blas.Transpose, scale float64) float64 {
	m := A.Rows
	n := B.Rows

	if m == 0 || n == 0 {
		return 0
	}

	var resid1, resid2 float64

	if trans == blas.NoTrans {
		// Equation 1: A*R - L*B = scale*C
		// Residual: A*R - L*B - scale*C
		res1 := zeros(m, n, n)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, A, R, 0, res1)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, L, B, 1, res1)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				res1.Data[i*res1.Stride+j] -= scale * C.Data[i*C.Stride+j]
			}
		}
		resid1 = dlange(lapack.MaxColumnSum, m, n, res1.Data, res1.Stride)

		// Equation 2: D*R - L*E = scale*F
		// Residual: D*R - L*E - scale*F
		res2 := zeros(m, n, n)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, D, R, 0, res2)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, L, E, 1, res2)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				res2.Data[i*res2.Stride+j] -= scale * F.Data[i*F.Stride+j]
			}
		}
		resid2 = dlange(lapack.MaxColumnSum, m, n, res2.Data, res2.Stride)
	} else {
		// Equation 1: A^T*R + D^T*L = scale*C
		// Residual: A^T*R + D^T*L - scale*C
		res1 := zeros(m, n, n)
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, A, R, 0, res1)
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, D, L, 1, res1)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				res1.Data[i*res1.Stride+j] -= scale * C.Data[i*C.Stride+j]
			}
		}
		resid1 = dlange(lapack.MaxColumnSum, m, n, res1.Data, res1.Stride)

		// Equation 2: R*B^T + L*E^T = -scale*F
		// Residual: R*B^T + L*E^T + scale*F
		res2 := zeros(m, n, n)
		blas64.Gemm(blas.NoTrans, blas.Trans, 1, R, B, 0, res2)
		blas64.Gemm(blas.NoTrans, blas.Trans, 1, L, E, 1, res2)
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				res2.Data[i*res2.Stride+j] += scale * F.Data[i*F.Stride+j]
			}
		}
		resid2 = dlange(lapack.MaxColumnSum, m, n, res2.Data, res2.Stride)
	}

	return math.Max(resid1, resid2)
}

// randomUpperTriangular generates a random upper triangular matrix.
func randomUpperTriangular(n, stride int, rnd *rand.Rand) blas64.General {
	if n == 0 {
		return blas64.General{Rows: 0, Cols: 0, Stride: max(1, stride)}
	}

	a := zeros(n, n, stride)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			a.Data[i*a.Stride+j] = rnd.NormFloat64()
		}
		// Ensure diagonal is not too small.
		if math.Abs(a.Data[i*a.Stride+i]) < 0.1 {
			a.Data[i*a.Stride+i] = 0.1 + rnd.Float64()
		}
	}
	return a
}
