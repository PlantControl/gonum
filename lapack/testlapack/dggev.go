// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

type Dggever interface {
	Dggev(jobvl lapack.LeftEVJob, jobvr lapack.RightEVJob, n int,
		a []float64, lda int, b []float64, ldb int,
		alphar, alphai, beta []float64,
		vl []float64, ldvl int, vr []float64, ldvr int,
		work []float64, lwork int) bool
}

func DggevTest(t *testing.T, impl Dggever) {
	testDggevWorkspace(t, impl)
	testDggevMinimumWorkspace(t, impl)
	testDggevEmpty(t, impl)

	for _, n := range []int{1, 2, 3, 4, 5} {
		testDggevDiagonal(t, impl, n)
	}

	testDggevEigenvaluesOnly(t, impl)
	testDggevEigenvectors(t, impl)
	testDggevEigenvectorResiduals(t, impl)
	testDggevComplexEigenvalues(t, impl)
	testDggevSingularPencil(t, impl)
	testDggevExtremeScaleEigenvalueRepresentation(t, impl)
	testDggevUnscaledEigenvalues(t, impl)
	testDggevFailureUnscalesEigenvalues(t, impl)
}

func testDggevExtremeScaleEigenvalueRepresentation(t *testing.T, impl Dggever) {
	const n = 2
	a := []float64{2e-105, -1e-105, 1e-105, 2e-105}
	b := []float64{1e239, 0, 0, 1e239}
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	work := make([]float64, 8*n)
	if !impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, n,
		a, n, b, n, alphar, alphai, beta, nil, 1, nil, 1, work, len(work)) {
		t.Fatal("extreme-scale solve did not converge")
	}
	for i := range n {
		if math.IsInf(alphar[i], 0) || math.IsInf(alphai[i], 0) || math.IsInf(beta[i], 0) ||
			math.IsNaN(alphar[i]) || math.IsNaN(alphai[i]) || math.IsNaN(beta[i]) {
			t.Fatalf("non-finite eigenvalue representation at %d: alpha=(%g,%g), beta=%g", i, alphar[i], alphai[i], beta[i])
		}
		if beta[i] == 0 || alphai[i] == 0 {
			t.Fatalf("lost finite complex eigenvalue at %d: alpha=(%g,%g), beta=%g", i, alphar[i], alphai[i], beta[i])
		}
	}
}

func testDggevMinimumWorkspace(t *testing.T, impl Dggever) {
	n := 4
	a := []float64{
		1, 2, 3, 4,
		5, 7, 11, 13,
		17, 19, 23, 29,
		31, 37, 41, 43,
	}
	b := []float64{
		2, 1, 0, 1,
		1, 3, 1, 0,
		0, 1, 4, 1,
		1, 0, 1, 5,
	}
	work := make([]float64, 8*n)
	if !impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, n,
		a, n, b, n, make([]float64, n), make([]float64, n), make([]float64, n),
		nil, 1, nil, 1, work, len(work)) {
		t.Fatal("minimum-workspace solve did not converge")
	}
}

func testDggevWorkspace(t *testing.T, impl Dggever) {
	for _, n := range []int{1, 5, 10, 20} {
		work := make([]float64, 1)
		impl.Dggev(lapack.LeftEVCompute, lapack.RightEVCompute, n,
			nil, max(1, n), nil, max(1, n),
			nil, nil, nil,
			nil, max(1, n), nil, max(1, n),
			work, -1)
		if work[0] < float64(8*n) {
			t.Errorf("n=%d: workspace query returned %v, want >= %d", n, work[0], 8*n)
		}
	}
}

func testDggevEmpty(t *testing.T, impl Dggever) {
	work := make([]float64, 1)
	ok := impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, 0,
		nil, 1, nil, 1,
		nil, nil, nil,
		nil, 1, nil, 1,
		work, 1)
	if !ok {
		t.Error("n=0: Dggev returned false")
	}
}

func testDggevDiagonal(t *testing.T, impl Dggever, n int) {
	// Test diagonal matrices A = diag(1,2,...,n), B = I.
	// Eigenvalues should be 1, 2, ..., n.
	a := zeros(n, n, n)
	b := eye(n, n)
	for i := 0; i < n; i++ {
		a.Data[i*a.Stride+i] = float64(i + 1)
	}

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)

	work := make([]float64, 1)
	impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, n,
		a.Data, a.Stride, b.Data, b.Stride,
		alphar, alphai, beta,
		nil, 1, nil, 1,
		work, -1)
	lwork := int(work[0])
	work = make([]float64, lwork)

	// Reset matrices.
	a = zeros(n, n, n)
	b = eye(n, n)
	for i := 0; i < n; i++ {
		a.Data[i*a.Stride+i] = float64(i + 1)
	}

	ok := impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, n,
		a.Data, a.Stride, b.Data, b.Stride,
		alphar, alphai, beta,
		nil, 1, nil, 1,
		work, lwork)

	if !ok {
		t.Errorf("n=%d diagonal: Dggev failed to converge", n)
		return
	}

	// Check eigenvalues. They should be 1, 2, ..., n (in some order).
	const tol = 1e-10
	evFound := make([]bool, n)
	for i := 0; i < n; i++ {
		if alphai[i] != 0 {
			t.Errorf("n=%d diagonal: expected real eigenvalue, got alphai[%d]=%v", n, i, alphai[i])
			continue
		}
		if beta[i] == 0 {
			t.Errorf("n=%d diagonal: unexpected infinite eigenvalue at %d", n, i)
			continue
		}
		ev := alphar[i] / beta[i]
		found := false
		for k := 0; k < n; k++ {
			if !evFound[k] && math.Abs(ev-float64(k+1)) < tol {
				evFound[k] = true
				found = true
				break
			}
		}
		if !found {
			t.Errorf("n=%d diagonal: unexpected eigenvalue %v", n, ev)
		}
	}
}

func testDggevEigenvaluesOnly(t *testing.T, impl Dggever) {
	// Test eigenvalue computation without eigenvectors.
	// Use diagonal A with diagonal B.
	// A = diag(2, 4, 6), B = diag(1, 2, 3) => eigenvalues are 2, 2, 2.
	n := 3
	a := zeros(n, n, n)
	b := zeros(n, n, n)
	a.Data[0] = 2
	a.Data[n+1] = 4
	a.Data[2*n+2] = 6
	b.Data[0] = 1
	b.Data[n+1] = 2
	b.Data[2*n+2] = 3

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)

	work := make([]float64, 1)
	impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, n,
		a.Data, a.Stride, b.Data, b.Stride,
		alphar, alphai, beta,
		nil, 1, nil, 1,
		work, -1)
	lwork := int(work[0])
	work = make([]float64, lwork)

	// Reset matrices.
	a = zeros(n, n, n)
	b = zeros(n, n, n)
	a.Data[0] = 2
	a.Data[n+1] = 4
	a.Data[2*n+2] = 6
	b.Data[0] = 1
	b.Data[n+1] = 2
	b.Data[2*n+2] = 3

	ok := impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, n,
		a.Data, a.Stride, b.Data, b.Stride,
		alphar, alphai, beta,
		nil, 1, nil, 1,
		work, lwork)

	if !ok {
		t.Error("eigenvalues only: Dggev failed to converge")
		return
	}

	// Check that all eigenvalues are 2.
	const tol = 1e-10
	for i := 0; i < n; i++ {
		if alphai[i] != 0 {
			t.Errorf("eigenvalues only: expected real eigenvalue, got alphai[%d]=%v", i, alphai[i])
			continue
		}
		if beta[i] == 0 {
			t.Errorf("eigenvalues only: unexpected infinite eigenvalue at %d", i)
			continue
		}
		ev := alphar[i] / beta[i]
		if math.Abs(ev-2) > tol {
			t.Errorf("eigenvalues only: eigenvalue[%d] = %v, want 2", i, ev)
		}
	}
}

func testDggevEigenvectors(t *testing.T, impl Dggever) {
	// Test eigenvector computation with diagonal matrices.
	// A = diag(1, 2, 3), B = I => eigenvalues 1, 2, 3 with standard basis eigenvectors.
	n := 3
	aOrig := zeros(n, n, n)
	bOrig := eye(n, n)
	for i := 0; i < n; i++ {
		aOrig.Data[i*aOrig.Stride+i] = float64(i + 1)
	}

	a := cloneGeneral(aOrig)
	b := cloneGeneral(bOrig)

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	vl := nanGeneral(n, n, n)
	vr := nanGeneral(n, n, n)

	work := make([]float64, 1)
	impl.Dggev(lapack.LeftEVCompute, lapack.RightEVCompute, n,
		nil, n, nil, n, nil, nil, nil, nil, n, nil, n, work, -1)
	lwork := int(work[0])
	work = make([]float64, lwork)

	ok := impl.Dggev(lapack.LeftEVCompute, lapack.RightEVCompute, n,
		a.Data, a.Stride, b.Data, b.Stride,
		alphar, alphai, beta,
		vl.Data, vl.Stride, vr.Data, vr.Stride,
		work, lwork)

	if !ok {
		t.Error("eigenvectors: Dggev failed to converge")
		return
	}

	// Verify eigenvalues.
	const tol = 1e-10
	for i := 0; i < n; i++ {
		if alphai[i] != 0 {
			t.Errorf("eigenvectors: expected real eigenvalue, got alphai[%d]=%v", i, alphai[i])
		}
		if beta[i] == 0 {
			t.Errorf("eigenvectors: unexpected infinite eigenvalue at %d", i)
		}
	}

	// Verify right eigenvector equation: A*v = lambda*B*v.
	// For diagonal A and identity B, eigenvectors should be standard basis.
	for j := 0; j < n; j++ {
		if beta[j] == 0 || alphai[j] != 0 {
			continue
		}
		lambda := alphar[j] / beta[j]

		// Compute A*v and lambda*B*v = lambda*v (since B=I).
		for i := 0; i < n; i++ {
			av := 0.0
			lbv := 0.0
			for k := 0; k < n; k++ {
				av += aOrig.Data[i*aOrig.Stride+k] * vr.Data[k*vr.Stride+j]
				lbv += lambda * bOrig.Data[i*bOrig.Stride+k] * vr.Data[k*vr.Stride+j]
			}
			if math.Abs(av-lbv) > tol*math.Max(1, math.Abs(av)) {
				t.Errorf("eigenvectors: right eigenvector residual at (%d,%d): |A*v - λ*B*v| = %v", i, j, math.Abs(av-lbv))
			}
		}
	}
}

func testDggevSingularPencil(t *testing.T, impl Dggever) {
	// Test singular pencils where B has zero rows/columns, producing
	// infinite eigenvalues (beta=0). Regression test for issue #1.
	for _, tc := range []struct {
		name string
		n    int
		a, b []float64
	}{
		{
			name: "9x9 Reinschke pencil",
			n:    9,
			a: []float64{
				0, 0, 1, 0, 0, 0, 0, 0, 0,
				2, 0, 0, 3, 4, 0, 0, 0, 0,
				0, 0, 5, 0, 0, 6, 0, 0, 0,
				0, 7, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 8, 9, 0, 10, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 11, 0,
				0, 12, 0, 0, 13, 0, 0, 0, 0,
				14, 0, 0, 0, 0, 0, 0, 0, 0,
				15, 0, 16, 0, 0, 0, 0, 0, 0,
			},
			b: func() []float64 {
				b := make([]float64, 81)
				for i := 0; i < 6; i++ {
					b[i*9+i] = 1
				}
				return b
			}(),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			n := tc.n
			a := make([]float64, n*n)
			copy(a, tc.a)
			b := make([]float64, n*n)
			copy(b, tc.b)

			alphar := make([]float64, n)
			alphai := make([]float64, n)
			beta := make([]float64, n)
			work := make([]float64, 1)
			impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, n,
				nil, n, nil, n, nil, nil, nil, nil, 1, nil, 1, work, -1)
			lwork := int(work[0])
			work = make([]float64, lwork)

			ok := impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, n,
				a, n, b, n,
				alphar, alphai, beta,
				nil, 1, nil, 1,
				work, lwork)

			if !ok {
				t.Fatal("Dggev did not converge on singular pencil")
			}

			nInf := 0
			for i := 0; i < n; i++ {
				if beta[i] == 0 {
					nInf++
				}
			}
			if nInf < 3 {
				t.Errorf("expected at least 3 infinite eigenvalues, got %d", nInf)
			}
		})
	}
}

func testDggevComplexEigenvalues(t *testing.T, impl Dggever) {
	// Test with matrices that produce complex conjugate eigenvalue pairs.
	// A = rotation-like matrix, B = identity.
	for _, tc := range []struct {
		name string
		n    int
		a    []float64
	}{
		{
			name: "4x4 rotation blocks",
			n:    4,
			a: []float64{
				0, -1, 0, 0,
				1, 0, 0, 0,
				0, 0, 0, -2,
				0, 0, 2, 0,
			},
		},
		{
			name: "6x6 mixed real and complex",
			n:    6,
			a: []float64{
				3, 1, 0, 0, 0, 0,
				-1, 3, 0, 0, 0, 0,
				0, 0, 5, 0, 0, 0,
				0, 0, 0, 0, -4, 0,
				0, 0, 0, 4, 0, 0,
				0, 0, 0, 0, 0, 7,
			},
		},
		{
			name: "5x5 dense with complex eigenvalues",
			n:    5,
			a: []float64{
				1, 2, 3, 4, 5,
				-2, 1, 0, 0, 0,
				0, 0, 0, -1, 2,
				0, 0, 1, 0, 3,
				0, 0, 0, 0, 4,
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			n := tc.n
			a := make([]float64, n*n)
			copy(a, tc.a)
			b := make([]float64, n*n)
			for i := 0; i < n; i++ {
				b[i*n+i] = 1
			}

			alphar := make([]float64, n)
			alphai := make([]float64, n)
			beta := make([]float64, n)
			vl := make([]float64, n*n)
			vr := make([]float64, n*n)

			work := make([]float64, 1)
			impl.Dggev(lapack.LeftEVCompute, lapack.RightEVCompute, n,
				nil, n, nil, n, nil, nil, nil, nil, n, nil, n, work, -1)
			lwork := int(work[0])
			work = make([]float64, lwork)

			ok := impl.Dggev(lapack.LeftEVCompute, lapack.RightEVCompute, n,
				a, n, b, n,
				alphar, alphai, beta,
				vl, n, vr, n,
				work, lwork)

			if !ok {
				t.Fatal("Dggev did not converge")
			}

			hasComplex := false
			for i := 0; i < n; i++ {
				if alphai[i] != 0 {
					hasComplex = true
				}
			}
			if !hasComplex {
				t.Error("expected complex eigenvalues but found none")
			}
			checkDggevNormalization(t, alphai, vl, n)
			checkDggevNormalization(t, alphai, vr, n)
		})
	}
}

func checkDggevNormalization(t *testing.T, alphai, v []float64, ldv int) {
	t.Helper()
	n := len(alphai)
	for j := 0; j < n; j++ {
		var vmax float64
		switch {
		case alphai[j] == 0:
			for i := 0; i < n; i++ {
				vmax = math.Max(vmax, math.Abs(v[i*ldv+j]))
			}
		case alphai[j] > 0:
			for i := 0; i < n; i++ {
				vmax = math.Max(vmax, math.Abs(v[i*ldv+j])+math.Abs(v[i*ldv+j+1]))
			}
		default:
			continue
		}
		if math.Abs(vmax-1) > 1e-14 {
			t.Errorf("eigenvector %d normalization=%v, want 1", j, vmax)
		}
	}
}

func testDggevEigenvectorResiduals(t *testing.T, impl Dggever) {
	fromRows := func(n, stride int, rows []float64) blas64.General {
		m := nanGeneral(n, n, stride)
		for i := 0; i < n; i++ {
			copy(m.Data[i*stride:i*stride+n], rows[i*n:i*n+n])
		}
		return m
	}
	for _, stride := range []int{3, 5} {
		testDggevResidualCase(t, impl, "upper-triangular", fromRows(3, stride,
			[]float64{1, 2, 0, 0, 3, 5, 0, 0, 7}), fromRows(3, stride,
			[]float64{1, 0, 0, 0, 1, 0, 0, 0, 1}), stride)
		testDggevResidualCase(t, impl, "complex-pair", fromRows(3, stride,
			[]float64{0, 1, 0, -2, -1, 3, 1, 0, 0.5}), fromRows(3, stride,
			[]float64{1, 0.2, 0, 0, 2, 0.1, 0.3, 0, 1}), stride)
	}

	rnd := rand.New(rand.NewPCG(1, 1))
	for n := 1; n <= 10; n++ {
		for _, extra := range []int{0, 3} {
			stride := n + extra
			for trial := 0; trial < 10; trial++ {
				a := randomGeneral(n, n, stride, rnd)
				b := randomGeneral(n, n, stride, rnd)
				if trial == 9 && n > 1 {
					// Singular B gives an infinite eigenvalue.
					for i := 0; i < n; i++ {
						b.Data[i*stride+n-1] = 0
					}
				}
				name := fmt.Sprintf("random n=%d stride=%d trial=%d", n, stride, trial)
				testDggevResidualCase(t, impl, name, a, b, stride)
			}
		}
	}
}

func testDggevResidualCase(t *testing.T, impl Dggever, name string, aOrig, bOrig blas64.General, ldv int) {
	t.Helper()
	n := aOrig.Rows
	a := cloneGeneral(aOrig)
	b := cloneGeneral(bOrig)
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	vl := nanGeneral(n, n, ldv)
	vr := nanGeneral(n, n, ldv)
	work := make([]float64, 1)
	impl.Dggev(lapack.LeftEVCompute, lapack.RightEVCompute, n, nil, a.Stride, nil, b.Stride,
		nil, nil, nil, nil, ldv, nil, ldv, work, -1)
	work = make([]float64, int(work[0]))
	if !impl.Dggev(lapack.LeftEVCompute, lapack.RightEVCompute, n, a.Data, a.Stride, b.Data, b.Stride,
		alphar, alphai, beta, vl.Data, ldv, vr.Data, ldv, work, len(work)) {
		t.Errorf("%s: Dggev did not converge", name)
		return
	}
	checkGeneralizedEigenvectorResiduals(t, name+" left", aOrig, bOrig, alphar, alphai, beta, vl.Data, ldv, true)
	checkGeneralizedEigenvectorResiduals(t, name+" right", aOrig, bOrig, alphar, alphai, beta, vr.Data, ldv, false)
	checkDggevNormalization(t, alphai, vl.Data, ldv)
	checkDggevNormalization(t, alphai, vr.Data, ldv)
}

// generalizedEigenvector returns the complex eigenvector of eigenvalue j
// stored in v, with a complex pair stored as (re, im) in columns j, j+1.
func generalizedEigenvector(n int, alphai, v []float64, ldv, j int) []complex128 {
	x := make([]complex128, n)
	for i := 0; i < n; i++ {
		switch {
		case alphai[j] == 0:
			x[i] = complex(v[i*ldv+j], 0)
		case alphai[j] > 0:
			x[i] = complex(v[i*ldv+j], v[i*ldv+j+1])
		default:
			x[i] = complex(v[i*ldv+j-1], -v[i*ldv+j])
		}
	}
	return x
}

// checkGeneralizedEigenvectorResiduals checks the normalized residuals
//
//	|(beta*A - alpha*B) x| / ((|beta|*|A| + |alpha|*|B|) * |x|)     right,
//	|yᴴ (beta*A - alpha*B)| / ((|beta|*|A| + |alpha|*|B|) * |y|)    left,
//
// for each eigenvalue alpha/beta, using max-abs vector norms and Frobenius
// matrix norms.
func checkGeneralizedEigenvectorResiduals(t *testing.T, name string, a, b blas64.General,
	alphar, alphai, beta, v []float64, ldv int, left bool) {
	t.Helper()
	n := a.Rows
	anorm := dlange(lapack.Frobenius, n, n, a.Data, a.Stride)
	bnorm := dlange(lapack.Frobenius, n, n, b.Data, b.Stride)
	tol := float64(max(n, 1)) * 100 * dlamchE
	for j := 0; j < n; j++ {
		alpha := complex(alphar[j], alphai[j])
		bet := complex(beta[j], 0)
		x := generalizedEigenvector(n, alphai, v, ldv, j)
		var xnorm, rnorm float64
		for _, xi := range x {
			xnorm = math.Max(xnorm, cmplx.Abs(xi))
		}
		for i := 0; i < n; i++ {
			var r complex128
			for k := 0; k < n; k++ {
				if left {
					r += (bet*complex(a.Data[k*a.Stride+i], 0) - alpha*complex(b.Data[k*b.Stride+i], 0)) * cmplx.Conj(x[k])
				} else {
					r += (bet*complex(a.Data[i*a.Stride+k], 0) - alpha*complex(b.Data[i*b.Stride+k], 0)) * x[k]
				}
			}
			rnorm = math.Max(rnorm, cmplx.Abs(r))
		}
		denom := (math.Abs(beta[j])*anorm + cmplx.Abs(alpha)*bnorm) * xnorm
		if xnorm == 0 || math.IsNaN(rnorm) {
			t.Errorf("%s: eigenvector %d is zero or NaN (|x|=%g, residual=%g)", name, j, xnorm, rnorm)
			continue
		}
		if rnorm == 0 {
			continue
		}
		if res := rnorm / denom; res > tol {
			t.Errorf("%s: eigenvector %d (alpha=%v, beta=%v) residual=%g, want <= %g", name, j, alpha, beta[j], res, tol)
		}
	}
}

// testDggevUnscaledEigenvalues checks that the scaling of a badly scaled A or
// B is undone on alphar, alphai and beta individually, as LAPACK does, so
// (A, c*B) returns (alpha, c*beta) when (A, B) returns (alpha, beta).
func testDggevUnscaledEigenvalues(t *testing.T, impl Dggever) {
	rnd := rand.New(rand.NewPCG(7, 7))
	for _, n := range []int{2, 3, 5, 8} {
		for _, sc := range []struct{ ca, cb float64 }{
			{1e-200, 1}, {1e200, 1}, {1, 1e-200}, {1, 1e200}, {1e-200, 1e200}, {1e200, 1e-200},
		} {
			a0 := randomGeneral(n, n, n, rnd)
			b0 := randomGeneral(n, n, n, rnd)
			if n > 2 {
				for i := 0; i < n; i++ {
					b0.Data[i*n+n-1] = 0
				}
			}
			ar0, ai0, beta0 := make([]float64, n), make([]float64, n), make([]float64, n)
			ar, ai, beta := make([]float64, n), make([]float64, n), make([]float64, n)
			run := func(a, b blas64.General, ar, ai, beta []float64) {
				work := make([]float64, 1)
				impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, n, nil, n, nil, n,
					nil, nil, nil, nil, 1, nil, 1, work, -1)
				work = make([]float64, int(work[0]))
				if !impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, n, a.Data, n, b.Data, n,
					ar, ai, beta, nil, 1, nil, 1, work, len(work)) {
					t.Fatalf("n=%d ca=%g cb=%g: Dggev did not converge", n, sc.ca, sc.cb)
				}
			}
			a, b := cloneGeneral(a0), cloneGeneral(b0)
			for i := range a.Data {
				a.Data[i] *= sc.ca
				b.Data[i] *= sc.cb
			}
			run(cloneGeneral(a0), cloneGeneral(b0), ar0, ai0, beta0)
			run(a, b, ar, ai, beta)
			anorm := 0.0
			for j := 0; j < n; j++ {
				anorm = math.Max(anorm, math.Hypot(ar0[j], ai0[j]))
			}
			bnorm := 0.0
			for j := 0; j < n; j++ {
				bnorm = math.Max(bnorm, math.Abs(beta0[j]))
			}
			const tol = 1e-10
			for j := 0; j < n; j++ {
				name := fmt.Sprintf("n=%d ca=%g cb=%g j=%d", n, sc.ca, sc.cb, j)
				if d := math.Hypot(ar[j]/sc.ca-ar0[j], ai[j]/sc.ca-ai0[j]); d > tol*anorm {
					t.Errorf("%s: alpha/ca=(%g,%g), want (%g,%g)", name, ar[j]/sc.ca, ai[j]/sc.ca, ar0[j], ai0[j])
				}
				if d := math.Abs(beta[j]/sc.cb - beta0[j]); d > tol*bnorm {
					t.Errorf("%s: beta/cb=%g, want %g", name, beta[j]/sc.cb, beta0[j])
				}
			}
		}
	}
}

// testDggevFailureUnscalesEigenvalues checks that the input scaling is undone
// on the eigenvalues QZ isolated before a failure, as LAPACK's DGGEV does by
// jumping to its unscaling code. A NaN in the 3×3 active block of one matrix
// makes QZ fail while the other matrix is badly scaled; the last row of the
// pencil is triangular, so balancing isolates its eigenvalue
// a[n-1,n-1]/b[n-1,n-1] before QZ starts.
func testDggevFailureUnscalesEigenvalues(t *testing.T, impl Dggever) {
	const n = 4
	for _, tc := range []struct {
		name string
		a, b []float64
	}{
		{"NaN A, huge B", dggevFailurePencil(math.NaN(), 1), dggevFailurePencil(1, 1e300)},
		{"NaN A, tiny B", dggevFailurePencil(math.NaN(), 1), dggevFailurePencil(1, 1e-300)},
		{"huge A, NaN B", dggevFailurePencil(1, 1e300), dggevFailurePencil(math.NaN(), 1)},
		{"tiny A, NaN B", dggevFailurePencil(1, 1e-300), dggevFailurePencil(math.NaN(), 1)},
	} {
		for _, wantv := range []bool{false, true} {
			jobvl, jobvr := lapack.LeftEVNone, lapack.RightEVNone
			if wantv {
				jobvl, jobvr = lapack.LeftEVCompute, lapack.RightEVCompute
			}
			wantAlpha, wantBeta := tc.a[n*n-1], tc.b[n*n-1]
			a := append([]float64(nil), tc.a...)
			b := append([]float64(nil), tc.b...)
			ar, ai, beta := make([]float64, n), make([]float64, n), make([]float64, n)
			vl, vr := make([]float64, n*n), make([]float64, n*n)
			work := make([]float64, 1)
			impl.Dggev(jobvl, jobvr, n, nil, n, nil, n, nil, nil, nil, nil, n, nil, n, work, -1)
			work = make([]float64, int(work[0]))
			if impl.Dggev(jobvl, jobvr, n, a, n, b, n, ar, ai, beta, vl, n, vr, n, work, len(work)) {
				t.Errorf("%s wantv=%v: Dggev converged, want QZ failure", tc.name, wantv)
				continue
			}
			if math.Abs(ar[n-1]-wantAlpha) > 1e-14*math.Abs(wantAlpha) || ai[n-1] != 0 ||
				math.Abs(beta[n-1]-wantBeta) > 1e-14*math.Abs(wantBeta) {
				t.Errorf("%s wantv=%v: isolated eigenvalue alpha=(%g,%g) beta=%g, want alpha=%g beta=%g",
					tc.name, wantv, ar[n-1], ai[n-1], beta[n-1], wantAlpha, wantBeta)
			}
		}
	}
}

// dggevFailurePencil returns a row-major 4×4 matrix scaled by c whose last
// row is zero except its diagonal and whose leading 3×3 block is dense and
// contains v. With v = NaN, QZ on a pencil built from it fails.
func dggevFailurePencil(v, c float64) []float64 {
	m := []float64{
		2, 1, 1, 1,
		1, v, 2, 1,
		1, 2, 3, 1,
		0, 0, 0, 0.5,
	}
	for i := range m {
		m[i] *= c
	}
	return m
}
