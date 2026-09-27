// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"math"
	"testing"

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
	testDggevComplexEigenvalues(t, impl)
	testDggevSingularPencil(t, impl)
	testDggevExtremeScaleEigenvalueRepresentation(t, impl)
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
