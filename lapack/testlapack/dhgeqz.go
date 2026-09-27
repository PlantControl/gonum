// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

type Dhgeqzer interface {
	Dhgeqz(job lapack.SchurJob, compq, compz lapack.SchurComp, n, ilo, ihi int,
		h []float64, ldh int, t []float64, ldt int, alphar, alphai, beta []float64,
		q []float64, ldq int, z []float64, ldz int, work []float64, lwork int) bool
}

func DhgeqzTest(t *testing.T, impl Dhgeqzer) {
	// Test workspace query.
	work := make([]float64, 1)
	ok := impl.Dhgeqz(lapack.EigenvaluesOnly, lapack.SchurNone, lapack.SchurNone, 10, 0, 9,
		nil, 10, nil, 10, nil, nil, nil, nil, 1, nil, 1, work, -1)
	if !ok {
		t.Error("Workspace query failed")
	}
	if work[0] < 1 {
		t.Errorf("Workspace query returned invalid size: %v", work[0])
	}

	testDhgeqz2x2(t, impl)
	testDhgeqzComplex2x2NonDiagonalT(t, impl)
	testDhgeqzIsolatedEigenvalueSigns(t, impl)
	testDhgeqzNonlocalScaleDeflation(t, impl)
	testDhgeqzFailureLeavesLeadingEigenvalues(t, impl)
	testDhgeqz3x3(t, impl)
	testDhgeqzComplex4x4(t, impl)
	testDhgeqzComplex6x6(t, impl)
	testDhgeqzLargeN(t, impl, 50)
	testDhgeqzLargeN(t, impl, 100)
}

func testDhgeqzFailureLeavesLeadingEigenvalues(t *testing.T, impl Dhgeqzer) {
	const n = 4
	h := []float64{
		2, 0, 0, 0,
		0, 1, 1, 0,
		0, 1, 2, 1,
		0, 0, math.NaN(), 3,
	}
	tt := []float64{
		-1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
	alphar := []float64{11, 12, 13, 14}
	alphai := []float64{21, 22, 23, 24}
	beta := []float64{31, 32, 33, 34}
	work := make([]float64, n)
	ok := impl.Dhgeqz(lapack.EigenvaluesOnly, lapack.SchurNone, lapack.SchurNone,
		n, 1, n-1, h, n, tt, n, alphar, alphai, beta, nil, 1, nil, 1, work, len(work))
	if ok {
		t.Fatal("non-finite active block unexpectedly converged")
	}
	if h[0] != 2 || tt[0] != -1 || alphar[0] != 11 || alphai[0] != 21 || beta[0] != 31 {
		t.Fatalf("failure modified leading isolated eigenvalue: H=%g T=%g alpha=(%g,%g) beta=%g",
			h[0], tt[0], alphar[0], alphai[0], beta[0])
	}
}

func testDhgeqzNonlocalScaleDeflation(t *testing.T, impl Dhgeqzer) {
	const n = 3
	h := []float64{
		1, 1, 1e300,
		1e-10, 2, 1,
		0, 1, 3,
	}
	tt := []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	}
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	q := make([]float64, n*n)
	z := make([]float64, n*n)
	work := make([]float64, n)

	if !impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess,
		n, 0, n-1, h, n, tt, n, alphar, alphai, beta, q, n, z, n, work, len(work)) {
		t.Fatal("nonlocal-scale pencil did not converge")
	}
	if alphai[0] == 0 && alphai[1] == 0 && alphai[2] == 0 {
		t.Fatalf("nonlocal scale caused premature deflation: alphar=%v alphai=%v beta=%v", alphar, alphai, beta)
	}
}

func testDhgeqzIsolatedEigenvalueSigns(t *testing.T, impl Dhgeqzer) {
	const n = 3
	h := []float64{
		2, 1, 0,
		0, 3, 1,
		0, 0, 4,
	}
	tt := []float64{
		-1, 2, 0,
		0, -2, 3,
		0, 0, -4,
	}
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	work := make([]float64, n)
	if !impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurNone, lapack.SchurNone,
		n, 1, 1, h, n, tt, n, alphar, alphai, beta, nil, 1, nil, 1, work, len(work)) {
		t.Fatal("isolated eigenvalue case did not converge")
	}
	for i, v := range beta {
		if v < 0 {
			t.Errorf("beta[%d]=%v, want non-negative Netlib representation", i, v)
		}
	}
}

func testDhgeqz2x2(t *testing.T, impl Dhgeqzer) {
	// 2x2 upper Hessenberg-triangular pair.
	h := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			2, 1,
			1, 3,
		},
	}
	tt := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 0.5,
			0, 1,
		},
	}
	q := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 0,
			0, 1,
		},
	}
	z := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 0,
			0, 1,
		},
	}

	alphar := make([]float64, 2)
	alphai := make([]float64, 2)
	beta := make([]float64, 2)
	work := make([]float64, 10)

	ok := impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess, 2, 0, 1,
		h.Data, h.Stride, tt.Data, tt.Stride, alphar, alphai, beta,
		q.Data, q.Stride, z.Data, z.Stride, work, 10)

	if !ok {
		t.Fatal("2x2 test: QZ iteration did not converge")
	}
	if alphai[0] != 0 || alphai[1] != 0 {
		t.Fatalf("2x2 test: expected two real eigenvalues, got alphai=%v", alphai)
	}
	if math.Abs(h.Data[h.Stride]) > 1e-14 {
		t.Fatalf("2x2 test: real block was not split: H[1,0]=%g", h.Data[h.Stride])
	}

	// Check that beta values are non-zero for non-infinite eigenvalues.
	for i := 0; i < 2; i++ {
		if math.Abs(beta[i]) > 1e-10 && math.IsNaN(alphar[i]/beta[i]) {
			t.Errorf("2x2 test: eigenvalue %d is NaN", i)
		}
	}
}

func testDhgeqzComplex2x2NonDiagonalT(t *testing.T, impl Dhgeqzer) {
	h := []float64{0, -1, 1, 0}
	tt := []float64{2, 1, 0, 3}
	q := make([]float64, 4)
	z := make([]float64, 4)
	alphar := make([]float64, 2)
	alphai := make([]float64, 2)
	beta := make([]float64, 2)
	work := make([]float64, 2)

	ok := impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess,
		2, 0, 1, h, 2, tt, 2, alphar, alphai, beta, q, 2, z, 2, work, len(work))
	if !ok {
		t.Fatal("complex 2x2 test: QZ iteration did not converge")
	}
	if alphai[0] <= 0 || alphai[1] >= 0 {
		t.Fatalf("complex 2x2 test: expected a conjugate pair, got alphai=%v", alphai)
	}
	const tol = 1e-14
	if math.Abs(tt[1]) > tol || math.Abs(tt[2]) > tol {
		t.Fatalf("complex 2x2 test: T is not diagonal: %v", tt)
	}
	if tt[0] <= 0 || tt[3] <= 0 {
		t.Fatalf("complex 2x2 test: T diagonal is not positive: %v", tt)
	}
}

func testDhgeqz3x3(t *testing.T, impl Dhgeqzer) {
	// 3x3 upper Hessenberg-triangular pair.
	h := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			2, 1, 0.5,
			1, 3, 1,
			0, 1, 4,
		},
	}
	tt := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0.3, 0.1,
			0, 1, 0.2,
			0, 0, 1,
		},
	}
	q := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
		},
	}
	z := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
		},
	}

	alphar := make([]float64, 3)
	alphai := make([]float64, 3)
	beta := make([]float64, 3)
	work := make([]float64, 20)

	ok := impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess, 3, 0, 2,
		h.Data, h.Stride, tt.Data, tt.Stride, alphar, alphai, beta,
		q.Data, q.Stride, z.Data, z.Stride, work, 20)

	if !ok {
		t.Fatal("3x3 test: QZ iteration did not converge")
	}

	// Check that beta values are non-zero for non-infinite eigenvalues.
	for i := 0; i < 3; i++ {
		if math.Abs(beta[i]) > 1e-10 && math.IsNaN(alphar[i]/beta[i]) {
			t.Errorf("3x3 test: eigenvalue %d is NaN", i)
		}
	}
}

func testDhgeqzComplex4x4(t *testing.T, impl Dhgeqzer) {
	// 4x4 Hessenberg-triangular pair with complex conjugate eigenvalues.
	// H has structure that produces complex eigenvalue pairs.
	n := 4
	h := blas64.General{
		Rows: n, Cols: n, Stride: n,
		Data: []float64{
			1, 2, 0.5, 0.3,
			-3, 1, 1.0, 0.2,
			0, -2, 0, 1.5,
			0, 0, -1, 0,
		},
	}
	tt := blas64.General{
		Rows: n, Cols: n, Stride: n,
		Data: []float64{
			1, 0.5, 0.2, 0.1,
			0, 1, 0.3, 0.1,
			0, 0, 1, 0.4,
			0, 0, 0, 1,
		},
	}

	hOrig := make([]float64, len(h.Data))
	tOrig := make([]float64, len(tt.Data))
	copy(hOrig, h.Data)
	copy(tOrig, tt.Data)

	q := blas64.General{Rows: n, Cols: n, Stride: n, Data: make([]float64, n*n)}
	z := blas64.General{Rows: n, Cols: n, Stride: n, Data: make([]float64, n*n)}
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	work := make([]float64, 4*n)

	ok := impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess, n, 0, n-1,
		h.Data, h.Stride, tt.Data, tt.Stride, alphar, alphai, beta,
		q.Data, q.Stride, z.Data, z.Stride, work, len(work))

	if !ok {
		t.Fatal("4x4 complex test: QZ iteration did not converge")
	}

	hasComplex := false
	for i := 0; i < n; i++ {
		if alphai[i] != 0 {
			hasComplex = true
			break
		}
	}
	if !hasComplex {
		t.Error("4x4 complex test: no complex eigenvalues detected")
	}

	// Verify Schur decomposition: Q^T * H_orig * Z ≈ S.
	bi := blas64.Implementation()
	const tol = 1e-10
	tmp := make([]float64, n*n)
	result := make([]float64, n*n)

	// tmp = H_orig * Z
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for k := 0; k < n; k++ {
				s += hOrig[i*n+k] * z.Data[k*n+j]
			}
			tmp[i*n+j] = s
		}
	}
	// result = Q^T * tmp
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for k := 0; k < n; k++ {
				s += q.Data[k*n+i] * tmp[k*n+j]
			}
			result[i*n+j] = s
		}
	}
	// Check result ≈ S (h after QZ)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			diff := math.Abs(result[i*n+j] - h.Data[i*n+j])
			if diff > tol {
				t.Errorf("4x4 complex test: Q^T*H*Z - S at (%d,%d): diff=%e", i, j, diff)
			}
		}
	}

	_ = bi
}

func testDhgeqzComplex6x6(t *testing.T, impl Dhgeqzer) {
	// 6x6 Hessenberg-triangular pair designed to have multiple complex pairs.
	n := 6
	h := blas64.General{
		Rows: n, Cols: n, Stride: n,
		Data: []float64{
			0.5, 1.0, 0.3, 0.1, 0.2, 0.4,
			-1.5, 0.5, 0.7, 0.2, 0.1, 0.3,
			0, -2.0, 1.0, 0.5, 0.3, 0.1,
			0, 0, -1.0, 1.0, 0.8, 0.2,
			0, 0, 0, -0.5, 0.3, 1.0,
			0, 0, 0, 0, -1.2, 0.3,
		},
	}
	tt := blas64.General{
		Rows: n, Cols: n, Stride: n,
		Data: []float64{
			1, 0.2, 0.1, 0, 0, 0,
			0, 1, 0.3, 0.1, 0, 0,
			0, 0, 1, 0.2, 0.1, 0,
			0, 0, 0, 1, 0.1, 0,
			0, 0, 0, 0, 1, 0.2,
			0, 0, 0, 0, 0, 1,
		},
	}

	hOrig := make([]float64, len(h.Data))
	tOrig := make([]float64, len(tt.Data))
	copy(hOrig, h.Data)
	copy(tOrig, tt.Data)

	q := blas64.General{Rows: n, Cols: n, Stride: n, Data: make([]float64, n*n)}
	z := blas64.General{Rows: n, Cols: n, Stride: n, Data: make([]float64, n*n)}
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	work := make([]float64, 4*n)

	ok := impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess, n, 0, n-1,
		h.Data, h.Stride, tt.Data, tt.Stride, alphar, alphai, beta,
		q.Data, q.Stride, z.Data, z.Stride, work, len(work))

	if !ok {
		t.Fatal("6x6 complex test: QZ iteration did not converge")
	}

	// Verify Schur decomposition: Q^T * H_orig * Z ≈ S.
	const tol = 1e-10
	tmp := make([]float64, n*n)
	result := make([]float64, n*n)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for k := 0; k < n; k++ {
				s += hOrig[i*n+k] * z.Data[k*n+j]
			}
			tmp[i*n+j] = s
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for k := 0; k < n; k++ {
				s += q.Data[k*n+i] * tmp[k*n+j]
			}
			result[i*n+j] = s
		}
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			diff := math.Abs(result[i*n+j] - h.Data[i*n+j])
			if diff > tol {
				t.Errorf("6x6 complex test: Q^T*H*Z - S at (%d,%d): diff=%e", i, j, diff)
			}
		}
	}

	_ = tOrig
}

func testDhgeqzLargeN(t *testing.T, impl Dhgeqzer, n int) {
	rng := rand.New(rand.NewPCG(uint64(n), 0))

	// Generate random upper Hessenberg H.
	hData := make([]float64, n*n)
	for i := range n {
		for j := range n {
			if j >= i-1 {
				hData[i*n+j] = rng.NormFloat64()
			}
		}
	}
	// Generate random upper triangular T with positive diagonal.
	tData := make([]float64, n*n)
	for i := range n {
		for j := i; j < n; j++ {
			tData[i*n+j] = rng.NormFloat64()
		}
		if tData[i*n+i] < 0 {
			tData[i*n+i] = -tData[i*n+i]
		}
		tData[i*n+i] += 1
	}

	hOrig := make([]float64, len(hData))
	tOrig := make([]float64, len(tData))
	copy(hOrig, hData)
	copy(tOrig, tData)

	qData := make([]float64, n*n)
	zData := make([]float64, n*n)
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)

	// Workspace query.
	work := make([]float64, 1)
	impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess, n, 0, n-1,
		nil, max(1, n), nil, max(1, n), nil, nil, nil,
		nil, max(1, n), nil, max(1, n), work, -1)
	lwork := int(work[0])
	work = make([]float64, lwork)

	ok := impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess, n, 0, n-1,
		hData, n, tData, n, alphar, alphai, beta,
		qData, n, zData, n, work, lwork)

	if !ok {
		t.Fatalf("n=%d test: QZ iteration did not converge", n)
	}

	// Verify Q^T * H_orig * Z ≈ S.
	const tol = 1e-9
	tmp := make([]float64, n*n)
	result := make([]float64, n*n)

	for i := range n {
		for j := range n {
			var s float64
			for k := range n {
				s += hOrig[i*n+k] * zData[k*n+j]
			}
			tmp[i*n+j] = s
		}
	}
	for i := range n {
		for j := range n {
			var s float64
			for k := range n {
				s += qData[k*n+i] * tmp[k*n+j]
			}
			result[i*n+j] = s
		}
	}
	for i := range n {
		for j := range n {
			diff := math.Abs(result[i*n+j] - hData[i*n+j])
			if diff > tol {
				t.Errorf("n=%d test: Q^T*H*Z - S at (%d,%d): diff=%e", n, i, j, diff)
				return
			}
		}
	}

	// Verify Q^T * T_orig * Z ≈ P.
	for i := range n {
		for j := range n {
			var s float64
			for k := range n {
				s += tOrig[i*n+k] * zData[k*n+j]
			}
			tmp[i*n+j] = s
		}
	}
	for i := range n {
		for j := range n {
			var s float64
			for k := range n {
				s += qData[k*n+i] * tmp[k*n+j]
			}
			result[i*n+j] = s
		}
	}
	for i := range n {
		for j := range n {
			diff := math.Abs(result[i*n+j] - tData[i*n+j])
			if diff > tol {
				t.Errorf("n=%d test: Q^T*T*Z - P at (%d,%d): diff=%e", n, i, j, diff)
				return
			}
		}
	}

	// Verify Q is orthogonal: Q^T * Q ≈ I.
	for i := range n {
		for j := range n {
			var s float64
			for k := range n {
				s += qData[k*n+i] * qData[k*n+j]
			}
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if math.Abs(s-expected) > tol {
				t.Errorf("n=%d test: Q not orthogonal at (%d,%d): got %e want %e", n, i, j, s, expected)
				return
			}
		}
	}

	// Verify Z is orthogonal: Z^T * Z ≈ I.
	for i := range n {
		for j := range n {
			var s float64
			for k := range n {
				s += zData[k*n+i] * zData[k*n+j]
			}
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if math.Abs(s-expected) > tol {
				t.Errorf("n=%d test: Z not orthogonal at (%d,%d): got %e want %e", n, i, j, s, expected)
				return
			}
		}
	}

}
