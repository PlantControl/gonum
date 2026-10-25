// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/cblas128"
)

type Zgesver interface {
	Zgesv(n, nrhs int, a []complex128, lda int, ipiv []int, b []complex128, ldb int) bool
}

func ZgesvTest(t *testing.T, impl Zgesver) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for _, n := range []int{0, 1, 2, 3, 5, 16, 64} {
		for _, nrhs := range []int{0, 1, 2, 5} {
			for _, ldExtra := range []int{0, 5} {
				lda := max(1, n) + ldExtra
				ldb := max(1, nrhs) + ldExtra
				zgesvTest(t, impl, rnd, n, nrhs, lda, ldb)
			}
		}
	}

	for _, n := range []int{2, 3, 5, 16, 64} {
		for _, nrhs := range []int{1, 3} {
			zgesvPivotTest(t, impl, rnd, n, nrhs)
		}
	}

	// Singular matrix path: zero column.
	t.Run("singular", func(t *testing.T) {
		n := 4
		a := make([]complex128, n*n)
		for i := range a {
			a[i] = complex(rnd.NormFloat64(), rnd.NormFloat64())
		}
		for i := 0; i < n; i++ {
			a[i*n+1] = 0
		}
		b := make([]complex128, n)
		for i := range b {
			b[i] = complex(rnd.NormFloat64(), rnd.NormFloat64())
		}
		ipiv := make([]int, n)
		if ok := impl.Zgesv(n, 1, a, n, ipiv, b, 1); ok {
			t.Errorf("Zgesv returned ok=true on singular matrix")
		}
	})
}

func zgesvTest(t *testing.T, impl Zgesver, rnd *rand.Rand, n, nrhs, lda, ldb int) {
	const tol = 1e-10

	name := fmt.Sprintf("n=%v,nrhs=%v,lda=%v,ldb=%v", n, nrhs, lda, ldb)

	// Well-conditioned A = diag-dominant random complex matrix.
	a := make([]complex128, max(1, n*lda))
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			a[i*lda+j] = complex(0.1*rnd.NormFloat64(), 0.1*rnd.NormFloat64())
		}
		if n > 0 {
			a[i*lda+i] += complex(float64(n), 0)
		}
	}
	// Exact solution.
	xWant := make([]complex128, max(1, n*ldb))
	for i := 0; i < n; i++ {
		for j := 0; j < nrhs; j++ {
			xWant[i*ldb+j] = complex(rnd.NormFloat64(), rnd.NormFloat64())
		}
	}
	// Compute B = A*X.
	A := cblas128.General{Rows: n, Cols: n, Stride: lda, Data: a}
	X := cblas128.General{Rows: n, Cols: nrhs, Stride: ldb, Data: xWant}
	B := cblas128.General{Rows: n, Cols: nrhs, Stride: ldb, Data: make([]complex128, max(1, n*ldb))}
	if n > 0 && nrhs > 0 {
		cblas128.Gemm(blas.NoTrans, blas.NoTrans, 1, A, X, 0, B)
	}

	aWork := make([]complex128, len(a))
	copy(aWork, a)
	bWork := make([]complex128, len(B.Data))
	copy(bWork, B.Data)
	ipiv := make([]int, n)
	for i := range ipiv {
		ipiv[i] = -1
	}

	ok := impl.Zgesv(n, nrhs, aWork, lda, ipiv, bWork, ldb)
	if !ok {
		t.Errorf("%s: unexpected failure in Zgesv", name)
		return
	}
	if n == 0 || nrhs == 0 {
		return
	}

	// Ensure ipiv populated.
	for _, ip := range ipiv {
		if ip == -1 {
			t.Errorf("%s: ipiv not fully set", name)
			return
		}
	}

	// Compare computed solution to exact.
	for i := 0; i < n; i++ {
		for j := 0; j < nrhs; j++ {
			diff := bWork[i*ldb+j] - xWant[i*ldb+j]
			if cmplx.Abs(diff) > tol {
				t.Errorf("%s: solution mismatch at (%d,%d): got %v want %v residual=%v", name, i, j, bWork[i*ldb+j], xWant[i*ldb+j], cmplx.Abs(diff))
				return
			}
		}
	}

	// Residual |A*xGot - B|.
	xGot := cblas128.General{Rows: n, Cols: nrhs, Stride: ldb, Data: bWork}
	resid := cblas128.General{Rows: n, Cols: nrhs, Stride: ldb, Data: make([]complex128, len(B.Data))}
	copy(resid.Data, B.Data)
	cblas128.Gemm(blas.NoTrans, blas.NoTrans, -1, A, xGot, 1, resid)
	for i := 0; i < n; i++ {
		for j := 0; j < nrhs; j++ {
			if cmplx.Abs(resid.Data[i*ldb+j]) > tol {
				t.Errorf("%s: residual too large at (%d,%d): %v", name, i, j, cmplx.Abs(resid.Data[i*ldb+j]))
				return
			}
		}
	}
}

// zgesvPivotTest solves with a general matrix whose factorization needs row
// interchanges.
func zgesvPivotTest(t *testing.T, impl Zgesver, rnd *rand.Rand, n, nrhs int) {
	lda, ldb := n+2, nrhs+1
	name := fmt.Sprintf("pivot/n=%v,nrhs=%v", n, nrhs)
	a := zluPivotMatrix(n, lda, rnd)
	x := make([]complex128, n*ldb)
	for i := range x {
		x[i] = complex(rnd.NormFloat64(), rnd.NormFloat64())
	}
	A := cblas128.General{Rows: n, Cols: n, Stride: lda, Data: a}
	X := cblas128.General{Rows: n, Cols: nrhs, Stride: ldb, Data: x}
	B := cblas128.General{Rows: n, Cols: nrhs, Stride: ldb, Data: make([]complex128, n*ldb)}
	cblas128.Gemm(blas.NoTrans, blas.NoTrans, 1, A, X, 0, B)
	lu := append([]complex128(nil), a...)
	got := append([]complex128(nil), B.Data...)
	ipiv := make([]int, n)
	if !impl.Zgesv(n, nrhs, lu, lda, ipiv, got, ldb) {
		t.Fatalf("%s: unexpected failure", name)
	}
	if ipiv[0] == 0 {
		t.Fatalf("%s: no row interchange at the first step", name)
	}
	if r := zluBackwardError(blas.NoTrans, A, cblas128.General{Rows: n, Cols: nrhs, Stride: ldb, Data: got}, B); r > 50 {
		t.Errorf("%s: backward error %v·n·eps, want <= 50", name, r)
	}
}
