// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

type Dtgex2er interface {
	Dtgex2(wantq, wantz bool, n int, a []float64, lda int, b []float64, ldb int,
		q []float64, ldq int, z []float64, ldz int, j1, n1, n2 int, work []float64, lwork int) bool
}

func Dtgex2Test(t *testing.T, impl Dtgex2er) {
	var query [1]float64
	impl.Dtgex2(false, false, 7, nil, 7, nil, 7, nil, 1, nil, 1, 1, 2, 2, query[:], -1)
	if query[0] != 32 {
		t.Fatalf("workspace query returned %v, want 32", query[0])
	}
	// Test 1×1 - 1×1 swap.
	testDtgex2_1x1(t, impl)

	// Test 1×1 - 2×2 swap.
	testDtgex2_1x2(t, impl)

	// Test 2×2 - 1×1 swap.
	testDtgex2_2x1(t, impl)

	// Test 2×2 - 2×2 swap.
	testDtgex2_2x2(t, impl)
	testDtgex2Embedded(t, impl)
}

func testDtgex2Embedded(t *testing.T, impl Dtgex2er) {
	const n = 5
	a := blas64.General{Rows: n, Cols: n, Stride: n, Data: []float64{
		9, 0.2, 0.3, 0.4, 0.5,
		0, 1, 0.6, 0.7, 0.8,
		0, 0, 2, 1, 0.9,
		0, 0, -1, 2, 1.1,
		0, 0, 0, 0, 7,
	}}
	b := blas64.General{Rows: n, Cols: n, Stride: n, Data: []float64{
		1, 0.1, 0.2, 0.3, 0.4,
		0, 1, 0.2, 0.3, 0.4,
		0, 0, 1, 0, 0.5,
		0, 0, 0, 1, 0.6,
		0, 0, 0, 0, 1,
	}}
	q := eye(n, n)
	z := eye(n, n)
	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)
	work := make([]float64, 2*n*n)
	if !impl.Dtgex2(true, true, n, a.Data, n, b.Data, n, q.Data, n, z.Data, n, 1, 1, 2, work, len(work)) {
		t.Fatal("embedded 1x1-2x2 swap failed")
	}
	checkDtgex2Decomposition(t, "embedded 1x1-2x2", aOrig, bOrig, a, b, q, z)
}

func testDtgex2_1x1(t *testing.T, impl Dtgex2er) {
	// 2×2 matrices with 1×1 blocks.
	a := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 2,
			0, 3,
		},
	}
	b := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 1,
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

	work := make([]float64, 10)
	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)

	ok := impl.Dtgex2(true, true, 2, a.Data, a.Stride, b.Data, b.Stride,
		q.Data, q.Stride, z.Data, z.Stride, 0, 1, 1, work, 10)

	if !ok {
		t.Error("1x1-1x1 swap failed")
		return
	}

	// Check that A is still quasi-upper-triangular.
	if math.Abs(a.Data[2]) > 1e-10 {
		t.Errorf("1x1-1x1 swap: A[1,0] = %v, want 0", a.Data[2])
	}

	// Check that B is still upper triangular.
	if math.Abs(b.Data[2]) > 1e-10 {
		t.Errorf("1x1-1x1 swap: B[1,0] = %v, want 0", b.Data[2])
	}
	checkDtgex2Decomposition(t, "1x1-1x1", aOrig, bOrig, a, b, q, z)
}

func testDtgex2_1x2(t *testing.T, impl Dtgex2er) {
	// 3×3 matrices: 1×1 block at (0,0), 2×2 block at (1,1).
	a := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0.5, 0.3,
			0, 2, 1,
			0, -1, 2,
		},
	}
	b := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0.2, 0.1,
			0, 1, 0.3,
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

	work := make([]float64, 20)
	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)

	ok := impl.Dtgex2(true, true, 3, a.Data, a.Stride, b.Data, b.Stride,
		q.Data, q.Stride, z.Data, z.Stride, 0, 1, 2, work, 20)

	if !ok {
		t.Fatal("1x1-2x2 swap failed")
	}
	checkDtgex2Decomposition(t, "1x1-2x2", aOrig, bOrig, a, b, q, z)
}

func testDtgex2_2x1(t *testing.T, impl Dtgex2er) {
	// 3×3 matrices: 2×2 block at (0,0), 1×1 block at (2,2).
	a := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			2, 1, 0.3,
			-1, 2, 0.5,
			0, 0, 3,
		},
	}
	b := blas64.General{
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

	work := make([]float64, 20)
	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)

	ok := impl.Dtgex2(true, true, 3, a.Data, a.Stride, b.Data, b.Stride,
		q.Data, q.Stride, z.Data, z.Stride, 0, 2, 1, work, 20)

	if !ok {
		t.Fatal("2x2-1x1 swap failed")
	}
	checkDtgex2Decomposition(t, "2x2-1x1", aOrig, bOrig, a, b, q, z)
}

func testDtgex2_2x2(t *testing.T, impl Dtgex2er) {
	// 4×4 matrices: 2×2 block at (0,0), 2×2 block at (2,2).
	a := blas64.General{
		Rows: 4, Cols: 4, Stride: 4,
		Data: []float64{
			2, 1, 0.1, 0.2,
			-1, 2, 0.2, 0.1,
			0, 0, 3, 1,
			0, 0, -1, 3,
		},
	}
	b := blas64.General{
		Rows: 4, Cols: 4, Stride: 4,
		Data: []float64{
			1, 0.1, 0.05, 0.02,
			0, 1, 0.1, 0.05,
			0, 0, 1, 0.1,
			0, 0, 0, 1,
		},
	}
	q := blas64.General{
		Rows: 4, Cols: 4, Stride: 4,
		Data: []float64{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		},
	}
	z := blas64.General{
		Rows: 4, Cols: 4, Stride: 4,
		Data: []float64{
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		},
	}

	work := make([]float64, 40)
	aOrig := cloneGeneral(a)
	bOrig := cloneGeneral(b)

	ok := impl.Dtgex2(true, true, 4, a.Data, a.Stride, b.Data, b.Stride,
		q.Data, q.Stride, z.Data, z.Stride, 0, 2, 2, work, 40)

	if !ok {
		t.Fatal("2x2-2x2 swap failed")
	}
	checkDtgex2Decomposition(t, "2x2-2x2", aOrig, bOrig, a, b, q, z)
}

func checkDtgex2Decomposition(t *testing.T, name string, aOrig, bOrig, a, b, q, z blas64.General) {
	t.Helper()
	const tol = 1e-10
	for label, tc := range map[string]struct {
		orig, got blas64.General
	}{"A": {aOrig, a}, "B": {bOrig, b}} {
		resid := residualGeneralizedSchur(tc.orig, tc.got, q, z)
		norm := dlange(lapack.MaxColumnSum, tc.orig.Rows, tc.orig.Cols, tc.orig.Data, tc.orig.Stride)
		if resid > tol*math.Max(1, norm) {
			t.Errorf("%s: %s decomposition residual=%g", name, label, resid)
		}
	}
}
