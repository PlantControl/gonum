// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"testing"

	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

type Dtgevcer interface {
	Dtgevc(side lapack.EVSide, howmny lapack.EVHowMany, selected []bool, n int,
		s []float64, lds int, p []float64, ldp int, vl []float64, ldvl int, vr []float64, ldvr int,
		mm int, work []float64) (m int, ok bool)
}

func DtgevcTest(t *testing.T, impl Dtgevcer) {
	// Test with 2x2 matrix pair in generalized Schur form.
	testDtgevc2x2(t, impl)

	// Test with 3x3 matrix pair.
	testDtgevc3x3(t, impl)
}

func testDtgevc2x2(t *testing.T, impl Dtgevcer) {
	// 2x2 quasi-triangular S and upper triangular P.
	s := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			2, 1,
			0, 3,
		},
	}
	p := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: []float64{
			1, 0.5,
			0, 1,
		},
	}
	vl := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: make([]float64, 4),
	}
	vr := blas64.General{
		Rows: 2, Cols: 2, Stride: 2,
		Data: make([]float64, 4),
	}

	work := make([]float64, 12)

	m, ok := impl.Dtgevc(lapack.EVBoth, lapack.EVAll, nil, 2,
		s.Data, s.Stride, p.Data, p.Stride, vl.Data, vl.Stride, vr.Data, vr.Stride,
		2, work)

	if !ok {
		t.Error("2x2 test: Dtgevc failed")
	}

	if m != 2 {
		t.Errorf("2x2 test: m = %d, want 2", m)
	}
}

func testDtgevc3x3(t *testing.T, impl Dtgevcer) {
	// 3x3 quasi-triangular S and upper triangular P.
	s := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			2, 1, 0.5,
			0, 3, 1,
			0, 0, 4,
		},
	}
	p := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0.3, 0.1,
			0, 1, 0.2,
			0, 0, 1,
		},
	}
	vl := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: make([]float64, 9),
	}
	vr := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: make([]float64, 9),
	}

	work := make([]float64, 18)

	m, ok := impl.Dtgevc(lapack.EVBoth, lapack.EVAll, nil, 3,
		s.Data, s.Stride, p.Data, p.Stride, vl.Data, vl.Stride, vr.Data, vr.Stride,
		3, work)

	if !ok {
		t.Error("3x3 test: Dtgevc failed")
	}

	if m != 3 {
		t.Errorf("3x3 test: m = %d, want 3", m)
	}

	// Test with selected eigenvectors.
	selected := []bool{true, false, true}
	vl = blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: make([]float64, 9),
	}
	vr = blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: make([]float64, 9),
	}

	m, ok = impl.Dtgevc(lapack.EVBoth, lapack.EVSelected, selected, 3,
		s.Data, s.Stride, p.Data, p.Stride, vl.Data, vl.Stride, vr.Data, vr.Stride,
		2, work)

	if !ok {
		t.Error("3x3 selected test: Dtgevc failed")
	}

	if m != 2 {
		t.Errorf("3x3 selected test: m = %d, want 2", m)
	}
}
