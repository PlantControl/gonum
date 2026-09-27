// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"testing"

	"plantcontrol.org/v1/gonum/blas/blas64"
)

type Dtgexcer interface {
	Dtgexc(wantq, wantz bool, n int, a []float64, lda int, b []float64, ldb int,
		q []float64, ldq int, z []float64, ldz int, ifst, ilst int, work []float64, lwork int) (ifstOut, ilstOut int, ok bool)
}

func DtgexcTest(t *testing.T, impl Dtgexcer) {
	// Test workspace query.
	work := make([]float64, 1)
	_, _, ok := impl.Dtgexc(false, false, 5, nil, 5, nil, 5, nil, 1, nil, 1, 0, 0, work, -1)
	if !ok {
		t.Error("Workspace query failed")
	}
	if work[0] < 1 {
		t.Errorf("Workspace query returned invalid size: %v", work[0])
	}
	if !dtgexcPanics(func() {
		impl.Dtgexc(false, false, 1, nil, 1, nil, 1, nil, 1, nil, 1, 0, 0, nil, -1)
	}) {
		t.Fatal("workspace query accepted a missing output slot")
	}
	for _, tc := range []struct {
		name       string
		ifst, ilst int
	}{
		{name: "IFST", ifst: -1, ilst: 0},
		{name: "ILST", ifst: 0, ilst: 2},
	} {
		t.Run("QueryValidates"+tc.name, func(t *testing.T) {
			if !dtgexcPanics(func() {
				impl.Dtgexc(false, false, 2, nil, 2, nil, 2, nil, 1, nil, 1,
					tc.ifst, tc.ilst, work, -1)
			}) {
				t.Fatal("workspace query accepted an invalid block position")
			}
		})
	}

	t.Run("SingleElementQuickReturn", func(t *testing.T) {
		work := make([]float64, 1)
		_, _, ok := impl.Dtgexc(false, false, 1, nil, 1, nil, 1, nil, 1, nil, 1, 0, 0, work, -1)
		if !ok {
			t.Fatal("workspace query failed")
		}
		if work[0] != 1 {
			t.Fatalf("workspace query returned %v, want 1", work[0])
		}
		ifst, ilst, ok := impl.Dtgexc(false, false, 1, nil, 1, nil, 1,
			nil, 1, nil, 1, 0, 0, work, 1)
		if !ok || ifst != 0 || ilst != 0 {
			t.Fatalf("quick return = (%d, %d, %v), want (0, 0, true)", ifst, ilst, ok)
		}
	})

	// Test forward move.
	testDtgexcForward(t, impl)

	// Test backward move.
	testDtgexcBackward(t, impl)
}

func dtgexcPanics(f func()) (panicked bool) {
	defer func() {
		panicked = recover() != nil
	}()
	f()
	return false
}

func testDtgexcForward(t *testing.T, impl Dtgexcer) {
	// 3x3 matrix pair in generalized Schur form.
	a := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0.5, 0.3,
			0, 2, 0.4,
			0, 0, 3,
		},
	}
	b := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0.2, 0.1,
			0, 1, 0.15,
			0, 0, 1,
		},
	}
	q := eye(3, 3)
	z := eye(3, 3)

	work := make([]float64, 100)

	// Move block from position 0 to position 2.
	ifstOut, ilstOut, ok := impl.Dtgexc(true, true, 3, a.Data, a.Stride, b.Data, b.Stride,
		q.Data, q.Stride, z.Data, z.Stride, 0, 2, work, 100)

	if !ok {
		t.Fatal("forward move failed")
	}
	if ifstOut != 0 || ilstOut != 2 {
		t.Fatalf("forward positions = (%d,%d), want (0,2)", ifstOut, ilstOut)
	}
	if work[0] != 4*3+16 {
		t.Fatalf("work[0]=%v, want %d", work[0], 4*3+16)
	}
}

func testDtgexcBackward(t *testing.T, impl Dtgexcer) {
	// 3x3 matrix pair in generalized Schur form.
	a := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0.5, 0.3,
			0, 2, 0.4,
			0, 0, 3,
		},
	}
	b := blas64.General{
		Rows: 3, Cols: 3, Stride: 3,
		Data: []float64{
			1, 0.2, 0.1,
			0, 1, 0.15,
			0, 0, 1,
		},
	}
	q := eye(3, 3)
	z := eye(3, 3)

	work := make([]float64, 100)

	// Move block from position 2 to position 0.
	ifstOut, ilstOut, ok := impl.Dtgexc(true, true, 3, a.Data, a.Stride, b.Data, b.Stride,
		q.Data, q.Stride, z.Data, z.Stride, 2, 0, work, 100)

	if !ok {
		t.Fatal("backward move failed")
	}
	if ifstOut != 2 || ilstOut != 0 {
		t.Fatalf("backward positions = (%d,%d), want (2,0)", ifstOut, ilstOut)
	}
}
