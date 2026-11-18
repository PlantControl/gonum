// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"
	"testing"
)

// TestDlaln2TiedPivotScale checks a real 2×2 solve whose off-diagonal
// entries tie for the largest magnitude. DLALN2 scans C column-wise, which
// decides the pivot and so the scale factor. The expected values are
// reference LAPACK 3.12 output.
func TestDlaln2TiedPivotScale(t *testing.T) {
	a := []float64{0, -3 * 0x1p-30, 3 * 0x1p-30, 0}
	b := []float64{0, -0x1p1002}
	const wantScale = 1.8665272370064379e-302
	wantX := []float64{2.2906492245333332e+08, 1.1453246122666667e+08}

	x := make([]float64, 2)
	scale, _, ok := Implementation{}.Dlaln2(true, 2, 1, 0x1p-80, 1, a, 2, 1, 1, b, 1, 3*0x1p-31, 0, x, 1)
	if !ok {
		t.Errorf("ok=false, want true")
	}
	if math.Abs(scale-wantScale) > 1e-15*wantScale {
		t.Errorf("scale=%v, want %v", scale, wantScale)
	}
	for i := range x {
		if math.Abs(x[i]-wantX[i]) > 1e-15*math.Abs(wantX[0]) {
			t.Errorf("x[%d]=%v, want %v", i, x[i], wantX[i])
		}
	}
}
