// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"
	"testing"
)

// TestDlasy2TiedPivotScale checks a 2×2 by 2×2 solve whose 4×4 system has
// tied pivot candidates, as standardized Schur blocks do. The unknown
// ordering decides the pivot and so the scale factor. The expected values
// are reference LAPACK 3.12 output.
func TestDlasy2TiedPivotScale(t *testing.T) {
	tl := []float64{-0x1p-29, 3 * 0x1p-30, 3 * 0x1p-30, -0x1p-29}
	tr := []float64{0x1p-29, 0x1p-30, -3 * 0x1p-30, 0x1p-29}
	b := []float64{-0x1p1002, 0, -3 * 0x1p1000, 0x1p1002}
	const wantScale = 2.1873366058669192e-303
	wantX := []float64{-1.8874368e+07, 0, -3.3554432e+07, -1.8874368e+07}

	x := make([]float64, 4)
	scale, xnorm, ok := Implementation{}.Dlasy2(false, true, 1, 2, 2, tl, 2, tr, 2, b, 2, x, 2)
	if !ok {
		t.Errorf("ok=false, want true")
	}
	if math.Abs(scale-wantScale) > 1e-15*wantScale {
		t.Errorf("scale=%v, want %v", scale, wantScale)
	}
	for i := range x {
		if math.Abs(x[i]-wantX[i]) > 1e-15*math.Abs(wantX[2]) {
			t.Errorf("x[%d]=%v, want %v", i, x[i], wantX[i])
		}
	}
	wantXnorm := math.Max(math.Abs(wantX[0])+math.Abs(wantX[1]), math.Abs(wantX[2])+math.Abs(wantX[3]))
	if math.Abs(xnorm-wantXnorm) > 1e-15*wantXnorm {
		t.Errorf("xnorm=%v, want %v", xnorm, wantXnorm)
	}
}
