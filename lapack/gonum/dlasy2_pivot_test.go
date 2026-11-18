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

// TestDlasy2TiedPivotScale1x2 checks the 1×2 and 2×1 solves when the
// off-diagonal entries of the 2×2 system tie for the largest magnitude.
// DLASY2 scans the system column-wise. The expected values are reference
// LAPACK 3.12 output.
func TestDlasy2TiedPivotScale1x2(t *testing.T) {
	for _, tc := range []struct {
		name      string
		n1, n2    int
		tl, tr, b []float64
		ldb       int
		wantScale float64
		wantX     []float64
	}{
		{
			name: "1x2", n1: 1, n2: 2,
			tl:        []float64{-0x1p-30},
			tr:        []float64{0, -4 * 0x1p-30, 4 * 0x1p-30, 0},
			b:         []float64{0x1p1000, 0x1p1000},
			ldb:       2,
			wantScale: 3.7330544740128757e-302,
			wantX:     []float64{7.579354051764706e+07, -1.2632256752941176e+08},
		},
		{
			name: "2x1", n1: 2, n2: 1,
			tl:        []float64{0x1p-30, -0x1p-29, -0x1p-29, 0x1p-30},
			tr:        []float64{0},
			b:         []float64{2 * 0x1p1000, 3 * 0x1p1000},
			ldb:       1,
			wantScale: 1.333233740718884e-302,
			wantX:     []float64{-4.0904450438095236e+08, -3.579139413333333e+08},
		},
	} {
		x := make([]float64, 2)
		scale, _, ok := Implementation{}.Dlasy2(true, false, -1, tc.n1, tc.n2, tc.tl, tc.n1, tc.tr, tc.n2, tc.b, tc.ldb, x, tc.ldb)
		if !ok {
			t.Errorf("%s: ok=false, want true", tc.name)
		}
		wantScale, wantX := tc.wantScale, tc.wantX
		if math.Abs(scale-wantScale) > 1e-15*wantScale {
			t.Errorf("%s: scale=%v, want %v", tc.name, scale, wantScale)
		}
		for i := range x {
			if math.Abs(x[i]-wantX[i]) > 1e-15*math.Abs(wantX[1]) {
				t.Errorf("%s: x[%d]=%v, want %v", tc.name, i, x[i], wantX[i])
			}
		}
	}
}
