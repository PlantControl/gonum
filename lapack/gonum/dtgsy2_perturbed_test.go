// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

// TestDtgsy2TransPerturbedPivotSign checks the transposed solve when Dgetc2
// replaces an exactly zero pivot by smin. DTGSY2 builds the system with the
// second equation negated, which fixes the sign of the perturbed solution.
// The expected values are reference LAPACK 3.12 output.
func TestDtgsy2TransPerturbedPivotSign(t *testing.T) {
	const (
		a = 1.0
		b = 1 - 0x1p-52
		d = 0.5
		e = 0.5 - 0x1p-53
	)
	const (
		wantR = -6755399441055743.0
		wantL = 13510798882111488.0
	)
	check := func(name string, scale, r, l float64, ok bool) {
		t.Helper()
		if ok {
			t.Errorf("%s: ok=true, want false for a perturbed system", name)
		}
		if scale != 1 || r != wantR || l != wantL {
			t.Errorf("%s: scale=%v R=%v L=%v, want scale=1 R=%v L=%v", name, scale, r, l, wantR, wantL)
		}
	}

	c, f := []float64{1}, []float64{2}
	scale, _, _, _, ok := Implementation{}.Dtgsy2(blas.Trans, 0, 1, 1,
		[]float64{a}, 1, []float64{b}, 1, c, 1, []float64{d}, 1, []float64{e}, 1, f, 1, 1, 0, make([]int, 4))
	check("Dtgsy2", scale, c[0], f[0], ok)

	c, f = []float64{1}, []float64{2}
	scale, _, ok = Implementation{}.Dtgsyl(blas.Trans, 0, 1, 1,
		[]float64{a}, 1, []float64{b}, 1, c, 1, []float64{d}, 1, []float64{e}, 1, f, 1, make([]float64, 1), 1, make([]int, 8))
	check("Dtgsyl", scale, c[0], f[0], ok)
}
