// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"math"
	"testing"
)

func TestDlaln2SminiDistinctStrides(t *testing.T) {
	for _, trans := range []bool{false, true} {
		nan := math.NaN()
		a := make([]float64, 4)
		b := []float64{1, 2, nan, nan, 3, 4, nan, nan}
		x := []float64{nan, nan, -7, nan, nan, -7}
		const smin = 0.5
		scale, xnorm, ok := Implementation{}.Dlaln2(trans, 2, 2, smin, 1, a, 2, 1, 1, b, 4, 0, 0, x, 3)
		if ok {
			t.Errorf("trans=%t: ok=true, want perturbed (false)", trans)
		}
		if scale != 1 {
			t.Errorf("trans=%t: scale=%v, want 1", trans, scale)
		}
		want := []float64{2, 4, -7, 6, 8, -7}
		for i, w := range want {
			if x[i] != w {
				t.Errorf("trans=%t: x=%v, want %v", trans, x, want)
				break
			}
		}
		if xnorm != 14 {
			t.Errorf("trans=%t: xnorm=%v, want 14", trans, xnorm)
		}
	}
}
