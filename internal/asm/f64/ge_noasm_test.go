// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 || noasm || gccgo || safe

package f64

import (
	"math"
	"testing"
)

func TestGemvTZeroBetaTail(t *testing.T) {
	for _, m := range []int{0, 1, 8} {
		const n = 9
		a := make([]float64, m*n)
		x := make([]float64, m)
		for i := range a {
			a[i] = 2
		}
		for i := range x {
			x[i] = 3
		}
		y := make([]float64, n+3)
		for i := range y[:n] {
			y[i] = math.NaN()
		}
		copy(y[n:], []float64{7, 8, 9})
		GemvT(uintptr(m), n, 0.5, a, n, x, 1, 0, y, 1)
		for i, got := range y {
			want := float64(3 * m)
			if i >= n {
				want = float64(i - n + 7)
			}
			if got != want {
				t.Errorf("m=%d: y[%d]=%g, want %g", m, i, got, want)
			}
		}
	}
}
