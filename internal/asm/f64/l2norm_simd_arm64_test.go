// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64_test

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/internal/asm/f64"
)

func TestL2NormIncZeroIncrement(t *testing.T) {
	for _, n := range []uintptr{0, 1, 17} {
		if got := f64.L2NormInc(nil, n, 0); got != 0 {
			t.Errorf("n=%d: got %g, want 0", n, got)
		}
	}
}

func TestL2NormSIMDRepeatedValue(t *testing.T) {
	for _, n := range []uintptr{1, 3, 16, 33} {
		for _, v := range []float64{0, 0x1p-600, 1, 0x1p600, math.Inf(1), math.NaN()} {
			want := math.Abs(v) * math.Sqrt(float64(n))
			checkL2Norm(t, f64.L2NormIncSIMD([]float64{v}, n, 0), want)
		}
	}
}
