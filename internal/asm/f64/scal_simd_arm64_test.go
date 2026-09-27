// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package f64_test

import (
	"testing"

	. "plantcontrol.org/v1/gonum/internal/asm/f64"
)

func TestScalUnitarySIMDLengths(t *testing.T) {
	const alpha = -2.5
	for n := 0; n <= 65; n++ {
		x := make([]float64, n)
		for i := range x {
			x[i] = float64(i) - 7.25
		}

		ScalUnitary(alpha, x)

		for i, got := range x {
			want := alpha * (float64(i) - 7.25)
			if got != want {
				t.Fatalf("n=%d: unexpected value at %d: got:%v want:%v", n, i, got, want)
			}
		}
	}
}

func TestScalUnitaryToSIMDInPlace(t *testing.T) {
	const alpha = 1.75
	for n := 0; n <= 65; n++ {
		x := make([]float64, n)
		for i := range x {
			x[i] = float64(i) - 3.5
		}

		ScalUnitaryTo(x, alpha, x)

		for i, got := range x {
			want := alpha * (float64(i) - 3.5)
			if got != want {
				t.Fatalf("n=%d: unexpected value at %d: got:%v want:%v", n, i, got, want)
			}
		}
	}
}
