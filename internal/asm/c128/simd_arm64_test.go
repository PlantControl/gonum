// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package c128

import (
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/cmplxs/cscalar"
)

func TestSIMDUnitaryOverlap(t *testing.T) {
	tests := []struct {
		name   string
		xStart int
		yStart int
	}{
		{name: "same", xStart: 1, yStart: 1},
		{name: "destination ahead", xStart: 0, yStart: 1},
		{name: "destination behind", xStart: 1, yStart: 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := make([]complex128, 15)
			for i := range got {
				got[i] = complex(float64(i+1), float64(2*i-3))
			}
			want := slices.Clone(got)
			for i, v := range want[test.xStart : test.xStart+12] {
				want[test.yStart+i] += (2 + 3i) * v
			}
			AxpyUnitary(2+3i, got[test.xStart:test.xStart+12], got[test.yStart:test.yStart+12])
			if !slices.Equal(got, want) {
				t.Fatalf("got %v want %v", got, want)
			}
		})
	}
}

func TestSIMDDotBeyondCutoff(t *testing.T) {
	for _, n := range []int{128, 129, 257} {
		x := make([]complex128, n)
		y := make([]complex128, n)
		var wantc, wantu complex128
		for i := range x {
			x[i] = complex(float64(i%7-3), float64(i%5-2))
			y[i] = complex(float64(i%11-5), float64(i%3-1))
			wantu += x[i] * y[i]
			wantc += cmplx.Conj(x[i]) * y[i]
		}
		if got := DotuUnitary(x, y); !cscalar.EqualWithinAbsOrRel(got, wantu, 1e-13, 1e-13) {
			t.Errorf("DotuUnitary n=%d: got %v want %v", n, got, wantu)
		}
		if got := DotcUnitary(x, y); !cscalar.EqualWithinAbsOrRel(got, wantc, 1e-13, 1e-13) {
			t.Errorf("DotcUnitary n=%d: got %v want %v", n, got, wantc)
		}
	}
}
