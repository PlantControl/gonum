// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo

package c64

import (
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
			got := make([]complex64, 15)
			for i := range got {
				got[i] = complex(float32(i+1), float32(2*i-3))
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

func TestSIMDUnitaryToOverlap(t *testing.T) {
	tests := []struct {
		name             string
		dstStart, xStart int
		yStart           int
	}{
		{name: "destination is x", dstStart: 0, xStart: 0, yStart: 16},
		{name: "destination is y", dstStart: 16, xStart: 0, yStart: 16},
		{name: "destination ahead of x", dstStart: 1, xStart: 0, yStart: 16},
		{name: "destination behind y", dstStart: 16, xStart: 0, yStart: 17},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			const n = 12
			got := make([]complex64, 32)
			for i := range got {
				got[i] = complex(float32(i+1), float32(2*i-3))
			}
			want := slices.Clone(got)
			wantDst := want[test.dstStart : test.dstStart+n]
			wantX := want[test.xStart : test.xStart+n]
			wantY := want[test.yStart : test.yStart+n]
			for i, v := range wantX {
				wantDst[i] = (2+3i)*v + wantY[i]
			}
			AxpyUnitaryTo(got[test.dstStart:test.dstStart+n], 2+3i, got[test.xStart:test.xStart+n], got[test.yStart:test.yStart+n])
			if !slices.Equal(got, want) {
				t.Fatalf("got %v want %v", got, want)
			}
		})
	}
}

func TestSIMDDotBeyondCutoff(t *testing.T) {
	for _, n := range []int{16, 17, 31, 32, 33, 129} {
		x := make([]complex64, n)
		y := make([]complex64, n)
		var wantc, wantu complex64
		for i := range x {
			x[i] = complex(float32(i%7-3), float32(i%5-2))
			y[i] = complex(float32(i%11-5), float32(i%3-1))
			wantu += x[i] * y[i]
			wantc += conj(x[i]) * y[i]
		}
		if got := DotuUnitary(x, y); !cscalar.EqualWithinAbsOrRel(complex128(got), complex128(wantu), 1e-5, 1e-5) {
			t.Errorf("DotuUnitary n=%d: got %v want %v", n, got, wantu)
		}
		if got := DotcUnitary(x, y); !cscalar.EqualWithinAbsOrRel(complex128(got), complex128(wantc), 1e-5, 1e-5) {
			t.Errorf("DotcUnitary n=%d: got %v want %v", n, got, wantc)
		}
	}
}

func TestSIMDScaleUnitaryToOverlap(t *testing.T) {
	got := make([]complex64, 15)
	for i := range got {
		got[i] = complex(float32(i+1), float32(2*i-3))
	}
	want := slices.Clone(got)
	for i, v := range want[:12] {
		want[i+1] = (2 + 3i) * v
	}
	ScalUnitaryTo(got[1:13], 2+3i, got[:12])
	if !slices.Equal(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}
