// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f32_test

import (
	"fmt"
	"math"
	"runtime"
	"slices"
	"testing"

	. "plantcontrol.org/v1/gonum/internal/asm/f32"
)

var axpyShortLengths = []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 64}
var axpyShortSink []float32

func TestAxpyUnitaryShort(t *testing.T) {
	for _, n := range axpyShortLengths {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			x := make([]float32, n+4)
			y := make([]float32, n+4)
			for i := range x {
				x[i] = float32(i%7+1) / 8
				y[i] = float32(i%5-2) / 4
			}
			x0, y0 := slices.Clone(x), slices.Clone(y)
			AxpyUnitary(-0.25, x[2:2+n], y[2:2+n])
			for i := 0; i < n; i++ {
				want := y0[i+2] - 0.25*x0[i+2]
				if y[i+2] != want {
					t.Fatalf("index %d: got %g want %g", i, y[i+2], want)
				}
			}
			if !slices.Equal(x, x0) || y[0] != y0[0] || y[1] != y0[1] || y[n+2] != y0[n+2] || y[n+3] != y0[n+3] {
				t.Fatal("input or guard changed")
			}
		})
	}
}

func TestAxpyUnitaryShortOverlap(t *testing.T) {
	for _, tc := range []struct {
		name       string
		xoff, yoff int
	}{
		{name: "alias", xoff: 1, yoff: 1},
		{name: "forward", xoff: 1, yoff: 2},
		{name: "reverse", xoff: 2, yoff: 1},
	} {
		for _, n := range []int{1, 2, 3, 7, 8, 15, 16, 17} {
			t.Run(fmt.Sprintf("%s/n=%d", tc.name, n), func(t *testing.T) {
				if runtime.GOARCH == "amd64" && tc.xoff != tc.yoff {
					t.Skip("amd64 assembly does not guarantee partial-overlap semantics")
				}
				data := make([]float32, n+3)
				for i := range data {
					data[i] = float32(i+1) / 8
				}
				want := slices.Clone(data)
				for i, v := range want[tc.xoff : tc.xoff+n] {
					want[tc.yoff+i] += -0.25 * v
				}
				AxpyUnitary(-0.25, data[tc.xoff:tc.xoff+n], data[tc.yoff:tc.yoff+n])
				if !slices.Equal(data, want) {
					t.Fatalf("got %v want %v", data, want)
				}
			})
		}
	}
}

func TestAxpyUnitaryShortDestination(t *testing.T) {
	if runtime.GOARCH == "amd64" {
		t.Skip("amd64 assembly does not guarantee bounds checks")
	}
	for _, tc := range []struct {
		name string
		x, y []float32
		want []float32
	}{
		{name: "n1/y0", x: []float32{2}},
		{name: "n2/y1", x: []float32{2, 3}, y: []float32{5}, want: []float32{7}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("short destination did not panic")
				}
				if !slices.Equal(tc.y, tc.want) {
					t.Fatalf("destination after panic: got %v want %v", tc.y, tc.want)
				}
			}()
			AxpyUnitary(1, tc.x, tc.y)
		})
	}
}

func TestAxpyUnitaryShortNonfinite(t *testing.T) {
	negativeZero := float32(math.Copysign(0, -1))
	tests := []struct {
		name, want string
		alpha, x   float32
		y          float32
	}{
		{name: "nan", want: "nan", alpha: 1, x: float32(math.NaN()), y: 1},
		{name: "infinity", want: "+inf", alpha: 1, x: float32(math.Inf(1)), y: 1},
		{name: "zero-times-infinity", want: "nan", alpha: 0, x: float32(math.Inf(1)), y: 1},
		{name: "negative-zero", want: "-zero", alpha: negativeZero, x: 1, y: negativeZero},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			y := []float32{test.y}
			AxpyUnitary(test.alpha, []float32{test.x}, y)
			switch test.want {
			case "nan":
				if !math.IsNaN(float64(y[0])) {
					t.Fatalf("got %g want NaN", y[0])
				}
			case "+inf":
				if !math.IsInf(float64(y[0]), 1) {
					t.Fatalf("got %g want +Inf", y[0])
				}
			case "-zero":
				if math.Float32bits(y[0]) != math.Float32bits(negativeZero) {
					t.Fatalf("got %g want negative zero", y[0])
				}
			}
		})
	}
}

func BenchmarkAxpyUnitaryShort(b *testing.B) {
	for _, n := range []int{0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 64, 256, 4096} {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			x, y := make([]float32, n), make([]float32, n)
			for i := range x {
				x[i] = float32(i%7+1) / 8
				y[i] = float32(i%5+1) / 8
			}
			alpha := float32(0.25)
			b.SetBytes(int64(n * 4))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				AxpyUnitary(alpha, x, y)
				alpha = -alpha
			}
			b.StopTimer()
			axpyShortSink = y
		})
	}
}
