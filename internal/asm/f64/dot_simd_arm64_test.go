// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && arm64 && !safe && !noasm && !gccgo
// +build go1.27,goexperiment.simd,arm64,!safe,!noasm,!gccgo

package f64_test

import (
	"fmt"
	"math"
	"slices"
	"testing"

	. "plantcontrol.org/v1/gonum/internal/asm/f64"
)

func TestDotUnitarySIMDEdgeLengths(t *testing.T) {
	lengths := make([]int, 81)
	for i := range lengths {
		lengths[i] = i
	}
	lengths = append(lengths, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025)
	for _, n := range lengths {
		for offset := 0; offset < 4; offset++ {
			t.Run(fmt.Sprintf("n=%d/offset=%d", n, offset), func(t *testing.T) {
				const guard = 4
				xStore := dotSIMDData(guard + offset + n + guard)
				yStore := dotSIMDData(guard + offset + n + 5 + guard)
				x := xStore[guard+offset : guard+offset+n]
				y := yStore[guard+offset : guard+offset+n+5]
				xOrig, yOrig := slices.Clone(xStore), slices.Clone(yStore)
				want := dotUnitaryARM64Reference(x, y)
				got := DotUnitary(x, y)
				dotSIMDCheck(t, got, want)
				if !dotSIMDEqualBits(xStore, xOrig) || !dotSIMDEqualBits(yStore, yOrig) {
					t.Fatal("read-only input or guard changed")
				}
			})
		}
	}
}

func TestDotUnitarySIMDOverlap(t *testing.T) {
	for _, n := range []int{0, 1, 15, 16, 17, 31, 32, 33, 64, 65} {
		for _, offsets := range [][2]int{{0, 0}, {0, 1}, {1, 0}, {1, 3}, {3, 1}} {
			t.Run(fmt.Sprintf("n=%d/x=%d/y=%d", n, offsets[0], offsets[1]), func(t *testing.T) {
				store := dotSIMDData(n + 7)
				orig := slices.Clone(store)
				x := store[offsets[0] : offsets[0]+n]
				y := store[offsets[1] : offsets[1]+n]
				want := dotUnitaryARM64Reference(slices.Clone(x), slices.Clone(y))
				dotSIMDCheck(t, DotUnitary(x, y), want)
				if !dotSIMDEqualBits(store, orig) {
					t.Fatal("overlapping read-only inputs changed")
				}
			})
		}
	}
}

func TestDotUnitarySIMDShortY(t *testing.T) {
	for _, n := range []int{16, 17, 31, 32, 33} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			x := dotSIMDData(n)
			yStore := dotSIMDData(n + 4)
			y := yStore[:n-1]
			xOrig, yOrig := slices.Clone(x), slices.Clone(yStore)
			defer func() {
				if recover() == nil {
					t.Fatal("did not panic with short y")
				}
				if !dotSIMDEqualBits(x, xOrig) || !dotSIMDEqualBits(yStore, yOrig) {
					t.Fatal("read-only input changed before panic")
				}
			}()
			DotUnitary(x, y)
		})
	}
}

func TestDotUnitarySIMDExceptional(t *testing.T) {
	for _, n := range []int{15, 16, 17, 23, 24, 25, 31, 32, 33, 63, 64, 65} {
		for _, pattern := range []string{
			"lane-overflow", "lane-cancellation", "tree-cancellation", "fma-rounding",
			"subnormal", "signed-zero", "infinity", "nan", "zero-times-infinity",
		} {
			t.Run(fmt.Sprintf("n=%d/%s", n, pattern), func(t *testing.T) {
				x := make([]float64, n)
				y := make([]float64, n)
				for i := range x {
					x[i] = 1
				}
				switch pattern {
				case "lane-overflow":
					y[0], y[8] = math.MaxFloat64, math.MaxFloat64
					if n > 16 {
						y[16] = -math.MaxFloat64
					}
				case "lane-cancellation":
					y[0], y[8] = math.MaxFloat64, -math.MaxFloat64
					if n > 16 {
						y[16] = math.MaxFloat64
					}
				case "tree-cancellation":
					y[0], y[2], y[4] = math.MaxFloat64, -math.MaxFloat64, math.MaxFloat64
				case "fma-rounding":
					x[0], y[0] = 1+0x1p-27, 1-0x1p-27
					x[8], y[8] = -1, 1
					x[2], y[2] = 0x1p-27, 0x1p-27
				case "subnormal":
					y[0], y[2], y[8] = 8*math.SmallestNonzeroFloat64, -4*math.SmallestNonzeroFloat64, math.SmallestNonzeroFloat64
					x[0], x[2], x[8] = 0.5, 0.5, 1
				case "signed-zero":
					for i := range y {
						y[i] = math.Copysign(0, -1)
					}
				case "infinity":
					y[0] = math.Inf(-1)
				case "nan":
					y[0] = math.NaN()
				case "zero-times-infinity":
					x[0], y[0] = 0, math.Inf(1)
				}
				xOrig, yOrig := slices.Clone(x), slices.Clone(y)
				want := dotUnitaryARM64Reference(x, y)
				dotSIMDCheck(t, DotUnitary(x, y), want)
				if !dotSIMDEqualBits(x, xOrig) || !dotSIMDEqualBits(y, yOrig) {
					t.Fatal("read-only input changed")
				}
			})
		}
	}
}

func dotUnitaryARM64Reference(x, y []float64) float64 {
	// Preserve this ARM64 kernel's lane accumulation and reduction order.
	if len(x) < 16 {
		sum := 0.0
		for i, v := range x {
			sum += v * y[i]
		}
		return sum
	}
	var sums [4][2]float64
	i := 0
	for ; i+7 < len(x); i += 8 {
		for vector := 0; vector < 4; vector++ {
			for lane := 0; lane < 2; lane++ {
				j := i + 2*vector + lane
				sums[vector][lane] = math.FMA(x[j], y[j], sums[vector][lane])
			}
		}
	}
	for lane := 0; lane < 2; lane++ {
		sums[0][lane] = (sums[0][lane] + sums[1][lane]) + (sums[2][lane] + sums[3][lane])
	}
	for ; i+1 < len(x); i += 2 {
		sums[0][0] = math.FMA(x[i], y[i], sums[0][0])
		sums[0][1] = math.FMA(x[i+1], y[i+1], sums[0][1])
	}
	sum := sums[0][0] + sums[0][1]
	for ; i < len(x); i++ {
		sum = math.FMA(x[i], y[i], sum)
	}
	return sum
}

func dotSIMDData(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i%19-9) / 16
	}
	return x
}

func dotSIMDCheck(t *testing.T, got, want float64) {
	t.Helper()
	if math.IsNaN(want) {
		if !math.IsNaN(got) {
			t.Fatalf("got %g want NaN", got)
		}
		return
	}
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("got %g (%#x) want %g (%#x)", got, math.Float64bits(got), want, math.Float64bits(want))
	}
}

func dotSIMDEqualBits(x, y []float64) bool {
	return slices.EqualFunc(x, y, func(a, b float64) bool {
		return math.Float64bits(a) == math.Float64bits(b)
	})
}
