// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package f64_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"testing"

	"plantcontrol.org/v1/gonum/internal/asm/f64"
)

func TestL2NormCompensatedAccuracy(t *testing.T) {
	rng := rand.New(rand.NewPCG(19, 37))
	for _, n := range []int{8, 9, 16, 17, 31, 32, 33, 64, 65, 257, 1024, 4096} {
		for _, exp := range []int{-450, 0, 450} {
			for _, distribution := range []string{"dense", "dominant", "rising", "product"} {
				t.Run(fmt.Sprintf("n=%d/exp=%d/%s", n, exp, distribution), func(t *testing.T) {
					x, y, strided, diff := make([]float64, n), make([]float64, n), make([]float64, 3*n), make([]float64, n)
					for i := range x {
						var v float64
						switch distribution {
						case "dense":
							v = 2*rng.Float64() - 1
						case "dominant":
							v = 0x1p-27
							if i == 0 {
								v = 1
							}
						case "rising":
							v = math.Ldexp(1, i*13%51-25)
						case "product":
							v = 1 + float64(i%63)*0x1p-27
						}
						x[i] = math.Ldexp(v, exp)
						strided[3*i] = x[i]
					}
					for i := range x {
						y[i] = -0.125 * x[(i+1)%n]
						diff[i] = x[i] - y[i]
					}
					want, wantDistance := l2NormReference(x), l2NormReference(diff)
					checkL2NormULP(t, f64.L2NormUnitarySIMD(x), want)
					checkL2NormULP(t, f64.L2NormIncSIMD(strided, uintptr(n), 3), want)
					checkL2NormULP(t, f64.L2DistanceUnitarySIMD(x, y), wantDistance)
					if runtime.GOARCH == "arm64" && n >= 32 {
						checkL2NormULP(t, f64.L2NormUnitary(x), want)
						checkL2NormULP(t, f64.L2DistanceUnitary(x, y), wantDistance)
						checkL2NormULP(t, f64.L2NormInc(strided, uintptr(n), 3), want)
					}
				})
			}
		}
	}
}

func checkL2NormULP(t *testing.T, got, want float64) {
	t.Helper()
	gb, wb := math.Float64bits(got), math.Float64bits(want)
	if gb < wb {
		gb, wb = wb, gb
	}
	if math.IsNaN(got) || math.IsInf(got, 0) || gb-wb > 1 {
		t.Fatalf("norm: got %.17g, want %.17g; error %d ULP", got, want, gb-wb)
	}
}
