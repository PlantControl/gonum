// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f64_test

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/internal/asm/f64"
)

var l2NormResult float64

func BenchmarkL2NormZeroLeading(b *testing.B) {
	for _, n := range []int{32, 4096} {
		for _, values := range []string{"zero", "ordinary", "tiny", "large"} {
			b.Run(fmt.Sprintf("n=%d/values=%s", n, values), func(b *testing.B) {
				x := make([]float64, n)
				for i := 1; i < n; i++ {
					switch values {
					case "ordinary":
						x[i] = float64(i%9 - 4)
					case "tiny":
						x[i] = math.Ldexp(float64(i%9-4), -600)
					case "large":
						x[i] = math.Ldexp(float64(i%9-4), 600)
					}
				}
				b.ReportAllocs()
				b.ResetTimer()
				var norm float64
				for i := 0; i < b.N; i++ {
					norm = f64.L2NormUnitary(x)
				}
				l2NormResult = norm
			})
		}
	}
}

func BenchmarkL2NormRange(b *testing.B) {
	for _, n := range []int{0, 1, 3, 8, 16, 31, 32, 33, 256, 4096} {
		for _, scale := range []struct {
			name string
			exp  int
		}{{"ordinary", 0}, {"tiny", -600}, {"large", 600}} {
			for _, inc := range []int{1, 2, 17} {
				b.Run(fmt.Sprintf("n=%d/scale=%s/inc=%d", n, scale.name, inc), func(b *testing.B) {
					x := make([]float64, n*inc)
					for i := 0; i < n; i++ {
						x[i*inc] = math.Ldexp(float64(i%9-4), scale.exp)
					}
					b.ReportAllocs()
					b.ResetTimer()
					var norm float64
					if inc == 1 {
						for i := 0; i < b.N; i++ {
							norm = f64.L2NormUnitary(x)
						}
					} else {
						for i := 0; i < b.N; i++ {
							norm = f64.L2NormInc(x, uintptr(n), uintptr(inc))
						}
					}
					l2NormResult = norm
				})
			}
		}
	}
}

func BenchmarkL2DistanceRange(b *testing.B) {
	for _, n := range []int{0, 1, 3, 8, 16, 31, 32, 33, 256, 4096} {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			x, y := make([]float64, n), make([]float64, n)
			for i := range x {
				x[i] = float64(i%9-4) * 0.25
				y[i] = float64(i%7-3) * 0.5
			}
			b.ReportAllocs()
			b.ResetTimer()
			var norm float64
			for i := 0; i < b.N; i++ {
				norm = f64.L2DistanceUnitary(x, y)
			}
			l2NormResult = norm
		})
	}
}
