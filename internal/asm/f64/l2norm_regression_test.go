// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package f64_test

import (
	"fmt"
	"math"
	"math/big"
	"testing"

	"plantcontrol.org/v1/gonum/internal/asm/f64"
)

func TestL2NormMagnitudes(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 257} {
		for _, exp := range []int{-1074, -1022, -600, -486, -485, -484, 0, 500, 510, 511, 512, 1000} {
			t.Run(fmt.Sprintf("n=%d/exp=%d", n, exp), func(t *testing.T) {
				x, y := make([]float64, n), make([]float64, n)
				for i := range x {
					x[i] = math.Ldexp(float64(i%9-4), exp)
					y[i] = math.Ldexp(float64(i%7-3), exp)
				}
				want := l2NormReference(x)
				checkL2Norm(t, f64.L2NormUnitary(x), want)
				diff := make([]float64, n)
				for i := range diff {
					diff[i] = x[i] - y[i]
				}
				checkL2Norm(t, f64.L2DistanceUnitary(x, y), l2NormReference(diff))
				for _, inc := range []int{1, 2, 5, 67} {
					strided := make([]float64, n*inc+2)
					for i := range strided {
						strided[i] = math.NaN()
					}
					for i, v := range x {
						strided[1+i*inc] = v
					}
					checkL2Norm(t, f64.L2NormInc(strided[1:], uintptr(n), uintptr(inc)), want)
					for i, v := range strided {
						if i >= 1 && i <= 1+(n-1)*inc && (i-1)%inc == 0 {
							if v != x[(i-1)/inc] {
								t.Fatalf("increment %d changed input %d", inc, i)
							}
						} else if !math.IsNaN(v) {
							t.Fatalf("increment %d changed guard %d", inc, i)
						}
					}
				}
			})
		}
	}
}

func TestL2NormZeroLeading(t *testing.T) {
	for _, n := range []int{31, 32, 33, 257} {
		for _, exp := range []int{-600, 0, 600} {
			x, zero := make([]float64, n), make([]float64, n)
			for i := 1; i < n; i++ {
				x[i] = math.Ldexp(float64(i%9-4), exp)
			}
			want := l2NormReference(x)
			checkL2Norm(t, f64.L2NormUnitary(x), want)
			checkL2Norm(t, f64.L2DistanceUnitary(x, zero), want)
			checkL2Norm(t, f64.L2NormInc(x, uintptr(n), 1), want)
		}
	}
}

func TestL2NormNonFinite(t *testing.T) {
	for _, n := range []int{1, 3, 8, 17, 33} {
		for pos := 0; pos < n; pos++ {
			for _, special := range []float64{math.Inf(1), math.Inf(-1), math.NaN()} {
				x, zero := make([]float64, n), make([]float64, n)
				for i := range x {
					x[i] = float64(i + 1)
				}
				x[pos] = special
				for _, withNaN := range []bool{false, true} {
					want := math.Abs(special)
					if withNaN {
						x[(pos+1)%n] = math.NaN()
						want = math.NaN()
					}
					checkL2Norm(t, f64.L2NormUnitary(x), want)
					checkL2Norm(t, f64.L2DistanceUnitary(x, zero), want)
					for _, inc := range []int{1, 3} {
						strided := make([]float64, n*inc)
						for i, v := range x {
							strided[i*inc] = v
						}
						checkL2Norm(t, f64.L2NormInc(strided, uintptr(n), uintptr(inc)), want)
					}
				}
			}
		}
	}
}

func TestL2NormMixedMagnitudes(t *testing.T) {
	for _, dominant := range []float64{math.SmallestNonzeroFloat64, 0x1p-500, 1, 0x1p500, math.MaxFloat64 / 2, math.MaxFloat64} {
		for _, n := range []int{2, 3, 8, 17, 33} {
			x, zero := make([]float64, n), make([]float64, n)
			for i := range x {
				x[i] = math.Ldexp(dominant, -53-i)
			}
			x[0], x[n-1] = dominant, -dominant
			want := l2NormReference(x)
			checkL2Norm(t, f64.L2NormUnitary(x), want)
			checkL2Norm(t, f64.L2DistanceUnitary(x, zero), want)
			strided := make([]float64, 5*n)
			for i, v := range x {
				strided[5*i] = v
			}
			checkL2Norm(t, f64.L2NormInc(strided, uintptr(n), 5), want)
		}
	}
	checkL2Norm(t, f64.L2DistanceUnitary([]float64{math.MaxFloat64, 1}, []float64{-math.MaxFloat64, 0}), math.Inf(1))
}

func l2NormReference(x []float64) float64 {
	sum := new(big.Float).SetPrec(256)
	v := new(big.Float).SetPrec(256)
	for _, f := range x {
		v.SetFloat64(f)
		sum.Add(sum, v.Mul(v, v))
	}
	want, _ := sum.Sqrt(sum).Float64()
	return want
}

func checkL2Norm(t *testing.T, got, want float64) {
	t.Helper()
	if got == want || math.IsNaN(got) && math.IsNaN(want) {
		return
	}
	if math.IsNaN(got) || math.IsNaN(want) || math.IsInf(got, 0) || math.IsInf(want, 0) ||
		math.Abs(got-want) > 3e-14*math.Abs(want)+2*math.SmallestNonzeroFloat64 {
		t.Fatalf("norm: got %g, want %g", got, want)
	}
}
