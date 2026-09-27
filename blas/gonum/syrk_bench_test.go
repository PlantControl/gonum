// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

type syrkBenchFloat interface{ float32 | float64 }

type syrkBenchFunc[T syrkBenchFloat] func(blas.Uplo, blas.Transpose, int, int, T, []T, int, T, []T, int)

func BenchmarkDsyrkSIMDShapes(b *testing.B) {
	benchmarkSyrkSIMDShapes(b, Implementation{}.Dsyrk)
}

func BenchmarkSsyrkSIMDShapes(b *testing.B) {
	benchmarkSyrkSIMDShapes(b, Implementation{}.Ssyrk)
}

func BenchmarkDsyrkCholeskyPanel(b *testing.B) {
	const (
		order = 256
		n     = 64
	)
	for _, k := range []int{64, 128, 192} {
		for _, shared := range []bool{true, false} {
			storage := "independent"
			if shared {
				storage = "shared"
			}
			b.Run(fmt.Sprintf("k=%d/%s", k, storage), func(b *testing.B) {
				backingOrig := make([]float64, order*order)
				for i := range backingOrig {
					backingOrig[i] = float64((i*17+11)%31-15) / 64
				}
				var backing, a, c []float64
				if shared {
					backing = append([]float64(nil), backingOrig...)
					a = backing[k:]
					c = backing[k*order+k:]
				} else {
					a = make([]float64, (k-1)*order+n)
					for i := 0; i < k; i++ {
						copy(a[i*order:i*order+n], backingOrig[i*order+k:i*order+k+n])
					}
					c = make([]float64, (n-1)*order+n)
					for i := 0; i < n; i++ {
						copy(c[i*order:i*order+n], backingOrig[(k+i)*order+k:(k+i)*order+k+n])
					}
				}
				aOrig := make([]float64, k*n)
				for i := 0; i < k; i++ {
					copy(aOrig[i*n:(i+1)*n], a[i*order:i*order+n])
				}
				cOrig := append([]float64(nil), c...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if i%16 == 0 {
						b.StopTimer()
						if shared {
							copy(backing, backingOrig)
						} else {
							copy(c, cOrig)
						}
						b.StartTimer()
					}
					Implementation{}.Dsyrk(blas.Upper, blas.Trans, n, k, -1, a, order, 1, c, order)
				}
				b.StopTimer()
				for i := 0; i < k; i++ {
					for j, want := range aOrig[i*n : (i+1)*n] {
						if a[i*order+j] != want {
							b.Fatalf("A[%d,%d] changed", i, j)
						}
					}
				}
				for i := 0; i < n; i++ {
					for _, v := range c[i*order+i : i*order+n] {
						if math.IsNaN(v) || math.IsInf(v, 0) {
							b.Fatal("nonfinite SYRK output")
						}
					}
				}
			})
		}
	}
}

func benchmarkSyrkSIMDShapes[T syrkBenchFloat](b *testing.B, syrk syrkBenchFunc[T]) {
	for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower} {
		for _, trans := range []blas.Transpose{blas.Trans, blas.NoTrans} {
			for _, n := range []int{4, 15, 16, 17, 64} {
				for _, k := range []int{16, 64, 256} {
					for _, pad := range []int{0, 3} {
						name := fmt.Sprintf("%c/%c/n=%d/k=%d/pad=%d", uplo, trans, n, k, pad)
						b.Run(name, func(b *testing.B) {
							rows, cols := k, n
							if trans == blas.NoTrans {
								rows, cols = n, k
							}
							lda, ldc := cols+pad, n+pad
							a := make([]T, rows*lda)
							for i := 0; i < rows; i++ {
								for j := 0; j < cols; j++ {
									a[i*lda+j] = T((i*17+j*11+3)%29-14) / 32
								}
							}
							c := make([]T, n*ldc)
							for i := range c {
								c[i] = T((i*7+5)%19-9) / 16
							}
							b.ReportAllocs()
							b.ResetTimer()
							for b.Loop() {
								syrk(uplo, trans, n, k, 0.75, a, lda, 0, c, ldc)
							}
							b.StopTimer()
							for i := 0; i < n; i++ {
								lo, hi := 0, i+1
								if uplo == blas.Upper {
									lo, hi = i, n
								}
								for _, v := range c[i*ldc+lo : i*ldc+hi] {
									if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
										b.Fatal("nonfinite SYRK output")
									}
								}
							}
						})
					}
				}
			}
		}
	}
}
