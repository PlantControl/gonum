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

type zeroGemmFloat interface{ float32 | float64 }
type zeroGemmFunc[T zeroGemmFloat] func(blas.Transpose, blas.Transpose, int, int, int, T, []T, int, []T, int, T, []T, int)

func TestDgemmZeroAlpha(t *testing.T) { testGemmZeroAlpha[float64](t, Implementation{}.Dgemm) }
func TestSgemmZeroAlpha(t *testing.T) { testGemmZeroAlpha[float32](t, Implementation{}.Sgemm) }

func testGemmZeroAlpha[T zeroGemmFloat](t *testing.T, gemm zeroGemmFunc[T]) {
	t.Helper()
	const m, n = 3, 5
	for _, ta := range []blas.Transpose{blas.NoTrans, blas.Trans, blas.ConjTrans} {
		for _, tb := range []blas.Transpose{blas.NoTrans, blas.Trans, blas.ConjTrans} {
			for _, k := range []int{0, 1, 17} {
				for _, alpha := range []T{0, T(math.Copysign(0, -1))} {
					for _, beta := range []T{0, 1, -0.5, T(math.NaN()), T(math.Inf(1))} {
						t.Run(fmt.Sprintf("%c%c/k=%d/alpha=%g/beta=%g", ta, tb, k, alpha, beta), func(t *testing.T) {
							ar, ac, br, bc := m, k, k, n
							if ta != blas.NoTrans {
								ar, ac = k, m
							}
							if tb != blas.NoTrans {
								br, bc = n, k
							}
							lda, ldb, ldc := ac+2, bc+2, n+2
							a := make([]T, max(0, (ar-1)*lda+ac))
							b := make([]T, max(0, (br-1)*ldb+bc))
							for i := range a {
								a[i] = T([]float64{math.NaN(), math.Inf(1), math.Inf(-1)}[i%3])
							}
							for i := range b {
								b[i] = T([]float64{math.Inf(-1), math.NaN(), math.Inf(1)}[i%3])
							}
							origA, origB := append([]T(nil), a...), append([]T(nil), b...)
							c := make([]T, m*ldc)
							for i := range c {
								c[i] = T([]float64{1, -2, math.NaN(), math.Inf(1), math.Copysign(0, -1), 73, 91}[i%ldc])
							}
							want := append([]T(nil), c...)
							for i := 0; i < m; i++ {
								for j := 0; j < n; j++ {
									if beta == 0 {
										want[i*ldc+j] = 0
									} else if beta != 1 {
										want[i*ldc+j] *= beta
									}
								}
							}
							gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
							checkZeroGemmValues(t, "C", c, want)
							checkZeroGemmValues(t, "A", a, origA)
							checkZeroGemmValues(t, "B", b, origB)
						})
					}
				}
			}
		}
	}
}

func checkZeroGemmValues[T zeroGemmFloat](t *testing.T, name string, got, want []T) {
	t.Helper()
	for i, g := range got {
		w := want[i]
		if math.IsNaN(float64(g)) && math.IsNaN(float64(w)) {
			continue
		}
		if math.Float64bits(float64(g)) != math.Float64bits(float64(w)) {
			t.Fatalf("%s[%d]=%g, want %g", name, i, g, w)
		}
	}
}

func TestGemmZeroAlphaValidation(t *testing.T) {
	testGemmZeroAlphaValidation[float64](t, Implementation{}.Dgemm)
	testGemmZeroAlphaValidation[float32](t, Implementation{}.Sgemm)
}
func testGemmZeroAlphaValidation[T zeroGemmFloat](t *testing.T, gemm zeroGemmFunc[T]) {
	t.Helper()
	for _, tc := range []struct {
		a, b, c int
		want    string
	}{{0, 0, 0, shortA}, {4, 0, 0, shortB}, {4, 4, 0, shortC}} {
		func() {
			defer func() {
				if got := recover(); got != tc.want {
					t.Errorf("panic=%v, want %q", got, tc.want)
				}
			}()
			gemm(blas.NoTrans, blas.NoTrans, 2, 2, 2, 0, make([]T, tc.a), 2, make([]T, tc.b), 2, 0, make([]T, tc.c), 2)
		}()
	}
	gemm(blas.NoTrans, blas.NoTrans, 0, 2, 2, 0, nil, 2, nil, 2, 0, nil, 2)
	gemm(blas.NoTrans, blas.NoTrans, 2, 0, 2, 0, nil, 2, nil, 1, 0, nil, 1)
}

func BenchmarkDgemmZeroAlphaControl(b *testing.B) {
	benchmarkGemmZeroAlpha[float64](b, Implementation{}.Dgemm)
}
func BenchmarkSgemmZeroAlphaControl(b *testing.B) {
	benchmarkGemmZeroAlpha[float32](b, Implementation{}.Sgemm)
}
func benchmarkGemmZeroAlpha[T zeroGemmFloat](b *testing.B, gemm zeroGemmFunc[T]) {
	for _, size := range []int{4, 32, 128} {
		for _, alpha := range []T{0, 1} {
			for _, tb := range []blas.Transpose{blas.NoTrans, blas.Trans} {
				b.Run(fmt.Sprintf("n=%d/alpha=%g/B=%c", size, alpha, tb), func(b *testing.B) {
					a, x, c := make([]T, size*size), make([]T, size*size), make([]T, size*size)
					for i := range a {
						a[i] = T(i%7-3) / 16
						x[i] = T(i%5-2) / 16
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						gemm(blas.Trans, tb, size, size, size, alpha, a, size, x, size, 0, c, size)
					}
					b.StopTimer()
					for _, v := range c {
						if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
							b.Fatal("nonfinite benchmark output")
						}
					}
				})
			}
		}
	}
}
