// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

type zeroSyrkFloat interface{ float32 | float64 }
type zeroSyrkFunc[T zeroSyrkFloat] func(blas.Uplo, blas.Transpose, int, int, T, []T, int, T, []T, int)

func TestDsyrkZeroBeta(t *testing.T) { testSyrkZeroBeta(t, Implementation{}.Dsyrk) }
func TestSsyrkZeroBeta(t *testing.T) { testSyrkZeroBeta(t, Implementation{}.Ssyrk) }

func testSyrkZeroBeta[T zeroSyrkFloat](t *testing.T, syrk zeroSyrkFunc[T]) {
	t.Helper()
	for _, dims := range [][2]int{{4, 3}, {17, 65}} {
		n, k := dims[0], dims[1]
		for _, ul := range []blas.Uplo{blas.Upper, blas.Lower} {
			for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans, blas.ConjTrans} {
				for _, beta := range []T{0, T(math.Copysign(0, -1))} {
					t.Run(fmt.Sprintf("n=%d/k=%d/%c/%c/beta=%g", n, k, ul, trans, beta), func(t *testing.T) {
						rows, cols := n, k
						if trans != blas.NoTrans {
							rows, cols = k, n
						}
						lda, ldc := cols+2, n+3
						a := make([]T, rows*lda)
						for i := 0; i < rows; i++ {
							for j := 0; j < cols; j++ {
								a[i*lda+j] = T((2*i+j)%5) - 2
							}
						}
						c := make([]T, n*ldc)
						for i := range c {
							c[i] = T([]float64{math.NaN(), math.Inf(1), math.Inf(-1), math.Copysign(0, -1), 73, 91, 117}[i%7])
						}
						want := slices.Clone(c)
						for i := 0; i < n; i++ {
							for j := 0; j < n; j++ {
								if ul == blas.Upper && j < i || ul == blas.Lower && j > i {
									continue
								}
								var sum T
								for l := 0; l < k; l++ {
									ai, aj := i*lda+l, j*lda+l
									if trans != blas.NoTrans {
										ai, aj = l*lda+i, l*lda+j
									}
									sum += a[ai] * a[aj]
								}
								want[i*ldc+j] = sum
							}
						}
						aOrig := slices.Clone(a)
						syrk(ul, trans, n, k, 1, a, lda, beta, c, ldc)
						checkZeroSyrkBits(t, c, want)
						checkZeroSyrkBits(t, a, aOrig)
					})
				}
			}
		}
	}
}

func checkZeroSyrkBits[T zeroSyrkFloat](t *testing.T, got, want []T) {
	t.Helper()
	for i, g := range got {
		w := want[i]
		if math.IsNaN(float64(g)) && math.IsNaN(float64(w)) {
			continue
		}
		if math.Float64bits(float64(g)) != math.Float64bits(float64(w)) {
			t.Fatalf("index %d: got %g, want %g", i, g, w)
		}
	}
}
