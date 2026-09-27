// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/gonum/internal/netlib"
)

func TestGemmNetlibBridge(t *testing.T) {
	testGemmNetlibBridge[float64](t, netlib.Implementation{}.Dgemm)
	testGemmNetlibBridge[float32](t, netlib.Implementation{}.Sgemm)
}
func testGemmNetlibBridge[T zeroGemmFloat](t *testing.T, gemm zeroGemmFunc[T]) {
	const m, n, k = 3, 4, 5
	for _, ta := range []blas.Transpose{blas.NoTrans, blas.Trans, blas.ConjTrans} {
		for _, tb := range []blas.Transpose{blas.NoTrans, blas.Trans, blas.ConjTrans} {
			t.Run(fmt.Sprintf("%c%c", ta, tb), func(t *testing.T) {
				ar, ac, br, bc := m, k, k, n
				if ta != blas.NoTrans {
					ar, ac = k, m
				}
				if tb != blas.NoTrans {
					br, bc = n, k
				}
				lda, ldb, ldc := ac+1, bc+2, n+3
				a, b, c := make([]T, ar*lda), make([]T, br*ldb), make([]T, m*ldc)
				for i := range a {
					a[i] = T(i%7-3) / 8
				}
				for i := range b {
					b[i] = T(i%5-2) / 4
				}
				for i := range c {
					c[i] = T(i%3 - 1)
				}
				want := append([]T(nil), c...)
				for i := 0; i < m; i++ {
					for j := 0; j < n; j++ {
						var sum T
						for l := 0; l < k; l++ {
							ai, bi := i*lda+l, l*ldb+j
							if ta != blas.NoTrans {
								ai = l*lda + i
							}
							if tb != blas.NoTrans {
								bi = j*ldb + l
							}
							sum += a[ai] * b[bi]
						}
						want[i*ldc+j] = -0.5*sum + 0.25*want[i*ldc+j]
					}
				}
				gemm(ta, tb, m, n, k, -0.5, a, lda, b, ldb, 0.25, c, ldc)
				checkZeroGemmValues(t, "reference product", c, want)
			})
		}
	}
}
