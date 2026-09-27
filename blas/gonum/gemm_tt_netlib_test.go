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

func TestGemmTTNetlib(t *testing.T) {
	t.Run("D", func(t *testing.T) { testGemmTTNetlib(t, Implementation{}.Dgemm, netlib.Implementation{}.Dgemm) })
	t.Run("S", func(t *testing.T) { testGemmTTNetlib(t, Implementation{}.Sgemm, netlib.Implementation{}.Sgemm) })
}

func testGemmTTNetlib[T zeroGemmFloat](t *testing.T, gemm, reference zeroGemmFunc[T]) {
	for _, shape := range [][3]int{{2, 4, 16}, {3, 5, 17}, {17, 31, 225}, {224, 64, 32}} {
		m, n, k := shape[0], shape[1], shape[2]
		for _, ta := range []blas.Transpose{blas.Trans, blas.ConjTrans} {
			for _, tb := range []blas.Transpose{blas.Trans, blas.ConjTrans} {
				t.Run(fmt.Sprintf("%c%c/%dx%dx%d", ta, tb, m, n, k), func(t *testing.T) {
					lda, ldb, ldc := m+3, k+2, n+4
					a, b, c := make([]T, k*lda), make([]T, n*ldb), make([]T, m*ldc)
					for i := range a {
						a[i] = T(i%11-5) / 16
					}
					for i := range b {
						b[i] = T(i%7-3) / 16
					}
					for i := range c {
						c[i] = T(i%5-2) / 8
					}
					want := append([]T(nil), c...)
					reference(ta, tb, m, n, k, -0.75, a, lda, b, ldb, -0.5, want, ldc)
					gemm(ta, tb, m, n, k, -0.75, a, lda, b, ldb, -0.5, c, ldc)
					// Netlib applies alpha after reduction; Gonum scales each term.
					// Exact cancellation can therefore differ only in zero's sign.
					for i := 0; i < m; i++ {
						for j := 0; j < n; j++ {
							p := i*ldc + j
							if c[p] == 0 && want[p] == 0 {
								want[p] = c[p]
							}
						}
					}
					checkZeroGemmValues(t, "TT reference product", c, want)
				})
			}
		}
	}
}
