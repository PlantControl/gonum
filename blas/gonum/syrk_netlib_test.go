// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/gonum/internal/netlib"
)

func TestDsyrkZeroBetaNetlib(t *testing.T) { testSyrkZeroBeta(t, netlib.Implementation{}.Dsyrk) }
func TestSsyrkZeroBetaNetlib(t *testing.T) { testSyrkZeroBeta(t, netlib.Implementation{}.Ssyrk) }

func TestDsyrkNetlibDifferential(t *testing.T) {
	testSyrkNetlibDifferential(t, Implementation{}.Dsyrk, netlib.Implementation{}.Dsyrk, 0)
}

func TestSsyrkNetlibDifferential(t *testing.T) {
	testSyrkNetlibDifferential(t, Implementation{}.Ssyrk, netlib.Implementation{}.Ssyrk, 0)
}

func testSyrkNetlibDifferential[T zeroSyrkFloat](t *testing.T, gonum, native zeroSyrkFunc[T], tol float64) {
	t.Helper()
	for _, dims := range [][2]int{{16, 16}, {17, 31}, {64, 128}} {
		n, k := dims[0], dims[1]
		for _, ul := range []blas.Uplo{blas.Upper, blas.Lower} {
			for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans, blas.ConjTrans} {
				for _, beta := range []T{0, 1, -0.5} {
					name := fmt.Sprintf("n=%d/k=%d/%c%c/beta=%g", n, k, ul, trans, beta)
					t.Run(name, func(t *testing.T) {
						rows, cols := n, k
						if trans != blas.NoTrans {
							rows, cols = k, n
						}
						lda, ldc := cols+3, n+5
						a := make([]T, rows*lda)
						for i := range a {
							a[i] = T((i%19)-9) / 32
						}
						gc := make([]T, n*ldc)
						for i := range gc {
							gc[i] = T((i%17)-8) / 16
						}
						nc := slices.Clone(gc)
						aOrig, cOrig := slices.Clone(a), slices.Clone(gc)
						gonum(ul, trans, n, k, T(-0.75), a, lda, beta, gc, ldc)
						native(ul, trans, n, k, T(-0.75), aOrig, lda, beta, nc, ldc)
						if !slices.Equal(a, aOrig) {
							t.Fatal("A changed")
						}
						for i := 0; i < n; i++ {
							for j := 0; j < ldc; j++ {
								p := i*ldc + j
								active := j < n && (ul == blas.Upper && j >= i || ul == blas.Lower && j <= i)
								if !active {
									if gc[p] != cOrig[p] || nc[p] != cOrig[p] {
										t.Fatalf("inactive C[%d,%d] changed", i, j)
									}
									continue
								}
								g, want := float64(gc[p]), float64(nc[p])
								if math.IsNaN(g) || math.IsInf(g, 0) || math.IsNaN(want) || math.IsInf(want, 0) ||
									math.Abs(g-want) > tol*math.Max(1, math.Abs(want)) {
									t.Fatalf("C[%d,%d]=%g, Netlib %g", i, j, g, want)
								}
							}
						}
					})
				}
			}
		}
	}
}
