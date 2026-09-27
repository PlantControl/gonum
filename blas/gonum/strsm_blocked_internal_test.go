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

func TestStrsmBlockedBoundaries(t *testing.T) {
	const eps = 0x1p-23
	for _, m := range []int{127, 128, 129, 257} {
		for _, n := range []int{15, 16, 17, 64} {
			for _, ul := range []blas.Uplo{blas.Upper, blas.Lower} {
				for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
					for _, diag := range []blas.Diag{blas.Unit, blas.NonUnit} {
						name := fmt.Sprintf("m=%d/n=%d/uplo=%c/trans=%c/diag=%c", m, n, ul, trans, diag)
						t.Run(name, func(t *testing.T) {
							lda, ldb := m+3, n+5
							a := strsmBlockedA(m, lda, ul, diag)
							aOrig := slices.Clone(a)
							rhs := strsmBlockedRHS(m, n, ldb)
							got, want := slices.Clone(rhs), slices.Clone(rhs)

							Implementation{}.Strsm(blas.Left, ul, trans, diag, m, n, 1, a, lda, got, ldb)
							for j := 0; j < n; j++ {
								// A single right-hand side is below the blocked
								// dispatch threshold and provides a public scalar oracle.
								Implementation{}.Strsm(blas.Left, ul, trans, diag, m, 1, 1, a, lda, want[j:], ldb)
							}

							tol := float32(8 * eps * float32(m))
							for i := 0; i < m; i++ {
								for j := 0; j < ldb; j++ {
									idx := i*ldb + j
									if j >= n {
										if math.Float32bits(got[idx]) != math.Float32bits(rhs[idx]) {
											t.Fatalf("B padding changed at [%d,%d]: got %g want %g", i, j, got[idx], rhs[idx])
										}
										continue
									}
									if !strsmBlockedClose(got[idx], want[idx], tol) {
										t.Fatalf("B[%d,%d]: got %g want scalar-column %g", i, j, got[idx], want[idx])
									}
								}
							}
							if !slices.EqualFunc(a, aOrig, func(x, y float32) bool {
								return math.Float32bits(x) == math.Float32bits(y)
							}) {
								t.Fatal("A changed")
							}
							residual := strsmBlockedResidual(ul, trans, diag, m, n, a, lda, got, ldb, rhs)
							if math.IsNaN(residual) || math.IsInf(residual, 0) || residual > float64(tol) {
								t.Fatalf("non-finite or excessive backward residual %g (tolerance %g)", residual, tol)
							}
						})
					}
				}
			}
		}
	}
}

func strsmBlockedA(n, lda int, ul blas.Uplo, diag blas.Diag) []float32 {
	const padding = float32(9.876543)
	a := make([]float32, n*lda)
	for i := range a {
		a[i] = padding
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			active := ul == blas.Upper && j > i || ul == blas.Lower && j < i
			switch {
			case active:
				a[i*lda+j] = float32((17*i+11*j+3)%19-9) / float32(16*n)
			case i == j && diag == blas.NonUnit:
				a[i*lda+j] = 2 + float32(i%7)/16
			default:
				a[i*lda+j] = float32(math.NaN())
			}
		}
	}
	return a
}

func strsmBlockedRHS(m, n, ldb int) []float32 {
	const padding = float32(-7.654321)
	b := make([]float32, m*ldb)
	for i := range b {
		b[i] = padding
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			b[i*ldb+j] = float32((7*i+3*j+5)%23-11) / 8
		}
	}
	return b
}

func strsmBlockedClose(got, want, tol float32) bool {
	if math.IsNaN(float64(got)) || math.IsNaN(float64(want)) ||
		math.IsInf(float64(got), 0) || math.IsInf(float64(want), 0) {
		return false
	}
	return float32(math.Abs(float64(got)-float64(want))) <= tol*(1+float32(math.Abs(float64(want))))
}

func strsmBlockedResidual(ul blas.Uplo, trans blas.Transpose, diag blas.Diag, m, n int, a []float32, lda int, x []float32, ldx int, rhs []float32) float64 {
	var maxRelative float64
	for i := 0; i < m; i++ {
		rowNorm := 0.0
		for k := 0; k < m; k++ {
			row, col := i, k
			if trans != blas.NoTrans {
				row, col = k, i
			}
			if ul == blas.Upper && col < row || ul == blas.Lower && col > row {
				continue
			}
			av := float64(a[row*lda+col])
			if diag == blas.Unit && row == col {
				av = 1
			}
			rowNorm += math.Abs(av)
		}
		for j := 0; j < n; j++ {
			sum, maxX := 0.0, 0.0
			for k := 0; k < m; k++ {
				row, col := i, k
				if trans != blas.NoTrans {
					row, col = k, i
				}
				if ul == blas.Upper && col < row || ul == blas.Lower && col > row {
					continue
				}
				av := float64(a[row*lda+col])
				if diag == blas.Unit && row == col {
					av = 1
				}
				xv := float64(x[k*ldx+j])
				sum += av * xv
				maxX = math.Max(maxX, math.Abs(xv))
			}
			rhsv := float64(rhs[i*ldx+j])
			relative := math.Abs(sum-rhsv) / math.Max(1, rowNorm*maxX+math.Abs(rhsv))
			maxRelative = math.Max(maxRelative, relative)
		}
	}
	return maxRelative
}
