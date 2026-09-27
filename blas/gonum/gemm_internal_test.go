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

func TestGemmBetaAndTails(t *testing.T) {
	t.Run("Dgemm", func(t *testing.T) { testGemmBetaAndTails(t, Implementation{}.Dgemm) })
	t.Run("Sgemm", func(t *testing.T) { testGemmBetaAndTails(t, Implementation{}.Sgemm) })
}

func testGemmBetaAndTails[T float32 | float64](t *testing.T, gemm func(blas.Transpose, blas.Transpose, int, int, int, T, []T, int, []T, int, T, []T, int)) {
	for _, dims := range [][3]int{{3, 5, 2}, {7, 59, 9}, {7, 61, 9}, {7, 67, 9}, {60, 60, 60}, {67, 65, 65}} {
		m, n, k := dims[0], dims[1], dims[2]
		for _, ta := range []blas.Transpose{blas.NoTrans, blas.Trans, blas.ConjTrans} {
			for _, tb := range []blas.Transpose{blas.NoTrans, blas.Trans, blas.ConjTrans} {
				for _, beta := range []T{0, 1, -1, 0.5} {
					for _, alpha := range []T{0, -0.75} {
						t.Run(fmt.Sprintf("%dx%dx%d/%c%c/alpha=%g/beta=%g", m, n, k, ta, tb, alpha, beta), func(t *testing.T) {
							ar, ac, br, bc := m, k, k, n
							if ta != blas.NoTrans {
								ar, ac = ac, ar
							}
							if tb != blas.NoTrans {
								br, bc = bc, br
							}
							lda, ldb, ldc := ac+3, bc+5, n+7
							a, b, c := make([]T, (ar-1)*lda+ac), make([]T, (br-1)*ldb+bc), make([]T, (m-1)*ldc+n)
							for _, x := range [][]T{a, b, c} {
								for i := range x {
									x[i] = T(math.Float64frombits(0x7ff8000000001234))
								}
							}
							for i := 0; i < m; i++ {
								for l := 0; l < k; l++ {
									ai := i*lda + l
									if ta != blas.NoTrans {
										ai = l*lda + i
									}
									a[ai] = T((2*i+l)%7) - 3
								}
							}
							for l := 0; l < k; l++ {
								for j := 0; j < n; j++ {
									bi := l*ldb + j
									if tb != blas.NoTrans {
										bi = j*ldb + l
									}
									b[bi] = T((3*l+j)%9) - 4
								}
							}
							if beta != 0 {
								for i := 0; i < m; i++ {
									for j := 0; j < n; j++ {
										c[i*ldc+j] = T((i+j)%7) - 3
									}
								}
							}
							want := slices.Clone(c)
							for i := 0; i < m; i++ {
								for j := 0; j < n; j++ {
									var sum T
									for l := 0; l < k; l++ {
										sum += (T((2*i+l)%7) - 3) * (T((3*l+j)%9) - 4)
									}
									want[i*ldc+j] = alpha * sum
									if beta != 0 {
										want[i*ldc+j] += beta * c[i*ldc+j]
									}
								}
							}
							aOrig, bOrig := slices.Clone(a), slices.Clone(b)
							gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
							for i, got := range c {
								if i%ldc >= n {
									if math.Float64bits(float64(got)) != math.Float64bits(float64(want[i])) {
										t.Fatalf("padding changed at %d", i)
									}
								} else if got != want[i] {
									t.Fatalf("index=%d got=%g want=%g", i, got, want[i])
								}
							}
							equalBits := func(a, b T) bool { return math.Float64bits(float64(a)) == math.Float64bits(float64(b)) }
							if !slices.EqualFunc(a, aOrig, equalBits) || !slices.EqualFunc(b, bOrig, equalBits) {
								t.Fatal("input changed")
							}
						})
					}
				}
			}
		}
	}
}
