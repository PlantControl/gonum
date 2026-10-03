// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

// BenchmarkDtrsmSmallRectangular covers the solve shapes used by controlsys
// MatLog_N50: the LU panel update and the real-augmented complex solve. Issue #10
// showed that downstream timings can depend on toolchain and executable layout
// even when this kernel's arithmetic is unchanged. Keep the downstream harness
// in testdata/issue10 alongside these focused kernel measurements.
//
// Each timed iteration includes restoring B, so repeated solves cannot turn the
// fixture into a different numerical workload. A and the known solution are
// bounded, and the untimed check verifies the fixture through the public API.
func BenchmarkDtrsmSmallRectangular(b *testing.B) {
	for _, shape := range []struct{ m, n, lda, ldb int }{
		{64, 36, 100, 100},
		{100, 50, 100, 50},
	} {
		for _, uplo := range []blas.Uplo{blas.Lower, blas.Upper} {
			b.Run(fmt.Sprintf("m=%d/n=%d/lda=%d/ldb=%d/uplo=%c", shape.m, shape.n, shape.lda, shape.ldb, uplo), func(b *testing.B) {
				m, n, lda, ldb := shape.m, shape.n, shape.lda, shape.ldb
				rnd := rand.New(rand.NewPCG(10, 50))
				a := make([]float64, m*lda)
				x := make([]float64, m*ldb)
				rhs := make([]float64, m*ldb)
				work := make([]float64, len(rhs))
				diag := blas.NonUnit
				if uplo == blas.Lower {
					diag = blas.Unit
				}
				for i := 0; i < m; i++ {
					for j := 0; j < m; j++ {
						a[i*lda+j] = (rnd.Float64() - 0.5) / float64(m)
					}
					a[i*lda+i] = 2
					for j := 0; j < n; j++ {
						x[i*ldb+j] = rnd.Float64() - 0.5
					}
				}
				// Construct B = A*X independently of the BLAS implementation.
				for i := 0; i < m; i++ {
					for k := 0; k < m; k++ {
						if (uplo == blas.Lower && k > i) || (uplo == blas.Upper && k < i) {
							continue
						}
						v := a[i*lda+k]
						if diag == blas.Unit && i == k {
							v = 1
						}
						for j := 0; j < n; j++ {
							rhs[i*ldb+j] += v * x[k*ldb+j]
						}
					}
				}
				impl := Implementation{}
				copy(work, rhs)
				impl.Dtrsm(blas.Left, uplo, blas.NoTrans, diag, m, n, 1, a, lda, work, ldb)
				for i := 0; i < m; i++ {
					for j := 0; j < n; j++ {
						if v := work[i*ldb+j]; math.IsNaN(v) || math.Abs(v-x[i*ldb+j]) > 1e-12 {
							b.Fatalf("unexpected solution at (%d,%d): got %g, want %g", i, j, v, x[i*ldb+j])
						}
					}
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(work, rhs)
					impl.Dtrsm(blas.Left, uplo, blas.NoTrans, diag, m, n, 1, a, lda, work, ldb)
				}
			})
		}
	}
}
