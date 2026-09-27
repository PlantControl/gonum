// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

// DgesvdBenchmark measures values-only and thin-vector decompositions. Workspace
// is reused; restoring the input matrix is included in the measured operation.
func DgesvdBenchmark(b *testing.B, impl Dgesvder) {
	for _, shape := range [][2]int{{16, 16}, {64, 64}, {128, 128}, {256, 256}, {512, 64}, {64, 512}, {256, 128}, {128, 256}} {
		m, n := shape[0], shape[1]
		for _, vectors := range []bool{false, true} {
			b.Run(fmt.Sprintf("m=%d/n=%d/vectors=%t", m, n, vectors), func(b *testing.B) {
				rnd := rand.New(rand.NewPCG(1, 1))
				lda, minmn := n+3, min(m, n)
				original, a := make([]float64, m*lda), make([]float64, m*lda)
				for i := range original {
					original[i] = rnd.NormFloat64()
				}
				s := make([]float64, minmn)
				var u, vt []float64
				job, ldu, ldvt := lapack.SVDNone, 1, 1
				if vectors {
					job, ldu, ldvt = lapack.SVDStore, minmn, n
					u, vt = make([]float64, m*ldu), make([]float64, minmn*ldvt)
				}
				work := make([]float64, 1)
				impl.Dgesvd(job, job, m, n, a, lda, s, u, ldu, vt, ldvt, work, -1)
				work = make([]float64, int(work[0]))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(a, original)
					if !impl.Dgesvd(job, job, m, n, a, lda, s, u, ldu, vt, ldvt, work, len(work)) {
						b.Fatal("SVD did not converge")
					}
				}
			})
		}
	}
}
