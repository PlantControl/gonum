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

func BenchmarkDgemmTT(b *testing.B) {
	benchmarkGemmTT[float64](b, Implementation{}.Dgemm)
}

func BenchmarkSgemmTT(b *testing.B) {
	benchmarkGemmTT[float32](b, Implementation{}.Sgemm)
}

func benchmarkGemmTT[T zeroGemmFloat](b *testing.B, gemm zeroGemmFunc[T]) {
	for _, shape := range [][3]int{
		{2, 4, 15}, {2, 4, 16}, {3, 5, 17},
		{4, 4, 4}, {16, 32, 224}, {224, 16, 32},
		{64, 32, 224}, {224, 64, 32}, {128, 32, 480}, {480, 128, 32},
		{17, 31, 225}, {225, 17, 31}, {128, 128, 128},
	} {
		m, n, k := shape[0], shape[1], shape[2]
		for _, pad := range []int{0, 7} {
			b.Run(fmt.Sprintf("m=%d/n=%d/k=%d/pad=%d", m, n, k, pad), func(b *testing.B) {
				lda, ldb, ldc := m+pad, k+pad, n+pad
				a, x, c := make([]T, k*lda), make([]T, n*ldb), make([]T, m*ldc)
				for i := range a {
					a[i] = T(i%11-5) / 16
				}
				for i := range x {
					x[i] = T(i%7-3) / 16
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					gemm(blas.Trans, blas.Trans, m, n, k, -1, a, lda, x, ldb, 0, c, ldc)
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
