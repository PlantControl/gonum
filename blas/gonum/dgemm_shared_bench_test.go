// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

func BenchmarkDgemmSharedBacking(b *testing.B) {
	for _, size := range []int{64, 128, 256} {
		for _, layout := range []string{"lu", "upper-cholesky"} {
			b.Run(fmt.Sprintf("%s/n%d", layout, size), func(b *testing.B) {
				trans, m, n, k, aoff, boff, coff := gemmSharedLayout(layout, size)
				data := make([]float64, size*size)
				for i := range data {
					data[i] = float64(i%29-14) / 32
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					impl.Dgemm(trans, blas.NoTrans, m, n, k, -0.75, data[aoff:], size, data[boff:], size, 0, data[coff:], size)
				}
			})
		}
	}
}

func BenchmarkSgemmSharedBacking(b *testing.B) {
	for _, size := range []int{64, 128, 256} {
		for _, layout := range []string{"lu", "upper-cholesky"} {
			b.Run(fmt.Sprintf("%s/n%d", layout, size), func(b *testing.B) {
				trans, m, n, k, aoff, boff, coff := gemmSharedLayout(layout, size)
				data := make([]float32, size*size)
				for i := range data {
					data[i] = float32(i%29-14) / 32
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					impl.Sgemm(trans, blas.NoTrans, m, n, k, -0.75, data[aoff:], size, data[boff:], size, 0, data[coff:], size)
				}
			})
		}
	}
}

func gemmSharedLayout(layout string, size int) (trans blas.Transpose, m, n, k, aoff, boff, coff int) {
	block := min(32, size/4)
	j := size / 4
	switch layout {
	case "lu":
		m, n, k = size-j-block, size-j-block, block
		aoff = (j+block)*size + j
		boff = j*size + j + block
		coff = (j+block)*size + j + block
		return blas.NoTrans, m, n, k, aoff, boff, coff
	case "upper-cholesky":
		m, n, k = block, size-j-block, j
		aoff = j
		boff = j + block
		coff = j*size + j + block
		return blas.Trans, m, n, k, aoff, boff, coff
	default:
		panic("unknown shared GEMM layout")
	}
}
