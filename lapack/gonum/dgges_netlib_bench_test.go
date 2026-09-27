// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
	"plantcontrol.org/v1/gonum/lapack/testlapack"
)

func BenchmarkDggesNetlibControl(b *testing.B) {
	b.Run("implementation=Gonum", func(b *testing.B) {
		testlapack.DggesControlBenchmark(b, Implementation{})
	})
	b.Run("implementation=Netlib", func(b *testing.B) {
		for _, mode := range []struct {
			name           string
			job, selection byte
		}{
			{"form", 'N', 'N'},
			{"vectors", 'V', 'N'},
			{"left", 'V', 'L'},
			{"unit", 'V', 'D'},
		} {
			for _, n := range []int{32, 64, 128, 256} {
				b.Run(fmt.Sprintf("mode=%s/n=%d", mode.name, n), func(b *testing.B) {
					rowA, rowB := testlapack.DggesBenchmarkPencil(n)
					inputA := netlibColMajor(n, n, rowA, n, n)
					inputB := netlibColMajor(n, n, rowB, n, n)
					a, bm := make([]float64, n*n), make([]float64, n*n)
					ar, ai, beta := make([]float64, n), make([]float64, n), make([]float64, n)
					var vsl, vsr []float64
					if mode.job == 'V' {
						vsl, vsr = make([]float64, n*n), make([]float64, n*n)
					}
					bwork := make([]int32, n)
					query := make([]float64, 1)
					if _, info := netlib.DggesWork(mode.job, mode.job, mode.selection, n,
						a, bm, ar, ai, beta, vsl, vsr, query, -1, bwork); info != 0 {
						b.Fatalf("Dgges query info=%d", info)
					}
					work := make([]float64, int(query[0]))
					wantSelected := 0
					if mode.selection != 'N' {
						wantSelected = n / 2
					}
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						copy(a, inputA)
						copy(bm, inputB)
						sdim, info := netlib.DggesWork(mode.job, mode.job, mode.selection, n,
							a, bm, ar, ai, beta, vsl, vsr, work, len(work), bwork)
						if info != 0 || sdim != wantSelected {
							b.Fatalf("Dgges info=%d, sdim=%d; want info=0, sdim=%d", info, sdim, wantSelected)
						}
					}
				})
			}
		}
	})
}
