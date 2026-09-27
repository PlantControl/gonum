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

func DlarftBenchmark(b *testing.B, impl interface {
	Dlarfter
	Dgelq2er
}) {
	for _, shape := range [][2]int{{64, 32}, {128, 64}, {256, 128}, {512, 256}} {
		n, ldv := shape[0], shape[1]
		for _, ldt := range []int{32, 64} {
			b.Run(fmt.Sprintf("n=%d/ldv=%d/ldt=%d", n, ldv, ldt), func(b *testing.B) {
				const k = 32
				rnd := rand.New(rand.NewPCG(1, 1))
				v := make([]float64, n*ldv)
				for i := 0; i < n; i++ {
					for j := 0; j < k; j++ {
						v[i*ldv+j] = rnd.NormFloat64()
					}
				}
				tau := make([]float64, k)
				impl.Dgeqr2(n, k, v, ldv, tau, make([]float64, k))
				t := make([]float64, k*ldt)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					impl.Dlarft(lapack.Forward, lapack.ColumnWise, n, k, v, ldv, tau, t, ldt)
				}
			})
		}
	}
	dlarftRowWiseBenchmark(b, impl)
}

func dlarftRowWiseBenchmark(b *testing.B, impl interface {
	Dlarfter
	Dgelq2er
}) {
	for _, n := range []int{64, 128, 256, 512} {
		for _, ldt := range []int{32, 64} {
			b.Run(fmt.Sprintf("RowWise/n=%d/ldv=%d/ldt=%d", n, n, ldt), func(b *testing.B) {
				const k = 32
				rnd := rand.New(rand.NewPCG(1, 1))
				v := make([]float64, k*n)
				for i := range v {
					v[i] = rnd.NormFloat64()
				}
				tau := make([]float64, k)
				impl.Dgelq2(k, n, v, n, tau, make([]float64, k))
				t := make([]float64, k*ldt)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					impl.Dlarft(lapack.Forward, lapack.RowWise, n, k, v, n, tau, t, ldt)
				}
			})
		}
	}
}
