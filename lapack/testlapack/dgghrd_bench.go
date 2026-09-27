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

func DgghrdBenchmark(b *testing.B, impl Dgghrder) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for _, n := range []int{10, 50, 100, 200, 500} {
		aOrig := make([]float64, n*n)
		bOrig := make([]float64, n*n)
		for i := range aOrig {
			aOrig[i] = rnd.NormFloat64()
		}
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				bOrig[i*n+j] = rnd.NormFloat64()
			}
			bOrig[i*n+i] += float64(n)
		}

		a := make([]float64, n*n)
		bm := make([]float64, n*n)
		q := make([]float64, n*n)
		z := make([]float64, n*n)

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(a, aOrig)
				copy(bm, bOrig)
				b.StartTimer()
				impl.Dgghrd(lapack.OrthoExplicit, lapack.OrthoExplicit, n, 0, n-1,
					a, n, bm, n, q, n, z, n)
			}
		})
	}
}

func DgghrdControlBenchmark(b *testing.B, impl Dgghrder) {
	modes := []struct {
		name         string
		compq, compz lapack.OrthoComp
	}{
		{name: "none", compq: lapack.OrthoNone, compz: lapack.OrthoNone},
		{name: "q", compq: lapack.OrthoExplicit, compz: lapack.OrthoNone},
		{name: "z", compq: lapack.OrthoNone, compz: lapack.OrthoExplicit},
		{name: "both", compq: lapack.OrthoExplicit, compz: lapack.OrthoExplicit},
	}
	sizes := []struct {
		n, extra int
	}{
		{31, 0}, {32, 0}, {33, 0},
		{63, 0}, {64, 0}, {65, 0},
		{128, 0}, {256, 0}, {256, 1}, {256, 3},
	}
	for _, mode := range modes {
		for _, size := range sizes {
			n := size.n
			stride := n + size.extra
			aOrig := make([]float64, n*stride)
			bOrig := make([]float64, n*stride)
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					aOrig[i*stride+j] = float64((i*17+j*11)%29-14) / 16
					if j >= i {
						bOrig[i*stride+j] = float64((i*7+j*13)%23-11) / 16
					}
				}
				bOrig[i*stride+i] += float64(n)
			}
			a := make([]float64, len(aOrig))
			bm := make([]float64, len(bOrig))
			var q, z []float64
			if mode.compq != lapack.OrthoNone {
				q = make([]float64, n*stride)
			}
			if mode.compz != lapack.OrthoNone {
				z = make([]float64, n*stride)
			}
			name := fmt.Sprintf("mode=%s/n=%d/stride=%d", mode.name, n, stride)
			b.Run(name, func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					copy(a, aOrig)
					copy(bm, bOrig)
					impl.Dgghrd(mode.compq, mode.compz, n, 0, n-1,
						a, stride, bm, stride, q, max(1, stride), z, max(1, stride))
				}
			})
		}
	}
}
