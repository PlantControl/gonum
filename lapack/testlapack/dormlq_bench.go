// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

type dormlqBenchmarker interface {
	Dgelqfer
	Dormlqer
}

func DormlqLeftBenchmark(b *testing.B, impl dormlqBenchmarker) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for _, size := range []struct {
		m, n, k int
	}{
		{128, 16, 95},
		{256, 32, 127},
		{256, 64, 160},
	} {
		m, n, k := size.m, size.n, size.k
		lda, ldc := m, n
		a := make([]float64, k*lda)
		for i := range a {
			a[i] = rnd.NormFloat64()
		}
		tau := make([]float64, k)
		query := make([]float64, 1)
		impl.Dgelqf(k, m, nil, lda, nil, query, -1)
		factorWork := make([]float64, int(query[0]))
		impl.Dgelqf(k, m, a, lda, tau, factorWork, len(factorWork))

		cOrig := make([]float64, m*ldc)
		for i := range cOrig {
			cOrig[i] = rnd.NormFloat64()
		}
		origNorm := frobeniusNorm(cOrig)

		impl.Dormlq(blas.Left, blas.NoTrans, m, n, k, nil, lda, nil, nil, ldc, query, -1)
		work := make([]float64, int(query[0]))
		c := make([]float64, len(cOrig))
		b.Run(fmt.Sprintf("m=%d/n=%d/k=%d", m, n, k), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if i%16 == 0 {
					b.StopTimer()
					copy(c, cOrig)
					b.StartTimer()
				}
				trans := blas.NoTrans
				if i%2 != 0 {
					trans = blas.Trans
				}
				impl.Dormlq(blas.Left, trans, m, n, k, a, lda, tau, c, ldc, work, len(work))
			}
			b.StopTimer()
			gotNorm := frobeniusNorm(c)
			if math.IsNaN(gotNorm) || math.IsInf(gotNorm, 0) || math.Abs(gotNorm-origNorm) > 1e-10*origNorm {
				b.Fatalf("invalid result norm: got %g want %g", gotNorm, origNorm)
			}
		})
	}
}

func frobeniusNorm(a []float64) float64 {
	var scale, sumsq float64
	for _, v := range a {
		if v == 0 {
			continue
		}
		av := math.Abs(v)
		if scale < av {
			r := scale / av
			sumsq = 1 + sumsq*r*r
			scale = av
		} else {
			r := av / scale
			sumsq += r * r
		}
	}
	return scale * math.Sqrt(sumsq)
}
