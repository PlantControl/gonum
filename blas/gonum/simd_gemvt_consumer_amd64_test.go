// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package gonum

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

// These cases execute the real transposed Dgemv method. A matched temporary
// overlay selects its internal candidate; production dispatch remains intact.
func simdGemvTShapeFixtures() []simdConsumerFixture {
	var fixtures []simdConsumerFixture
	impl := Implementation{}
	for _, shape := range [][3]int{{1, 8, 0}, {3, 8, 3}, {4, 8, 0}, {5, 8, 3}, {8, 8, 0}, {16, 8, 3}, {64, 8, 3}, {8, 7, 3}, {8, 9, 3}, {64, 64, 0}} {
		m, n, pad := shape[0], shape[1], shape[2]
		lda := n + pad
		for _, beta := range []float64{0, .75} {
			a, x, y := make([]float64, (m-1)*lda+n), make([]float64, m), make([]float64, n+3)
			for i := range a {
				a[i] = math.NaN()
			}
			for i := 0; i < m; i++ {
				x[i] = float64(i%7-3) * .125
				for j := 0; j < n; j++ {
					a[i*lda+j] = float64((i+j)%13-6) * .0625
				}
			}
			for i := range y {
				y[i] = float64(i%9-4) * .25
			}
			want := append([]float64(nil), y...)
			for j := 0; j < n; j++ {
				if beta == 0 {
					want[j] = 0
					y[j] = math.NaN()
				} else {
					want[j] *= beta
				}
			}
			for i := 0; i < m; i++ {
				scale := float64(.5 * x[i])
				for j := 0; j < n; j++ {
					product := float64(scale * a[i*lda+j])
					want[j] += product
				}
			}
			original := append([]float64(nil), y...)
			fixtures = append(fixtures, simdConsumerFixture{
				name: fmt.Sprintf("m=%d/n=%d/pad=%d/beta=%g", m, n, pad, beta),
				run: func() {
					copy(y, original)
					simdConsumerClearAVX()
					impl.Dgemv(blas.Trans, m, n, .5, a, lda, x, 1, beta, y, 1)
				},
				check: func(t testing.TB) { simdConsumerCheck(t, y, want) },
			})
		}
	}
	return fixtures
}

func TestSIMDGemvTShapeFixtures(t *testing.T) {
	for _, f := range simdGemvTShapeFixtures() {
		t.Run(f.name, func(t *testing.T) {
			for range 3 {
				f.run()
				f.check(t)
			}
		})
	}
}
func BenchmarkSIMDGemvTShapes(b *testing.B) {
	for _, f := range simdGemvTShapeFixtures() {
		b.Run(f.name, func(b *testing.B) {
			f.run()
			f.check(b)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				f.run()
			}
			b.StopTimer()
			f.check(b)
		})
	}
}
