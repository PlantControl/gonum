// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package gonum

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

func simdGemvShapeFixtures() []simdConsumerFixture {
	var fixtures []simdConsumerFixture
	impl := Implementation{}
	for _, shape := range [][3]int{
		{4, 8, 0}, {5, 8, 3}, {8, 8, 3}, {12, 8, 0}, {16, 8, 3}, {64, 8, 3},
		{4, 16, 3}, {8, 16, 0}, {16, 32, 3}, {64, 32, 3}, {64, 64, 0}, {128, 64, 3},
		{4, 7, 3}, {8, 15, 0}, {16, 31, 3}, {64, 63, 3}, {64, 65, 3}, {128, 129, 3},
	} {
		m, n, pad := shape[0], shape[1], shape[2]
		lda := n + pad
		for _, beta := range []float64{0, .75} {
			a, x, y := make([]float64, (m-1)*lda+n), make([]float64, n), make([]float64, m)
			for i := range a {
				a[i] = -99
			}
			for j := range x {
				x[j] = float64(j%7-3) * .125
			}
			want := make([]float64, m)
			for i := range y {
				y[i] = float64(i%9-4) * .25
				var sum float64
				for j := 0; j < n; j++ {
					a[i*lda+j] = float64((i+j)%13-6) * .0625
					sum += a[i*lda+j] * x[j]
				}
				want[i] = .5*sum + beta*y[i]
			}
			original := append([]float64(nil), y...)
			fixtures = append(fixtures, simdConsumerFixture{
				name: fmt.Sprintf("m=%d/n=%d/pad=%d/beta=%g", m, n, pad, beta),
				run: func() {
					copy(y, original)
					simdConsumerClearAVX()
					impl.Dgemv(blas.NoTrans, m, n, .5, a, lda, x, 1, beta, y, 1)
				},
				check: func(t testing.TB) { simdConsumerCheck(t, y, want) },
			})
		}
	}
	return fixtures
}
func TestSIMDGemvShapeFixtures(t *testing.T) {
	for _, f := range simdGemvShapeFixtures() {
		t.Run(f.name, func(t *testing.T) {
			for range 3 {
				f.run()
				f.check(t)
			}
		})
	}
}
func BenchmarkSIMDGemvShapes(b *testing.B) {
	for _, f := range simdGemvShapeFixtures() {
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
