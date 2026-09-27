// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package gonum

import (
	"fmt"
	"simd/archsimd"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

type simdComplexDotConsumerFixture struct {
	name  string
	run   func()
	check func(testing.TB)
}

func TestSIMDComplexDotConsumerFixtures(t *testing.T) {
	for _, fixture := range simdComplexDotConsumerFixtures() {
		t.Run(fixture.name, func(t *testing.T) {
			for range 3 {
				fixture.run()
				fixture.check(t)
			}
		})
	}
}

func BenchmarkSIMDComplexDotConsumers(b *testing.B) {
	for _, fixture := range simdComplexDotConsumerFixtures() {
		b.Run(fixture.name, func(b *testing.B) {
			fixture.run()
			fixture.check(b)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fixture.run()
			}
			b.StopTimer()
			fixture.check(b)
		})
	}
}

func simdComplexDotConsumerClearAVX() {
	if archsimd.X86.AVX() {
		archsimd.ClearAVXUpperBits()
	}
}

func simdComplexDotConsumerFixtures() []simdComplexDotConsumerFixture {
	var fixtures []simdComplexDotConsumerFixture
	impl := Implementation{}
	for _, n := range []int{7, 31, 63, 65, 127, 4096} {
		for _, inc := range []int{1, 3} {
			for _, conjugate := range []bool{false, true} {
				x, y := make([]complex64, (n-1)*inc+1), make([]complex64, (n-1)*inc+1)
				for i := range x {
					x[i] = complex(float32(i%11-5)/16, float32(i%7-3)/8)
					y[i] = complex(float32(i%13-6)/8, float32(i%5-2)/16)
				}
				originalX, originalY := slices.Clone(x), slices.Clone(y)
				// Dyadic inputs and bounded sums are exactly representable in
				// float32; the independent complex128 sum is an exact oracle.
				var want complex128
				for i := 0; i < n; i++ {
					xv := complex128(x[i*inc])
					if conjugate {
						xv = complex(real(xv), -imag(xv))
					}
					want += xv * complex128(y[i*inc])
				}
				var result complex64
				name := "Cdotu"
				if conjugate {
					name = "Cdotc"
				}
				fixtures = append(fixtures, simdComplexDotConsumerFixture{
					name: fmt.Sprintf("%s/n=%d/inc=%d", name, n, inc),
					run: func() {
						if conjugate {
							simdComplexDotConsumerClearAVX()
							result = impl.Cdotc(n, x, inc, y, inc)
						} else {
							simdComplexDotConsumerClearAVX()
							result = impl.Cdotu(n, x, inc, y, inc)
						}
					},
					check: func(t testing.TB) {
						if complex128(result) != want {
							t.Fatalf("got=%v want=%v", result, want)
						}
						if !slices.Equal(x, originalX) || !slices.Equal(y, originalY) {
							t.Fatal("dot modified an input")
						}
					},
				})
			}
		}
	}
	for _, dimensions := range [][2]int{{8, 31}, {16, 65}} {
		for _, incX := range []int{1, 3} {
			m, n := dimensions[0], dimensions[1]
			lda, incY := n+3, 2
			a := make([]complex64, m*lda)
			x, y := make([]complex64, (n-1)*incX+1), make([]complex64, (m-1)*incY+1)
			for i := range a {
				a[i] = complex(float32(i%11-5)/16, float32(i%7-3)/16)
			}
			for i := range x {
				x[i] = complex(float32(i%5-2)/8, float32(i%7-3)/16)
			}
			for i := range y {
				y[i] = complex(float32(i%3-1)/4, float32(i%5-2)/8)
			}
			originalA, originalX, originalY := slices.Clone(a), slices.Clone(x), slices.Clone(y)
			want := make([]complex128, len(y))
			for i, value := range y {
				want[i] = complex128(value)
			}
			for i := 0; i < m; i++ {
				var sum complex128
				for j := 0; j < n; j++ {
					sum += complex128(a[i*lda+j]) * complex128(x[j*incX])
				}
				want[i*incY] += (0.5 - 0.25i) * sum
			}
			fixtures = append(fixtures, simdComplexDotConsumerFixture{
				name: fmt.Sprintf("Cgemv/m=%d/n=%d/incX=%d/incY=2", m, n, incX),
				run: func() {
					copy(y, originalY)
					simdComplexDotConsumerClearAVX()
					// beta=1 keeps this consumer focused on its real dot calls.
					impl.Cgemv(blas.NoTrans, m, n, 0.5-0.25i, a, lda, x, incX, 1, y, incY)
				},
				check: func(t testing.TB) {
					for i, value := range y {
						if complex128(value) != want[i] {
							t.Fatalf("index=%d got=%v want=%v", i, value, want[i])
						}
					}
					if !slices.Equal(a, originalA) || !slices.Equal(x, originalX) {
						t.Fatal("gemv modified its matrix or x input")
					}
				},
			})
		}
	}
	return fixtures
}
