// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 && go1.27 && goexperiment.simd && !safe && !noasm && !gccgo

package gonum

import (
	"fmt"
	"math"
	"math/cmplx"
	"simd/archsimd"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

// These fixtures call the real BLAS methods. A temporary Go build overlay may
// replace their selected internal kernel calls; production dispatch is intact.
// Each mutation starts from the same input and includes its reset copy in timing.
type simdConsumerFixture struct {
	name  string
	run   func()
	check func(testing.TB)
}

func TestSIMDConsumerFixtures(t *testing.T) {
	for _, fixture := range simdConsumerFixtures() {
		t.Run(fixture.name, func(t *testing.T) {
			for range 3 {
				fixture.run()
				fixture.check(t)
			}
		})
	}
}

func BenchmarkSIMDConsumers(b *testing.B) {
	for _, fixture := range simdConsumerFixtures() {
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

func simdConsumerClearAVX() {
	if archsimd.X86.AVX() {
		archsimd.ClearAVXUpperBits()
	}
}

func simdConsumerFixtures() []simdConsumerFixture {
	var fixtures []simdConsumerFixture
	impl := Implementation{}
	for _, n := range []int{31, 256} {
		for _, inc := range []int{1, 3} {
			for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
				lda := n + 3
				a, af := make([]float64, n*lda), make([]float32, n*lda)
				x, xf := make([]float64, (n-1)*inc+1), make([]float32, (n-1)*inc+1)
				for i := range a {
					a[i] = float64(i%13-6) / 64
					af[i] = float32(a[i])
				}
				for i := range x {
					x[i] = float64(i%7+1) / 8
					xf[i] = float32(x[i])
				}
				original, originalf := slices.Clone(x), slices.Clone(xf)
				want := slices.Clone(x)
				for i := 0; i < n; i++ {
					var sum float64
					for j := 0; j < n; j++ {
						if trans == blas.NoTrans && j >= i {
							sum += a[i*lda+j] * original[j*inc]
						} else if trans == blas.Trans && j <= i {
							sum += a[j*lda+i] * original[j*inc]
						}
					}
					want[i*inc] = sum
				}
				fixtures = append(fixtures, simdConsumerFixture{
					name: fmt.Sprintf("Dtrmv/n=%d/inc=%d/trans=%c", n, inc, trans),
					run: func() {
						copy(x, original)
						simdConsumerClearAVX()
						impl.Dtrmv(blas.Upper, trans, blas.NonUnit, n, a, lda, x, inc)
					},
					check: func(t testing.TB) { simdConsumerCheck(t, x, want) },
				}, simdConsumerFixture{
					name: fmt.Sprintf("Strmv/n=%d/inc=%d/trans=%c", n, inc, trans),
					run: func() {
						copy(xf, originalf)
						simdConsumerClearAVX()
						impl.Strmv(blas.Upper, trans, blas.NonUnit, n, af, lda, xf, inc)
					},
					check: func(t testing.TB) { simdConsumerCheck(t, xf, want) },
				})
			}
		}
	}
	for _, n := range []int{8, 64} {
		for _, inc := range []int{1, 3} {
			lda := n + 3
			a, x, y := make([]float32, n*lda), make([]float32, (n-1)*inc+1), make([]float32, (n-1)*inc+1)
			for i := range a {
				a[i] = float32(i%7-3) / 16
			}
			for i := range x {
				x[i], y[i] = float32(i%5+1)/8, float32(i%7-3)/4
			}
			original := slices.Clone(a)
			want := make([]float64, len(a))
			for i, value := range a {
				want[i] = float64(value)
			}
			for i := 0; i < n; i++ {
				for j := 0; j < n; j++ {
					want[i*lda+j] += 0.5 * float64(x[i*inc]) * float64(y[j*inc])
				}
			}
			fixtures = append(fixtures, simdConsumerFixture{
				name: fmt.Sprintf("Sger/n=%d/inc=%d", n, inc),
				run: func() {
					copy(a, original)
					simdConsumerClearAVX()
					impl.Sger(n, n, 0.5, x, inc, y, inc, a, lda)
				},
				check: func(t testing.TB) { simdConsumerCheck(t, a, want) },
			})
		}
	}
	for _, n := range []int{31, 4096} {
		x := make([]float64, 2*n-1)
		var squares, result float64
		for i := range x {
			x[i] = float64(i%7+1) / 8
			if i%2 == 0 {
				squares += x[i] * x[i]
			}
		}
		want := []float64{math.Sqrt(squares)}
		fixtures = append(fixtures, simdConsumerFixture{
			name: fmt.Sprintf("Dnrm2/n=%d/inc=2", n),
			run: func() {
				simdConsumerClearAVX()
				result = impl.Dnrm2(n, x, 2)
			},
			check: func(t testing.TB) { simdConsumerCheck(t, []float64{result}, want) },
		})
	}
	for _, dims := range [][2]int{{8, 31}, {64, 129}} {
		for _, conjugate := range []bool{false, true} {
			m, n := dims[0], dims[1]
			lda := n + 3
			a, x, y := make([]complex128, m*lda), make([]complex128, 3*m-2), make([]complex128, 2*n-1)
			for i := range a {
				a[i] = complex(float64(i%7-3)/16, 0.125)
			}
			for i := range x {
				x[i] = complex(float64(i%5+1)/8, -0.25)
			}
			for i := range y {
				y[i] = complex(float64(i%7-3)/4, 0.5)
			}
			original, want := slices.Clone(a), slices.Clone(a)
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					yv := y[2*j]
					if conjugate {
						yv = cmplx.Conj(yv)
					}
					want[i*lda+j] += (0.5 - 0.25i) * x[3*i] * yv
				}
			}
			name := "Zgeru"
			if conjugate {
				name = "Zgerc"
			}
			fixtures = append(fixtures, simdConsumerFixture{
				name: fmt.Sprintf("%s/m=%d/n=%d/incX=3/incY=2", name, m, n),
				run: func() {
					copy(a, original)
					simdConsumerClearAVX()
					if conjugate {
						impl.Zgerc(m, n, 0.5-0.25i, x, 3, y, 2, a, lda)
					} else {
						impl.Zgeru(m, n, 0.5-0.25i, x, 3, y, 2, a, lda)
					}
				},
				check: func(t testing.TB) {
					for i, value := range a {
						if value != want[i] {
							t.Fatalf("index=%d got=%v want=%v", i, value, want[i])
						}
					}
				},
			})
		}
	}
	for _, dims := range [][2]int{{8, 31}, {64, 129}} {
		for _, conjugate := range []bool{false, true} {
			m, n := dims[0], dims[1]
			lda := n + 3
			a, x, y := make([]complex64, m*lda), make([]complex64, 3*m-2), make([]complex64, 2*n-1)
			for i := range a {
				a[i] = complex(float32(i%7-3)/16, 0.125)
			}
			for i := range x {
				x[i] = complex(float32(i%5+1)/8, -0.25)
			}
			for i := range y {
				y[i] = complex(float32(i%7-3)/4, 0.5)
			}
			original, want := slices.Clone(a), slices.Clone(a)
			for i := 0; i < m; i++ {
				for j := 0; j < n; j++ {
					yv := y[2*j]
					if conjugate {
						yv = complex64(cmplx.Conj(complex128(yv)))
					}
					want[i*lda+j] += (0.5 - 0.25i) * x[3*i] * yv
				}
			}
			name := "Cgeru"
			if conjugate {
				name = "Cgerc"
			}
			fixtures = append(fixtures, simdConsumerFixture{
				name: fmt.Sprintf("%s/m=%d/n=%d/incX=3/incY=2", name, m, n),
				run: func() {
					copy(a, original)
					simdConsumerClearAVX()
					if conjugate {
						impl.Cgerc(m, n, 0.5-0.25i, x, 3, y, 2, a, lda)
					} else {
						impl.Cgeru(m, n, 0.5-0.25i, x, 3, y, 2, a, lda)
					}
				},
				check: func(t testing.TB) {
					for i, value := range a {
						if value != want[i] {
							t.Fatalf("index=%d got=%v want=%v", i, value, want[i])
						}
					}
				},
			})
		}
	}
	return fixtures
}

func simdConsumerCheck[T ~float32 | ~float64](t testing.TB, got []T, want []float64) {
	t.Helper()
	for i, value := range got {
		x := float64(value)
		if math.IsNaN(x) || math.IsInf(x, 0) || math.Abs(x-want[i]) > 1e-12*math.Max(1, math.Abs(want[i])) {
			t.Fatalf("index=%d got=%g want=%g", i, x, want[i])
		}
	}
}
