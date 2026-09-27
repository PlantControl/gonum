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

func TestGemmSharedBacking(t *testing.T) {
	t.Run("Dgemm", func(t *testing.T) {
		testGemmSharedBacking(t, Implementation{}.Dgemm, math.Ldexp(1, 500), 1e-14)
	})
	t.Run("Sgemm", func(t *testing.T) {
		testGemmSharedBacking(t, Implementation{}.Sgemm, float32(math.Ldexp(1, 50)), 2e-5)
	})
}

func testGemmSharedBacking[T float32 | float64](t *testing.T, gemm func(blas.Transpose, blas.Transpose, int, int, int, T, []T, int, []T, int, T, []T, int), large T, tolerance float64) {
	const size = 64
	for _, layout := range []string{"lu", "upper-cholesky"} {
		for _, scenario := range []string{"zero-coefficients", "finite-extreme-cancellation"} {
			t.Run(fmt.Sprintf("%s/%s", layout, scenario), func(t *testing.T) {
				trans, m, n, k, aoff, boff, coff := gemmSharedLayout(layout, size)
				data := make([]T, size*size)
				for i := range data {
					data[i] = T((i%17)-8) / 16
				}
				for i := 0; i < m; i++ {
					for l := 0; l < k; l++ {
						ai := aoff + i*size + l
						if trans != blas.NoTrans {
							ai = aoff + l*size + i
						}
						switch scenario {
						case "zero-coefficients":
							if l%4 == 0 {
								data[ai] = 0
							} else if l%4 == 1 {
								data[ai] = T(math.Copysign(0, -1))
							} else {
								data[ai] = T((i+l)%5-2) / 4
							}
						case "finite-extreme-cancellation":
							if l%4 < 2 {
								data[ai] = large
							} else {
								data[ai] = -large
							}
						}
					}
				}
				for l := 0; l < k; l++ {
					for j := 0; j < n; j++ {
						bi := boff + l*size + j
						switch scenario {
						case "zero-coefficients":
							switch l % 4 {
							case 0:
								data[bi] = T(math.NaN())
							case 1:
								data[bi] = T(math.Inf(1))
							default:
								data[bi] = T((l+2*j)%7-3) / 8
							}
						case "finite-extreme-cancellation":
							data[bi] = large
						}
					}
				}

				before := append([]T(nil), data...)
				want := append([]T(nil), data...)
				for i := 0; i < m; i++ {
					for j := 0; j < n; j++ {
						value := before[coff+i*size+j]
						for l := 0; l < k; l++ {
							ai := aoff + i*size + l
							if trans != blas.NoTrans {
								ai = aoff + l*size + i
							}
							scale := before[ai]
							if scale == 0 {
								continue
							}
							value += scale * before[boff+l*size+j]
						}
						want[coff+i*size+j] = value
					}
				}

				gemm(trans, blas.NoTrans, m, n, k, 1, data[aoff:], size, data[boff:], size, 1, data[coff:], size)
				for idx, got := range data {
					row, col := idx/size, idx%size
					crow, ccol := coff/size, coff%size
					inC := row >= crow && row < crow+m && col >= ccol && col < ccol+n
					if !inC {
						if !gemmSharedEqualBits(got, before[idx]) {
							t.Fatalf("storage outside C changed at %d", idx)
						}
						continue
					}
					if !gemmSharedSameClass(got, want[idx]) {
						t.Fatalf("C[%d,%d] classification: got %g want %g", row-crow, col-ccol, got, want[idx])
					}
					if !gemmSharedClose(got, want[idx], tolerance) {
						t.Fatalf("C[%d,%d]: got %g want %g", row-crow, col-ccol, got, want[idx])
					}
				}
			})
		}
	}
}

func gemmSharedEqualBits[T float32 | float64](a, b T) bool {
	switch av := any(a).(type) {
	case float32:
		return math.Float32bits(av) == math.Float32bits(any(b).(float32))
	case float64:
		return math.Float64bits(av) == math.Float64bits(any(b).(float64))
	default:
		panic("unexpected floating-point type")
	}
}

func gemmSharedSameClass[T float32 | float64](got, want T) bool {
	g, w := float64(got), float64(want)
	return math.IsNaN(g) == math.IsNaN(w) && math.IsInf(g, 1) == math.IsInf(w, 1) && math.IsInf(g, -1) == math.IsInf(w, -1)
}

func gemmSharedClose[T float32 | float64](got, want T, tolerance float64) bool {
	g, w := float64(got), float64(want)
	if math.IsNaN(w) {
		return math.IsNaN(g)
	}
	if math.IsInf(w, 0) {
		return g == w
	}
	return !math.IsNaN(g) && !math.IsInf(g, 0) && math.Abs(g-w) <= tolerance*(1+math.Abs(w))
}
