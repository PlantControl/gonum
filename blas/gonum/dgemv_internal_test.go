// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

func TestDgemvZeroBetaPreservesUnusedY(t *testing.T) {
	for _, shape := range [][2]int{{0, 9}, {8, 0}, {1, 9}, {8, 9}, {16, 32}, {64, 64}} {
		m, n := shape[0], shape[1]
		lda := n + 3
		for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
			xn, yn := n, m
			if trans == blas.Trans {
				xn, yn = m, n
			}
			for _, incY := range []int{1, 2, -1} {
				t.Run(fmt.Sprintf("%dx%d/T=%c/incY=%d", m, n, trans, incY), func(t *testing.T) {
					a := make([]float64, m*lda)
					x := make([]float64, xn)
					for i := range a {
						a[i] = 2
					}
					for i := range x {
						x[i] = 3
					}
					step := incY
					if step < 0 {
						step = -step
					}
					length, first := 0, 0
					if yn != 0 {
						length = (yn-1)*step + 1
						if incY < 0 {
							first = length - 1
						}
					}
					y := make([]float64, length+3)
					for i := range y {
						y[i] = float64(i + 7)
					}
					for i := 0; i < yn; i++ {
						y[first+i*incY] = math.NaN()
					}
					want := slices.Clone(y)
					if m != 0 && n != 0 {
						for i := 0; i < yn; i++ {
							want[first+i*incY] = float64(3 * xn)
						}
					}
					Implementation{}.Dgemv(trans, m, n, 0.5, a, lda, x, 1, 0, y, incY)
					for i, got := range y {
						if got != want[i] && !(math.IsNaN(got) && math.IsNaN(want[i])) {
							t.Errorf("y[%d]=%g, want %g", i, got, want[i])
						}
					}
				})
			}
		}
	}
}
