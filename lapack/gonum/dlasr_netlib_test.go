// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDlasrNetlibDifferential(t *testing.T) {
	for _, shape := range [][2]int{{5, 4}, {63, 65}, {64, 64}, {65, 63}, {65, 65}, {79, 65}, {80, 65}, {81, 65}} {
		m, n := shape[0], shape[1]
		for _, side := range []blas.Side{blas.Left, blas.Right} {
			for _, pivot := range []lapack.Pivot{lapack.Variable, lapack.Top, lapack.Bottom} {
				for _, direct := range []lapack.Direct{lapack.Forward, lapack.Backward} {
					for _, mixedIdentity := range []bool{false, true} {
						name := fmt.Sprintf("m=%d/n=%d/side=%c/pivot=%c/direct=%c/mixedIdentity=%t", m, n, side, pivot, direct, mixedIdentity)
						t.Run(name, func(t *testing.T) {
							lda := n + 3
							a := make([]float64, m*lda)
							for i := 0; i < m; i++ {
								for j := 0; j < n; j++ {
									a[i*lda+j] = float64(3*i-2*j+1) / 7
								}
							}
							rotations := n - 1
							if side == blas.Left {
								rotations = m - 1
							}
							c, s := make([]float64, rotations), make([]float64, rotations)
							for i := range c {
								theta := float64(i+1) * 0.37
								c[i], s[i] = math.Cos(theta), math.Sin(theta)
								if mixedIdentity && i%2 == 0 {
									c[i], s[i] = 1, 0
								}
							}
							got := append([]float64(nil), a...)
							Implementation{}.Dlasr(side, pivot, direct, m, n, c, s, got, lda)
							ldcol := m + 2
							wantCol := netlibColMajor(m, n, a, lda, ldcol)
							netlib.Dlasr(byte(side), byte(pivot), byte(direct), m, n, c, s, wantCol, ldcol)
							want := netlibRowMajor(m, n, wantCol, ldcol, lda)
							checkDlasrForwardError(t, side, m, n, rotations, a, lda, got, want)
						})
					}
				}
			}
		}
	}
}

func checkDlasrForwardError(t *testing.T, side blas.Side, m, n, rotations int, a []float64, lda int, got, want []float64) {
	t.Helper()
	const eps = 0x1p-52
	gamma := float64(3*rotations) * eps / (1 - float64(3*rotations)*eps)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			g, w := got[i*lda+j], want[i*lda+j]
			if math.IsNaN(g) || math.IsInf(g, 0) || math.IsNaN(w) || math.IsInf(w, 0) {
				t.Fatalf("A[%d,%d] is non-finite: got %g, want %g", i, j, g, w)
			}
			norm := 0.0
			if side == blas.Left {
				for k := 0; k < m; k++ {
					norm = math.Hypot(norm, a[k*lda+j])
				}
			} else {
				for k := 0; k < n; k++ {
					norm = math.Hypot(norm, a[i*lda+k])
				}
			}
			// Each output follows at most rotations plane rotations. Each rotation
			// uses two products and one sum, giving gamma_(3*rotations) forward
			// error scaled by the input vector norm. Account for rounding in both
			// the Gonum and Netlib evaluations.
			bound := 2 * gamma * norm
			if math.Abs(g-w) > bound {
				t.Fatalf("A[%d,%d]=%g, want %g (error=%g bound=%g)", i, j, g, w, math.Abs(g-w), bound)
			}
		}
	}
}
