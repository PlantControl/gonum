// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDgeqrfNetlib(t *testing.T) {
	for _, shape := range [][2]int{{63, 63}, {64, 65}, {65, 64}, {129, 129}, {258, 129}, {129, 258}} {
		m, n := shape[0], shape[1]
		t.Run(fmt.Sprintf("m=%d/n=%d", m, n), func(t *testing.T) {
			lda := n + 3
			a0 := factorMatrix(m, n, lda)
			ga := append([]float64(nil), a0...)
			gtau := make([]float64, min(m, n))
			gq := []float64{0}
			Implementation{}.Dgeqrf(m, n, ga, lda, gtau, gq, -1)
			Implementation{}.Dgeqrf(m, n, ga, lda, gtau, make([]float64, int(gq[0])), int(gq[0]))
			checkFactorRowPadding(t, ga, m, n, lda)

			nlda := m + 2
			na := factorColMajor(m, n, a0, lda, nlda)
			ntau, nq := make([]float64, min(m, n)), []float64{0}
			if info := netlib.Dgeqrf(m, n, na, nlda, ntau, nq, -1); info != 0 {
				t.Fatalf("query info=%d", info)
			}
			if info := netlib.Dgeqrf(m, n, na, nlda, ntau, make([]float64, int(nq[0])), int(nq[0])); info != 0 {
				t.Fatalf("info=%d", info)
			}
			checkFactorColPadding(t, na, m, n, nlda)
			naRow := factorRowMajor(m, n, na, nlda, lda)
			checkQRFactor(t, "Gonum", m, n, a0, lda, ga, gtau)
			checkQRFactor(t, "Netlib", m, n, a0, lda, naRow, ntau)
		})
	}
}

func checkQRFactor(t *testing.T, name string, m, n int, original []float64, lda int, a, tau []float64) {
	t.Helper()
	k := min(m, n)
	q := make([]float64, m*m)
	for i := 0; i < m; i++ {
		q[i*m+i] = 1
	}
	for h := 0; h < k; h++ {
		for i := 0; i < m; i++ {
			dot := q[i*m+h]
			for r := h + 1; r < m; r++ {
				dot += q[i*m+r] * a[r*lda+h]
			}
			dot *= tau[h]
			q[i*m+h] -= dot
			for r := h + 1; r < m; r++ {
				q[i*m+r] -= dot * a[r*lda+h]
			}
		}
	}
	maxA, maxErr := 0.0, 0.0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for p := 0; p < k && p <= j; p++ {
				sum += q[i*m+p] * a[p*lda+j]
			}
			maxA = math.Max(maxA, math.Abs(original[i*lda+j]))
			maxErr = math.Max(maxErr, math.Abs(sum-original[i*lda+j]))
		}
	}
	if math.IsNaN(maxErr) || math.IsInf(maxErr, 0) || maxErr > 128*0x1p-52*math.Max(1, maxA)*float64(max(m, n)) {
		t.Errorf("%s reconstruction residual=%g", name, maxErr)
	}
	if r := factorOrthoResidual(q, m); math.IsNaN(r) || math.IsInf(r, 0) || r > 128*0x1p-52*float64(m) {
		t.Errorf("%s orthogonality residual=%g", name, r)
	}
}

func BenchmarkDgeqrfNetlib(b *testing.B) {
	for _, shape := range [][2]int{{32, 32}, {128, 128}, {256, 256}, {64, 32}, {256, 128}, {32, 64}, {128, 256}} {
		m, n := shape[0], shape[1]
		b.Run(fmt.Sprintf("m=%d/n=%d", m, n), func(b *testing.B) {
			row0 := factorMatrix(m, n, n)
			col0 := factorColMajor(m, n, row0, n, m)
			b.Run("Gonum", func(b *testing.B) {
				b.ReportAllocs()
				a, tau, q := make([]float64, len(row0)), make([]float64, min(m, n)), []float64{0}
				Implementation{}.Dgeqrf(m, n, a, n, tau, q, -1)
				work := make([]float64, int(q[0]))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(a, row0)
					Implementation{}.Dgeqrf(m, n, a, n, tau, work, len(work))
				}
			})
			b.Run("Netlib", func(b *testing.B) {
				b.ReportAllocs()
				a, tau, q := make([]float64, len(col0)), make([]float64, min(m, n)), []float64{0}
				netlib.Dgeqrf(m, n, a, m, tau, q, -1)
				work := make([]float64, int(q[0]))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(a, col0)
					if info := netlib.Dgeqrf(m, n, a, m, tau, work, len(work)); info != 0 {
						b.Fatal(info)
					}
				}
			})
		})
	}
}
