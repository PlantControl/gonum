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

func TestDgetrfNetlib(t *testing.T) {
	for _, shape := range [][2]int{{63, 63}, {64, 65}, {65, 64}, {129, 129}, {258, 129}, {129, 258}} {
		m, n := shape[0], shape[1]
		t.Run(fmt.Sprintf("m=%d/n=%d", m, n), func(t *testing.T) {
			lda := n + 3
			a0 := factorPivotMatrix(m, n, lda)
			ga := append([]float64(nil), a0...)
			gp := make([]int, min(m, n))
			if !(Implementation{}).Dgetrf(m, n, ga, lda, gp) {
				t.Fatal("Gonum singular")
			}
			checkFactorRowPadding(t, ga, m, n, lda)
			nlda := m + 2
			na := factorColMajor(m, n, a0, lda, nlda)
			np := make([]int32, min(m, n))
			if info := netlib.Dgetrf(m, n, na, nlda, np); info != 0 {
				t.Fatalf("Netlib info=%d", info)
			}
			checkFactorColPadding(t, na, m, n, nlda)
			nr := factorRowMajor(m, n, na, nlda, lda)
			npi := make([]int, len(np))
			for i, p := range np {
				npi[i] = int(p) - 1
			}
			checkLUFactor(t, "Gonum", m, n, a0, lda, ga, gp)
			checkLUFactor(t, "Netlib", m, n, a0, lda, nr, npi)
		})
	}
}

func TestDgetrfNetlibSingular(t *testing.T) {
	a := []float64{1, 2, 2, 4}
	ga := append([]float64(nil), a...)
	gp := make([]int, 2)
	gok := Implementation{}.Dgetrf(2, 2, ga, 2, gp)
	na := factorColMajor(2, 2, a, 2, 2)
	np := make([]int32, 2)
	info := netlib.Dgetrf(2, 2, na, 2, np)
	if gok || info == 0 {
		t.Fatalf("singular status: Gonum ok=%v Netlib info=%d", gok, info)
	}
}

func checkLUFactor(t *testing.T, name string, m, n int, original []float64, lda int, lu []float64, piv []int) {
	t.Helper()
	pa := append([]float64(nil), original...)
	k := min(m, n)
	for i, p := range piv {
		if p < i || p >= m {
			t.Fatalf("%s pivot[%d]=%d", name, i, p)
		}
		if p != i {
			for j := 0; j < n; j++ {
				pa[i*lda+j], pa[p*lda+j] = pa[p*lda+j], pa[i*lda+j]
			}
		}
	}
	maxA, maxErr := 0.0, 0.0
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			for p := 0; p < k; p++ {
				l := 0.0
				if i == p {
					l = 1
				} else if i > p {
					l = lu[i*lda+p]
				}
				u := 0.0
				if p <= j {
					u = lu[p*lda+j]
				}
				sum += l * u
			}
			maxA = math.Max(maxA, math.Abs(pa[i*lda+j]))
			maxErr = math.Max(maxErr, math.Abs(sum-pa[i*lda+j]))
		}
	}
	if math.IsNaN(maxErr) || math.IsInf(maxErr, 0) || maxErr > 128*0x1p-52*math.Max(1, maxA)*float64(k) {
		t.Errorf("%s residual=%g", name, maxErr)
	}
}

func BenchmarkDgetrfNetlib(b *testing.B) {
	for _, n := range []int{32, 128, 256} {
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			row0 := factorPivotMatrix(n, n, n)
			col0 := factorColMajor(n, n, row0, n, n)
			b.Run("Gonum", func(b *testing.B) {
				b.ReportAllocs()
				a := make([]float64, len(row0))
				p := make([]int, n)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(a, row0)
					if !(Implementation{}).Dgetrf(n, n, a, n, p) {
						b.Fatal("singular")
					}
				}
			})
			b.Run("Netlib", func(b *testing.B) {
				b.ReportAllocs()
				a := make([]float64, len(col0))
				p := make([]int32, n)
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					copy(a, col0)
					if info := netlib.Dgetrf(n, n, a, n, p); info != 0 {
						b.Fatal(info)
					}
				}
			})
		})
	}
}
