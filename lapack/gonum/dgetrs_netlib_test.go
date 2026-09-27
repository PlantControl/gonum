// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDgetrsNetlib(t *testing.T) {
	for _, test := range []struct {
		n, nrhs int
		scale   float64
	}{
		{n: 128, nrhs: 16, scale: 1},
		{n: 129, nrhs: 17, scale: 1},
		{n: 192, nrhs: 1, scale: 1},
		{n: 192, nrhs: 1, scale: 1e-200},
		{n: 192, nrhs: 1, scale: 1e200},
		{n: 192, nrhs: 16, scale: 1e-200},
		{n: 192, nrhs: 16, scale: 1e200},
		{n: 256, nrhs: 64, scale: 1},
	} {
		for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
			n, nrhs := test.n, test.nrhs
			t.Run(fmt.Sprintf("n=%d/nrhs=%d/scale=%g/trans=%c", n, nrhs, test.scale, trans), func(t *testing.T) {
				lda, ldb := n+3, nrhs+2
				a0 := factorPivotMatrix(n, n, lda)
				b0 := solveRHS(n, nrhs, ldb)
				scaleSolveRHS(b0, n, nrhs, ldb, test.scale)
				ga := append([]float64(nil), a0...)
				gp := make([]int, n)
				if !(Implementation{}).Dgetrf(n, n, ga, lda, gp) {
					t.Fatal("Gonum singular")
				}
				gb := append([]float64(nil), b0...)
				(Implementation{}).Dgetrs(trans, n, nrhs, ga, lda, gp, gb, ldb)
				checkFactorRowPadding(t, ga, n, n, lda)
				checkFactorRowPadding(t, gb, n, nrhs, ldb)
				checkSolveResidual(t, "Gonum", trans, n, nrhs, a0, lda, b0, ldb, gb, ldb)

				nlda, nldb := n+2, n+4
				na := factorColMajor(n, n, a0, lda, nlda)
				np := make([]int32, n)
				if info := netlib.Dgetrf(n, n, na, nlda, np); info != 0 {
					t.Fatalf("Netlib factor info=%d", info)
				}
				nb := factorColMajor(n, nrhs, b0, ldb, nldb)
				if info := netlib.Dgetrs(byte(trans), n, nrhs, na, nlda, np, nb, nldb); info != 0 {
					t.Fatalf("Netlib solve info=%d", info)
				}
				checkFactorColPadding(t, na, n, n, nlda)
				checkFactorColPadding(t, nb, n, nrhs, nldb)
				nx := factorRowMajor(n, nrhs, nb, nldb, ldb)
				checkSolveResidual(t, "Netlib", trans, n, nrhs, a0, lda, b0, ldb, nx, ldb)
				checkSolveMatchesNetlib(t, gb, nx, n, nrhs, ldb, ldb)
			})
		}
	}
}

func BenchmarkDgetrsNetlib(b *testing.B) {
	for _, n := range []int{128, 256} {
		for _, nrhs := range []int{1, 16, 64} {
			for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
				b.Run(fmt.Sprintf("n=%d/nrhs=%d/trans=%c", n, nrhs, trans), func(b *testing.B) {
					rowA := factorPivotMatrix(n, n, n)
					rowB := solveRHS(n, nrhs, nrhs)
					gp := make([]int, n)
					if !(Implementation{}).Dgetrf(n, n, rowA, n, gp) {
						b.Fatal("Gonum singular")
					}
					colA := factorColMajor(n, n, factorPivotMatrix(n, n, n), n, n)
					np := make([]int32, n)
					if info := netlib.Dgetrf(n, n, colA, n, np); info != 0 {
						b.Fatal(info)
					}
					colB := factorColMajor(n, nrhs, rowB, nrhs, n)
					b.Run("Gonum", func(b *testing.B) {
						b.ReportAllocs()
						x := make([]float64, len(rowB))
						b.ResetTimer()
						for i := 0; i < b.N; i++ {
							copy(x, rowB)
							(Implementation{}).Dgetrs(trans, n, nrhs, rowA, n, gp, x, nrhs)
						}
					})
					b.Run("Netlib", func(b *testing.B) {
						b.ReportAllocs()
						x := make([]float64, len(colB))
						b.ResetTimer()
						for i := 0; i < b.N; i++ {
							copy(x, colB)
							if info := netlib.Dgetrs(byte(trans), n, nrhs, colA, n, np, x, n); info != 0 {
								b.Fatal(info)
							}
						}
					})
				})
			}
		}
	}
}
