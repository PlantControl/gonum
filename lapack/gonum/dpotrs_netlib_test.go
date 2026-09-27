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

func TestDpotrsNetlib(t *testing.T) {
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
		for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower} {
			n, nrhs := test.n, test.nrhs
			t.Run(fmt.Sprintf("n=%d/nrhs=%d/scale=%g/uplo=%c", n, nrhs, test.scale, uplo), func(t *testing.T) {
				lda, ldb := n+3, nrhs+2
				a0 := factorSPD(n, lda)
				b0 := solveRHS(n, nrhs, ldb)
				scaleSolveRHS(b0, n, nrhs, ldb, test.scale)
				ga := append([]float64(nil), a0...)
				if !(Implementation{}).Dpotrf(uplo, n, ga, lda) {
					t.Fatal("Gonum not positive definite")
				}
				gb := append([]float64(nil), b0...)
				(Implementation{}).Dpotrs(uplo, n, nrhs, ga, lda, gb, ldb)
				checkFactorRowPadding(t, ga, n, n, lda)
				checkFactorRowPadding(t, gb, n, nrhs, ldb)
				checkSolveResidual(t, "Gonum", blas.NoTrans, n, nrhs, a0, lda, b0, ldb, gb, ldb)

				nlda, nldb := n+2, n+4
				na := factorColMajor(n, n, a0, lda, nlda)
				if info := netlib.Dpotrf(byte(uplo), n, na, nlda); info != 0 {
					t.Fatalf("Netlib factor info=%d", info)
				}
				nb := factorColMajor(n, nrhs, b0, ldb, nldb)
				if info := netlib.Dpotrs(byte(uplo), n, nrhs, na, nlda, nb, nldb); info != 0 {
					t.Fatalf("Netlib solve info=%d", info)
				}
				checkFactorColPadding(t, na, n, n, nlda)
				checkFactorColPadding(t, nb, n, nrhs, nldb)
				nx := factorRowMajor(n, nrhs, nb, nldb, ldb)
				checkSolveResidual(t, "Netlib", blas.NoTrans, n, nrhs, a0, lda, b0, ldb, nx, ldb)
				checkSolveMatchesNetlib(t, gb, nx, n, nrhs, ldb, ldb)
			})
		}
	}
}

func BenchmarkDpotrsNetlib(b *testing.B) {
	for _, n := range []int{128, 256} {
		for _, nrhs := range []int{1, 16, 64} {
			for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower} {
				b.Run(fmt.Sprintf("n=%d/nrhs=%d/uplo=%c", n, nrhs, uplo), func(b *testing.B) {
					rowA := factorSPD(n, n)
					if !(Implementation{}).Dpotrf(uplo, n, rowA, n) {
						b.Fatal("Gonum not positive definite")
					}
					colA := factorColMajor(n, n, factorSPD(n, n), n, n)
					if info := netlib.Dpotrf(byte(uplo), n, colA, n); info != 0 {
						b.Fatal(info)
					}
					rowB := solveRHS(n, nrhs, nrhs)
					colB := factorColMajor(n, nrhs, rowB, nrhs, n)
					b.Run("Gonum", func(b *testing.B) {
						b.ReportAllocs()
						x := make([]float64, len(rowB))
						b.ResetTimer()
						for i := 0; i < b.N; i++ {
							copy(x, rowB)
							(Implementation{}).Dpotrs(uplo, n, nrhs, rowA, n, x, nrhs)
						}
					})
					b.Run("Netlib", func(b *testing.B) {
						b.ReportAllocs()
						x := make([]float64, len(colB))
						b.ResetTimer()
						for i := 0; i < b.N; i++ {
							copy(x, colB)
							if info := netlib.Dpotrs(byte(uplo), n, nrhs, colA, n, x, n); info != 0 {
								b.Fatal(info)
							}
						}
					})
				})
			}
		}
	}
}
