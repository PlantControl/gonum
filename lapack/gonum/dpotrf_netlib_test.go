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
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDpotrfNetlib(t *testing.T) {
	for _, n := range []int{63, 64, 65, 129} {
		for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower} {
			t.Run(fmt.Sprintf("n=%d/uplo=%c", n, uplo), func(t *testing.T) {
				lda := n + 3
				a0 := factorSPD(n, lda)
				ga := append([]float64(nil), a0...)
				if !(Implementation{}).Dpotrf(uplo, n, ga, lda) {
					t.Fatal("Gonum not positive definite")
				}
				checkFactorRowPadding(t, ga, n, n, lda)
				nlda := n + 2
				na := factorColMajor(n, n, a0, lda, nlda)
				if info := netlib.Dpotrf(byte(uplo), n, na, nlda); info != 0 {
					t.Fatalf("Netlib info=%d", info)
				}
				checkFactorColPadding(t, na, n, n, nlda)
				checkCholeskyFactor(t, "Gonum", uplo, n, a0, lda, ga, lda)
				checkCholeskyFactor(t, "Netlib", uplo, n, a0, lda, factorRowMajor(n, n, na, nlda, lda), lda)
			})
		}
	}
}

func TestDpotrfNetlibNotPositiveDefinite(t *testing.T) {
	for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower} {
		a := []float64{1, 2, 2, 1}
		ga := append([]float64(nil), a...)
		gok := Implementation{}.Dpotrf(uplo, 2, ga, 2)
		na := factorColMajor(2, 2, a, 2, 2)
		info := netlib.Dpotrf(byte(uplo), 2, na, 2)
		if gok || info == 0 {
			t.Errorf("uplo=%c: Gonum ok=%v Netlib info=%d", uplo, gok, info)
		}
	}
}

func checkCholeskyFactor(t *testing.T, name string, uplo blas.Uplo, n int, original []float64, lda int, factor []float64, ldf int) {
	t.Helper()
	maxA, maxErr := 0.0, 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			sum := 0.0
			if uplo == blas.Lower {
				for k := 0; k <= min(i, j); k++ {
					sum += factor[i*ldf+k] * factor[j*ldf+k]
				}
			} else {
				for k := 0; k <= min(i, j); k++ {
					sum += factor[k*ldf+i] * factor[k*ldf+j]
				}
			}
			maxA = math.Max(maxA, math.Abs(original[i*lda+j]))
			maxErr = math.Max(maxErr, math.Abs(sum-original[i*lda+j]))
		}
	}
	if math.IsNaN(maxErr) || math.IsInf(maxErr, 0) || maxErr > 128*0x1p-52*math.Max(1, maxA)*float64(n) {
		t.Errorf("%s residual=%g", name, maxErr)
	}
}

func BenchmarkDpotrfNetlib(b *testing.B) {
	for _, n := range []int{32, 128, 256} {
		for _, uplo := range []blas.Uplo{blas.Upper, blas.Lower} {
			b.Run(fmt.Sprintf("n=%d/uplo=%c", n, uplo), func(b *testing.B) {
				row0 := factorSPD(n, n)
				col0 := factorColMajor(n, n, row0, n, n)
				b.Run("Gonum", func(b *testing.B) {
					b.ReportAllocs()
					a := make([]float64, len(row0))
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						copy(a, row0)
						if !(Implementation{}).Dpotrf(uplo, n, a, n) {
							b.Fatal("not positive definite")
						}
					}
				})
				b.Run("Netlib", func(b *testing.B) {
					b.ReportAllocs()
					a := make([]float64, len(col0))
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						copy(a, col0)
						if info := netlib.Dpotrf(byte(uplo), n, a, n); info != 0 {
							b.Fatal(info)
						}
					}
				})
			})
		}
	}
}
