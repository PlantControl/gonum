// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

func BenchmarkDtrsmSizes(b *testing.B) {
	sizes := []int{31, 32, 33, 63, 64, 65, 128, 256}
	for _, order := range sizes {
		benchmarkDtrsmCase(b, blas.Left, blas.Lower, blas.NoTrans, order, 16)
		benchmarkDtrsmCase(b, blas.Left, blas.Upper, blas.Trans, order, 16)
		benchmarkDtrsmCase(b, blas.Right, blas.Lower, blas.Trans, 16, order)
		benchmarkDtrsmCase(b, blas.Right, blas.Upper, blas.NoTrans, 16, order)
	}
	for _, order := range []int{64, 128, 256} {
		benchmarkDtrsmCase(b, blas.Left, blas.Lower, blas.NoTrans, order, 1)
		benchmarkDtrsmCase(b, blas.Left, blas.Lower, blas.NoTrans, order, 64)
	}
}

func benchmarkDtrsmCase(b *testing.B, side blas.Side, uplo blas.Uplo, trans blas.Transpose, m, n int) {
	order := m
	if side == blas.Right {
		order = n
	}
	name := fmt.Sprintf("%c/%c/%c/m%d_n%d", side, uplo, trans, m, n)
	b.Run(name, func(b *testing.B) {
		a := make([]float64, order*order)
		for i := 0; i < order; i++ {
			a[i*order+i] = 2 + float64(i%7)/16
			for j := 0; j < order; j++ {
				if (uplo == blas.Upper && j > i) || (uplo == blas.Lower && j < i) {
					a[i*order+j] = float64((i+j)%11-5) / float64(32*order)
				}
			}
		}
		want := make([]float64, m*n)
		for i := range want {
			want[i] = float64(i%17-8) / 8
		}
		data := append([]float64(nil), want...)
		b.ReportAllocs()
		b.SetBytes(int64(m * n * 8))
		for b.Loop() {
			impl.Dtrsm(side, uplo, trans, blas.NonUnit, m, n, 0.75, a, order, data, n)
			b.StopTimer()
			copy(data, want)
			b.StartTimer()
		}
	})
}

func BenchmarkDsyrkSizes(b *testing.B) {
	for _, n := range []int{31, 32, 33, 63, 64, 65, 128, 256} {
		for _, k := range []int{16, n} {
			benchmarkDsyrkCase(b, blas.Upper, blas.NoTrans, n, k)
			benchmarkDsyrkCase(b, blas.Lower, blas.NoTrans, n, k)
			benchmarkDsyrkCase(b, blas.Upper, blas.Trans, n, k)
			benchmarkDsyrkCase(b, blas.Lower, blas.Trans, n, k)
		}
	}
}

func benchmarkDsyrkCase(b *testing.B, uplo blas.Uplo, trans blas.Transpose, n, k int) {
	name := fmt.Sprintf("%c/%c/n%d_k%d", uplo, trans, n, k)
	b.Run(name, func(b *testing.B) {
		rows, cols := n, k
		if trans != blas.NoTrans {
			rows, cols = k, n
		}
		a := make([]float64, rows*cols)
		for i := range a {
			a[i] = float64(i%19-9) / 16
		}
		want := make([]float64, n*n)
		for i := range want {
			want[i] = float64(i%13-6) / 8
		}
		data := append([]float64(nil), want...)
		b.ReportAllocs()
		b.SetBytes(int64(n * n * 8))
		for b.Loop() {
			impl.Dsyrk(uplo, trans, n, k, 0.75, a, cols, 0.5, data, n)
			b.StopTimer()
			copy(data, want)
			b.StartTimer()
		}
	})
}
