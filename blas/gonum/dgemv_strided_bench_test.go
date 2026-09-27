// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

var dgemvStridedOutputSink float64

func BenchmarkDgemvStridedOutput(b *testing.B) {
	tests := []struct {
		m, n, ldaPad int
		incX, incY   int
	}{
		{m: 31, n: 8, incX: 1, incY: 2},
		{m: 32, n: 8, incX: 1, incY: 2},
		{m: 33, n: 8, incX: 1, incY: 2},
		{m: 31, n: 32, incX: 1, incY: 32},
		{m: 32, n: 32, incX: 1, incY: 32},
		{m: 33, n: 32, incX: 1, incY: 32},
		{m: 32, n: 32, ldaPad: 3, incX: 1, incY: 32},
		{m: 64, n: 8, incX: 1, incY: 2},
		{m: 64, n: 32, incX: 1, incY: 32},
		{m: 128, n: 32, incX: 1, incY: 32},
		{m: 128, n: 224, incX: 1, incY: 2},
		{m: 224, n: 32, incX: 1, incY: 64},
		{m: 224, n: 224, incX: 1, incY: 32},
		{m: 224, n: 224, ldaPad: 3, incX: 1, incY: 32},
		{m: 255, n: 255, incX: 1, incY: 32},
		{m: 255, n: 255, ldaPad: 3, incX: 1, incY: 32},
		{m: 256, n: 255, incX: 1, incY: 32},
		{m: 512, n: 255, incX: 1, incY: 64},
		{m: 32, n: 32, incX: 2, incY: 32},
		{m: 64, n: 32, incX: 2, incY: 32},
		{m: 224, n: 255, incX: 2, incY: 32},
		{m: 32, n: 32, incX: 1, incY: 1},
		{m: 64, n: 32, incX: 1, incY: 1},
		{m: 224, n: 224, incX: 1, incY: 1},
		{m: 512, n: 255, incX: 1, incY: 1},
	}
	for _, test := range tests {
		name := fmt.Sprintf("m=%d/n=%d/ldaPad=%d/incX=%d/incY=%d", test.m, test.n, test.ldaPad, test.incX, test.incY)
		b.Run(name, func(b *testing.B) {
			benchmarkDgemvStridedOutput(b, test.m, test.n, test.ldaPad, test.incX, test.incY)
		})
	}
}

func benchmarkDgemvStridedOutput(b *testing.B, m, n, ldaPad, incX, incY int) {
	lda := n + ldaPad
	a := make([]float64, (m-1)*lda+n)
	x := make([]float64, (n-1)*incX+1)
	y := make([]float64, (m-1)*incY+1)
	for i := range a {
		a[i] = float64(i%17-8) / 16
	}
	for i := range x {
		x[i] = 1 + float64(i%13)/16
	}
	for i := range y {
		y[i] = float64(i%11-5) / 8
	}
	var impl Implementation
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		impl.Dgemv(blas.NoTrans, m, n, 1, a, lda, x, incX, 0, y, incY)
	}
	b.StopTimer()
	dgemvStridedOutputSink = y[(m-1)*incY]
}
