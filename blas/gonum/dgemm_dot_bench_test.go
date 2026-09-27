// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gonum

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
)

var dgemmTransposedBSink float64

func BenchmarkDgemmTransposedB(b *testing.B) {
	for _, test := range []struct {
		m, n, k        int
		ldaPad, ldbPad int
		ldcPad         int
	}{
		{m: 1, n: 1, k: 7},
		{m: 1, n: 1, k: 15},
		{m: 1, n: 1, k: 16},
		{m: 1, n: 1, k: 17},
		{m: 1, n: 2, k: 15},
		{m: 1, n: 2, k: 16},
		{m: 1, n: 2, k: 17},
		{m: 1, n: 3, k: 15},
		{m: 1, n: 3, k: 16},
		{m: 1, n: 3, k: 17},
		{m: 1, n: 3, k: 480},
		{m: 4, n: 4, k: 7},
		{m: 4, n: 4, k: 8, ldaPad: 3, ldbPad: 5, ldcPad: 2},
		{m: 8, n: 8, k: 15},
		{m: 8, n: 8, k: 16, ldaPad: 3, ldbPad: 5, ldcPad: 2},
		{m: 8, n: 8, k: 17},
		{m: 32, n: 32, k: 31},
		{m: 32, n: 32, k: 32, ldaPad: 3, ldbPad: 5, ldcPad: 2},
		{m: 32, n: 32, k: 33},
		{m: 32, n: 31, k: 480},
		{m: 32, n: 32, k: 480},
		{m: 32, n: 33, k: 480},
		{m: 96, n: 224, k: 32},
		{m: 224, n: 224, k: 32},
		{m: 224, n: 480, k: 32},
		{m: 224, n: 480, k: 32, ldaPad: 3, ldbPad: 5, ldcPad: 2},
		{m: 96, n: 32, k: 224},
		{m: 224, n: 32, k: 480},
		{m: 224, n: 32, k: 480, ldaPad: 3, ldbPad: 5, ldcPad: 2},
		{m: 128, n: 128, k: 128},
		{m: 128, n: 128, k: 129, ldaPad: 3, ldbPad: 5, ldcPad: 2},
		{m: 256, n: 256, k: 256},
	} {
		name := fmt.Sprintf("m=%d/n=%d/k=%d/ldaPad=%d/ldbPad=%d/ldcPad=%d", test.m, test.n, test.k, test.ldaPad, test.ldbPad, test.ldcPad)
		b.Run(name, func(b *testing.B) {
			benchmarkDgemmTransposedB(b, test.m, test.n, test.k, test.ldaPad, test.ldbPad, test.ldcPad)
		})
	}
}

func benchmarkDgemmTransposedB(b *testing.B, m, n, k, ldaPad, ldbPad, ldcPad int) {
	lda := k + ldaPad
	ldb := k + ldbPad
	ldc := n + ldcPad
	a := make([]float64, (m-1)*lda+k)
	bb := make([]float64, (n-1)*ldb+k)
	c := make([]float64, (m-1)*ldc+n)
	for i := range a {
		a[i] = float64(i%17-8) / 16
	}
	for i := range bb {
		bb[i] = float64(i%13-6) / 12
	}
	var impl Implementation
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		impl.Dgemm(blas.NoTrans, blas.Trans, m, n, k, 1, a, lda, bb, ldb, 0, c, ldc)
	}
	b.StopTimer()
	dgemmTransposedBSink = c[0]
}
