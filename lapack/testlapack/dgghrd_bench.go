// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
)

func DgghrdBenchmark(b *testing.B, impl Dgghrder) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for _, n := range []int{10, 50, 100, 200, 500} {
		aOrig := make([]float64, n*n)
		bOrig := make([]float64, n*n)
		for i := range aOrig {
			aOrig[i] = rnd.NormFloat64()
		}
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				bOrig[i*n+j] = rnd.NormFloat64()
			}
			bOrig[i*n+i] += float64(n)
		}

		a := make([]float64, n*n)
		bm := make([]float64, n*n)
		q := make([]float64, n*n)
		z := make([]float64, n*n)

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(a, aOrig)
				copy(bm, bOrig)
				b.StartTimer()
				impl.Dgghrd(lapack.OrthoExplicit, lapack.OrthoExplicit, n, 0, n-1,
					a, n, bm, n, q, n, z, n)
			}
		})
	}
}
