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

func DggbalBenchmark(b *testing.B, impl Dggbaler) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for _, n := range []int{10, 50, 100, 200} {
		aOrig := make([]float64, n*n)
		bOrig := make([]float64, n*n)
		for i := range aOrig {
			aOrig[i] = rnd.NormFloat64()
			bOrig[i] = rnd.NormFloat64()
		}
		a := make([]float64, len(aOrig))
		bm := make([]float64, len(bOrig))
		lscale := make([]float64, n)
		rscale := make([]float64, n)
		work := make([]float64, 6*n)

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(a, aOrig)
				copy(bm, bOrig)
				b.StartTimer()
				impl.Dggbal(lapack.Scale, n, a, n, bm, n, lscale, rscale, work)
			}
		})
	}
}
