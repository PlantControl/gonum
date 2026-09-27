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

func DhgeqzBenchmark(b *testing.B, impl Dhgeqzer) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for _, n := range []int{10, 50, 100, 200, 500} {
		hOrig := make([]float64, n*n)
		tOrig := make([]float64, n*n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				if i <= j+1 {
					hOrig[i*n+j] = rnd.NormFloat64()
				}
			}
		}
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				tOrig[i*n+j] = rnd.NormFloat64()
			}
			tOrig[i*n+i] += float64(n)
		}

		h := make([]float64, n*n)
		t := make([]float64, n*n)
		alphar := make([]float64, n)
		alphai := make([]float64, n)
		beta := make([]float64, n)
		q := make([]float64, n*n)
		z := make([]float64, n*n)
		workQuery := make([]float64, 1)
		impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess,
			n, 0, n-1, nil, max(1, n), nil, max(1, n), nil, nil, nil,
			nil, max(1, n), nil, max(1, n), workQuery, -1)
		lwork := max(4*n, int(workQuery[0]))
		work := make([]float64, lwork)

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(h, hOrig)
				copy(t, tOrig)
				b.StartTimer()
				impl.Dhgeqz(lapack.EigenvaluesAndSchur, lapack.SchurHess, lapack.SchurHess,
					n, 0, n-1, h, n, t, n, alphar, alphai, beta,
					q, n, z, n, work, len(work))
			}
		})
	}
}
