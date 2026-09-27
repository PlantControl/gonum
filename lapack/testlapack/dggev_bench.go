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

func DggevBenchmark(b *testing.B, impl Dggever) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for _, n := range []int{10, 50, 100, 200, 500} {
		a := make([]float64, n*n)
		aOrig := make([]float64, n*n)
		bm := make([]float64, n*n)
		bOrig := make([]float64, n*n)
		for i := range aOrig {
			aOrig[i] = rnd.NormFloat64()
			bOrig[i] = rnd.NormFloat64()
		}
		for i := 0; i < n; i++ {
			bOrig[i*n+i] += float64(n)
		}

		alphar := make([]float64, n)
		alphai := make([]float64, n)
		beta := make([]float64, n)

		work := make([]float64, 1)
		impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, n,
			nil, n, nil, n, nil, nil, nil, nil, 1, nil, 1, work, -1)
		work = make([]float64, int(work[0]))

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(a, aOrig)
				copy(bm, bOrig)
				b.StartTimer()
				impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, n,
					a, n, bm, n,
					alphar, alphai, beta,
					nil, 1, nil, 1,
					work, len(work))
			}
		})
	}
}

func DggevRightEVBenchmark(b *testing.B, impl Dggever) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for _, n := range []int{10, 50, 100, 200} {
		a := make([]float64, n*n)
		aOrig := make([]float64, n*n)
		bm := make([]float64, n*n)
		bOrig := make([]float64, n*n)
		for i := range aOrig {
			aOrig[i] = rnd.NormFloat64()
			bOrig[i] = rnd.NormFloat64()
		}
		for i := 0; i < n; i++ {
			bOrig[i*n+i] += float64(n)
		}

		alphar := make([]float64, n)
		alphai := make([]float64, n)
		beta := make([]float64, n)
		vr := make([]float64, n*n)

		work := make([]float64, 1)
		impl.Dggev(lapack.LeftEVNone, lapack.RightEVCompute, n,
			nil, n, nil, n, nil, nil, nil, nil, 1, nil, n, work, -1)
		work = make([]float64, int(work[0]))

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(a, aOrig)
				copy(bm, bOrig)
				b.StartTimer()
				impl.Dggev(lapack.LeftEVNone, lapack.RightEVCompute, n,
					a, n, bm, n,
					alphar, alphai, beta,
					nil, 1, vr, n,
					work, len(work))
			}
		})
	}
}

func DggevSingularBenchmark(b *testing.B, impl Dggever) {
	rnd := rand.New(rand.NewPCG(1, 1))
	for _, n := range []int{10, 50, 100, 200} {
		a := make([]float64, n*n)
		aOrig := make([]float64, n*n)
		bm := make([]float64, n*n)
		bOrig := make([]float64, n*n)
		for i := range aOrig {
			aOrig[i] = rnd.NormFloat64()
			bOrig[i] = rnd.NormFloat64()
		}
		// Make B singular by zeroing a row.
		for j := 0; j < n; j++ {
			bOrig[0*n+j] = 0
		}

		alphar := make([]float64, n)
		alphai := make([]float64, n)
		beta := make([]float64, n)

		work := make([]float64, 1)
		impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, n,
			nil, n, nil, n, nil, nil, nil, nil, 1, nil, 1, work, -1)
		work = make([]float64, int(work[0]))

		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				copy(a, aOrig)
				copy(bm, bOrig)
				b.StartTimer()
				impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, n,
					a, n, bm, n,
					alphar, alphai, beta,
					nil, 1, nil, 1,
					work, len(work))
			}
		})
	}
}
