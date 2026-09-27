// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlapack

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
)

// DlasrBenchmark measures all rotation layouts and application orders. Matrix
// contents are not restored because the rotations are normalized.
func DlasrBenchmark(b *testing.B, impl Dlasrer) {
	shapes := []struct {
		m, n, padding int
	}{
		{1, 1, 0},
		{1, 7, 3},
		{7, 1, 3},
		{2, 7, 0},
		{7, 2, 3},
		{15, 17, 0},
		{16, 16, 3},
		{17, 15, 3},
		{31, 31, 0},
		{32, 15, 0},
		{32, 16, 3},
		{32, 17, 3},
		{32, 32, 3},
		{33, 33, 0},
		{33, 33, 3},
		{64, 31, 0},
		{64, 32, 3},
		{64, 33, 0},
		{64, 33, 3},
		{63, 65, 0},
		{63, 65, 3},
		{64, 64, 0},
		{64, 64, 3},
		{65, 63, 0},
		{65, 63, 3},
		{65, 65, 0},
		{65, 65, 3},
		{79, 65, 0},
		{79, 65, 3},
		{80, 65, 0},
		{80, 65, 3},
		{81, 65, 0},
		{81, 65, 3},
		{64, 256, 0},
		{64, 256, 3},
		{256, 64, 0},
		{256, 64, 3},
		{256, 256, 0},
		{256, 256, 3},
	}
	for _, side := range []blas.Side{blas.Left, blas.Right} {
		for _, pivot := range []lapack.Pivot{lapack.Variable, lapack.Top, lapack.Bottom} {
			for _, direct := range []lapack.Direct{lapack.Forward, lapack.Backward} {
				for _, shape := range shapes {
					patterns := []string{"dense"}
					if side == blas.Right && pivot == lapack.Variable && shape.m >= 64 && shape.n >= 64 {
						patterns = append(patterns, "sparse", "identity")
					}
					for _, pattern := range patterns {
						name := fmt.Sprintf("side=%c/pivot=%c/direct=%c/pattern=%s/m=%d/n=%d/lda=%d", side, pivot, direct, pattern, shape.m, shape.n, shape.n+shape.padding)
						b.Run(name, func(b *testing.B) {
							rnd := rand.New(rand.NewPCG(1, 1))
							lda := shape.n + shape.padding
							a := make([]float64, shape.m*lda)
							for i := range a {
								a[i] = rnd.NormFloat64()
							}
							nrot := shape.m - 1
							if side == blas.Right {
								nrot = shape.n - 1
							}
							c := make([]float64, nrot)
							s := make([]float64, nrot)
							for i := range c {
								c[i] = 1
								if pattern == "identity" || pattern == "sparse" && i%8 != 0 {
									continue
								}
								theta := float64(i+1) * math.Pi / float64(nrot+1)
								c[i], s[i] = math.Cos(theta), math.Sin(theta)
							}
							b.ReportAllocs()
							b.ResetTimer()
							for i := 0; i < b.N; i++ {
								impl.Dlasr(side, pivot, direct, shape.m, shape.n, c, s, a, lda)
							}
							b.StopTimer()
							allZero := true
							for i := 0; i < shape.m; i++ {
								for _, v := range a[i*lda : i*lda+shape.n] {
									if math.IsNaN(v) || math.IsInf(v, 0) {
										b.Fatal("non-finite matrix result")
									}
									allZero = allZero && v == 0
								}
							}
							if allZero {
								b.Fatal("zero matrix result")
							}
						})
					}
				}
			}
		}
	}
}

func DlasrVariableBenchmark(b *testing.B, impl Dlasrer) {
	for _, direct := range []lapack.Direct{lapack.Forward, lapack.Backward} {
		for _, tc := range []struct {
			m, n, padding int
			pattern       string
		}{
			{2, 15, 0, "dense"}, {2, 16, 0, "dense"}, {2, 17, 3, "dense"},
			{2, 31, 0, "dense"}, {2, 32, 0, "dense"}, {2, 33, 3, "dense"},
			{4, 16, 0, "dense"}, {4, 17, 3, "sparse"},
			{32, 16, 0, "dense"}, {32, 32, 3, "dense"}, {32, 256, 0, "dense"},
			{32, 256, 3, "sparse"}, {32, 256, 3, "identity"},
		} {
			b.Run(fmt.Sprintf("direct=%c/pattern=%s/m=%d/n=%d/lda=%d", direct, tc.pattern, tc.m, tc.n, tc.n+tc.padding), func(b *testing.B) {
				rnd := rand.New(rand.NewPCG(2, 1))
				lda := tc.n + tc.padding
				a := make([]float64, tc.m*lda)
				for i := range a {
					a[i] = rnd.Float64() - 0.5
				}
				c, s := make([]float64, tc.m-1), make([]float64, tc.m-1)
				for i := range c {
					c[i] = 1
					if tc.pattern == "identity" || tc.pattern == "sparse" && i != len(c)/2 {
						continue
					}
					theta := float64(i+1) * math.Pi / float64(2*len(c)+1)
					c[i], s[i] = math.Cos(theta), math.Sin(theta)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					impl.Dlasr(blas.Left, lapack.Variable, direct, tc.m, tc.n, c, s, a, lda)
				}
				b.StopTimer()
				for i := 0; i < tc.m; i++ {
					for _, v := range a[i*lda : i*lda+tc.n] {
						if math.IsNaN(v) || math.IsInf(v, 0) {
							b.Fatal("non-finite matrix result")
						}
					}
				}
			})
		}
	}
}
