// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDggbalNetlibDifferential(t *testing.T) {
	const n = 5
	rnd := rand.New(rand.NewPCG(13, 13))
	scaledA := make([]float64, n*n)
	scaledB := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			scale := math.Pow10(4 * (i - j))
			scaledA[i*n+j] = rnd.NormFloat64() * scale
			scaledB[i*n+j] = rnd.NormFloat64() * scale
		}
	}
	fixtures := []struct {
		name string
		a, b []float64
	}{
		{name: "ScaledDense", a: scaledA, b: scaledB},
		{
			name: "IsolatedEnds",
			a: []float64{
				1, 2, 3, 4, 5,
				0, 6, 7, 8, 9,
				0, 10, 11, 12, 13,
				0, 14, 15, 16, 17,
				0, 0, 0, 0, 18,
			},
			b: []float64{
				2, -1, 3, -2, 4,
				0, 5, -3, 6, 1,
				0, 2, 7, -4, 3,
				0, -5, 1, 8, -2,
				0, 0, 0, 0, 9,
			},
		},
	}
	for _, fixture := range fixtures {
		for _, job := range []struct {
			g lapack.BalanceJob
			n byte
		}{
			{g: lapack.BalanceNone, n: 'N'},
			{g: lapack.Permute, n: 'P'},
			{g: lapack.Scale, n: 'S'},
			{g: lapack.PermuteScale, n: 'B'},
		} {
			t.Run(fmt.Sprintf("%s/Job=%c", fixture.name, job.n), func(t *testing.T) {
				ga, gb := append([]float64(nil), fixture.a...), append([]float64(nil), fixture.b...)
				na, nb := append([]float64(nil), fixture.a...), append([]float64(nil), fixture.b...)
				gl, gr := make([]float64, n), make([]float64, n)
				nl, nr := make([]float64, n), make([]float64, n)
				gilo, gihi := Implementation{}.Dggbal(job.g, n, ga, n, gb, n, gl, gr, make([]float64, 6*n))
				nilo, nihi, info := netlib.Dggbal(job.n, n, na, nb, nl, nr)
				if info != 0 || gilo != nilo || gihi != nihi {
					t.Fatalf("Gonum range=(%d,%d), Netlib=(%d,%d,info=%d)", gilo, gihi, nilo, nihi, info)
				}
				for i := range ga {
					checkCloseNetlib(t, fmt.Sprintf("A[%d]", i), ga[i], na[i])
					checkCloseNetlib(t, fmt.Sprintf("B[%d]", i), gb[i], nb[i])
				}
				for i := range gl {
					checkCloseNetlib(t, fmt.Sprintf("lscale[%d]", i), gl[i], nl[i])
					checkCloseNetlib(t, fmt.Sprintf("rscale[%d]", i), gr[i], nr[i])
				}
			})
		}
	}
}
