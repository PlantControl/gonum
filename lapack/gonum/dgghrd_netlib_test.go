// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDgghrdNetlibDifferential(t *testing.T) {
	const n = 6
	rnd := rand.New(rand.NewPCG(17, 17))
	aOrig := make([]float64, n*n)
	bOrig := make([]float64, n*n)
	for i := range aOrig {
		aOrig[i] = rnd.NormFloat64()
		bOrig[i] = rnd.NormFloat64()
	}
	comps := []struct {
		g lapack.OrthoComp
		n byte
	}{
		{g: lapack.OrthoNone, n: 'N'},
		{g: lapack.OrthoExplicit, n: 'I'},
		{g: lapack.OrthoPostmul, n: 'V'},
	}
	for _, active := range []struct {
		name     string
		ilo, ihi int
	}{
		{name: "Full", ilo: 0, ihi: n - 1},
		{name: "Interior", ilo: 1, ihi: n - 2},
	} {
		for _, compq := range comps {
			for _, compz := range comps {
				name := fmt.Sprintf("%s/Q=%c/Z=%c", active.name, compq.n, compz.n)
				t.Run(name, func(t *testing.T) {
					ga, gb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
					na, nb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
					gq, gz := identityData(n), identityData(n)
					nq, nz := identityData(n), identityData(n)
					Implementation{}.Dgghrd(compq.g, compz.g, n, active.ilo, active.ihi,
						ga, n, gb, n, gq, n, gz, n)
					info := netlib.Dgghrd(compq.n, compz.n, n, active.ilo, active.ihi,
						na, nb, nq, nz)
					if info != 0 {
						t.Fatalf("Netlib info=%d", info)
					}
					for i := range ga {
						checkCloseNetlib(t, fmt.Sprintf("A[%d]", i), ga[i], na[i])
						checkCloseNetlib(t, fmt.Sprintf("B[%d]", i), gb[i], nb[i])
						if compq.g != lapack.OrthoNone {
							checkCloseNetlib(t, fmt.Sprintf("Q[%d]", i), gq[i], nq[i])
						}
						if compz.g != lapack.OrthoNone {
							checkCloseNetlib(t, fmt.Sprintf("Z[%d]", i), gz[i], nz[i])
						}
					}
				})
			}
		}
	}
}
