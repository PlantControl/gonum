// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"testing"

	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDtgexcNetlibSplitBlock(t *testing.T) {
	const n = 8
	aOrig := []float64{
		1, 1, 1.1, 1.3, 2, 3, -4.7, 3.3,
		1, 1, 3.7, 7.9, 4, 5.3, 3.3, -0.9,
		0, 0, 2, -3, 3.4, 6.5, 5.2, 1.8,
		0, 0, 4, 2, -5.3, -8.9, -0.2, -0.5,
		0, 0, 0, 0, 4.2, 2, 3.3, 2.3,
		0, 0, 0, 0, 3.7, 4.2, 9.9, 8.8,
		0, 0, 0, 0, 0, 0, 9.9, 8.8,
		0, 0, 0, 0, 0, 0, -9.9, 9.9,
	}
	for _, move := range []struct {
		name       string
		ifst, ilst int
	}{{"Down", 0, 4}, {"Up", 4, 0}} {
		t.Run(move.name, func(t *testing.T) {
			bOrig := identityData(n)
			ga, gb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
			na, nb := append([]float64(nil), aOrig...), append([]float64(nil), bOrig...)
			q, z := identityData(n), identityData(n)
			work := make([]float64, 4*n+16)
			gifst, gilst, gok := Implementation{}.Dtgexc(true, true, n, ga, n, gb, n, q, n, z, n, move.ifst, move.ilst, work, len(work))
			nifst, nilst, info := netlib.Dtgexc(n, na, nb, move.ifst, move.ilst)
			if gok != (info == 0) || gifst != nifst || gilst != nilst {
				t.Fatalf("Gonum=(%d,%d,%v), Netlib=(%d,%d,info=%d)", gifst, gilst, gok, nifst, nilst, info)
			}
			for i := 0; i < n-1; i++ {
				if (ga[(i+1)*n+i] == 0) != (na[(i+1)*n+i] == 0) {
					t.Fatalf("block split differs at %d: Gonum subdiagonal=%g, Netlib subdiagonal=%g", i, ga[(i+1)*n+i], na[(i+1)*n+i])
				}
			}
		})
	}
}
