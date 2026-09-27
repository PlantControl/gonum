// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDggbakNetlibDifferential(t *testing.T) {
	const (
		n   = 5
		m   = 3
		ilo = 1
		ihi = 3
	)
	lscale := []float64{1, 10, 0.1, 100, 3}
	rscale := []float64{2, 0.5, 2, 4, 0}
	vOrig := []float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
		10, 11, 12,
		13, 14, 15,
	}
	for _, job := range []struct {
		g lapack.BalanceJob
		n byte
	}{
		{g: lapack.BalanceNone, n: 'N'},
		{g: lapack.Permute, n: 'P'},
		{g: lapack.Scale, n: 'S'},
		{g: lapack.PermuteScale, n: 'B'},
	} {
		for _, side := range []struct {
			g blas.Side
			n byte
		}{
			{g: blas.Left, n: 'L'},
			{g: blas.Right, n: 'R'},
		} {
			t.Run(fmt.Sprintf("Job=%c/Side=%c", job.n, side.n), func(t *testing.T) {
				gv := append([]float64(nil), vOrig...)
				nv := append([]float64(nil), vOrig...)
				Implementation{}.Dggbak(job.g, side.g, n, ilo, ihi, lscale, rscale, m, gv, m)
				info := netlib.Dggbak(job.n, side.n, n, ilo, ihi, lscale, rscale, m, nv)
				if info != 0 {
					t.Fatalf("Netlib info=%d", info)
				}
				for i := range gv {
					checkCloseNetlib(t, fmt.Sprintf("V[%d]", i), gv[i], nv[i])
				}
			})
		}
	}
}
