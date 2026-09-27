// Copyright ©2026 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netlib && darwin && cgo

package gonum

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/blas"

	"plantcontrol.org/v1/gonum/lapack/gonum/internal/netlib"
)

func TestDtgsy2NetlibSuccessiveBlocks(t *testing.T) {
	const n = 3
	a := []float64{
		2, 0.1, 0.2,
		0, 1, 2,
		0, -1, 1,
	}
	b := []float64{
		4, 0.2, 0.3,
		0, 3, 1,
		0, -0.5, 3,
	}
	d := []float64{
		1, 0.1, -0.2,
		0, 2, 0.2,
		0, 0, 3,
	}
	e := []float64{
		1.5, -0.1, 0.2,
		0, 2.5, 0.3,
		0, 0, 0.7,
	}
	c := []float64{1, -2, 0.5, 0.7, 1.2, -0.3, -0.8, 0.4, 2}
	f := []float64{-0.5, 1, 0.2, 1.5, -0.7, 0.9, 0.3, -1.1, 0.6}
	for _, tc := range []struct {
		trans blas.Transpose
		ijob  int
	}{
		{trans: blas.NoTrans, ijob: 0},
		{trans: blas.NoTrans, ijob: 1},
		{trans: blas.NoTrans, ijob: 2},
		{trans: blas.Trans, ijob: -3},
		{trans: blas.Trans, ijob: 0},
		{trans: blas.Trans, ijob: 7},
	} {
		t.Run(fmt.Sprintf("Trans=%c/IJob=%d", tc.trans, tc.ijob), func(t *testing.T) {
			gc, gf := append([]float64(nil), c...), append([]float64(nil), f...)
			nc, nf := append([]float64(nil), c...), append([]float64(nil), f...)
			gscale, gsum, gscal, gpq, gok := Implementation{}.Dtgsy2(tc.trans, tc.ijob, n, n,
				a, n, b, n, gc, n, d, n, e, n, gf, n, 1, 0, make([]int, 2*n+2))
			nscale, nsum, nscal, npq, info := netlib.Dtgsy2(byte(tc.trans), tc.ijob, n, n,
				a, b, nc, d, e, nf, 1, 0)
			if gok != (info == 0) || gpq != npq {
				t.Fatalf("Gonum=(pq=%d,ok=%v), Netlib=(pq=%d,info=%d)", gpq, gok, npq, info)
			}
			checkCloseNetlib(t, "scale", gscale, nscale)
			checkCloseNetlib(t, "rdsum", gsum, nsum)
			checkCloseNetlib(t, "rdscal", gscal, nscal)
			for i := range gc {
				checkCloseNetlib(t, fmt.Sprintf("C[%d]", i), gc[i], nc[i])
				checkCloseNetlib(t, fmt.Sprintf("F[%d]", i), gf[i], nf[i])
			}
		})
	}
}

func TestDtgsy2NetlibScaling(t *testing.T) {
	const small = 1e-100
	a, b := []float64{small}, []float64{2 * small}
	d, e := []float64{3 * small}, []float64{5 * small}
	for _, trans := range []blas.Transpose{blas.NoTrans, blas.Trans} {
		t.Run(fmt.Sprintf("Trans=%c", trans), func(t *testing.T) {
			gc, gf := []float64{1e300}, []float64{-1e300}
			nc, nf := append([]float64(nil), gc...), append([]float64(nil), gf...)
			gscale, gsum, gscal, gpq, gok := Implementation{}.Dtgsy2(trans, 0, 1, 1,
				a, 1, b, 1, gc, 1, d, 1, e, 1, gf, 1, 1, 0, make([]int, 4))
			nscale, nsum, nscal, npq, info := netlib.Dtgsy2(byte(trans), 0, 1, 1,
				a, b, nc, d, e, nf, 1, 0)
			if gok != (info == 0) || gpq != npq {
				t.Fatalf("Gonum=(pq=%d,ok=%v), Netlib=(pq=%d,info=%d)", gpq, gok, npq, info)
			}
			if !(gscale < 1) || !(nscale < 1) {
				t.Fatalf("scaling not exercised: Gonum=%g Netlib=%g", gscale, nscale)
			}
			checkCloseNetlib(t, "scale", gscale, nscale)
			checkCloseNetlib(t, "rdsum", gsum, nsum)
			checkCloseNetlib(t, "rdscal", gscal, nscal)
			checkCloseNetlib(t, "C", gc[0], nc[0])
			checkCloseNetlib(t, "F", gf[0], nf[0])
		})
	}
}
