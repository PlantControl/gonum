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

func TestDtgsylNetlibBlockDifferential(t *testing.T) {
	for _, test := range []struct {
		name       string
		m, n       int
		a, b, d, e []float64
	}{
		{
			name: "1x2",
			m:    1,
			n:    2,
			a:    []float64{2},
			b:    []float64{1, 2, -1, 1},
			d:    []float64{3},
			e:    []float64{2, 0.2, 0, 3},
		},
		{
			name: "2x1",
			m:    2,
			n:    1,
			a:    []float64{1, 2, -1, 1},
			b:    []float64{4},
			d:    []float64{2, 0.2, 0, 3},
			e:    []float64{1.5},
		},
		{
			name: "2x2",
			m:    2,
			n:    2,
			a:    []float64{1, 2, -1, 1},
			b:    []float64{3, 1, -0.5, 3},
			d:    []float64{2, 0.2, 0, 3},
			e:    []float64{1.5, -0.1, 0, 2.5},
		},
		{
			name: "3x2-multiblock",
			m:    3,
			n:    2,
			a: []float64{
				-3, 0.1, 0.2,
				0, 1, 2,
				0, -1, 1,
			},
			b: []float64{4, 0.7, 0, -2},
			d: []float64{
				1, 0.1, -0.2,
				0, 2, 0,
				0, 0, 3,
			},
			e: []float64{1.5, -0.2, 0, 0.8},
		},
	} {
		for _, ijob := range []int{3, 4} {
			t.Run(fmt.Sprintf("%s/ijob=%d", test.name, ijob), func(t *testing.T) {
				gc := make([]float64, test.m*test.n)
				gf := make([]float64, test.m*test.n)
				nc := make([]float64, test.m*test.n)
				nf := make([]float64, test.m*test.n)
				work := make([]float64, 1)
				gscale, gdif, gok := Implementation{}.Dtgsyl(blas.NoTrans, ijob, test.m, test.n,
					test.a, test.m, test.b, test.n, gc, test.n,
					test.d, test.m, test.e, test.n, gf, test.n,
					work, len(work), make([]int, test.m+test.n+6))
				nscale, ndif, info := netlib.Dtgsyl(byte(blas.NoTrans), ijob, test.m, test.n,
					test.a, test.b, nc, test.d, test.e, nf)
				if gok != (info == 0) {
					t.Fatalf("Gonum ok=%v, Netlib info=%d", gok, info)
				}
				checkCloseNetlib(t, "scale", gscale, nscale)
				checkCloseNetlib(t, "dif", gdif, ndif)
			})
		}
	}
}

func TestDtgsylNetlibSolveAndEstimate(t *testing.T) {
	const (
		m = 3
		n = 2
	)
	a := []float64{
		-3, 0.1, 0.2,
		0, 1, 2,
		0, -1, 1,
	}
	b := []float64{4, 0.7, 0, -2}
	d := []float64{
		1, 0.1, -0.2,
		0, 2, 0,
		0, 0, 3,
	}
	e := []float64{1.5, -0.2, 0, 0.8}
	c := []float64{1, -2, 0.5, 0.7, -0.3, 1.2}
	f := []float64{-0.5, 1, 1.5, -0.7, 0.3, -1.1}
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
			work := make([]float64, 2*m*n)
			gscale, gdif, gok := Implementation{}.Dtgsyl(tc.trans, tc.ijob, m, n,
				a, m, b, n, gc, n, d, m, e, n, gf, n,
				work, len(work), make([]int, m+n+6))
			nscale, ndif, info := netlib.Dtgsyl(byte(tc.trans), tc.ijob, m, n,
				a, b, nc, d, e, nf)
			if gok != (info == 0) {
				t.Fatalf("Gonum ok=%v, Netlib info=%d", gok, info)
			}
			checkCloseNetlib(t, "scale", gscale, nscale)
			checkCloseNetlib(t, "dif", gdif, ndif)
			for i := range gc {
				checkCloseNetlib(t, fmt.Sprintf("C[%d]", i), gc[i], nc[i])
				checkCloseNetlib(t, fmt.Sprintf("F[%d]", i), gf[i], nf[i])
			}
		})
	}
}
